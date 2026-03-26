#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

load_ilp_inputs = importlib.import_module("ilp.data_loader").load_ilp_inputs
merge_ilp_inputs_multi_hardware = importlib.import_module("ilp.data_loader").merge_ilp_inputs_multi_hardware
ILPInputData = importlib.import_module("ilp.data_loader").ILPInputData
ILPConfig = importlib.import_module("ilp.model_builder").ILPConfig
solve_partition_ilp = importlib.import_module("ilp.solve").solve_partition_ilp


def _default_paths(config_dir: Path, model_name: str):
    stats = config_dir / f"{model_name}_metrics_stats.csv"
    if not stats.exists():
        stats = config_dir / "metrics_stats.csv"

    run_dirs = sorted([p for p in config_dir.glob("run_*") if p.is_dir()])
    if run_dirs:
        ref_run = run_dirs[0]
        graph_edges = ref_run / f"{model_name}_graph_edges.csv"
        transfer_edges = ref_run / f"{model_name}_transfer_edges.csv"
    else:
        graph_edges = config_dir / f"{model_name}_graph_edges.csv"
        transfer_edges = config_dir / f"{model_name}_transfer_edges.csv"

    if not graph_edges.exists() or not transfer_edges.exists():
        raise FileNotFoundError(
            "Could not resolve graph/transfer artifacts in config_dir. "
            f"Expected either run_*/ files or direct files in: {config_dir}"
        )
    return stats, graph_edges, transfer_edges


def _parse_budget_list(text: str) -> List[float]:
    vals = []
    for chunk in text.split(","):
        c = chunk.strip()
        if not c:
            continue
        vals.append(float(c))
    if not vals:
        raise ValueError("At least one GPU budget must be provided")
    return vals


def _load_profile(
    cdir: Path,
    model: str,
    k_sigma: float,
    strict_graph_mapping: bool,
    strict_transfer_mapping: bool,
    strict_sample_quality: bool,
    strict_transfer_calibration: bool,
    strict_graph_trace_source: bool,
):
    stats_csv, graph_csv, transfer_csv = _default_paths(cdir, model)
    return load_ilp_inputs(
        metrics_stats_csv=str(stats_csv),
        graph_edges_csv=str(graph_csv),
        transfer_edges_csv=str(transfer_csv),
        k_sigma=k_sigma,
        strict_graph_mapping=strict_graph_mapping,
        strict_transfer_mapping=strict_transfer_mapping,
        strict_sample_quality=strict_sample_quality,
        strict_transfer_calibration=strict_transfer_calibration,
        strict_graph_trace_source=strict_graph_trace_source,
    )


def _load_data(
    config_dirs: List[Path],
    model: str,
    k_sigma: float,
    strict_graph_mapping: bool,
    strict_transfer_mapping: bool,
    strict_sample_quality: bool,
    strict_transfer_calibration: bool,
    strict_graph_trace_source: bool,
    hw_aggregate: str,
    hw_dispersion_k: float,
):
    profiles = [
        _load_profile(
            cdir=cdir,
            model=model,
            k_sigma=k_sigma,
            strict_graph_mapping=strict_graph_mapping,
            strict_transfer_mapping=strict_transfer_mapping,
            strict_sample_quality=strict_sample_quality,
            strict_transfer_calibration=strict_transfer_calibration,
            strict_graph_trace_source=strict_graph_trace_source,
        )
        for cdir in config_dirs
    ]

    if len(profiles) == 1:
        return profiles[0]

    return merge_ilp_inputs_multi_hardware(
        profiles=profiles,
        strategy=hw_aggregate,
        dispersion_k=hw_dispersion_k,
        strict_schema=True,
    )


def _clone_without_topology(data: Any):
    return ILPInputData(
        nodes=list(data.nodes),
        node_cost_gpu_ms=dict(data.node_cost_gpu_ms),
        node_cost_cpu_ms=dict(data.node_cost_cpu_ms),
        node_energy_gpu_j=dict(data.node_energy_gpu_j),
        node_energy_cpu_j=dict(data.node_energy_cpu_j),
        node_mem_gpu_mb=dict(data.node_mem_gpu_mb),
        node_mem_cpu_mb=dict(data.node_mem_cpu_mb),
        edges=[],
        edge_transfer_ms={},
    )


def _clone_without_transfer_edges(data: Any):
    return ILPInputData(
        nodes=list(data.nodes),
        node_cost_gpu_ms=dict(data.node_cost_gpu_ms),
        node_cost_cpu_ms=dict(data.node_cost_cpu_ms),
        node_energy_gpu_j=dict(data.node_energy_gpu_j),
        node_energy_cpu_j=dict(data.node_energy_cpu_j),
        node_mem_gpu_mb=dict(data.node_mem_gpu_mb),
        node_mem_cpu_mb=dict(data.node_mem_cpu_mb),
        edges=list(data.edges),
        edge_transfer_ms={e: 0.0 for e in data.edges},
    )


def _solve_variant(data: Any, cfg: Any, backend: str) -> Dict[str, Any]:
    sol = solve_partition_ilp(data, cfg, backend=backend)
    return {
        "status": sol.status,
        "backend": sol.backend,
        "objective": sol.objective_value,
        "gpu_mem_mb": sol.gpu_mem_used_mb,
        "cpu_mem_mb": sol.cpu_mem_used_mb,
        "layers_gpu": sum(1 for _, d in sol.assignment.items() if d == "GPU"),
        "layers_cpu": sum(1 for _, d in sol.assignment.items() if d == "CPU"),
        "cut_edges": len(sol.cut_edges),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run ILP ablation suite for one model and multiple GPU budgets")
    parser.add_argument("--config_dir", default=None)
    parser.add_argument("--config_dirs", default=None, help="Comma-separated batch directories for multi-hardware aggregation")
    parser.add_argument("--model", required=True)
    parser.add_argument("--gpu_budgets_mb", required=True, help="Comma-separated budgets, e.g. 400,600,800,1000")
    parser.add_argument("--cpu_mem_budget_mb", type=float, default=1e18)
    parser.add_argument("--k_sigma", type=float, default=1.0)
    parser.add_argument("--strict_graph_mapping", action="store_true")
    parser.add_argument("--strict_transfer_mapping", action="store_true")
    parser.add_argument("--allow_low_quality_stats", action="store_true")
    parser.add_argument("--allow_transfer_calibration_fallback", action="store_true")
    parser.add_argument("--allow_fallback_graph_trace", action="store_true")
    parser.add_argument("--w_time", type=float, default=1.0)
    parser.add_argument("--w_energy", type=float, default=0.0)
    parser.add_argument("--w_transfer", type=float, default=1.0)
    parser.add_argument("--backend", choices=["auto", "pulp", "exhaustive"], default="auto")
    parser.add_argument("--hw_aggregate", choices=["max", "mean"], default="max")
    parser.add_argument("--hw_dispersion_k", type=float, default=0.0)
    parser.add_argument("--output_csv", default=None)
    parser.add_argument("--output_json", default=None)
    args = parser.parse_args()

    config_dirs: List[Path]
    if args.config_dirs:
        config_dirs = [Path(p.strip()) for p in args.config_dirs.split(",") if p.strip()]
        if not config_dirs:
            raise ValueError("--config_dirs was provided but no valid paths were found")
    else:
        if not args.config_dir:
            raise ValueError("Provide --config_dir or --config_dirs")
        config_dirs = [Path(args.config_dir)]

    for cdir in config_dirs:
        if not cdir.exists():
            raise FileNotFoundError(f"config_dir does not exist: {cdir}")

    base_data = _load_data(
        config_dirs=config_dirs,
        model=args.model,
        k_sigma=args.k_sigma,
        strict_graph_mapping=args.strict_graph_mapping,
        strict_transfer_mapping=args.strict_transfer_mapping,
        strict_sample_quality=not args.allow_low_quality_stats,
        strict_transfer_calibration=not args.allow_transfer_calibration_fallback,
        strict_graph_trace_source=not args.allow_fallback_graph_trace,
        hw_aggregate=args.hw_aggregate,
        hw_dispersion_k=args.hw_dispersion_k,
    )
    no_robust_data = _load_data(
        config_dirs=config_dirs,
        model=args.model,
        k_sigma=0.0,
        strict_graph_mapping=args.strict_graph_mapping,
        strict_transfer_mapping=args.strict_transfer_mapping,
        strict_sample_quality=not args.allow_low_quality_stats,
        strict_transfer_calibration=not args.allow_transfer_calibration_fallback,
        strict_graph_trace_source=not args.allow_fallback_graph_trace,
        hw_aggregate=args.hw_aggregate,
        hw_dispersion_k=args.hw_dispersion_k,
    )

    budgets = _parse_budget_list(args.gpu_budgets_mb)

    variant_data = {
        "full_model": base_data,
        "no_topology": _clone_without_topology(base_data),
        "no_transfer_edges": _clone_without_transfer_edges(base_data),
        "no_robustification": no_robust_data,
    }

    rows: List[Dict[str, Any]] = []

    for b in budgets:
        cfg = ILPConfig(
            w_time=args.w_time,
            w_energy=args.w_energy,
            w_transfer=args.w_transfer,
            gpu_mem_budget_mb=b,
            cpu_mem_budget_mb=args.cpu_mem_budget_mb,
        )

        full_result = _solve_variant(variant_data["full_model"], cfg, backend=args.backend)
        for variant_name, vdata in variant_data.items():
            result = _solve_variant(vdata, cfg, backend=args.backend)
            row = {
                "model": args.model,
                "variant": variant_name,
                "gpu_budget_mb": b,
                "cpu_budget_mb": args.cpu_mem_budget_mb,
                "k_sigma": args.k_sigma,
                "w_time": args.w_time,
                "w_energy": args.w_energy,
                "w_transfer": args.w_transfer,
                "ilp_status": result["status"],
                "backend": result["backend"],
                "ilp_objective": result["objective"],
                "ilp_gpu_mem_mb": result["gpu_mem_mb"],
                "ilp_cpu_mem_mb": result["cpu_mem_mb"],
                "ilp_layers_gpu": result["layers_gpu"],
                "ilp_layers_cpu": result["layers_cpu"],
                "ilp_cut_edges": result["cut_edges"],
                "full_model_objective": full_result["objective"],
            }
            if full_result["status"] in {"optimal", "feasible"} and result["status"] in {"optimal", "feasible"}:
                row["delta_vs_full_obj"] = float(result["objective"]) - float(full_result["objective"])
            else:
                row["delta_vs_full_obj"] = float("nan")
            rows.append(row)

    out_df = pd.DataFrame(rows).sort_values(by=["model", "variant", "gpu_budget_mb"], kind="stable")

    base_config_dir = config_dirs[0]
    out_csv = Path(args.output_csv) if args.output_csv else (base_config_dir / f"{args.model}_ablation_suite.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)

    summary = {
        "model": args.model,
        "rows": int(len(out_df)),
        "variants": list(variant_data.keys()),
        "gpu_budgets_mb": budgets,
        "output_csv": str(out_csv),
    }

    out_json = Path(args.output_json) if args.output_json else (base_config_dir / f"{args.model}_ablation_summary.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=4)

    print("=" * 80)
    print("ILP ABLATION SUITE")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Rows: {len(out_df)}")
    print(f"Variants: {', '.join(variant_data.keys())}")
    print(f"CSV: {out_csv}")
    print(f"JSON: {out_json}")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
