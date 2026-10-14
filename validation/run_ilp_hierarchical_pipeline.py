#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

load_ilp_inputs = importlib.import_module("ilp.data_loader").load_ilp_inputs
ILPConfig = importlib.import_module("ilp.model_builder").ILPConfig
solve_partition_ilp = importlib.import_module("ilp.solve").solve_partition_ilp
refine_solution_hierarchical_local = importlib.import_module("ilp.solve").refine_solution_hierarchical_local
save_ilp_solution = importlib.import_module("ilp.export_solution").save_ilp_solution


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
        if c:
            vals.append(float(c))
    if not vals:
        raise ValueError("At least one GPU budget must be provided")
    return vals


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run robust profiled-backward + hierarchical-refinement ILP pipeline and full simulation evaluation"
    )
    parser.add_argument("--config_dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--gpu_budgets_mb", required=True)
    parser.add_argument("--cpu_mem_budget_mb", type=float, default=1e18)
    parser.add_argument("--k_sigma", type=float, default=1.0)
    parser.add_argument("--k_sigma_time", type=float, default=None)
    parser.add_argument("--k_sigma_energy", type=float, default=None)
    parser.add_argument("--w_time", type=float, default=1.0)
    parser.add_argument("--w_energy", type=float, default=1.0)
    parser.add_argument("--w_transfer", type=float, default=1.0)
    parser.add_argument("--w_fragmentation", type=float, default=0.0)
    parser.add_argument("--w_congestion", type=float, default=0.0)
    parser.add_argument("--congestion_knee_ms", type=float, default=0.0)
    parser.add_argument("--backend", choices=["auto", "pulp", "exhaustive"], default="auto")
    parser.add_argument("--local_refine_budget", type=int, default=0)
    parser.add_argument("--backward_meta_model_json", default=None, help="Deprecated and ignored. Backward costs now come from profiling artifacts.")
    parser.add_argument("--meta_validation_ratio", type=float, default=0.25, help="Deprecated and ignored.")
    parser.add_argument("--meta_ridge_lambda", type=float, default=1e-6, help="Deprecated and ignored.")
    parser.add_argument("--meta_seed", type=int, default=42, help="Deprecated and ignored.")
    parser.add_argument("--meta_blend", type=float, default=1.0, help="Deprecated and ignored.")
    parser.add_argument("--no_topology_check", action="store_true", help="Skip real topological peak-memory validation stage")
    parser.add_argument("--output_dir", default=None)

    args = parser.parse_args()

    load_execution_plan = None
    PreparedTopologyMemory = None
    if not args.no_topology_check:
        load_execution_plan = importlib.import_module("runtime.plan_representation").load_execution_plan
        PreparedTopologyMemory = importlib.import_module("ilp.topology_memory").PreparedTopologyMemory

    config_dir = Path(args.config_dir)
    if not config_dir.exists():
        raise FileNotFoundError(f"config_dir does not exist: {config_dir}")

    budgets = _parse_budget_list(args.gpu_budgets_mb)
    output_dir = Path(args.output_dir) if args.output_dir else (config_dir / "ilp_hierarchical_pipeline")
    output_dir.mkdir(parents=True, exist_ok=True)

    stats_csv, graph_csv, transfer_csv = _default_paths(config_dir, args.model)

    if args.backward_meta_model_json:
        print(
            "[deprecated] Ignoring --backward_meta_model_json/--meta_*; "
            "backward costs now come directly from profiling artifacts."
        )

    rows: List[Dict[str, Any]] = []

    for budget in budgets:
        data = load_ilp_inputs(
            metrics_stats_csv=str(stats_csv),
            graph_edges_csv=str(graph_csv),
            transfer_edges_csv=str(transfer_csv),
            k_sigma=args.k_sigma,
            k_sigma_time=args.k_sigma_time,
            k_sigma_energy=args.k_sigma_energy,
        )
        cfg = ILPConfig(
            w_time=args.w_time,
            w_energy=args.w_energy,
            w_transfer=args.w_transfer,
            w_fragmentation=args.w_fragmentation,
            w_congestion=args.w_congestion,
            congestion_knee_ms=args.congestion_knee_ms,
            gpu_mem_budget_mb=budget,
            cpu_mem_budget_mb=args.cpu_mem_budget_mb,
        )

        base_sol = solve_partition_ilp(data, cfg, backend=args.backend)
        final_sol = base_sol
        if args.local_refine_budget > 0:
            final_sol = refine_solution_hierarchical_local(
                data=data,
                cfg=cfg,
                base_solution=base_sol,
                max_assignment_changes=args.local_refine_budget,
            )

        budget_tag = str(int(budget)) if float(budget).is_integer() else str(budget).replace(".", "p")
        budget_dir = output_dir / f"budget_{budget_tag}"
        exported = save_ilp_solution(final_sol, str(budget_dir / "ilp_solution"))

        peak_gpu_mb = None
        peak_cpu_mb = None
        topology_summary = None
        if not args.no_topology_check:
            plan = load_execution_plan(
                assignment_csv=exported["assignment_csv"],
                cut_edges_csv=exported["cut_edges_csv"],
            )
            prepared = PreparedTopologyMemory.prepare(graph_csv, stats_csv)
            peak_gpu_mb, peak_cpu_mb = prepared.evaluate(
                forward_assignment=plan.assignment_forward,
                backward_assignment=plan.assignment_backward,
                activation_strategy=getattr(plan, "activation_strategies", None),
            )

            topology_summary = budget_dir / "topology_memory_summary.json"
            topology_summary.parent.mkdir(parents=True, exist_ok=True)
            topology_summary.write_text(
                json.dumps({"peak_gpu_mem_mb": peak_gpu_mb, "peak_cpu_mem_mb": peak_cpu_mb}, indent=4),
                encoding="utf-8",
            )

        rows.append(
            {
                "model": args.model,
                "gpu_budget_mb": budget,
                "base_backend": base_sol.backend,
                "final_backend": final_sol.backend,
                "base_objective": base_sol.objective_value,
                "final_objective": final_sol.objective_value,
                "delta_objective": float(final_sol.objective_value - base_sol.objective_value),
                "topology_peak_gpu_mem_mb": peak_gpu_mb if peak_gpu_mb is not None else float("nan"),
                "topology_peak_cpu_mem_mb": peak_cpu_mb if peak_cpu_mb is not None else float("nan"),
                "topology_feasible": (
                    (peak_gpu_mb <= budget and peak_cpu_mb <= args.cpu_mem_budget_mb)
                    if peak_gpu_mb is not None
                    else None
                ),
                "solution_dir": str((budget_dir / "ilp_solution")),
                "topology_memory_summary_json": str(topology_summary) if topology_summary is not None else "",
            }
        )

    df = pd.DataFrame(rows).sort_values(by=["gpu_budget_mb"], kind="stable")
    out_csv = output_dir / f"{args.model}_hierarchical_pipeline_results.csv"
    df.to_csv(out_csv, index=False)

    summary = {
        "model": args.model,
        "rows": int(len(df)),
        "output_csv": str(out_csv),
        "backward_cost_source": "profiled",
        "local_refine_budget": int(args.local_refine_budget),
        "w_fragmentation": float(args.w_fragmentation),
        "w_congestion": float(args.w_congestion),
        "congestion_knee_ms": float(args.congestion_knee_ms),
    }
    out_json = output_dir / f"{args.model}_hierarchical_pipeline_summary.json"
    out_json.write_text(json.dumps(summary, indent=4), encoding="utf-8")

    print("=" * 80)
    print("ILP HIERARCHICAL PIPELINE")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Rows: {len(df)}")
    print("Backward costs: profiled")
    print(f"Results CSV: {out_csv}")
    print(f"Summary JSON: {out_json}")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
