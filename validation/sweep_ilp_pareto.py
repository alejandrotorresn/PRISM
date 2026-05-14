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
merge_ilp_inputs_multi_hardware = importlib.import_module("ilp.data_loader").merge_ilp_inputs_multi_hardware
ILPConfig = importlib.import_module("ilp.model_builder").ILPConfig
ILPProblemData = importlib.import_module("ilp.model_builder").ILPProblemData
build_problem_data_dual = importlib.import_module("ilp.model_builder").build_problem_data_dual
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


def _build_effective_from_dual(data, cfg) -> Any:
    """Build baselines with the same dual objective used by solve_partition_ilp.

    The ILP solver accounts for forward and backward costs independently
    (build_problem_data_dual). This helper collapses them into per-node
    effective costs (fwd + bwd) so that greedy / all-cpu / all-gpu baselines
    are evaluated in the same objective space, enabling apples-to-apples
    comparison in Phase 1 Pareto reports.
    """
    dual = build_problem_data_dual(data, cfg)
    return ILPProblemData(
        objective_node_gpu={
            n: dual.objective_fwd_gpu[n] + dual.objective_bwd_gpu[n]
            for n in data.nodes
        },
        objective_node_cpu={
            n: dual.objective_fwd_cpu[n] + dual.objective_bwd_cpu[n]
            for n in data.nodes
        },
        objective_edge_cut={
            e: dual.objective_edge_cut_forward[e] + dual.objective_edge_cut_backward[e]
            for e in data.edges
        },
        gpu_mem=dual.gpu_mem,
        cpu_mem=dual.cpu_mem,
    )


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


def _eval_fixed_policy(policy: str, data, cfg: Any) -> Dict[str, float | str | int]:
    problem = _build_effective_from_dual(data, cfg)

    if policy == "all_gpu":
        bits = {n: 1 for n in data.nodes}
    elif policy == "all_cpu":
        bits = {n: 0 for n in data.nodes}
    else:
        raise ValueError(f"Unsupported policy: {policy}")

    gpu_mem = sum(problem.gpu_mem[n] for n in data.nodes if bits[n] == 1)
    cpu_mem = sum(problem.cpu_mem[n] for n in data.nodes if bits[n] == 0)
    feasible = (gpu_mem <= cfg.gpu_mem_budget_mb) and (cpu_mem <= cfg.cpu_mem_budget_mb)

    obj = 0.0
    for n in data.nodes:
        obj += problem.objective_node_gpu[n] if bits[n] == 1 else problem.objective_node_cpu[n]

    # No boundary cuts for homogeneous policies.
    cut_edges = 0

    return {
        "status": "feasible" if feasible else "infeasible",
        "objective_value": float(obj) if feasible else float("inf"),
        "gpu_mem_used_mb": float(gpu_mem),
        "cpu_mem_used_mb": float(cpu_mem),
        "layers_gpu": int(sum(1 for v in bits.values() if v == 1)),
        "layers_cpu": int(sum(1 for v in bits.values() if v == 0)),
        "cut_edges": cut_edges,
    }


def _eval_bits_policy(bits: Dict[str, int], data, cfg: Any) -> Dict[str, float | str | int]:
    problem = _build_effective_from_dual(data, cfg)

    gpu_mem = sum(problem.gpu_mem[n] for n in data.nodes if bits[n] == 1)
    cpu_mem = sum(problem.cpu_mem[n] for n in data.nodes if bits[n] == 0)
    feasible = (gpu_mem <= cfg.gpu_mem_budget_mb) and (cpu_mem <= cfg.cpu_mem_budget_mb)

    obj = 0.0
    for n in data.nodes:
        obj += problem.objective_node_gpu[n] if bits[n] == 1 else problem.objective_node_cpu[n]

    cut_edges = 0
    for e in data.edges:
        u, v = e
        if bits[u] != bits[v]:
            obj += problem.objective_edge_cut[e]
            cut_edges += 1

    return {
        "status": "feasible" if feasible else "infeasible",
        "objective_value": float(obj) if feasible else float("inf"),
        "gpu_mem_used_mb": float(gpu_mem),
        "cpu_mem_used_mb": float(cpu_mem),
        "layers_gpu": int(sum(1 for v in bits.values() if v == 1)),
        "layers_cpu": int(sum(1 for v in bits.values() if v == 0)),
        "cut_edges": cut_edges,
    }


def _greedy_bits(data, cfg: Any) -> Dict[str, int] | None:
    problem = _build_effective_from_dual(data, cfg)

    # Start from per-layer local best (ignores graph cuts by design).
    bits = {
        n: 1 if problem.objective_node_gpu[n] <= problem.objective_node_cpu[n] else 0
        for n in data.nodes
    }

    def _gpu_mem() -> float:
        return float(sum(problem.gpu_mem[n] for n in data.nodes if bits[n] == 1))

    def _cpu_mem() -> float:
        return float(sum(problem.cpu_mem[n] for n in data.nodes if bits[n] == 0))

    gpu_mem = _gpu_mem()
    cpu_mem = _cpu_mem()

    while gpu_mem > cfg.gpu_mem_budget_mb:
        candidates = []
        for n in data.nodes:
            if bits[n] != 1:
                continue
            new_cpu_mem = cpu_mem + problem.cpu_mem[n]
            if new_cpu_mem > cfg.cpu_mem_budget_mb:
                continue
            obj_penalty = max(0.0, problem.objective_node_cpu[n] - problem.objective_node_gpu[n])
            gpu_reduction = max(problem.gpu_mem[n], 1e-12)
            score = obj_penalty / gpu_reduction
            candidates.append((score, n))
        if not candidates:
            return None
        _, pick = min(candidates)
        bits[pick] = 0
        gpu_mem -= problem.gpu_mem[pick]
        cpu_mem += problem.cpu_mem[pick]

    while cpu_mem > cfg.cpu_mem_budget_mb:
        candidates = []
        for n in data.nodes:
            if bits[n] != 0:
                continue
            new_gpu_mem = gpu_mem + problem.gpu_mem[n]
            if new_gpu_mem > cfg.gpu_mem_budget_mb:
                continue
            obj_penalty = max(0.0, problem.objective_node_gpu[n] - problem.objective_node_cpu[n])
            cpu_reduction = max(problem.cpu_mem[n], 1e-12)
            score = obj_penalty / cpu_reduction
            candidates.append((score, n))
        if not candidates:
            return None
        _, pick = min(candidates)
        bits[pick] = 1
        gpu_mem += problem.gpu_mem[pick]
        cpu_mem -= problem.cpu_mem[pick]

    return bits


def _eval_greedy_policy(data, cfg: Any) -> Dict[str, float | str | int]:
    bits = _greedy_bits(data, cfg)
    if bits is None:
        return {
            "status": "infeasible",
            "objective_value": float("inf"),
            "gpu_mem_used_mb": float("nan"),
            "cpu_mem_used_mb": float("nan"),
            "layers_gpu": 0,
            "layers_cpu": 0,
            "cut_edges": 0,
        }
    return _eval_bits_policy(bits, data, cfg)


def _phase_assignment_counts(ilp) -> Dict[str, int]:
    forward_assignment = getattr(ilp, "forward_assignment", None) or ilp.assignment
    backward_assignment = getattr(ilp, "backward_assignment", None) or ilp.assignment
    backward_cut_edges = getattr(ilp, "backward_cut_edges", None) or []
    cross_phase_edges = getattr(ilp, "cross_phase_edges", None) or []

    return {
        "ilp_layers_gpu_forward": sum(1 for _, d in forward_assignment.items() if d == "GPU"),
        "ilp_layers_cpu_forward": sum(1 for _, d in forward_assignment.items() if d == "CPU"),
        "ilp_layers_gpu_backward": sum(1 for _, d in backward_assignment.items() if d == "GPU"),
        "ilp_layers_cpu_backward": sum(1 for _, d in backward_assignment.items() if d == "CPU"),
        "ilp_cut_edges_forward": len(ilp.cut_edges),
        "ilp_cut_edges_backward": len(backward_cut_edges),
        "ilp_cross_phase_edges": len(cross_phase_edges),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Sweep ILP over GPU memory budgets and compare baselines")
    parser.add_argument("--config_dir", default=None)
    parser.add_argument("--config_dirs", default=None, help="Comma-separated batch directories for multi-hardware aggregation")
    parser.add_argument("--model", required=True)
    parser.add_argument("--gpu_budgets_mb", required=True, help="Comma-separated budgets, e.g. 400,600,800,1000")
    parser.add_argument("--cpu_mem_budget_mb", type=float, default=1e18)
    parser.add_argument("--memory_model", choices=["nodal_sum", "peak_approx"], default="peak_approx")
    parser.add_argument("--peak_activation_overlap", type=float, default=0.35)
    parser.add_argument("--k_sigma", type=float, default=1.0)
    parser.add_argument("--k_sigma_time", type=float, default=None)
    parser.add_argument("--k_sigma_energy", type=float, default=None)
    parser.add_argument("--strict_graph_mapping", action="store_true", help="Fail if graph edges cannot be mapped to metrics layers")
    parser.add_argument("--strict_transfer_mapping", action="store_true", help="Fail if matched graph edges miss transfer costs")
    parser.add_argument("--allow_low_quality_stats", action="store_true", help="Allow ILP sweep on metrics_stats.csv rows flagged as low quality (diagnostic only)")
    parser.add_argument("--allow_transfer_calibration_fallback", action="store_true", help="Allow ILP sweep when transfer calibration fell back to neutral defaults (diagnostic only)")
    parser.add_argument("--allow_fallback_graph_trace", action="store_true", help="Allow ILP sweep from fallback_leaf_modules graph traces (diagnostic only)")
    parser.add_argument("--w_time", type=float, default=1.0)
    parser.add_argument("--w_energy", type=float, default=0.0)
    parser.add_argument("--w_transfer", type=float, default=1.0)
    parser.add_argument("--w_fragmentation", type=float, default=0.0)
    parser.add_argument("--backend", choices=["auto", "pulp", "exhaustive"], default="auto")
    parser.add_argument("--hw_aggregate", choices=["max", "mean"], default="max", help="How to aggregate costs across hardware profiles")
    parser.add_argument("--hw_dispersion_k", type=float, default=0.0, help="If hw_aggregate=mean, use mean + k*std across hardware profiles")
    parser.add_argument("--output_csv", default=None)
    parser.add_argument("--output_json", default=None)
    args = parser.parse_args()

    config_dirs: list[Path]
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

    profiles = []
    for cdir in config_dirs:
        stats_csv, graph_csv, transfer_csv = _default_paths(cdir, args.model)
        profile = load_ilp_inputs(
            metrics_stats_csv=str(stats_csv),
            graph_edges_csv=str(graph_csv),
            transfer_edges_csv=str(transfer_csv),
            k_sigma=args.k_sigma,
            k_sigma_time=args.k_sigma_time,
            k_sigma_energy=args.k_sigma_energy,
            strict_graph_mapping=args.strict_graph_mapping,
            strict_transfer_mapping=args.strict_transfer_mapping,
            strict_sample_quality=not args.allow_low_quality_stats,
            strict_transfer_calibration=not args.allow_transfer_calibration_fallback,
            strict_graph_trace_source=not args.allow_fallback_graph_trace,
        )
        profiles.append(profile)

    if len(profiles) == 1:
        data = profiles[0]
    else:
        data = merge_ilp_inputs_multi_hardware(
            profiles=profiles,
            strategy=args.hw_aggregate,
            dispersion_k=args.hw_dispersion_k,
            strict_schema=True,
        )

    budgets = _parse_budget_list(args.gpu_budgets_mb)
    rows = []

    for b in budgets:
        cfg = ILPConfig(
            w_time=args.w_time,
            w_energy=args.w_energy,
            w_transfer=args.w_transfer,
            w_fragmentation=args.w_fragmentation,
            gpu_mem_budget_mb=b,
            cpu_mem_budget_mb=args.cpu_mem_budget_mb,
            memory_model=args.memory_model,
            peak_activation_overlap=args.peak_activation_overlap,
        )

        ilp = solve_partition_ilp(data, cfg, backend=args.backend)
        all_cpu = _eval_fixed_policy("all_cpu", data, cfg)
        all_gpu = _eval_fixed_policy("all_gpu", data, cfg)
        greedy = _eval_greedy_policy(data, cfg)
        phase_counts = _phase_assignment_counts(ilp)

        rows.append({
            "model": args.model,
            "gpu_budget_mb": b,
            "cpu_budget_mb": args.cpu_mem_budget_mb,
            "backend": ilp.backend,
            "ilp_status": ilp.status,
            "ilp_objective": ilp.objective_value,
            "ilp_gpu_mem_mb": ilp.gpu_mem_used_mb,
            "ilp_cpu_mem_mb": ilp.cpu_mem_used_mb,
            "ilp_layers_gpu": sum(1 for _, d in ilp.assignment.items() if d == "GPU"),
            "ilp_layers_cpu": sum(1 for _, d in ilp.assignment.items() if d == "CPU"),
            "ilp_cut_edges": len(ilp.cut_edges),
            **phase_counts,
            "all_cpu_status": all_cpu["status"],
            "all_cpu_objective": all_cpu["objective_value"],
            "all_cpu_gpu_mem_mb": all_cpu["gpu_mem_used_mb"],
            "all_cpu_cpu_mem_mb": all_cpu["cpu_mem_used_mb"],
            "all_gpu_status": all_gpu["status"],
            "all_gpu_objective": all_gpu["objective_value"],
            "all_gpu_gpu_mem_mb": all_gpu["gpu_mem_used_mb"],
            "all_gpu_cpu_mem_mb": all_gpu["cpu_mem_used_mb"],
            "greedy_status": greedy["status"],
            "greedy_objective": greedy["objective_value"],
            "greedy_gpu_mem_mb": greedy["gpu_mem_used_mb"],
            "greedy_cpu_mem_mb": greedy["cpu_mem_used_mb"],
            "greedy_layers_gpu": greedy["layers_gpu"],
            "greedy_layers_cpu": greedy["layers_cpu"],
            "greedy_cut_edges": greedy["cut_edges"],
        })

    out_df = pd.DataFrame(rows).sort_values(by=["gpu_budget_mb"], kind="stable")

    base_config_dir = config_dirs[0]
    out_csv = Path(args.output_csv) if args.output_csv else (base_config_dir / f"{args.model}_pareto_sweep.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)

    summary = {
        "model": args.model,
        "rows": int(len(out_df)),
        "gpu_budgets_mb": budgets,
        "backend_requested": args.backend,
        "output_csv": str(out_csv),
        "best_feasible_row": None,
    }

    feasible = out_df[out_df["ilp_status"].isin(["optimal", "feasible"])]
    if len(feasible) > 0:
        best_idx = feasible["ilp_objective"].astype(float).idxmin()
        summary["best_feasible_row"] = out_df.loc[best_idx].to_dict()

    out_json = Path(args.output_json) if args.output_json else (base_config_dir / f"{args.model}_pareto_summary.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=4)

    print("=" * 80)
    print("ILP PARETO SWEEP")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Rows: {len(out_df)}")
    print(f"CSV: {out_csv}")
    print(f"JSON: {out_json}")
    if summary["best_feasible_row"] is not None:
        r = summary["best_feasible_row"]
        print("Best feasible:")
        print(f"  gpu_budget_mb={r['gpu_budget_mb']}, ilp_objective={r['ilp_objective']}")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
