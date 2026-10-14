#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

plan_mod = importlib.import_module("runtime.plan_representation")
topology_mod = importlib.import_module("ilp.topology_memory")

ExecutionPlan = plan_mod.ExecutionPlan
load_execution_plan = plan_mod.load_execution_plan
infer_ilp_input_paths = plan_mod.infer_ilp_input_paths

PreparedTopologyMemory = topology_mod.PreparedTopologyMemory


def _resolve_solution_paths(solution_dir: Path):
    assignment_csv = solution_dir / "ilp_assignment.csv"
    cut_edges_csv = solution_dir / "ilp_cut_edges.csv"
    summary_json = solution_dir / "ilp_solution_summary.json"

    if not assignment_csv.exists():
        raise FileNotFoundError(f"Missing ILP assignment file: {assignment_csv}")
    if not cut_edges_csv.exists():
        raise FileNotFoundError(f"Missing ILP cut edges file: {cut_edges_csv}")

    return assignment_csv, cut_edges_csv, summary_json


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate an already-solved ILP plan's real peak memory "
        "(src/ilp/topology_memory.py's validated topological walk) against a "
        "GPU/CPU budget, independent of the solver's own estimate."
    )
    parser.add_argument("--config_dir", required=True, help="Path to ILP config directory (batch folder)")
    parser.add_argument("--model", required=True, help="Model name prefix, e.g. simple_mlp")
    parser.add_argument("--solution_dir", default=None, help="Path containing ilp_assignment.csv and ilp_cut_edges.csv")

    parser.add_argument("--metrics_stats_csv", default=None)
    parser.add_argument("--graph_edges_csv", default=None)

    parser.add_argument("--gpu_mem_budget_mb", type=float, default=1e18)
    parser.add_argument("--cpu_mem_budget_mb", type=float, default=1e18)

    parser.add_argument("--output_dir", default=None, help="Where to save topology_memory_summary.json and _breakdown.csv")

    args = parser.parse_args()

    config_dir = Path(args.config_dir)
    if not config_dir.exists():
        raise FileNotFoundError(f"config_dir does not exist: {config_dir}")

    solution_dir = Path(args.solution_dir) if args.solution_dir else (config_dir / "ilp_solution")
    assignment_csv, cut_edges_csv, ilp_summary_json = _resolve_solution_paths(solution_dir)

    if args.metrics_stats_csv and args.graph_edges_csv:
        metrics_stats_csv = Path(args.metrics_stats_csv)
        graph_edges_csv = Path(args.graph_edges_csv)
    else:
        paths = infer_ilp_input_paths(config_dir=config_dir, model_name=args.model)
        metrics_stats_csv = paths.metrics_stats_csv
        graph_edges_csv = paths.graph_edges_csv

    plan: ExecutionPlan = load_execution_plan(assignment_csv=assignment_csv, cut_edges_csv=cut_edges_csv)

    prepared = PreparedTopologyMemory.prepare(graph_edges_csv, metrics_stats_csv)
    peak_gpu_mb, peak_cpu_mb = prepared.evaluate(
        forward_assignment=plan.assignment_forward,
        backward_assignment=plan.assignment_backward,
        activation_strategy=getattr(plan, "activation_strategies", None),
    )

    feasible = peak_gpu_mb <= args.gpu_mem_budget_mb and peak_cpu_mb <= args.cpu_mem_budget_mb
    status = "ok" if feasible else "budget_exceeded"

    output_dir = Path(args.output_dir) if args.output_dir else (solution_dir / "topology_memory")
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_payload = {
        "status": status,
        "peak_gpu_mem_mb": peak_gpu_mb,
        "peak_cpu_mem_mb": peak_cpu_mb,
        "gpu_mem_budget_mb": args.gpu_mem_budget_mb,
        "cpu_mem_budget_mb": args.cpu_mem_budget_mb,
        "diagnostics": prepared.diagnostics,
        "inputs": {
            "config_dir": str(config_dir),
            "solution_dir": str(solution_dir),
            "assignment_csv": str(assignment_csv),
            "cut_edges_csv": str(cut_edges_csv),
            "ilp_summary_json": str(ilp_summary_json) if ilp_summary_json.exists() else None,
            "metrics_stats_csv": str(metrics_stats_csv),
            "graph_edges_csv": str(graph_edges_csv),
        },
    }

    summary_path = output_dir / "topology_memory_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary_payload, f, indent=4)

    breakdown_df = pd.DataFrame(
        [
            {"metric": "peak_gpu_mem_mb", "value": peak_gpu_mb},
            {"metric": "peak_cpu_mem_mb", "value": peak_cpu_mb},
            {"metric": "gpu_mem_budget_mb", "value": args.gpu_mem_budget_mb},
            {"metric": "cpu_mem_budget_mb", "value": args.cpu_mem_budget_mb},
        ]
    )
    breakdown_path = output_dir / "topology_memory_breakdown.csv"
    breakdown_df.to_csv(breakdown_path, index=False)

    print("=" * 80)
    print("ILP PLAN VALIDATION (real topological peak memory)")
    print("=" * 80)
    print(f"Status: {status}")
    print(f"Peak GPU mem (MB): {peak_gpu_mb:.6f} (budget {args.gpu_mem_budget_mb:.1f})")
    print(f"Peak CPU mem (MB): {peak_cpu_mb:.6f} (budget {args.cpu_mem_budget_mb:.1f})")
    print(f"Summary JSON: {summary_path}")
    print(f"Breakdown CSV: {breakdown_path}")
    print("=" * 80)

    return 0 if status == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
