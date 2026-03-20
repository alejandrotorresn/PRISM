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
sim_mod = importlib.import_module("runtime.simulator")

ExecutionPlan = plan_mod.ExecutionPlan
load_execution_plan = plan_mod.load_execution_plan
infer_ilp_input_paths = plan_mod.infer_ilp_input_paths
load_graph_edges = plan_mod.load_graph_edges
load_transfer_costs = plan_mod.load_transfer_costs

SimulationConfig = sim_mod.SimulationConfig
simulate_plan = sim_mod.simulate_plan
simulate_plan_phase4 = sim_mod.simulate_plan_phase4


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
        description="Validate ILP plan topology and simulate expected objective decomposition"
    )
    parser.add_argument("--config_dir", required=True, help="Path to ILP config directory (batch folder)")
    parser.add_argument("--model", required=True, help="Model name prefix, e.g. simple_mlp")
    parser.add_argument("--solution_dir", default=None, help="Path containing ilp_assignment.csv and ilp_cut_edges.csv")

    parser.add_argument("--metrics_stats_csv", default=None)
    parser.add_argument("--graph_edges_csv", default=None)
    parser.add_argument("--transfer_edges_csv", default=None)

    parser.add_argument("--mode", choices=["robust", "nominal"], default="robust")
    parser.add_argument("--k_sigma", type=float, default=1.0)
    parser.add_argument("--w_time", type=float, default=1.0)
    parser.add_argument("--w_energy", type=float, default=0.0)
    parser.add_argument("--w_transfer", type=float, default=1.0)
    parser.add_argument("--gpu_mem_budget_mb", type=float, default=1e18)
    parser.add_argument("--cpu_mem_budget_mb", type=float, default=1e18)

    parser.add_argument("--strict_graph_subset", action="store_true")
    parser.add_argument("--strict_transfer_mapping", action="store_true")
    parser.add_argument("--strict_topology", action="store_true")

    parser.add_argument("--output_dir", default=None, help="Where to save simulation_summary.json and simulation_breakdown.csv")

    args = parser.parse_args()

    config_dir = Path(args.config_dir)
    if not config_dir.exists():
        raise FileNotFoundError(f"config_dir does not exist: {config_dir}")

    solution_dir = Path(args.solution_dir) if args.solution_dir else (config_dir / "ilp_solution")
    assignment_csv, cut_edges_csv, ilp_summary_json = _resolve_solution_paths(solution_dir)

    if args.metrics_stats_csv and args.graph_edges_csv and args.transfer_edges_csv:
        metrics_stats_csv = Path(args.metrics_stats_csv)
        graph_edges_csv = Path(args.graph_edges_csv)
        transfer_edges_csv = Path(args.transfer_edges_csv)
    else:
        paths = infer_ilp_input_paths(config_dir=config_dir, model_name=args.model)
        metrics_stats_csv = paths.metrics_stats_csv
        graph_edges_csv = paths.graph_edges_csv
        transfer_edges_csv = paths.transfer_edges_csv

    plan: ExecutionPlan = load_execution_plan(assignment_csv=assignment_csv, cut_edges_csv=cut_edges_csv)
    graph_edges = load_graph_edges(graph_edges_csv)
    transfer_costs = load_transfer_costs(transfer_edges_csv)

    cfg = SimulationConfig(
        mode=args.mode,
        k_sigma=args.k_sigma,
        w_time=args.w_time,
        w_energy=args.w_energy,
        w_transfer=args.w_transfer,
        gpu_mem_budget_mb=args.gpu_mem_budget_mb,
        cpu_mem_budget_mb=args.cpu_mem_budget_mb,
        strict_transfer_mapping=args.strict_transfer_mapping,
        strict_graph_subset=args.strict_graph_subset,
        strict_topology=args.strict_topology,
    )

    if getattr(plan, "activation_strategies", None):
        result = simulate_plan_phase4(
            plan=plan,
            metrics_stats_csv=metrics_stats_csv,
            graph_edges=graph_edges,
            transfer_costs=transfer_costs,
            cfg=cfg,
            activation_strategies=plan.activation_strategies,
        )
    else:
        result = simulate_plan(
            plan=plan,
            metrics_stats_csv=metrics_stats_csv,
            graph_edges=graph_edges,
            transfer_costs=transfer_costs,
            cfg=cfg,
        )

    output_dir = Path(args.output_dir) if args.output_dir else (solution_dir / "simulation")
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_payload = {
        **result.to_dict(),
        "inputs": {
            "config_dir": str(config_dir),
            "solution_dir": str(solution_dir),
            "assignment_csv": str(assignment_csv),
            "cut_edges_csv": str(cut_edges_csv),
            "ilp_summary_json": str(ilp_summary_json) if ilp_summary_json.exists() else None,
            "metrics_stats_csv": str(metrics_stats_csv),
            "graph_edges_csv": str(graph_edges_csv),
            "transfer_edges_csv": str(transfer_edges_csv),
        },
        "config": {
            "mode": args.mode,
            "k_sigma": args.k_sigma,
            "w_time": args.w_time,
            "w_energy": args.w_energy,
            "w_transfer": args.w_transfer,
            "gpu_mem_budget_mb": args.gpu_mem_budget_mb,
            "cpu_mem_budget_mb": args.cpu_mem_budget_mb,
            "strict_graph_subset": args.strict_graph_subset,
            "strict_transfer_mapping": args.strict_transfer_mapping,
            "strict_topology": args.strict_topology,
        },
    }

    summary_path = output_dir / "simulation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary_payload, f, indent=4)

    breakdown_df = pd.DataFrame(
        [
            {"metric": "objective_value", "value": result.objective_value},
            {"metric": "total_time_ms", "value": result.total_time_ms},
            {"metric": "total_energy_j", "value": result.total_energy_j},
            {"metric": "total_transfer_ms", "value": result.total_transfer_ms},
            {"metric": "gpu_mem_used_mb", "value": result.gpu_mem_used_mb},
            {"metric": "cpu_mem_used_mb", "value": result.cpu_mem_used_mb},
            {"metric": "layers_total", "value": result.layers_total},
            {"metric": "layers_gpu", "value": result.layers_gpu},
            {"metric": "layers_cpu", "value": result.layers_cpu},
            {"metric": "cut_edges_count", "value": result.cut_edges_count},
        ]
    )
    breakdown_path = output_dir / "simulation_breakdown.csv"
    breakdown_df.to_csv(breakdown_path, index=False)

    print("=" * 80)
    print("ILP PLAN VALIDATION + SIMULATION")
    print("=" * 80)
    print(f"Status: {result.status}")
    print(f"Objective: {result.objective_value:.6f}")
    print(f"Time (ms): {result.total_time_ms:.6f}")
    print(f"Energy (J): {result.total_energy_j:.6f}")
    print(f"Transfer (ms): {result.total_transfer_ms:.6f}")
    print(f"GPU mem used (MB): {result.gpu_mem_used_mb:.6f}")
    print(f"CPU mem used (MB): {result.cpu_mem_used_mb:.6f}")
    print(f"Layers: total={result.layers_total}, gpu={result.layers_gpu}, cpu={result.layers_cpu}")
    print(f"Cut edges: {result.cut_edges_count}")

    if result.warnings:
        print("Warnings:")
        for msg in result.warnings:
            print(f"  - {msg}")

    if result.violations:
        print("Violations:")
        for msg in result.violations:
            print(f"  - {msg}")

    print(f"Summary JSON: {summary_path}")
    print(f"Breakdown CSV: {breakdown_path}")
    print("=" * 80)

    return 0 if result.status == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
