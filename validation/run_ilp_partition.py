#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

load_ilp_inputs = importlib.import_module("ilp.data_loader").load_ilp_inputs
save_ilp_solution = importlib.import_module("ilp.export_solution").save_ilp_solution
ILPConfig = importlib.import_module("ilp.model_builder").ILPConfig
solve_partition_ilp = importlib.import_module("ilp.solve").solve_partition_ilp


def _default_paths(config_dir: Path, model_name: str):
    stats = config_dir / f"{model_name}_metrics_stats.csv"
    if not stats.exists():
        stats = config_dir / "metrics_stats.csv"

    run_dirs = sorted([p for p in config_dir.glob("run_*") if p.is_dir()])
    if not run_dirs:
        raise FileNotFoundError(f"No run_* directories found in: {config_dir}")

    ref_run = run_dirs[0]
    graph_edges = ref_run / f"{model_name}_graph_edges.csv"
    transfer_edges = ref_run / f"{model_name}_transfer_edges.csv"
    return stats, graph_edges, transfer_edges


def main() -> int:
    parser = argparse.ArgumentParser(description="Solve robust CPU/GPU layer partitioning ILP")
    parser.add_argument("--config_dir", required=True, help="Path to batch directory containing run_* and metrics_stats")
    parser.add_argument("--model", required=True, help="Model artifact prefix (e.g., simple_mlp)")
    parser.add_argument("--metrics_stats_csv", default=None, help="Explicit metrics stats CSV path")
    parser.add_argument("--graph_edges_csv", default=None, help="Explicit graph edges CSV path")
    parser.add_argument("--transfer_edges_csv", default=None, help="Explicit transfer edges CSV path")
    parser.add_argument("--k_sigma", type=float, default=1.0, help="Robustness factor for mu + k*sigma")
    parser.add_argument("--strict_graph_mapping", action="store_true", help="Fail if graph edges cannot be mapped to metrics layers")
    parser.add_argument("--strict_transfer_mapping", action="store_true", help="Fail if matched graph edges miss transfer costs")
    parser.add_argument("--w_time", type=float, default=1.0)
    parser.add_argument("--w_energy", type=float, default=0.0)
    parser.add_argument("--w_transfer", type=float, default=1.0)
    parser.add_argument("--gpu_mem_budget_mb", type=float, default=1e18)
    parser.add_argument("--cpu_mem_budget_mb", type=float, default=1e18)
    parser.add_argument("--backend", choices=["auto", "pulp", "exhaustive"], default="auto")
    parser.add_argument("--output_dir", default=None, help="Output folder for ILP solution files")
    args = parser.parse_args()

    config_dir = Path(args.config_dir)
    if not config_dir.exists():
        raise FileNotFoundError(f"config_dir does not exist: {config_dir}")

    if args.metrics_stats_csv and args.graph_edges_csv and args.transfer_edges_csv:
        stats_csv = Path(args.metrics_stats_csv)
        graph_csv = Path(args.graph_edges_csv)
        transfer_csv = Path(args.transfer_edges_csv)
    else:
        stats_csv, graph_csv, transfer_csv = _default_paths(config_dir, args.model)

    output_dir = Path(args.output_dir) if args.output_dir else (config_dir / "ilp_solution")

    data = load_ilp_inputs(
        metrics_stats_csv=str(stats_csv),
        graph_edges_csv=str(graph_csv),
        transfer_edges_csv=str(transfer_csv),
        k_sigma=args.k_sigma,
        strict_graph_mapping=args.strict_graph_mapping,
        strict_transfer_mapping=args.strict_transfer_mapping,
    )

    cfg = ILPConfig(
        w_time=args.w_time,
        w_energy=args.w_energy,
        w_transfer=args.w_transfer,
        gpu_mem_budget_mb=args.gpu_mem_budget_mb,
        cpu_mem_budget_mb=args.cpu_mem_budget_mb,
    )

    sol = solve_partition_ilp(data, cfg, backend=args.backend)
    out = save_ilp_solution(sol, str(output_dir))

    print("=" * 80)
    print("ILP PARTITION RESULT")
    print("=" * 80)
    print(f"Status: {sol.status}")
    print(f"Backend: {sol.backend}")
    print(f"Objective: {sol.objective_value:.6f}")
    print(f"GPU mem used (MB): {sol.gpu_mem_used_mb:.3f}")
    print(f"CPU mem used (MB): {sol.cpu_mem_used_mb:.3f}")
    print(f"Layers assigned: {len(sol.assignment)}")
    print(f"Cut edges: {len(sol.cut_edges)}")
    print(f"Assignment CSV: {out['assignment_csv']}")
    print(f"Cut edges CSV: {out['cut_edges_csv']}")
    print(f"Summary JSON: {out['summary_json']}")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
