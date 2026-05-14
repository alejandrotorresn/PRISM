#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from ilp.data_loader import load_ilp_inputs
from ilp.export_solution import save_ilp_solution
from ilp.model_builder import ILPConfig
from ilp.solve import refine_solution_hierarchical_local, solve_partition_ilp
from runtime.plan_representation import infer_ilp_input_paths, load_execution_plan, load_graph_edges, load_transfer_costs
from runtime.simulator import SimulationConfig, simulate_plan


def _parse_csv_list(text: str) -> List[str]:
    values = [item.strip() for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one comma-separated value")
    return values


def _parse_float_list(text: str) -> List[float]:
    return [float(item) for item in _parse_csv_list(text)]


def _default_paths(config_dir: Path, model_name: str) -> Tuple[Path, Path, Path]:
    stats = config_dir / f"{model_name}_metrics_stats.csv"
    if not stats.exists():
        stats = config_dir / "metrics_stats.csv"

    run_dirs = sorted([path for path in config_dir.glob("run_*") if path.is_dir()])
    if run_dirs:
        ref_run = run_dirs[0]
        graph_edges = ref_run / f"{model_name}_graph_edges.csv"
        transfer_edges = ref_run / f"{model_name}_transfer_edges.csv"
    else:
        graph_edges = config_dir / f"{model_name}_graph_edges.csv"
        transfer_edges = config_dir / f"{model_name}_transfer_edges.csv"

    if not stats.exists() or not graph_edges.exists() or not transfer_edges.exists():
        raise FileNotFoundError(
            "Could not resolve metrics/graph/transfer artifacts for case: "
            f"config_dir={config_dir}, model={model_name}"
        )
    return stats, graph_edges, transfer_edges


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a controlled ILP hyperparameter sweep across multiple real cases")
    parser.add_argument("--config_dirs", required=True, help="Comma-separated config directories")
    parser.add_argument("--models", required=True, help="Comma-separated model names aligned with --config_dirs")
    parser.add_argument("--gpu_budgets_mb", required=True, help="Comma-separated GPU budgets aligned with --config_dirs")
    parser.add_argument("--local_refine_budget", type=int, default=2)
    parser.add_argument("--w_fragmentation_grid", default="0.0,0.02,0.05")
    parser.add_argument("--k_sigma_time_grid", default="0.0,0.5,1.0")
    parser.add_argument("--k_sigma_energy_grid", default="0.0,0.5,1.0")
    parser.add_argument("--w_time", type=float, default=1.0)
    parser.add_argument("--w_energy", type=float, default=1.0)
    parser.add_argument("--w_transfer", type=float, default=1.0)
    parser.add_argument("--w_congestion", type=float, default=0.0)
    parser.add_argument("--congestion_knee_ms", type=float, default=0.0)
    parser.add_argument("--k_sigma", type=float, default=1.0)
    parser.add_argument("--output_dir", required=True)
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    config_dirs = [Path(path) for path in _parse_csv_list(args.config_dirs)]
    models = _parse_csv_list(args.models)
    gpu_budgets_mb = _parse_float_list(args.gpu_budgets_mb)
    if not (len(config_dirs) == len(models) == len(gpu_budgets_mb)):
        raise ValueError("--config_dirs, --models and --gpu_budgets_mb must have the same length")

    w_fragmentation_grid = _parse_float_list(args.w_fragmentation_grid)
    k_sigma_time_grid = _parse_float_list(args.k_sigma_time_grid)
    k_sigma_energy_grid = _parse_float_list(args.k_sigma_energy_grid)
    cases = [
        {
            "config_dir": config_dir,
            "model": model,
            "gpu_budget_mb": gpu_budget_mb,
        }
        for config_dir, model, gpu_budget_mb in zip(config_dirs, models, gpu_budgets_mb)
    ]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    combo_total = (
        len(w_fragmentation_grid)
        * len(k_sigma_time_grid)
        * len(k_sigma_energy_grid)
    )
    all_rows: List[Dict[str, Any]] = []

    for case in cases:
        config_dir = case["config_dir"]
        model = case["model"]
        gpu_budget_mb = float(case["gpu_budget_mb"])

        stats_csv, graph_csv, transfer_csv = _default_paths(config_dir, model)
        case_output_dir = output_dir / model
        case_output_dir.mkdir(parents=True, exist_ok=True)

        inferred = infer_ilp_input_paths(config_dir=config_dir, model_name=model)
        graph_edges = load_graph_edges(
            inferred.graph_edges_csv,
            transfer_edges_csv=inferred.transfer_edges_csv,
            measured_layers=None,
        )
        transfer_costs = load_transfer_costs(
            inferred.transfer_edges_csv,
            graph_edges_csv=inferred.graph_edges_csv,
            measured_layers=None,
        )

        print(f"[case] {model} budget={gpu_budget_mb} combos={combo_total}", flush=True)

        combos = itertools.product(
            w_fragmentation_grid,
            k_sigma_time_grid,
            k_sigma_energy_grid,
        )
        for idx, (w_fragmentation, k_sigma_time, k_sigma_energy) in enumerate(combos, start=1):
            data = load_ilp_inputs(
                metrics_stats_csv=str(stats_csv),
                graph_edges_csv=str(graph_csv),
                transfer_edges_csv=str(transfer_csv),
                k_sigma=args.k_sigma,
                k_sigma_time=k_sigma_time,
                k_sigma_energy=k_sigma_energy,
                strict_sample_quality=False,
                strict_transfer_calibration=False,
                strict_graph_trace_source=False,
            )

            cfg = ILPConfig(
                w_time=args.w_time,
                w_energy=args.w_energy,
                w_transfer=args.w_transfer,
                w_fragmentation=w_fragmentation,
                w_congestion=args.w_congestion,
                congestion_knee_ms=args.congestion_knee_ms,
                gpu_mem_budget_mb=gpu_budget_mb,
                cpu_mem_budget_mb=1e18,
            )
            solution = solve_partition_ilp(data, cfg, backend="pulp")
            if args.local_refine_budget > 0:
                solution = refine_solution_hierarchical_local(
                    data=data,
                    cfg=cfg,
                    base_solution=solution,
                    max_assignment_changes=args.local_refine_budget,
                )

            tag = (
                f"wf_{w_fragmentation:.2f}__kt_{k_sigma_time:.1f}__ke_{k_sigma_energy:.1f}"
                .replace(".", "p")
            )
            run_dir = case_output_dir / "sweep_runs" / tag
            exported = save_ilp_solution(solution, str(run_dir / "ilp_solution"))
            plan = load_execution_plan(
                assignment_csv=exported["assignment_csv"],
                cut_edges_csv=exported["cut_edges_csv"],
            )

            sim_cfg = SimulationConfig(
                mode="robust",
                k_sigma=args.k_sigma,
                k_sigma_time=k_sigma_time,
                k_sigma_energy=k_sigma_energy,
                w_time=args.w_time,
                w_energy=args.w_energy,
                w_transfer=args.w_transfer,
                gpu_mem_budget_mb=gpu_budget_mb,
                cpu_mem_budget_mb=1e18,
            )
            sim_result = simulate_plan(
                plan=plan,
                metrics_stats_csv=inferred.metrics_stats_csv,
                graph_edges=graph_edges,
                transfer_costs=transfer_costs,
                cfg=sim_cfg,
            )

            sim_summary_path = run_dir / "simulation_summary.json"
            sim_summary_path.parent.mkdir(parents=True, exist_ok=True)
            sim_summary_path.write_text(json.dumps(sim_result.to_dict(), indent=2), encoding="utf-8")

            all_rows.append(
                {
                    "model": model,
                    "config_dir": str(config_dir),
                    "gpu_budget_mb": gpu_budget_mb,
                    "local_refine_budget": args.local_refine_budget,
                    "w_fragmentation": w_fragmentation,
                    "w_congestion": float(args.w_congestion),
                    "congestion_knee_ms": float(args.congestion_knee_ms),
                    "k_sigma_time": k_sigma_time,
                    "k_sigma_energy": k_sigma_energy,
                    "solver_status": solution.status,
                    "solver_backend": solution.backend,
                    "ilp_objective": solution.objective_value,
                    "gpu_mem_used_mb": solution.gpu_mem_used_mb,
                    "cpu_mem_used_mb": solution.cpu_mem_used_mb,
                    "cut_edges_total": len(solution.cut_edges),
                    "layers_gpu_fwd": sum(1 for device in plan.assignment_forward.values() if device == "GPU"),
                    "layers_gpu_bwd": sum(1 for device in plan.assignment_backward.values() if device == "GPU"),
                    "sim_status": sim_result.status,
                    "sim_objective": sim_result.objective_value,
                    "sim_time_ms": sim_result.total_time_ms,
                    "sim_energy_j": sim_result.total_energy_j,
                    "sim_transfer_ms": sim_result.total_transfer_ms,
                    "sim_gpu_mem_mb": sim_result.gpu_mem_used_mb,
                    "sim_cpu_mem_mb": sim_result.cpu_mem_used_mb,
                    "sim_warnings": len(sim_result.warnings),
                    "sim_violations": len(sim_result.violations),
                    "solution_dir": str(run_dir / "ilp_solution"),
                    "simulation_summary_json": str(sim_summary_path),
                }
            )
            if idx % 10 == 0 or idx == combo_total:
                print(f"[progress] {model} {idx}/{combo_total} complete", flush=True)

    raw_df = pd.DataFrame(all_rows)
    raw_csv = output_dir / "controlled_hparam_sweep_results.csv"
    raw_df.to_csv(raw_csv, index=False)

    baseline_key = {
        "w_fragmentation": 0.0,
        "k_sigma_time": 1.0,
        "k_sigma_energy": 1.0,
    }

    baseline_rows = raw_df[
        (raw_df["w_fragmentation"] == baseline_key["w_fragmentation"])
        & (raw_df["k_sigma_time"] == baseline_key["k_sigma_time"])
        & (raw_df["k_sigma_energy"] == baseline_key["k_sigma_energy"])
    ].copy()
    baseline_by_model = baseline_rows.set_index("model").to_dict(orient="index")

    scored_rows = []
    for _, row in raw_df.iterrows():
        baseline = baseline_by_model[row["model"]]
        baseline_sim_objective = float(baseline["sim_objective"])
        scored_rows.append(
            {
                **row.to_dict(),
                "baseline_sim_objective": baseline_sim_objective,
                "baseline_sim_warnings": int(baseline["sim_warnings"]),
                "baseline_sim_violations": int(baseline["sim_violations"]),
                "sim_objective_improvement_pct": (
                    ((baseline_sim_objective - float(row["sim_objective"])) / baseline_sim_objective) * 100.0
                    if baseline_sim_objective
                    else 0.0
                ),
                "warning_delta": int(row["sim_warnings"]) - int(baseline["sim_warnings"]),
                "violation_delta": int(row["sim_violations"]) - int(baseline["sim_violations"]),
            }
        )

    scored_df = pd.DataFrame(scored_rows)
    scored_csv = output_dir / "controlled_hparam_sweep_scored.csv"
    scored_df.to_csv(scored_csv, index=False)

    valid_df = scored_df[
        (scored_df["sim_status"] == "ok")
        & (scored_df["sim_violations"] == 0)
        & (scored_df["violation_delta"] <= 0)
        & (scored_df["warning_delta"] <= 0)
    ].copy()

    top5_df = (
        valid_df.sort_values(
            by=["model", "sim_objective_improvement_pct", "sim_transfer_ms", "cut_edges_total"],
            ascending=[True, False, True, True],
            kind="stable",
        )
        .groupby("model", as_index=False)
        .head(5)
    )
    top5_csv = output_dir / "controlled_hparam_sweep_top5_per_model.csv"
    top5_df.to_csv(top5_csv, index=False)

    shared_df = (
        valid_df.groupby(["w_fragmentation", "k_sigma_time", "k_sigma_energy"], as_index=False)
        .agg(
            models_covered=("model", "nunique"),
            mean_improvement_pct=("sim_objective_improvement_pct", "mean"),
            min_improvement_pct=("sim_objective_improvement_pct", "min"),
            max_warnings=("sim_warnings", "max"),
            max_violations=("sim_violations", "max"),
        )
    )
    shared_df = shared_df[shared_df["models_covered"] == len(cases)].copy()
    shared_df = shared_df.sort_values(
        by=["mean_improvement_pct", "min_improvement_pct", "w_fragmentation"],
        ascending=[False, False, True],
        kind="stable",
    )
    shared_csv = output_dir / "controlled_hparam_sweep_shared_configs.csv"
    shared_df.to_csv(shared_csv, index=False)

    recommendation: Dict[str, Any] = {
        "baseline_key": baseline_key,
        "cases": [
            {
                "config_dir": str(case["config_dir"]),
                "model": case["model"],
                "gpu_budget_mb": float(case["gpu_budget_mb"]),
            }
            for case in cases
        ],
        "best_shared_config": None,
        "best_per_model": {},
        "artifacts": {
            "raw_csv": str(raw_csv),
            "scored_csv": str(scored_csv),
            "top5_per_model_csv": str(top5_csv),
            "shared_csv": str(shared_csv),
        },
    }
    if not shared_df.empty:
        recommendation["best_shared_config"] = shared_df.iloc[0].to_dict()
    for model, group in valid_df.groupby("model"):
        best_row = group.sort_values(
            by=["sim_objective_improvement_pct", "sim_transfer_ms", "cut_edges_total"],
            ascending=[False, True, True],
            kind="stable",
        ).iloc[0]
        recommendation["best_per_model"][model] = best_row.to_dict()

    recommendation_json = output_dir / "controlled_hparam_sweep_recommendation.json"
    recommendation_json.write_text(json.dumps(recommendation, indent=2), encoding="utf-8")

    print(f"[done] raw={raw_csv}", flush=True)
    print(f"[done] scored={scored_csv}", flush=True)
    print(f"[done] top5={top5_csv}", flush=True)
    print(f"[done] shared={shared_csv}", flush=True)
    print(f"[done] recommendation={recommendation_json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())