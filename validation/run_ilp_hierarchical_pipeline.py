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
    parser.add_argument("--no_simulate", action="store_true", help="Skip runtime simulation validation stage")
    parser.add_argument("--output_dir", default=None)

    args = parser.parse_args()

    load_execution_plan = None
    infer_ilp_input_paths = None
    load_graph_edges = None
    load_transfer_costs = None
    SimulationConfig = None
    simulate_plan = None
    if not args.no_simulate:
        load_execution_plan = importlib.import_module("runtime.plan_representation").load_execution_plan
        infer_ilp_input_paths = importlib.import_module("runtime.plan_representation").infer_ilp_input_paths
        load_graph_edges = importlib.import_module("runtime.plan_representation").load_graph_edges
        load_transfer_costs = importlib.import_module("runtime.plan_representation").load_transfer_costs
        SimulationConfig = importlib.import_module("runtime.simulator").SimulationConfig
        simulate_plan = importlib.import_module("runtime.simulator").simulate_plan

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

        sim = None
        sim_summary = None
        if not args.no_simulate:
            plan = load_execution_plan(
                assignment_csv=exported["assignment_csv"],
                cut_edges_csv=exported["cut_edges_csv"],
            )
            inferred = infer_ilp_input_paths(config_dir=config_dir, model_name=args.model)
            measured_layers = set(data.nodes)
            graph_edges = load_graph_edges(
                inferred.graph_edges_csv,
                transfer_edges_csv=inferred.transfer_edges_csv,
                measured_layers=measured_layers,
            )
            transfer_costs = load_transfer_costs(
                inferred.transfer_edges_csv,
                graph_edges_csv=inferred.graph_edges_csv,
                measured_layers=measured_layers,
            )
            sim_cfg = SimulationConfig(
                mode="robust",
                k_sigma=args.k_sigma,
                k_sigma_time=args.k_sigma_time,
                k_sigma_energy=args.k_sigma_energy,
                w_time=args.w_time,
                w_energy=args.w_energy,
                w_transfer=args.w_transfer,
                gpu_mem_budget_mb=budget,
                cpu_mem_budget_mb=args.cpu_mem_budget_mb,
            )
            sim = simulate_plan(
                plan=plan,
                metrics_stats_csv=inferred.metrics_stats_csv,
                graph_edges=graph_edges,
                transfer_costs=transfer_costs,
                cfg=sim_cfg,
            )

            sim_summary = budget_dir / "simulation_summary.json"
            sim_summary.parent.mkdir(parents=True, exist_ok=True)
            sim_summary.write_text(json.dumps(sim.to_dict(), indent=4), encoding="utf-8")

        rows.append(
            {
                "model": args.model,
                "gpu_budget_mb": budget,
                "base_backend": base_sol.backend,
                "final_backend": final_sol.backend,
                "base_objective": base_sol.objective_value,
                "final_objective": final_sol.objective_value,
                "delta_objective": float(final_sol.objective_value - base_sol.objective_value),
                "sim_status": sim.status if sim is not None else "skipped",
                "sim_objective": sim.objective_value if sim is not None else float("nan"),
                "sim_total_time_ms": sim.total_time_ms if sim is not None else float("nan"),
                "sim_total_energy_j": sim.total_energy_j if sim is not None else float("nan"),
                "sim_total_transfer_ms": sim.total_transfer_ms if sim is not None else float("nan"),
                "sim_gpu_mem_mb": sim.gpu_mem_used_mb if sim is not None else float("nan"),
                "sim_cpu_mem_mb": sim.cpu_mem_used_mb if sim is not None else float("nan"),
                "sim_violations": len(sim.violations) if sim is not None else 0,
                "sim_warnings": len(sim.warnings) if sim is not None else 0,
                "solution_dir": str((budget_dir / "ilp_solution")),
                "simulation_summary_json": str(sim_summary) if sim_summary is not None else "",
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
