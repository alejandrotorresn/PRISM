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
merge_ilp_inputs_multi_hardware = importlib.import_module("ilp.data_loader").merge_ilp_inputs_multi_hardware
save_ilp_solution = importlib.import_module("ilp.export_solution").save_ilp_solution
ILPConfig = importlib.import_module("ilp.model_builder").ILPConfig
ILPConfig4 = importlib.import_module("ilp.model_builder").ILPConfig4
solve_partition_ilp = importlib.import_module("ilp.solve").solve_partition_ilp
solve_partition_ilp_phase4 = importlib.import_module("ilp.solve").solve_partition_ilp_phase4
refine_solution_hierarchical_local = importlib.import_module("ilp.solve").refine_solution_hierarchical_local


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


def main() -> int:
    parser = argparse.ArgumentParser(description="Solve robust CPU/GPU layer partitioning ILP")
    parser.add_argument("--config_dir", default=None, help="Path to batch directory containing run_* and metrics_stats")
    parser.add_argument("--config_dirs", default=None, help="Comma-separated batch directories for multi-hardware aggregation")
    parser.add_argument("--model", required=True, help="Model artifact prefix (e.g., simple_mlp)")
    parser.add_argument("--metrics_stats_csv", default=None, help="Explicit metrics stats CSV path")
    parser.add_argument("--graph_edges_csv", default=None, help="Explicit graph edges CSV path")
    parser.add_argument("--transfer_edges_csv", default=None, help="Explicit transfer edges CSV path")
    parser.add_argument("--k_sigma", type=float, default=1.0, help="Robustness factor for mu + k*sigma")
    parser.add_argument("--k_sigma_time", type=float, default=None, help="Optional robustness factor for time costs (overrides --k_sigma for time only)")
    parser.add_argument("--k_sigma_energy", type=float, default=None, help="Optional robustness factor for energy costs (overrides --k_sigma for energy only)")
    parser.add_argument("--strict_graph_mapping", action="store_true", help="Fail if graph edges cannot be mapped to metrics layers")
    parser.add_argument("--strict_transfer_mapping", action="store_true", help="Fail if matched graph edges miss transfer costs")
    parser.add_argument("--allow_low_quality_stats", action="store_true", help="Allow ILP solve on metrics_stats.csv rows flagged as low quality (diagnostic only)")
    parser.add_argument("--allow_transfer_calibration_fallback", action="store_true", help="Allow ILP solve when transfer calibration fell back to neutral defaults (diagnostic only)")
    parser.add_argument("--allow_fallback_graph_trace", action="store_true", help="Allow ILP solve from fallback_leaf_modules graph traces (diagnostic only)")
    parser.add_argument("--w_time", type=float, default=1.0)
    parser.add_argument("--w_energy", type=float, default=0.0)
    parser.add_argument("--w_transfer", type=float, default=1.0)
    parser.add_argument("--w_fragmentation", type=float, default=0.0, help="Regularization penalty per cut/cross-phase fragmentation event")
    parser.add_argument("--w_congestion", type=float, default=0.0, help="Weight for explicit frontier-congestion overflow penalty")
    parser.add_argument("--congestion_knee_ms", type=float, default=0.0, help="Congestion knee in aggregated cut-transfer ms (0 = auto)")
    parser.add_argument("--gpu_mem_budget_mb", type=float, default=1e18)
    parser.add_argument("--cpu_mem_budget_mb", type=float, default=1e18)
    parser.add_argument("--memory_model", choices=["nodal_sum", "peak_approx"], default="peak_approx")
    parser.add_argument("--peak_activation_overlap", type=float, default=0.35)
    parser.add_argument("--backward_meta_model_json", default=None, help="Deprecated and ignored. Backward costs now come from profiling artifacts.")
    parser.add_argument("--backward_meta_blend", type=float, default=1.0, help="Deprecated and ignored. Backward costs now come from profiling artifacts.")
    parser.add_argument("--local_refine_budget", type=int, default=0, help="Hierarchical local-search budget (max number of assignment flips)")
    parser.add_argument("--backend", choices=["auto", "pulp", "exhaustive"], default="auto")
    parser.add_argument("--hw_aggregate", choices=["max", "mean"], default="max", help="How to aggregate costs across hardware profiles")
    parser.add_argument("--hw_dispersion_k", type=float, default=0.0, help="If hw_aggregate=mean, use mean + k*std across hardware profiles")
    parser.add_argument("--output_dir", default=None, help="Output folder for ILP solution files")
    parser.add_argument("--phase4_activation", action="store_true", help="Enable Phase 4 activation-strategy optimization")
    parser.add_argument("--phase4_backend", choices=["auto", "greedy", "exhaustive"], default="auto")
    parser.add_argument("--phase4_enable_recompute", action="store_true")
    parser.add_argument("--phase4_enable_checkpoint", action="store_true")
    parser.add_argument("--phase4_w_io", type=float, default=0.0)
    parser.add_argument("--phase4_recompute_penalty", type=float, default=0.5)
    parser.add_argument("--phase4_checkpoint_penalty", type=float, default=0.3)
    parser.add_argument("--no_simulate", action="store_true", help="Disable automatic post-solve simulation")
    parser.add_argument("--simulate_mode", choices=["robust", "nominal"], default="robust")
    parser.add_argument("--strict_graph_subset", action="store_true")
    parser.add_argument("--strict_topology", action="store_true")
    args = parser.parse_args()

    load_execution_plan = None
    infer_ilp_input_paths = None
    load_graph_edges = None
    load_transfer_costs = None
    SimulationConfig = None
    simulate_plan = None
    simulate_plan_phase4 = None
    if not args.no_simulate:
        load_execution_plan = importlib.import_module("runtime.plan_representation").load_execution_plan
        infer_ilp_input_paths = importlib.import_module("runtime.plan_representation").infer_ilp_input_paths
        load_graph_edges = importlib.import_module("runtime.plan_representation").load_graph_edges
        load_transfer_costs = importlib.import_module("runtime.plan_representation").load_transfer_costs
        SimulationConfig = importlib.import_module("runtime.simulator").SimulationConfig
        simulate_plan = importlib.import_module("runtime.simulator").simulate_plan
        simulate_plan_phase4 = importlib.import_module("runtime.simulator").simulate_plan_phase4

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

    base_output_dir = config_dirs[0]
    output_dir = Path(args.output_dir) if args.output_dir else (base_output_dir / "ilp_solution")

    if args.metrics_stats_csv and args.graph_edges_csv and args.transfer_edges_csv and len(config_dirs) == 1:
        stats_csv = Path(args.metrics_stats_csv)
        graph_csv = Path(args.graph_edges_csv)
        transfer_csv = Path(args.transfer_edges_csv)
        data = load_ilp_inputs(
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
    else:
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

    if args.backward_meta_model_json:
        print(
            "[deprecated] Ignoring --backward_meta_model_json/--backward_meta_blend; "
            "backward costs now come directly from profiling artifacts."
        )

    if args.phase4_activation:
        cfg4 = ILPConfig4(
            w_time=args.w_time,
            w_energy=args.w_energy,
            w_transfer=args.w_transfer,
            w_fragmentation=args.w_fragmentation,
            w_congestion=args.w_congestion,
            congestion_knee_ms=args.congestion_knee_ms,
            gpu_mem_budget_mb=args.gpu_mem_budget_mb,
            cpu_mem_budget_mb=args.cpu_mem_budget_mb,
            memory_model=args.memory_model,
            peak_activation_overlap=args.peak_activation_overlap,
            w_io=args.phase4_w_io,
            w_recompute_penalty=args.phase4_recompute_penalty,
            w_checkpoint_penalty=args.phase4_checkpoint_penalty,
            enable_recompute=args.phase4_enable_recompute,
            enable_checkpoint=args.phase4_enable_checkpoint,
        )
        sol = solve_partition_ilp_phase4(data, cfg4, backend=args.phase4_backend)
    else:
        cfg = ILPConfig(
            w_time=args.w_time,
            w_energy=args.w_energy,
            w_transfer=args.w_transfer,
            w_fragmentation=args.w_fragmentation,
            w_congestion=args.w_congestion,
            congestion_knee_ms=args.congestion_knee_ms,
            gpu_mem_budget_mb=args.gpu_mem_budget_mb,
            cpu_mem_budget_mb=args.cpu_mem_budget_mb,
            memory_model=args.memory_model,
            peak_activation_overlap=args.peak_activation_overlap,
        )
        sol = solve_partition_ilp(data, cfg, backend=args.backend)

    if args.local_refine_budget > 0:
        refine_cfg = ILPConfig(
            w_time=args.w_time,
            w_energy=args.w_energy,
            w_transfer=args.w_transfer,
            w_fragmentation=args.w_fragmentation,
            gpu_mem_budget_mb=args.gpu_mem_budget_mb,
            cpu_mem_budget_mb=args.cpu_mem_budget_mb,
            memory_model=args.memory_model,
            peak_activation_overlap=args.peak_activation_overlap,
        )
        sol = refine_solution_hierarchical_local(
            data=data,
            cfg=refine_cfg,
            base_solution=sol,
            max_assignment_changes=args.local_refine_budget,
        )
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
    print(f"Fragmentation regularizer: {args.w_fragmentation}")
    print(f"Memory model: {args.memory_model} (peak_activation_overlap={args.peak_activation_overlap})")
    print(f"Congestion term: w_congestion={args.w_congestion}, knee_ms={args.congestion_knee_ms} (0=auto)")
    if args.local_refine_budget > 0:
        print(f"Local hierarchical refinement budget: {args.local_refine_budget}")
    if len(config_dirs) > 1:
        print(f"Hardware profiles merged: {len(config_dirs)}")
        print(f"Aggregation: {args.hw_aggregate} (k={args.hw_dispersion_k})")
    print(f"Assignment CSV: {out['assignment_csv']}")
    print(f"Cut edges CSV: {out['cut_edges_csv']}")
    print(f"Summary JSON: {out['summary_json']}")

    solve_status_ok = str(sol.status).lower() in {"optimal", "feasible"}
    exit_code = 0 if solve_status_ok else 2

    if not args.no_simulate:
        if not solve_status_ok or not sol.assignment:
            print("-" * 80)
            print("POST-SOLVE SIMULATION")
            print("-" * 80)
            print(
                "Simulation skipped: ILP solution is not feasible/optimal "
                f"(status={sol.status})."
            )
        else:
            sim_plan = load_execution_plan(
                assignment_csv=out["assignment_csv"],
                cut_edges_csv=out["cut_edges_csv"],
            )

            inferred = infer_ilp_input_paths(config_dir=config_dirs[0], model_name=args.model)
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
                mode=args.simulate_mode,
                k_sigma=args.k_sigma,
                k_sigma_time=args.k_sigma_time,
                k_sigma_energy=args.k_sigma_energy,
                w_time=args.w_time,
                w_energy=args.w_energy,
                w_transfer=args.w_transfer,
                gpu_mem_budget_mb=args.gpu_mem_budget_mb,
                cpu_mem_budget_mb=args.cpu_mem_budget_mb,
                memory_model=args.memory_model,
                peak_activation_overlap=args.peak_activation_overlap,
                strict_transfer_mapping=args.strict_transfer_mapping,
                strict_graph_subset=args.strict_graph_subset,
                strict_topology=args.strict_topology,
            )

            if getattr(sim_plan, "activation_strategies", None):
                sim_result = simulate_plan_phase4(
                    plan=sim_plan,
                    metrics_stats_csv=inferred.metrics_stats_csv,
                    graph_edges=graph_edges,
                    transfer_costs=transfer_costs,
                    cfg=sim_cfg,
                    activation_strategies=sim_plan.activation_strategies,
                )
            else:
                sim_result = simulate_plan(
                    plan=sim_plan,
                    metrics_stats_csv=inferred.metrics_stats_csv,
                    graph_edges=graph_edges,
                    transfer_costs=transfer_costs,
                    cfg=sim_cfg,
                )

            sim_out_dir = output_dir / "simulation"
            sim_out_dir.mkdir(parents=True, exist_ok=True)

            sim_summary_path = sim_out_dir / "simulation_summary.json"
            import json

            with open(sim_summary_path, "w") as f:
                json.dump(
                    {
                        **sim_result.to_dict(),
                        "inputs": {
                            "metrics_stats_csv": str(inferred.metrics_stats_csv),
                            "graph_edges_csv": str(inferred.graph_edges_csv),
                            "transfer_edges_csv": str(inferred.transfer_edges_csv),
                        },
                    },
                    f,
                    indent=4,
                )

            print("-" * 80)
            print("POST-SOLVE SIMULATION")
            print("-" * 80)
            print(f"Simulation status: {sim_result.status}")
            print(f"Simulation objective: {sim_result.objective_value:.6f}")
            print(f"Simulation summary JSON: {sim_summary_path}")
            if sim_result.warnings:
                print("Simulation warnings:")
                for msg in sim_result.warnings:
                    print(f"  - {msg}")
            if sim_result.violations:
                print("Simulation violations:")
                for msg in sim_result.violations:
                    print(f"  - {msg}")
            if sim_result.status != "ok":
                exit_code = 2
    print("=" * 80)

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
