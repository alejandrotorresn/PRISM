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
_build_effective_memory = importlib.import_module("ilp.model_builder")._build_effective_memory
solve_partition_ilp = importlib.import_module("ilp.solve").solve_partition_ilp
solve_partition_ilp_phase4 = importlib.import_module("ilp.solve").solve_partition_ilp_phase4
refine_solution_hierarchical_local = importlib.import_module("ilp.solve").refine_solution_hierarchical_local
PreparedTopologyMemory = importlib.import_module("ilp.topology_memory").PreparedTopologyMemory
correct_solution_memory = importlib.import_module("ilp.topology_memory").correct_solution_memory
build_ordered_candidate_blocks = importlib.import_module("runner.recompute_curve_calibration").build_ordered_candidate_blocks
select_gpu_resident_candidate_blocks = importlib.import_module("runner.recompute_curve_calibration").select_gpu_resident_candidate_blocks
measure_recompute_curve = importlib.import_module("runner.recompute_curve_calibration").measure_recompute_curve
measure_real_peak_for_plan = importlib.import_module("runner.hybrid_overhead_calibration").measure_real_peak_for_plan


def _default_paths(config_dir: Path, model_name: str):
    stats = config_dir / f"{model_name}_metrics_stats.csv"
    if not stats.exists():
        stats = config_dir / "metrics_stats.csv"

    run_dirs = sorted([p for p in config_dir.glob("run_*") if p.is_dir()])
    ref_run = None
    for rdir in run_dirs:
        if (rdir / f"{model_name}_graph_edges.csv").exists() and (rdir / f"{model_name}_transfer_edges.csv").exists():
            ref_run = rdir
            break

    if ref_run:
        graph_edges = ref_run / f"{model_name}_graph_edges.csv"
        transfer_edges = ref_run / f"{model_name}_transfer_edges.csv"
    else:
        graph_edges = config_dir / f"{model_name}_graph_edges.csv"
        transfer_edges = config_dir / f"{model_name}_transfer_edges.csv"

    if not graph_edges.exists() or not transfer_edges.exists():
        raise FileNotFoundError(
            "Could not resolve graph/transfer artifacts in config_dir. "
            f"Expected either valid run_*/ files or direct files in: {config_dir}"
        )
    return stats, graph_edges, transfer_edges


def _resolve_hybrid_overhead_ratio(config_dirs: "list[Path]", model_name: str) -> float:
    # Averages the measured hybrid_overhead_ratio (see
    # src/runner/hybrid_overhead_calibration.py) across every run_*/ this
    # model/batch was profiled with. Each value is a REAL measurement
    # (memory_reserved()/memory_allocated() from one real forward+backward
    # pass through PRISM's own hybrid execution engine, not a plain native
    # call) -- never a hardcoded constant. Only "status": "ok" entries are
    # counted; a missing file or a non-"ok" status for a given run simply
    # does not contribute (older profiling runs, from before this
    # calibration pass existed, silently fall back to ratio=1.0, i.e. no
    # correction, rather than failing).
    import json

    ratios: list[float] = []
    for cdir in config_dirs:
        for run_dir in sorted(cdir.glob("run_*")):
            candidate = run_dir / f"{model_name}_hybrid_overhead.json"
            if not candidate.exists():
                continue
            try:
                payload = json.loads(candidate.read_text())
            except (OSError, ValueError):
                continue
            if payload.get("status") == "ok":
                ratio = payload.get("hybrid_overhead_ratio")
                if isinstance(ratio, (int, float)) and ratio >= 1.0:
                    ratios.append(float(ratio))

    if not ratios:
        return 1.0
    return sum(ratios) / len(ratios)


def _load_calibration_model(args: argparse.Namespace):
    """Loads the real model+input used to validate a candidate plan against
    the hybrid engine's REAL transient memory overhead -- shared by
    _validate_gpu_expansion_against_real_overhead and
    _validate_final_plan_against_real_overhead so the model is loaded (and
    its CUDA memory paid for) at most once per run_ilp_partition.py
    invocation, regardless of how many of those two validations run.

    Returns (model, example_input) on success, or (None, None) with a
    printed warning on any failure (no CUDA, no --calibration_batch_size, or
    the real model failing to load) -- never raises. Callers treat (None,
    None) as "validation unavailable, fall back to topology_memory alone",
    same posture every calibration-dependent feature in this project uses.
    """
    import torch

    if not torch.cuda.is_available():
        print(
            "[real_overhead] WARNING: CUDA not available -- cannot validate any plan against the "
            "hybrid engine's real transient overhead, relying on topology_memory only.",
            flush=True,
        )
        return None, None

    if not args.calibration_batch_size:
        print(
            "[real_overhead] WARNING: no --calibration_batch_size set -- cannot validate any plan "
            "against the hybrid engine's real transient overhead, relying on topology_memory only "
            "(see fix/08 README).",
            flush=True,
        )
        return None, None

    build_model_input_target = importlib.import_module("models.factory").build_model_input_target
    from types import SimpleNamespace

    mf_args = SimpleNamespace(
        model=args.model,
        precision=args.calibration_precision,
        batch_size=args.calibration_batch_size,
        input_size=224,
        seq_length=args.calibration_seq_length,
        datasets_root=args.calibration_datasets_root,
        require_datasets=False,
    )
    torch_dtype = torch.bfloat16 if args.calibration_precision == "bf16" else torch.float32
    try:
        model, inp, _target, _data_info = build_model_input_target(mf_args, torch_dtype)
        model = model.cuda()
        if isinstance(inp, dict):
            inp = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in inp.items()}
        elif torch.is_tensor(inp):
            inp = inp.cuda()
        return model, inp
    except Exception as exc:
        print(
            f"[real_overhead] WARNING: could not load the real model for real-overhead validation "
            f"({exc}) -- relying on topology_memory only.",
            flush=True,
        )
        return None, None


def _validate_gpu_expansion_against_real_overhead(
    expanded: dict,
    base_fwd: dict,
    base_bwd: dict,
    data,
    args: argparse.Namespace,
    activation_strategy,
    prepared_topology: "PreparedTopologyMemory",
    calib_model,
    calib_inp,
) -> dict:
    """Re-validates expand_solution_memory's candidate plan against the
    hybrid execution engine's REAL transient memory overhead (allocator
    fragmentation from async transfer/prefetch, cuDNN algorithm selection),
    which topology_memory.py's sequential walk never models -- see
    fix/08_gpu_memory_expansion_and_underestimation/README.md: a generic,
    pre-expansion overhead ratio (src/runner/hybrid_overhead_calibration.py)
    does not describe the specific cut pattern expansion actually produces,
    and trusting it stalled a real resnet50 training run in an infinite
    CUDA OOM retry loop.

    Measures the EXACT candidate plan with measure_real_peak_for_plan
    (same hybrid execution engine, one real forward+backward pass), and
    binary-searches over how many of expanded["flip_order"]'s moves (in
    order) to keep, so the final plan is backed by a real measurement
    instead of only topology_memory's estimate. Falls back to `expanded`
    unchanged (with real_reserved_mb=None) when there is nothing to
    validate, or when calibration prerequisites are unavailable -- same
    behavior as before this validation existed, but explicitly flagged as
    unvalidated rather than silently assumed safe.
    """
    flip_order = expanded.get("flip_order") or []
    if not flip_order:
        return {**expanded, "real_reserved_mb": None}

    if calib_model is None:
        # _load_calibration_model already printed why (no CUDA, no
        # --calibration_batch_size, or model load failure).
        return {**expanded, "real_reserved_mb": None}

    model, inp = calib_model, calib_inp

    def _plan_for_k(k: int) -> tuple[dict, dict]:
        fwd_k = dict(base_fwd)
        bwd_k = dict(base_bwd)
        for name in flip_order[:k]:
            fwd_k[name] = "GPU"
            bwd_k[name] = "GPU"
        return fwd_k, bwd_k

    def _measure(k: int):
        fwd_k, bwd_k = _plan_for_k(k)
        return measure_real_peak_for_plan(
            model=model,
            example_input=inp,
            forward_assignment=fwd_k,
            backward_assignment=bwd_k,
            activation_strategy=activation_strategy,
            gpu_id=0,
        )

    n = len(flip_order)
    measured0 = _measure(0)
    reserved0 = measured0.get("real_reserved_mb")
    feasible0 = measured0["status"] == "ok" and reserved0 is not None and reserved0 <= args.gpu_mem_budget_mb
    best_k, best_measured = 0, measured0
    if not feasible0:
        print(
            f"[gpu_expansion] WARNING: even the PRE-expansion plan exceeds the real measured peak "
            f"({reserved0}MB vs budget {args.gpu_mem_budget_mb}MB, status={measured0['status']}) -- "
            "this points to a blind spot in topology_memory.py/correct_solution_memory itself, "
            "outside this validation's scope. Using the pre-expansion plan as-is (no worse than "
            "before this validation existed).",
            flush=True,
        )
    else:
        measuredN = _measure(n)
        reservedN = measuredN.get("real_reserved_mb")
        if measuredN["status"] == "ok" and reservedN is not None and reservedN <= args.gpu_mem_budget_mb:
            best_k, best_measured = n, measuredN
        else:
            lo, hi = 0, n
            while hi - lo > 1:
                mid = (lo + hi) // 2
                measured_mid = _measure(mid)
                reserved_mid = measured_mid.get("real_reserved_mb")
                if measured_mid["status"] == "ok" and reserved_mid is not None and reserved_mid <= args.gpu_mem_budget_mb:
                    lo, best_k, best_measured = mid, mid, measured_mid
                else:
                    hi = mid

    fwd_final, bwd_final = _plan_for_k(best_k)
    peak_gpu, peak_cpu = prepared_topology.evaluate(fwd_final, bwd_final, activation_strategy)
    print(
        f"[gpu_expansion] Real-overhead validation: kept {best_k}/{n} expansion move(s) "
        f"(real_reserved_mb={best_measured.get('real_reserved_mb')})",
        flush=True,
    )
    return {
        "forward_assignment": fwd_final,
        "backward_assignment": bwd_final,
        "peak_gpu_mb": peak_gpu,
        "peak_cpu_mb": peak_cpu,
        "flips_applied": best_k,
        "flip_order": flip_order[:best_k],
        "feasible": peak_gpu <= args.gpu_mem_budget_mb and peak_cpu <= args.cpu_mem_budget_mb,
        "real_reserved_mb": best_measured.get("real_reserved_mb"),
    }


def _validate_final_plan_against_real_overhead(
    final_fwd: dict,
    final_bwd: dict,
    activation_strategy,
    data,
    args: argparse.Namespace,
    prepared_topology: "PreparedTopologyMemory",
    calib_model,
    calib_inp,
) -> Optional[dict]:
    """Measures the plan that is ABOUT to be used -- after expansion (if any)
    and correct_solution_memory's own repair have both already run -- with
    ONE real forward+backward pass, and uses that measurement for two
    different things with the metric each actually needs (see fix/08
    README's "duplicación de buffers de transferencia" finding):

      - real_reserved_mb (what the CUDA allocator actually reserves) is
        compared against args.gpu_mem_budget_mb to decide feasibility --
        topology_memory.py's sequential walk structurally cannot see the
        duplicate cross-device buffer async-transfer/prefetch keeps alive
        at each cut edge (src/runtime/hybrid_executor.py lines ~2093-2116),
        which got measurably worse (-29.8% -> -35.0% real-vs-predicted
        error for resnet50 batch_192) specifically when a tight budget
        forces a cut right at the network's largest-activation layer.
      - real_allocated_mb (tensor payload only) replaces the reported
        gpu_mem_used_mb -- it is the SAME metric hybrid_execution_summary.json's
        peak_gpu_mem_mb reports (torch.cuda.max_memory_allocated, see
        hybrid_executor.py:2320), so "predicted" and "real" become a
        consistent, comparable pair again instead of the allocated-vs-
        reserved mismatch that originally made this gap look like it had
        gotten worse than it had.

    If the real measurement exceeds budget (not observed in production as
    of this writing, but not assumed away either), binary-searches removing
    the biggest GPU-resident layers first (prepared_topology.layer_gpu_weight,
    descending -- the exact ranking correct_solution_memory already uses)
    until a real measurement confirms it fits.

    Returns None (caller makes no changes) when there is nothing to measure
    with (no CUDA / no --calibration_batch_size / model load failed --
    calib_model is None, already warned by _load_calibration_model).
    """
    if calib_model is None:
        return None

    def _measure(fwd: dict, bwd: dict):
        return measure_real_peak_for_plan(
            model=calib_model,
            example_input=calib_inp,
            forward_assignment=fwd,
            backward_assignment=bwd,
            activation_strategy=activation_strategy,
            gpu_id=0,
        )

    measured = _measure(final_fwd, final_bwd)
    reserved = measured.get("real_reserved_mb")
    feasible = measured["status"] == "ok" and reserved is not None and reserved <= args.gpu_mem_budget_mb

    fwd, bwd = dict(final_fwd), dict(final_bwd)
    flips_removed = 0

    if not feasible:
        print(
            f"[real_overhead] WARNING: the final plan (after topology_memory's own repair) exceeds "
            f"the real measured peak ({reserved}MB vs budget {args.gpu_mem_budget_mb}MB, "
            f"status={measured['status']}) -- topology_memory's repair was not enough. Trimming "
            "further using real measurements.",
            flush=True,
        )
        removal_order = sorted(
            {name for name, dev in fwd.items() if dev == "GPU"} | {name for name, dev in bwd.items() if dev == "GPU"},
            key=prepared_topology.layer_gpu_weight,
            reverse=True,
        )
        n = len(removal_order)

        def _plan_removing(k: int) -> tuple[dict, dict]:
            fwd_k, bwd_k = dict(final_fwd), dict(final_bwd)
            for name in removal_order[:k]:
                fwd_k[name] = "CPU"
                bwd_k[name] = "CPU"
            return fwd_k, bwd_k

        # Binary search the smallest k in [0, n] (biggest-layers-first
        # removal order) whose real measurement fits. k=0 is already known
        # infeasible (that's why we're in this branch).
        lo, hi = 0, n
        measured = _measure(*_plan_removing(n))
        if measured["status"] == "ok" and measured.get("real_reserved_mb") is not None and measured["real_reserved_mb"] <= args.gpu_mem_budget_mb:
            while hi - lo > 1:
                mid = (lo + hi) // 2
                m = _measure(*_plan_removing(mid))
                r = m.get("real_reserved_mb")
                if m["status"] == "ok" and r is not None and r <= args.gpu_mem_budget_mb:
                    hi, measured = mid, m
                else:
                    lo = mid
        else:
            # Even moving every GPU layer to CPU does not fit for real --
            # degenerate case, use it anyway (best effort, logged loudly).
            hi = n
        flips_removed = hi
        fwd, bwd = _plan_removing(hi)

    print(
        f"[real_overhead] Final plan validated: real_allocated_mb={measured.get('real_allocated_mb')} "
        f"real_reserved_mb={measured.get('real_reserved_mb')} (budget={args.gpu_mem_budget_mb:.1f}MB)"
        + (f", removed {flips_removed} additional layer(s)" if flips_removed else ""),
        flush=True,
    )
    return {
        "forward_assignment": fwd,
        "backward_assignment": bwd,
        "flips_removed": flips_removed,
        "real_allocated_mb": measured.get("real_allocated_mb"),
        "real_reserved_mb": measured.get("real_reserved_mb"),
        "feasible": measured["status"] == "ok"
        and measured.get("real_reserved_mb") is not None
        and measured["real_reserved_mb"] <= args.gpu_mem_budget_mb,
    }


def _solve_with_calibrated_recompute_curve(data, args: argparse.Namespace, base_cfg4: "ILPConfig4"):
    """Automates the 4-step flow documented in
    fix/07_gpu_recompute_joint_decision/README.md: recompute's real memory
    benefit is non-additive and depends on which OTHER layers are already
    GPU-resident (confirmed empirically: a curve calibrated assuming "every
    layer on GPU" does not transfer to a real, tightly memory-constrained
    case where most layers end up on CPU). So:
      1. Solve a no-recompute baseline first, to get the REAL CPU/GPU split.
      2. Keep only candidate blocks that baseline already put fully on GPU
         (recompute only relieves memory for GPU-resident nodes).
      3. Measure the REAL recompute curve in THAT device context.
      4. Re-solve with backend="joint" using the calibrated curve.
    Never raises and never silently assumes an unmeasured benefit: any
    failure along the way (no GPU-resident candidates, CUDA unavailable,
    calibration error) logs why and falls back to the plain no-recompute
    baseline solution.
    """
    baseline_cfg4 = ILPConfig4(**{**base_cfg4.__dict__, "enable_recompute": False, "enable_checkpoint": False})
    baseline_sol = solve_partition_ilp_phase4(data, baseline_cfg4, backend="greedy")
    if baseline_sol.status not in ("optimal", "feasible"):
        return baseline_sol  # nothing to calibrate against an infeasible/failed baseline

    baseline_fwd = dict(baseline_sol.forward_assignment or baseline_sol.assignment)
    baseline_bwd = dict(baseline_sol.backward_assignment or baseline_sol.assignment)

    _, _, _, gpu_act_mem, _, _ = _build_effective_memory(data, baseline_cfg4)
    all_blocks = build_ordered_candidate_blocks(
        leaf_names=data.nodes, gpu_act_mem=gpu_act_mem, shared_call_site_layers=data.shared_call_site_layers,
    )
    candidate_blocks = select_gpu_resident_candidate_blocks(all_blocks, baseline_fwd)
    if not candidate_blocks:
        print(
            "[recompute_curve] No candidate block is fully GPU-resident in the no-recompute "
            "baseline; skipping calibration, using the baseline solution as-is.",
            flush=True,
        )
        return baseline_sol

    import torch
    from types import SimpleNamespace

    if not torch.cuda.is_available():
        print("[recompute_curve] CUDA not available; skipping calibration, using the baseline solution as-is.", flush=True)
        return baseline_sol

    build_model_input_target = importlib.import_module("models.factory").build_model_input_target
    mf_args = SimpleNamespace(
        model=args.model,
        precision=args.calibration_precision,
        batch_size=args.calibration_batch_size,
        input_size=224,
        seq_length=args.calibration_seq_length,
        datasets_root=args.calibration_datasets_root,
        require_datasets=False,
    )
    torch_dtype = torch.bfloat16 if args.calibration_precision == "bf16" else torch.float32
    model, inp, target, _data_info = build_model_input_target(mf_args, torch_dtype)
    model = model.cuda()
    if isinstance(inp, dict):
        inp = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in inp.items()}
    elif torch.is_tensor(inp):
        inp = inp.cuda()
    if torch.is_tensor(target):
        target = target.cuda()

    fractions = tuple(float(x) for x in str(args.calibration_fractions).split(",") if x.strip())
    result = measure_recompute_curve(
        model=model,
        example_input=inp,
        candidate_blocks_ordered=candidate_blocks,
        target_data=target,
        fractions=fractions,
        baseline_forward_assignment=baseline_fwd,
        baseline_backward_assignment=baseline_bwd,
    )
    if result.status != "ok" or len(result.points) < 2:
        print(
            f"[recompute_curve] Calibration did not produce a usable curve "
            f"(status={result.status} reason={result.reason}); using the baseline solution as-is.",
            flush=True,
        )
        return baseline_sol

    curve_points = [(p.n_blocks, p.peak_gpu_mem_mb) for p in result.points]
    print(
        f"[recompute_curve] Calibrated {len(candidate_blocks)} GPU-resident candidate block(s) "
        f"(of {len(all_blocks)} total); curve points (n_blocks, peak_mb): {curve_points}",
        flush=True,
    )

    joint_cfg4 = ILPConfig4(**{
        **base_cfg4.__dict__,
        "enable_recompute": True,
        "recompute_block_order": candidate_blocks,
        "recompute_curve_points": curve_points,
    })
    return solve_partition_ilp_phase4(data, joint_cfg4, backend="joint")


def _resolve_regime_and_weight_policy(args: argparse.Namespace) -> tuple[str, bool]:
    regime = str(args.regime).strip().lower()
    if regime not in {"deterministic", "diagnostic"}:
        raise ValueError(f"Unsupported --regime value: {args.regime}")

    if regime == "deterministic":
        if args.allow_low_quality_stats:
            raise ValueError("Deterministic regime does not allow --allow_low_quality_stats")
        if args.allow_fallback_graph_trace:
            raise ValueError("Deterministic regime does not allow --allow_fallback_graph_trace")
        if not args.strict_graph_mapping:
            raise ValueError("Deterministic regime requires --strict_graph_mapping")
        if not args.strict_transfer_mapping:
            raise ValueError("Deterministic regime requires --strict_transfer_mapping")

    enforce_convex = bool(args.enforce_convex_weights or regime == "deterministic")
    if enforce_convex:
        weight_sum = float(args.w_time + args.w_energy)
        if abs(weight_sum - 1.0) > 1e-9:
            raise ValueError(
                "Convex weighting required: w_time + w_energy must be 1.0; "
                f"got {weight_sum} (w_time={args.w_time}, w_energy={args.w_energy})"
            )

    return regime, enforce_convex


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
    parser.add_argument("--strict_metric_validity", action="store_true", help="Require strictly positive CPU/GPU mean timings for all layers")
    parser.add_argument("--w_time", type=float, default=1.0)
    parser.add_argument("--w_energy", type=float, default=0.0)
    parser.add_argument("--w_transfer", type=float, default=1.0)
    parser.add_argument("--w_fragmentation", type=float, default=0.0, help="Regularization penalty per cut/cross-phase fragmentation event")
    parser.add_argument("--w_congestion", type=float, default=0.0, help="Weight for explicit frontier-congestion overflow penalty")
    parser.add_argument("--congestion_knee_ms", type=float, default=0.0, help="Congestion knee in aggregated cut-transfer ms (0 = auto)")
    parser.add_argument("--gpu_mem_budget_mb", type=float, default=1e18)
    parser.add_argument("--cpu_mem_budget_mb", type=float, default=1e18)
    parser.add_argument(
        "--disable_hybrid_overhead_correction",
        action="store_true",
        help="Skip correcting gpu_mem_budget_mb by the measured hybrid-execution "
        "memory overhead ratio (src/runner/hybrid_overhead_calibration.py). "
        "Default is to apply the correction when a measurement is available.",
    )
    parser.add_argument(
        "--memory_model",
        choices=["topological", "topological_tight"],
        default="topological",
    )
    parser.add_argument("--peak_activation_overlap", type=float, default=0.35)
    parser.add_argument(
        "--disable_topology_memory_correction",
        action="store_true",
        help="Skip the post-solve real topological peak-memory check/repair "
        "(src/ilp/topology_memory.py). Default is to replace the solver's own "
        "(approximate, aggregate-sum) gpu_mem_used_mb/cpu_mem_used_mb with the "
        "real sequential-walk estimate, and greedily move layers to CPU if the "
        "solver's plan doesn't actually fit under it. Only available for a "
        "single config_dir (not multi-hardware --config_dirs merges).",
    )
    parser.add_argument(
        "--topology_memory_repair_budget",
        type=int,
        default=200,
        help="Max number of layer flips the topology-memory repair loop may "
        "apply when the solver's plan doesn't fit the real estimate.",
    )
    parser.add_argument(
        "--enable_gpu_expansion",
        action="store_true",
        help="Before the shrink-only topology-memory repair above, greedily move CPU-resident "
        "layers (highest real time-savings first) to GPU as long as the REAL (topology_memory) "
        "peak still fits the budget. Off by default -- addresses a confirmed, measured gap where "
        "the MILP's own memory constraint (a simultaneous-sum over-approximation, see "
        "src/ilp/solve.py) believes the GPU budget is fully used when the real sequential peak is "
        "only 40-55% of it on resnet50, leaving most of the network needlessly on CPU. See "
        "src/ilp/topology_memory.py::expand_solution_memory/rebalance_solution_memory.",
    )
    parser.add_argument(
        "--gpu_expansion_budget",
        type=int,
        default=200,
        help="Max number of CPU->GPU layer moves --enable_gpu_expansion may apply.",
    )
    parser.add_argument("--backward_meta_model_json", default=None, help="Deprecated and ignored. Backward costs now come from profiling artifacts.")
    parser.add_argument("--backward_meta_blend", type=float, default=1.0, help="Deprecated and ignored. Backward costs now come from profiling artifacts.")
    parser.add_argument("--local_refine_budget", type=int, default=0, help="Hierarchical local-search budget (max number of assignment flips)")
    parser.add_argument("--backend", choices=["auto", "pulp", "exhaustive"], default="auto")
    parser.add_argument("--hw_aggregate", choices=["max", "mean"], default="max", help="How to aggregate costs across hardware profiles")
    parser.add_argument("--hw_dispersion_k", type=float, default=0.0, help="If hw_aggregate=mean, use mean + k*std across hardware profiles")
    parser.add_argument("--output_dir", default=None, help="Output folder for ILP solution files")
    parser.add_argument("--phase4_activation", action="store_true", help="Enable Phase 4 activation-strategy optimization")
    parser.add_argument("--phase4_backend", choices=["auto", "greedy", "exhaustive", "joint"], default="auto")
    parser.add_argument(
        "--phase4_calibrate_recompute_curve",
        action="store_true",
        help="Automate the 4-step flow in fix/07_gpu_recompute_joint_decision/README.md: solve a "
        "no-recompute baseline, measure the REAL (non-additive) recompute memory curve for the "
        "blocks that baseline already put fully on GPU, then re-solve with --phase4_backend joint "
        "using that calibrated curve. Requires --phase4_backend joint and --calibration_batch_size. "
        "Loads the real model to measure real GPU memory -- needs CUDA; falls back to the plain "
        "no-recompute baseline (never silently assumes an unmeasured benefit) if CUDA is "
        "unavailable, no candidate block is GPU-resident in the baseline, or calibration fails.",
    )
    parser.add_argument(
        "--calibration_batch_size",
        type=int,
        default=None,
        help="Batch size for the real model+input used to calibrate the recompute curve "
        "(required with --phase4_calibrate_recompute_curve; independent of --gpu_mem_budget_mb).",
    )
    parser.add_argument("--calibration_precision", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--calibration_seq_length", type=int, default=128, help="For sequence models (gpt2_small, distilgpt2, bert_base)")
    parser.add_argument("--calibration_datasets_root", default="datasets")
    parser.add_argument(
        "--calibration_fractions",
        default="0.0,0.25,0.5,0.75,1.0",
        help="Comma-separated fractions of GPU-resident candidate blocks to probe when "
        "calibrating the recompute curve.",
    )
    parser.add_argument("--phase4_enable_recompute", action="store_true")
    parser.add_argument("--phase4_enable_checkpoint", action="store_true")
    parser.add_argument("--phase4_w_io", type=float, default=0.0)
    parser.add_argument("--phase4_recompute_penalty", type=float, default=0.5)
    parser.add_argument("--phase4_checkpoint_penalty", type=float, default=0.3)
    parser.add_argument(
        "--phase4_target_gpu_utilization",
        type=float,
        default=0.95,
        help="Fraction of gpu_mem_budget_mb the Phase 4 greedy heuristic "
        "(src/ilp/solve.py::_solve_phase4_greedy) aims to use once triggered "
        "-- default 0.95: the baseline solve already enforces gpu_mem_budget_mb "
        "as a hard constraint, so this should only trigger extra recompute/"
        "checkpoint near the real edge of that budget, not as an independent "
        "~40%% haircut on an already-correct, already-feasible solution "
        "(see src/ilp/model_builder.py::ILPConfig4.target_gpu_utilization).",
    )
    parser.add_argument("--regime", choices=["deterministic", "diagnostic"], default="diagnostic")
    parser.add_argument("--enforce_convex_weights", action="store_true", help="Require w_time + w_energy == 1")
    args = parser.parse_args()

    regime, enforce_convex_weights = _resolve_regime_and_weight_policy(args)

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
            strict_metric_validity=bool(args.strict_metric_validity),
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
                strict_metric_validity=bool(args.strict_metric_validity),
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

    # Paths for the post-solve real topological memory check (see below).
    # Only meaningful for a single config_dir: a multi-hardware merge has no
    # single graph_edges.csv/metrics_stats.csv pair the real walk can use.
    topology_csvs = (str(stats_csv), str(graph_csv)) if len(config_dirs) == 1 else None

    if args.memory_model == "topological_tight":
        # The tight per-critical-point GPU memory constraint (see
        # src/ilp/solve.py::_build_tight_topological_gpu_constraints,
        # fix/09_.../README.md) needs the SAME PreparedTopologyMemory the
        # post-solve repair already builds from topology_csvs -- built here
        # (before solving, not after) so the solver itself can use it.
        if topology_csvs is None:
            raise ValueError(
                "--memory_model topological_tight requires a single --config_dir "
                "(a multi-hardware merge has no single graph_edges.csv/metrics_stats.csv "
                "the real topological walk can use)."
            )
        stats_csv_path, graph_csv_path = topology_csvs
        data.prepared_topology = PreparedTopologyMemory.prepare(graph_csv_path, stats_csv_path)

    if args.backward_meta_model_json:
        print(
            "[deprecated] Ignoring --backward_meta_model_json/--backward_meta_blend; "
            "backward costs now come directly from profiling artifacts."
        )

    if not args.disable_hybrid_overhead_correction:
        hybrid_overhead_ratio = _resolve_hybrid_overhead_ratio(config_dirs, args.model)
        if hybrid_overhead_ratio > 1.0:
            corrected_budget = args.gpu_mem_budget_mb / hybrid_overhead_ratio
            print(
                f"[hybrid_overhead] Measured hybrid-execution memory overhead ratio={hybrid_overhead_ratio:.3f} "
                f"-> effective gpu_mem_budget_mb {args.gpu_mem_budget_mb:.1f} -> {corrected_budget:.1f} "
                "(see src/runner/hybrid_overhead_calibration.py)"
            )
            args.gpu_mem_budget_mb = corrected_budget

    if args.phase4_activation:
        cfg4 = ILPConfig4(
            w_time=args.w_time,
            w_energy=args.w_energy,
            enforce_convex_weights=enforce_convex_weights,
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
            target_gpu_utilization=args.phase4_target_gpu_utilization,
        )
        if args.phase4_calibrate_recompute_curve:
            if args.phase4_backend != "joint":
                raise ValueError("--phase4_calibrate_recompute_curve requires --phase4_backend joint")
            if not args.calibration_batch_size:
                raise ValueError("--phase4_calibrate_recompute_curve requires --calibration_batch_size")
            sol = _solve_with_calibrated_recompute_curve(data, args, cfg4)
        else:
            sol = solve_partition_ilp_phase4(data, cfg4, backend=args.phase4_backend)
    else:
        cfg = ILPConfig(
            w_time=args.w_time,
            w_energy=args.w_energy,
            enforce_convex_weights=enforce_convex_weights,
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
            enforce_convex_weights=enforce_convex_weights,
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

    if topology_csvs is not None and not args.disable_topology_memory_correction and sol.status in ("optimal", "feasible"):
        stats_csv_path, graph_csv_path = topology_csvs
        prepared_topology = PreparedTopologyMemory.prepare(graph_csv_path, stats_csv_path)
        base_fwd = sol.forward_assignment or sol.assignment
        base_bwd = sol.backward_assignment or sol.assignment
        activation_strategy = getattr(sol, "activation_strategies", None)
        # Loaded at most once per invocation, shared by the expansion
        # validation below (if --enable_gpu_expansion) and the always-on
        # final-plan validation after it -- (None, None) when calibration
        # prerequisites are unavailable, both validations degrade gracefully.
        calib_model, calib_inp = _load_calibration_model(args)
        if args.enable_gpu_expansion:
            expand_solution_memory = importlib.import_module("ilp.topology_memory").expand_solution_memory
            expanded = expand_solution_memory(
                prepared=prepared_topology,
                forward_assignment=base_fwd,
                backward_assignment=base_bwd,
                activation_strategy=activation_strategy,
                node_cost_cpu_fwd_ms=data.node_cost_cpu_fwd_ms,
                node_cost_cpu_bwd_ms=data.node_cost_cpu_bwd_ms,
                node_cost_gpu_fwd_ms=data.node_cost_gpu_fwd_ms,
                node_cost_gpu_bwd_ms=data.node_cost_gpu_bwd_ms,
                gpu_budget_mb=args.gpu_mem_budget_mb,
                cpu_budget_mb=args.cpu_mem_budget_mb,
                expand_budget=args.gpu_expansion_budget,
            )
            # Re-validate against the hybrid engine's REAL transient memory
            # overhead before the shrink-only repair below -- topology_memory's
            # walk never models allocator fragmentation/async-transfer overhead,
            # and trusting it alone stalled a real resnet50 run in an infinite
            # OOM retry loop (fix/08 README). Falls back to `expanded` unchanged
            # when there is nothing to validate against (see that function).
            validated = _validate_gpu_expansion_against_real_overhead(
                expanded, base_fwd, base_bwd, data, args, activation_strategy, prepared_topology,
                calib_model, calib_inp,
            )
            correction = correct_solution_memory(
                prepared=prepared_topology,
                forward_assignment=validated["forward_assignment"],
                backward_assignment=validated["backward_assignment"],
                activation_strategy=activation_strategy,
                gpu_budget_mb=args.gpu_mem_budget_mb,
                cpu_budget_mb=args.cpu_mem_budget_mb,
                repair_budget=args.topology_memory_repair_budget,
            )
            correction["flips_applied"] = validated["flips_applied"] + correction["flips_applied"]
            correction["expand_flips_applied"] = validated["flips_applied"]
            correction["real_reserved_mb"] = validated.get("real_reserved_mb")
            print(
                f"[topology_memory] GPU expansion: {correction['expand_flips_applied']} layer(s) moved CPU->GPU"
                + (
                    f" (real_reserved_mb={correction['real_reserved_mb']:.1f}MB, validated)"
                    if correction["real_reserved_mb"] is not None
                    else " (NOT validated against real transient overhead)"
                ),
                flush=True,
            )
        else:
            correction = correct_solution_memory(
                prepared=prepared_topology,
                forward_assignment=base_fwd,
                backward_assignment=base_bwd,
                activation_strategy=activation_strategy,
                gpu_budget_mb=args.gpu_mem_budget_mb,
                cpu_budget_mb=args.cpu_mem_budget_mb,
                repair_budget=args.topology_memory_repair_budget,
            )
        print(
            f"[topology_memory] real peak: gpu={correction['peak_gpu_mb']:.1f}MB "
            f"(solver said {sol.gpu_mem_used_mb:.1f}MB) cpu={correction['peak_cpu_mb']:.1f}MB "
            f"(solver said {sol.cpu_mem_used_mb:.1f}MB); "
            f"flips_applied={correction['flips_applied']} feasible={correction['feasible']}",
            flush=True,
        )
        if correction["flips_applied"] > 0:
            new_fwd = correction["forward_assignment"]
            new_bwd = correction["backward_assignment"]
            sol.forward_assignment = new_fwd
            sol.backward_assignment = new_bwd
            sol.assignment = dict(new_fwd)
            sol.cut_edges = [e for e in data.edges if new_fwd.get(e[0]) != new_fwd.get(e[1])]
            sol.backward_cut_edges = [e for e in data.edges if new_bwd.get(e[0]) != new_bwd.get(e[1])]
            sol.cross_phase_edges = [(n, n) for n in data.nodes if new_fwd.get(n) != new_bwd.get(n)]
        sol.gpu_mem_used_mb = correction["peak_gpu_mb"]
        sol.cpu_mem_used_mb = correction["peak_cpu_mb"]
        if not correction["feasible"]:
            print(
                "[topology_memory] WARNING: real peak memory still exceeds budget after "
                f"{correction['flips_applied']} repair flips (repair_budget="
                f"{args.topology_memory_repair_budget}). Consider a smaller batch size.",
                flush=True,
            )

        # Always runs (with or without --enable_gpu_expansion) when
        # calibration prerequisites are available: topology_memory.py's
        # sequential walk cannot see the duplicate cross-device buffer
        # async-transfer/prefetch keeps alive at each cut edge, which
        # measurably worsens for tightly-constrained plans that cut right at
        # a large-activation layer (fix/08 README). Replaces the reported
        # gpu_mem_used_mb with a real measurement instead of leaving it on
        # an estimate already known to be unreliable for this exact case.
        final_check = _validate_final_plan_against_real_overhead(
            sol.forward_assignment or sol.assignment,
            sol.backward_assignment or sol.assignment,
            activation_strategy,
            data,
            args,
            prepared_topology,
            calib_model,
            calib_inp,
        )
        if final_check is not None:
            if final_check["flips_removed"] > 0:
                new_fwd = final_check["forward_assignment"]
                new_bwd = final_check["backward_assignment"]
                sol.forward_assignment = new_fwd
                sol.backward_assignment = new_bwd
                sol.assignment = dict(new_fwd)
                sol.cut_edges = [e for e in data.edges if new_fwd.get(e[0]) != new_fwd.get(e[1])]
                sol.backward_cut_edges = [e for e in data.edges if new_bwd.get(e[0]) != new_bwd.get(e[1])]
                sol.cross_phase_edges = [(n, n) for n in data.nodes if new_fwd.get(n) != new_bwd.get(n)]
            if final_check["real_allocated_mb"] is not None:
                sol.gpu_mem_used_mb = final_check["real_allocated_mb"]
            if not final_check["feasible"]:
                print(
                    "[real_overhead] WARNING: final plan still exceeds the real measured peak even "
                    "after removing every GPU-resident layer this validation could. Consider a "
                    "smaller batch size.",
                    flush=True,
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
    print(f"Regime: {regime}")
    print(f"Convex time/energy weights enforced: {enforce_convex_weights}")
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

    print("=" * 80)

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
