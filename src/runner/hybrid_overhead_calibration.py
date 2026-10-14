"""Measures the real memory overhead of PRISM's own hybrid execution engine.

Why this exists
----------------
`training_profiler.py` measures each layer's memory cost running natively
(a plain `nn.Module` call, one device, no wrapper). But PRISM's real
runtime (`src/runtime/hybrid_executor.py`) runs dual-placed layers through
`_run_layer_with_dual_placement` -- a custom `torch.autograd.Function` that
moves layers between devices, saves CPU snapshots, and reuses pinned
buffers. That machinery produces a much more fragmented CUDA allocation
pattern than a plain native forward/backward ever would.

Confirmed empirically (resnet152, batch_96, real ILP plan): the CUDA
allocator's real `memory_reserved()` peak ran 52.8% higher than
`memory_allocated()` (3,434MB vs 2,247MB) specifically when executed
through this hybrid engine -- a gap the ILP's budget never accounted for,
because its input data (the profiled per-layer costs) was always measured
natively. This module measures that gap directly, once per model, by
running ONE real forward+backward pass through the actual hybrid execution
engine (not a reimplementation) with a representative CPU/GPU split, and
reports the ratio so callers can correct the GPU memory budget they hand to
the ILP -- without changing the ILP's own optimization logic at all.

This is deliberately a SEPARATE, additive module, not a change inside
`TrainingProfiler`: a failure here must never be able to break the
profiling run that was already working. Every public entry point catches
its own exceptions and returns a "skipped"/"error" status with ratio=1.0
(no correction) instead of propagating.
"""
from __future__ import annotations

import gc
import logging
from typing import Any, Dict, List

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

try:
    from runtime.device_plan import DevicePlan
    from runtime.hybrid_executor import (
        _DeviceAwareFXInterpreter,
        _prepare_fx_run_args,
        _try_symbolic_trace,
    )
except ModuleNotFoundError:
    import sys
    from pathlib import Path

    src_dir = str(Path(__file__).resolve().parents[1])
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    from runtime.device_plan import DevicePlan
    from runtime.hybrid_executor import (
        _DeviceAwareFXInterpreter,
        _prepare_fx_run_args,
        _try_symbolic_trace,
    )


def _leaf_module_names(model: nn.Module) -> List[str]:
    return [name for name, module in model.named_modules() if name and len(list(module.children())) == 0]


def _infer_batch_size(input_data: Any) -> int:
    if isinstance(input_data, torch.Tensor):
        if input_data.dim() == 0:
            raise ValueError("Cannot infer batch size from scalar tensor input")
        return int(input_data.size(0))
    if isinstance(input_data, dict):
        candidates = [int(v.size(0)) for v in input_data.values() if isinstance(v, torch.Tensor) and v.dim() > 0]
        if not candidates:
            raise ValueError("Cannot infer batch size from dict input without batched tensors")
        return min(candidates)
    raise TypeError(f"Unsupported input_data type for batch inference: {type(input_data)}")


def _slice_batch(input_data: Any, end: int) -> Any:
    if isinstance(input_data, torch.Tensor):
        return input_data[:end]
    if isinstance(input_data, dict):
        out: Dict[str, Any] = {}
        for key, value in input_data.items():
            if isinstance(value, torch.Tensor) and value.dim() > 0 and value.size(0) >= end:
                out[key] = value[:end]
            else:
                out[key] = value
        return out
    raise TypeError(f"Unsupported input_data type for batch slicing: {type(input_data)}")


def _empty_runtime_features() -> Dict[str, Any]:
    return {
        "recompute_layers": set(),
        "checkpoint_layers": set(),
        "backward_relocation_layers": set(),
        "prefetch_layers": set(),
        "unsupported_activation_layers": set(),
        "recompute_count": 0,
        "checkpoint_count": 0,
        "backward_relocation_count": 0,
        "prefetch_count": 0,
        "prefetch_bytes": 0.0,
    }


def _build_representative_dual_plan(leaf_names: List[str]) -> "DevicePlan":
    # Alternating split: every other leaf layer stays fully GPU-resident
    # ("retain"), the rest are forward=CPU/backward=GPU (the dominant
    # direction observed in real ILP solutions -- see
    # hybrid_executor.py's objective_cross_phase_cpu_to_gpu). Not meant to
    # match any real ILP solution exactly: it only needs to exercise the
    # SAME device-switching/pinned-buffer machinery a real dual-placement
    # plan would, at a representative scale, to measure the engine's own
    # allocator overhead.
    assignment_forward: Dict[str, str] = {}
    assignment_backward: Dict[str, str] = {}
    cross_phase_edges = []
    for i, name in enumerate(leaf_names):
        if i % 2 == 0:
            assignment_forward[name] = "GPU"
            assignment_backward[name] = "GPU"
        else:
            assignment_forward[name] = "CPU"
            assignment_backward[name] = "GPU"
            cross_phase_edges.append((name, name))
    return DevicePlan(
        assignment_forward=assignment_forward,
        assignment_backward=assignment_backward,
        cut_edges_forward=[],
        cut_edges_backward=[],
        cross_phase_edges=cross_phase_edges,
        activation_strategies={},
    )


def _run_one_pass_and_measure(traced: Any, plan: "DevicePlan", x: Any, gpu_id: int) -> Dict[str, float]:
    """Runs ONE real forward+backward pass of `traced` through the actual
    hybrid execution engine with the given `plan`, and returns the real
    peak allocated/reserved memory (MB). Shared by both
    `_run_one_calibration_pass` (synthetic representative plan) and
    `measure_real_peak_for_plan` (an exact candidate plan) so the OOM
    handling/instrumentation lives in exactly one place. Raises
    torch.cuda.OutOfMemoryError (left to the caller) or any other
    exception from a genuinely broken plan/model -- callers wrap this.
    """
    # See fix/08_gpu_memory_expansion_and_underestimation/README.md: reference
    # cycles through cross-device autograd bookkeeping (confirmed empirically
    # in convergence_validation/prism_loop.py: +73.6MB/step linear growth,
    # reclaimed in full by gc.collect(), independent of CUDA allocator
    # fragmentation) leave real, unreclaimed CUDA memory behind after a
    # forward+backward pass. This function is called REPEATEDLY in the same
    # process by a binary search (see run_ilp_partition.py's
    # _validate_gpu_expansion_against_real_overhead/_validate_final_plan_against_real_overhead),
    # so a leftover cycle from one call contaminates every later call's own
    # "real" measurement -- biasing the search toward rejecting candidates
    # that would actually fit. gc.collect() before AND after each probe, same
    # convention already established in recompute_curve_calibration.py.
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(gpu_id)
    torch.cuda.reset_peak_memory_stats(gpu_id)

    pinned_cache: Dict[str, torch.Tensor] = {}
    interp = _DeviceAwareFXInterpreter(
        traced,
        plan,
        gpu_id,
        {"bytes": 0.0, "events": 0.0},
        {},
        [],
        _empty_runtime_features(),
        True,  # enable_async_transfer
        True,  # enable_prefetch
        None,  # transfer_stream
        pinned_output_buffer_cache=pinned_cache,
    )
    out = interp.run(*_prepare_fx_run_args(traced, x))
    loss = out.sum() if torch.is_tensor(out) else next(v for v in out if torch.is_tensor(v)).sum()
    loss.backward()
    torch.cuda.synchronize(gpu_id)

    result = {
        "allocated_mb": torch.cuda.max_memory_allocated(gpu_id) / (1024**2),
        "reserved_mb": torch.cuda.max_memory_reserved(gpu_id) / (1024**2),
    }
    del out, loss, interp, pinned_cache
    gc.collect()
    return result


def _run_one_calibration_pass(model: nn.Module, example_input: Any, gpu_id: int) -> Dict[str, Any]:
    traced = _try_symbolic_trace(model)
    if traced is None:
        return {"hybrid_overhead_ratio": 1.0, "status": "skipped", "reason": "symbolic_trace unsupported for this model"}

    leaf_names = _leaf_module_names(model)
    if len(leaf_names) < 2:
        return {"hybrid_overhead_ratio": 1.0, "status": "skipped", "reason": "too few leaf layers to build a representative split"}

    plan = _build_representative_dual_plan(leaf_names)
    target_batch = _infer_batch_size(example_input)
    current_batch = target_batch
    min_batch = 1

    while current_batch >= min_batch:
        x = _slice_batch(example_input, current_batch)
        try:
            model.to("cpu")
            measured = _run_one_pass_and_measure(traced, plan, x, gpu_id)
            allocated_mb = measured["allocated_mb"]
            reserved_mb = measured["reserved_mb"]
            ratio = (reserved_mb / allocated_mb) if allocated_mb > 1e-6 else 1.0
            return {
                "hybrid_overhead_ratio": float(ratio),
                "status": "ok",
                "reason": None,
                "calibration_batch_size": current_batch,
                "calibration_target_batch_size": target_batch,
                "calibration_allocated_mb": allocated_mb,
                "calibration_reserved_mb": reserved_mb,
            }
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            next_batch = max(min_batch, current_batch // 2)
            if next_batch == current_batch:
                break
            current_batch = next_batch
            continue

    return {"hybrid_overhead_ratio": 1.0, "status": "skipped", "reason": "calibration OOM'd even at minimum batch size"}


def measure_hybrid_overhead_ratio(model: nn.Module, example_input: Any, gpu_id: int = 0) -> Dict[str, Any]:
    """Measures memory_reserved()/memory_allocated() for one real
    forward+backward pass of `model` through PRISM's actual hybrid
    execution engine, using a representative (not model-specific) CPU/GPU
    split. Returns a dict always containing "hybrid_overhead_ratio" (float,
    1.0 if calibration could not run) and "status"
    ("ok"|"skipped"|"error"). Never raises.

    Restores `model`'s original device placement before returning,
    regardless of outcome.
    """
    if not torch.cuda.is_available():
        return {"hybrid_overhead_ratio": 1.0, "status": "skipped", "reason": "CUDA not available"}

    try:
        original_device = next(model.parameters()).device
    except StopIteration:
        original_device = torch.device("cpu")

    try:
        return _run_one_calibration_pass(model, example_input, gpu_id)
    except Exception as exc:  # defensive: calibration must never break the real profiling run
        logger.warning("Hybrid overhead calibration failed (non-fatal, using ratio=1.0): %s", exc)
        return {"hybrid_overhead_ratio": 1.0, "status": "error", "reason": str(exc)}
    finally:
        try:
            model.to(original_device)
        except Exception:
            pass


def _run_real_plan_pass(
    model: nn.Module,
    example_input: Any,
    forward_assignment: Dict[str, str],
    backward_assignment: Dict[str, str],
    activation_strategy: Any,
    gpu_id: int,
) -> Dict[str, Any]:
    traced = _try_symbolic_trace(model)
    if traced is None:
        return {"status": "skipped", "reason": "symbolic_trace unsupported for this model"}

    plan = DevicePlan(
        assignment_forward=dict(forward_assignment),
        assignment_backward=dict(backward_assignment),
        cut_edges_forward=[],
        cut_edges_backward=[],
        cross_phase_edges=[
            (n, n) for n in forward_assignment if forward_assignment.get(n) != backward_assignment.get(n)
        ],
        activation_strategies=dict(activation_strategy or {}),
    )
    x = example_input
    try:
        model.to("cpu")
        measured = _run_one_pass_and_measure(traced, plan, x, gpu_id)
        return {
            "status": "ok",
            "reason": None,
            "real_allocated_mb": measured["allocated_mb"],
            "real_reserved_mb": measured["reserved_mb"],
        }
    except torch.cuda.OutOfMemoryError as exc:
        torch.cuda.empty_cache()
        return {"status": "error", "reason": f"OOM: {exc}", "real_allocated_mb": None, "real_reserved_mb": None}


def measure_real_peak_for_plan(
    model: nn.Module,
    example_input: Any,
    forward_assignment: Dict[str, str],
    backward_assignment: Dict[str, str],
    activation_strategy: Any = None,
    gpu_id: int = 0,
) -> Dict[str, Any]:
    """Measures the REAL peak GPU memory (allocated and reserved) of one
    forward+backward pass of `model` through PRISM's actual hybrid
    execution engine, using the EXACT candidate plan given -- not the
    synthetic representative split `measure_hybrid_overhead_ratio` uses.

    This exists because that representative-split ratio is measured once,
    before any post-solve GPU expansion (src/ilp/topology_memory.py's
    expand_solution_memory) decides anything, so it describes a different
    cut pattern than whatever plan expansion actually produces -- confirmed
    to matter in practice: fix/08_gpu_memory_expansion_and_underestimation/
    README.md documents a real resnet50 training run that stalled in an
    infinite CUDA OOM retry loop under an expanded plan the generic ratio
    had deemed safe. Measuring the specific candidate plan directly is the
    only way to validate it against the hybrid engine's real transient
    overhead (allocator fragmentation from async transfer/prefetch double
    buffering, cuDNN algorithm selection variance) that
    topology_memory.py's sequential walk never models.

    Returns a dict always containing "status" ("ok"|"skipped"|"error") and,
    when "ok", "real_reserved_mb"/"real_allocated_mb" (floats). Never
    raises. Restores `model`'s original device placement before returning.
    """
    if not torch.cuda.is_available():
        return {"status": "skipped", "reason": "CUDA not available", "real_reserved_mb": None, "real_allocated_mb": None}

    try:
        original_device = next(model.parameters()).device
    except StopIteration:
        original_device = torch.device("cpu")

    try:
        return _run_real_plan_pass(model, example_input, forward_assignment, backward_assignment, activation_strategy, gpu_id)
    except Exception as exc:  # defensive: calibration must never break the real solve/run
        logger.warning("Real-plan overhead measurement failed (non-fatal): %s", exc)
        return {"status": "error", "reason": str(exc), "real_reserved_mb": None, "real_allocated_mb": None}
    finally:
        try:
            model.to(original_device)
        except Exception:
            pass
