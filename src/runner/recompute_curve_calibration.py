"""Measures the REAL (non-additive) memory-savings curve of recompute/
checkpoint, for a real model/batch, before the ILP ever assumes a number.

Why this exists
----------------
`src/ilp/solve.py`'s joint device+activation-strategy MILP
(`enable_joint_activation_strategy`) models each `recompute[n]`'s memory
saving as additive and independent: marking more nodes always subtracts
more from the GPU memory total. Confirmed empirically (controlled, GPU-real
experiments; see `fix/07_gpu_recompute_joint_decision/README.md`) that this
is false for both a vision model (resnet50) and a language model
(distilgpt2): real peak memory improves as more blocks are marked
recompute, up to a point, then gets WORSE than doing nothing at all once
too many are marked simultaneously. Two simpler hypotheses (a fixed memory
cost per checkpoint "frame"; an inherently "bad" block regardless of
context) were each tested with isolated synthetic experiments and both
refuted -- the effect is a genuine interaction between multiple
simultaneously-recomputed blocks of the REAL model, not reducible to a
closed-form additive or per-frame formula found so far.

Rather than invent an unvalidated formula, this module measures the real
curve directly: real peak GPU memory for a handful of points along "how
many of the candidate blocks (ordered by their own theoretical savings,
highest first) are marked recompute" -- then the caller (the ILP's
solve.py) fits a piecewise-linear interpolation through these measured
points and feeds THAT to the MILP, instead of a theoretical per-node term.

Same defensive posture as `hybrid_overhead_calibration.py`: a failure here
must never break the pipeline that was already working. Every public entry
point catches its own exceptions and returns a "skipped"/"error" status
with an empty curve (equivalent to "assume no recompute savings at all",
the safest fallback) instead of propagating.

A critical, non-obvious requirement for correctness, confirmed empirically
while building this module: measuring several of these points IN THE SAME
PROCESS, one after another, silently gives WRONG, non-reproducible numbers
unless `gc.collect()` runs between measurements -- `torch.cuda.
empty_cache()` + `reset_peak_memory_stats()` alone are NOT enough. This
matches the exact reference-cycle failure mode already found and fixed
elsewhere in this project (see PLAN_ACCION_JURADO.md item 42: a Python
reference cycle through the hybrid engine's own autograd bookkeeping that
only Python's cyclic GC, not plain refcounting, ever breaks) -- without the
explicit `gc.collect()` here, a GPU-memory-still-resident leftover from one
probe measurably contaminates the NEXT probe's peak reading, enough to
flip a measured delta from negative to positive. Verified directly: the
exact same probe, re-measured later in the same process after
`gc.collect()` was added, reproduces bit-identical peak readings.
"""
from __future__ import annotations

import gc
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

try:
    from runtime.device_plan import DevicePlan
    from runtime.hybrid_executor import run_hybrid_training
except ModuleNotFoundError:
    import sys
    from pathlib import Path

    src_dir = str(Path(__file__).resolve().parents[1])
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    from runtime.device_plan import DevicePlan
    from runtime.hybrid_executor import run_hybrid_training


@dataclass
class RecomputeCurvePoint:
    n_blocks: int
    fraction: float
    peak_gpu_mem_mb: float


@dataclass
class RecomputeCurveResult:
    status: str  # "ok" | "skipped" | "error"
    reason: str | None = None
    points: List[RecomputeCurvePoint] = field(default_factory=list)


def _default_fractions() -> Sequence[float]:
    return (0.0, 0.25, 0.5, 0.75, 1.0)


def _n_blocks_for_fraction(total_blocks: int, fraction: float) -> int:
    return max(0, min(total_blocks, round(fraction * total_blocks)))


def _block_id_for_layer_name(layer_name: str) -> str:
    # Kept in sync with src/ilp/solve.py's and src/runtime/hybrid_executor.py's
    # own copies of this exact function (duplicated rather than imported,
    # same reasoning as those two: this module is a lower layer than `ilp`,
    # reaching into its private helpers would invert the real dependency
    # direction). Identifies which repeated block (nn.ModuleList-style, e.g.
    # "layer1.0" for resnet, "transformer.h.5" for GPT-2) a leaf layer
    # belongs to, by truncating its name at the last purely-numeric path
    # segment. A layer with no numeric segment (e.g. "fc") is its own block
    # of size 1 -- build_ordered_candidate_blocks drops those (see below).
    parts = layer_name.split(".")
    last_numeric_idx = None
    for i, part in enumerate(parts):
        if part.isdigit():
            last_numeric_idx = i
    if last_numeric_idx is None:
        return layer_name
    return ".".join(parts[: last_numeric_idx + 1])


def build_ordered_candidate_blocks(
    leaf_names: List[str],
    gpu_act_mem: Dict[str, float],
    shared_call_site_layers: Any = None,
    exclude_substrings: Sequence[str] = ("downsample",),
) -> List[List[str]]:
    """Groups `leaf_names` into repeated-block candidates (see
    _block_id_for_layer_name) and orders them from highest to lowest
    theoretical savings (summed `gpu_act_mem`, typically
    model_builder.py's own `_build_effective_memory` output for the SAME
    data the ILP is about to solve -- reusing that keeps this ordering
    consistent with what the MILP itself believes about each node's
    activation cost).

    Drops: singleton "blocks" (a layer with no repeated-block structure,
    e.g. "fc", "avgpool" -- recompute/checkpoint on an isolated single
    leaf measurably does NOT save real memory, see fix/07's README);
    any layer whose name contains one of `exclude_substrings` (default:
    "downsample" -- a resnet block's shortcut branch has its own,
    non-matching block-id scheme, see fix/07's README); and any layer in
    `shared_call_site_layers` (a module reused at multiple call sites --
    see ILPInputData.shared_call_site_layers' docstring, src/ilp/
    data_loader.py -- never independently strategizable).
    """
    shared = set(shared_call_site_layers or ())
    groups: Dict[str, List[str]] = {}
    for name in leaf_names:
        if name in shared or any(s in name for s in exclude_substrings):
            continue
        groups.setdefault(_block_id_for_layer_name(name), []).append(name)
    candidate_groups = [g for g in groups.values() if len(g) > 1]
    candidate_groups.sort(key=lambda g: sum(gpu_act_mem.get(n, 0.0) for n in g), reverse=True)
    return candidate_groups


def select_gpu_resident_candidate_blocks(
    candidate_blocks_ordered: List[List[str]],
    baseline_forward_assignment: Dict[str, str],
) -> List[List[str]]:
    """Filters `candidate_blocks_ordered` down to the blocks that are
    ALREADY entirely GPU-resident in `baseline_forward_assignment` --
    confirmed empirically (see fix/07_gpu_recompute_joint_decision/
    README.md) to matter a lot: a block that is mostly on CPU in the real
    baseline isn't a meaningful recompute candidate at all (recompute only
    relieves memory for a node whose backward already runs on GPU; the
    MILP's own block_recompute[b] <= xb[n] constraint, src/ilp/solve.py,
    requires EVERY node in the block to be GPU-resident before recompute
    does anything for it). Preserves the input ordering.
    """
    return [
        group for group in candidate_blocks_ordered
        if group and all(baseline_forward_assignment.get(n) == "GPU" for n in group)
    ]


def measure_recompute_curve(
    model: nn.Module,
    example_input: Any,
    candidate_blocks_ordered: List[List[str]],
    target_data: torch.Tensor | None = None,
    fractions: Sequence[float] | None = None,
    gpu_id: int = 0,
    lr: float = 1e-6,
    baseline_forward_assignment: Dict[str, str] | None = None,
    baseline_backward_assignment: Dict[str, str] | None = None,
) -> RecomputeCurveResult:
    """Measures real peak GPU memory for a handful of points along "how
    many of `candidate_blocks_ordered`'s groups are marked recompute",
    always taking a PREFIX of the ordered list (the caller orders groups
    from highest to lowest theoretical savings -- typically each group's
    own summed `gpu_act_mem` from the ILP's already-profiled data).

    `candidate_blocks_ordered`: list of groups, each group a list of layer
    names that must share one activation-strategy decision (e.g. all leaf
    modules of one resnet Bottleneck, or all leaves of one GPT-2 transformer
    block) -- the same granularity `_block_id_for_layer_name`
    (`src/ilp/solve.py`, `src/runtime/hybrid_executor.py`) already groups
    by elsewhere in this project. This module does not compute that
    ordering itself: it is a pure measurement tool, not a ranking policy.

    `baseline_forward_assignment`/`baseline_backward_assignment`: the REAL
    device plan to measure the curve against -- typically the ILP's own
    no-recompute baseline solve for this exact model/batch/budget.
    Defaults to "every leaf on GPU" when omitted (useful for a
    quick/isolated check, e.g. in a unit test), but confirmed empirically
    to NOT transfer well to the ILP's real, usually mostly-CPU, operating
    point for a tightly memory-constrained case: the curve's absolute
    values (and sometimes its very shape) depend on how much of the REST
    of the network is already GPU-resident when recompute is applied, not
    just on which blocks are marked. Pass the ILP's real baseline
    assignment whenever one is available -- see
    select_gpu_resident_candidate_blocks for narrowing
    candidate_blocks_ordered to the blocks that assignment can actually
    benefit from.

    Returns `status="ok"` with `points` filled in on success, or
    `status="skipped"/"error"` with an empty `points` list (equivalent to
    "assume no recompute savings") -- never raises.
    """
    if not torch.cuda.is_available():
        return RecomputeCurveResult(status="skipped", reason="CUDA not available")
    if not candidate_blocks_ordered:
        return RecomputeCurveResult(status="skipped", reason="no candidate blocks provided")

    try:
        original_device = next(model.parameters()).device
    except StopIteration:
        original_device = torch.device("cpu")

    leaf_names = [n for n, m in model.named_modules() if n and len(list(m.children())) == 0]
    all_gpu_fwd = {n: "GPU" for n in leaf_names}
    all_gpu_bwd = {n: "GPU" for n in leaf_names}
    plan_fwd = {n: (baseline_forward_assignment or all_gpu_fwd).get(n, "CPU") for n in leaf_names}
    plan_bwd = {n: (baseline_backward_assignment or all_gpu_bwd).get(n, "CPU") for n in leaf_names}

    fractions = tuple(_default_fractions() if fractions is None else fractions)
    total_blocks = len(candidate_blocks_ordered)

    points: List[RecomputeCurvePoint] = []
    try:
        model.to(f"cuda:{gpu_id}" if torch.cuda.device_count() > 1 else "cuda")
        seen_n_blocks: set = set()
        for fraction in fractions:
            n_blocks = _n_blocks_for_fraction(total_blocks, fraction)
            if n_blocks in seen_n_blocks:
                continue  # duplicate rounding (e.g. two fractions mapping to the same n_blocks) -- skip the repeat measurement
            seen_n_blocks.add(n_blocks)

            targets = [n for group in candidate_blocks_ordered[:n_blocks] for n in group]
            activation_strategies = {n: "recompute" for n in targets}
            plan = DevicePlan(
                assignment_forward=plan_fwd,
                assignment_backward=plan_bwd,
                cut_edges_forward=[],
                cut_edges_backward=[],
                cross_phase_edges=[],
                activation_strategies=activation_strategies,
            )

            # See module docstring: gc.collect() before AND after each
            # probe is required for correctness, not just hygiene -- a
            # leftover reference cycle from the PREVIOUS probe otherwise
            # contaminates THIS probe's peak reading.
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(gpu_id)

            result = run_hybrid_training(
                model=model,
                input_data=example_input,
                plan=plan,
                target_data=target_data,
                activation_strategies=activation_strategies,
                steps=1,
                lr=lr,
                gpu_id=gpu_id,
                strict_plan=False,
                execution_mode="dag",
            )
            points.append(
                RecomputeCurvePoint(
                    n_blocks=n_blocks,
                    fraction=float(n_blocks) / total_blocks if total_blocks > 0 else 0.0,
                    peak_gpu_mem_mb=float(result.peak_gpu_mem_mb),
                )
            )
            del result, plan
            gc.collect()

        points.sort(key=lambda p: p.n_blocks)
        return RecomputeCurveResult(status="ok", points=points)
    except torch.cuda.OutOfMemoryError as exc:
        torch.cuda.empty_cache()
        logger.warning(
            "Recompute curve calibration hit OOM partway through (keeping %d point(s) already "
            "measured): %s", len(points), exc,
        )
        points.sort(key=lambda p: p.n_blocks)
        return RecomputeCurveResult(
            status="ok" if points else "error",
            reason=None if points else f"OOM before any point could be measured: {exc}",
            points=points,
        )
    except Exception as exc:  # defensive: calibration must never break the real pipeline
        logger.warning("Recompute curve calibration failed (non-fatal, no curve): %s", exc)
        return RecomputeCurveResult(status="error", reason=str(exc))
    finally:
        try:
            model.to(original_device)
        except Exception:
            pass
        gc.collect()
        torch.cuda.empty_cache()
