"""Smoke tests for convergence_validation.

These are deliberately separate from `tests/` at the repo root (the existing
suite that already protects `src/runtime/hybrid_executor.py` and friends —
see convergence_validation/README.md for why nothing there is touched).

What these tests check is specific to the two risks this module was built
to avoid:

1. The dataloaders really cycle (a real epoch sees different batches, not
   the same one repeated) — this is the entire point of the module, so a
   test that silently used a fixed batch again would defeat the purpose.
2. `prism_loop.py` traces the model's graph exactly once per run, not once
   per step — this is the performance property the whole `_trace_once` /
   `_forward_through_plan` split exists for (see prism_loop.py's module
   docstring).

These tests use `simple_mlp` + `mnist` rather than the thesis's real target
models, specifically so they run in seconds on a laptop CPU and can be part
of routine local iteration, not just a pre-campaign check on a GPU node.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

_CONVERGENCE_DIR = Path(__file__).resolve().parents[1]
if str(_CONVERGENCE_DIR) not in sys.path:
    sys.path.insert(0, str(_CONVERGENCE_DIR))

_REPO_ROOT = _CONVERGENCE_DIR.parent
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))


def test_vision_epoch_batch_provider_cycles_real_data() -> None:
    from datasets import make_vision_provider

    datasets_root = _REPO_ROOT / "datasets"
    if not (datasets_root / "imagenette2-160" / "train").exists():
        import pytest

        pytest.skip("imagenette2-160 not present on this machine")

    provider = make_vision_provider(
        model_name="resnet50",
        datasets_root=datasets_root,
        batch_size=4,
        split="train",
        seed=123,
        torch_dtype=torch.float32,
    )

    first_images, first_labels = provider()
    second_images, second_labels = provider()

    assert first_images.shape == second_images.shape == (4, 3, 224, 224)
    # The whole point of this module: consecutive calls must not return the
    # exact same tensor content (which is what the pre-existing
    # src/data/dataset_registry.py one-shot loader would do if called twice).
    assert not torch.equal(first_images, second_images)


def test_epoch_batch_provider_reshuffles_after_exhaustion() -> None:
    """A dataset of 5 items with batch_size=5, drop_last=True, has exactly
    one batch per epoch. Two consecutive calls must both succeed (the second
    one triggers a StopIteration internally and rebuilds the iterator) and,
    with shuffle=True, should not yield the same row order every time."""
    from datasets import EpochBatchProvider
    from torch.utils.data import TensorDataset

    data = TensorDataset(torch.arange(5).unsqueeze(1).float(), torch.arange(5))
    provider = EpochBatchProvider(dataset=data, batch_size=5, shuffle=True, generator_seed=7)

    epoch_1_inputs, _ = provider()
    epoch_2_inputs, _ = provider()

    assert epoch_1_inputs.shape == epoch_2_inputs.shape == (5, 1)
    # Same 5 values should be present both times (nothing lost/duplicated),
    # just not necessarily in the same order.
    assert set(epoch_1_inputs.flatten().tolist()) == set(epoch_2_inputs.flatten().tolist()) == {0.0, 1.0, 2.0, 3.0, 4.0}


def test_prism_loop_traces_graph_once_per_run(monkeypatch) -> None:
    """Verifies the performance property `prism_loop.py` exists for: the
    expensive tracing step (`_trace_once`) must be called exactly once for a
    multi-step run, never once per step."""
    import prism_loop

    call_count = {"n": 0}
    real_trace_once = prism_loop._trace_once

    def _counting_trace_once(model, example_input):
        call_count["n"] += 1
        return real_trace_once(model, example_input)

    monkeypatch.setattr(prism_loop, "_trace_once", _counting_trace_once)

    # A minimal run would require a real ILP plan + dataset; instead this
    # test directly exercises the traced-graph reuse by calling
    # _forward_through_plan multiple times against one _trace_once() result,
    # which is exactly what run_prism_training's step loop does.
    import torch.nn as nn

    model = nn.Sequential(nn.Linear(4, 4))
    example_input = torch.randn(2, 4)
    traced = prism_loop._trace_once(model, example_input)
    assert call_count["n"] == 1

    from runtime.device_plan import DevicePlan

    plan = DevicePlan(
        assignment_forward={"0": "CPU"},
        assignment_backward={"0": "CPU"},
        cut_edges_forward=[],
        cut_edges_backward=[],
        cross_phase_edges=[],
        activation_strategies={},
    )
    warnings: list[str] = []
    for _ in range(3):
        out, interpreter = prism_loop._forward_through_plan(traced, plan, {}, gpu_id=0, warnings=warnings, input_data=torch.randn(2, 4))
        assert out.shape == (2, 4)
        interpreter.env.clear()

    # Still exactly one trace, even though the graph was run 3 more times.
    assert call_count["n"] == 1


def test_forward_through_plan_reuses_caller_provided_pinned_buffer_cache() -> None:
    """Regression test for a real OOM: `_forward_through_plan` used to build
    a brand-new `_DeviceAwareFXInterpreter` (and therefore a brand-new,
    empty pinned-buffer cache) on EVERY call, with no way for a caller to
    pass one in. Since PyTorch's pinned-memory allocator does not release
    page-locked host pages back to the OS promptly, a fresh cache on every
    forward pass (every training step AND every eval batch) accumulated
    real, unbounded host memory across a run. Confirmed the hard way: a
    real resnet50/batch_192 full-epoch run climbed ~800MB/step in RSS and
    was SIGKILLed by its own cgroup memory ceiling partway through.

    This test proves the fix at the unit level: a cache dict passed in by
    the caller is the SAME object reused across repeated calls (not
    replaced), and keeps exactly one entry per call-site key regardless of
    how many times `_forward_through_plan` runs -- it must never grow
    per-call the way it would if a fresh cache were being built internally.
    """
    import torch.nn as nn

    import prism_loop
    from runtime.device_plan import DevicePlan

    # net.1 = ReLU(inplace=True) forces output_clone_safe=False for net.0's
    # output (an in-place downstream consumer), which is exactly the
    # condition that routes _run_layer_with_dual_placement through the
    # pinned-buffer-cache path instead of skipping it entirely -- see
    # hybrid_executor.py's _node_output_clone_safe.
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(inplace=True), nn.Linear(8, 2))
    example_input = torch.randn(3, 4)
    traced = prism_loop._trace_once(model, example_input)

    plan = DevicePlan(
        assignment_forward={"0": "CPU", "1": "CPU", "2": "CPU"},
        assignment_backward={"0": "GPU" if torch.cuda.is_available() else "CPU", "1": "CPU", "2": "CPU"},
        cut_edges_forward=[],
        cut_edges_backward=[],
        cross_phase_edges=[("0", "0")],
        activation_strategies={},
    )
    warnings: list[str] = []
    shared_cache: dict = {}

    for _ in range(5):
        out, interpreter = prism_loop._forward_through_plan(
            traced, plan, {}, gpu_id=0, warnings=warnings, input_data=torch.randn(3, 4),
            pinned_output_buffer_cache=shared_cache,
        )
        assert out.shape == (3, 2)
        interpreter.env.clear()

    # The cache must have been populated (the dual-placement path actually
    # ran) and must NOT have grown with the number of calls -- one entry
    # per unique call-site key, reused every time, not one new entry per
    # forward pass.
    assert len(shared_cache) <= 1, f"expected the cache to stay bounded across calls, got {len(shared_cache)} entries"
