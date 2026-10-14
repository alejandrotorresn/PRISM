"""Correctness tests for grouped checkpoint/recompute on decoder-only
(torch.export) graphs -- see src/runtime/hybrid_executor.py's
_detect_checkpoint_groups / _run_checkpoint_group.

Why this exists
----------------
Phase 4's checkpoint/recompute strategies only reduce real memory when
applied to a WHOLE named layer's worth of computation (confirmed
empirically: checkpointing a single decomposed ATen op in isolation saves
~0%, checkpointing/recomputing a whole transformer block together saves
~70-74% -- see fix/06 README). `call_module` (vision models) already gets
this right for free. This module adds the equivalent for `call_function`
(decoder-only torch.export graphs), where one named layer decomposes into
many individual ATen nodes that must be grouped and checkpointed/recomputed
together, not one at a time.

These tests use a tiny real GPT2LMHeadModel (2 layers, small hidden size)
instead of a synthetic nn.Module, because torch.export tracing in this
runtime only activates for a genuine `transformers.PreTrainedModel` (see
`is_decoder_only_export_candidate`) -- there is no generic decoder-only
path to exercise with an arbitrary toy module.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_CONVERGENCE_DIR = Path(__file__).resolve().parents[1]
if str(_CONVERGENCE_DIR) not in sys.path:
    sys.path.insert(0, str(_CONVERGENCE_DIR))

_REPO_ROOT = _CONVERGENCE_DIR.parent
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))


def _tiny_gpt2():
    from transformers import GPT2Config, GPT2LMHeadModel

    config = GPT2Config(
        vocab_size=64,
        n_positions=32,
        n_embd=16,
        n_layer=2,
        n_head=2,
        n_inner=32,
    )
    torch.manual_seed(0)
    model = GPT2LMHeadModel(config)
    model.eval()
    return model


def _run_with_strategy(model, input_ids, target, strategy: str | None, block_prefix: str = "transformer.h.0"):
    # node_layer_names only ever names LEAF submodules (confirmed
    # empirically -- see src/runtime/hybrid_executor.py's
    # _detect_checkpoint_groups docstring): "transformer.h.0" itself is
    # never a key. Phase 4 marking a whole block for relief in practice
    # means marking EVERY leaf under that block with the same strategy --
    # replicate that here, not a single (non-existent) container-level key.
    from runtime.device_plan import DevicePlan
    from runtime.hybrid_executor import run_hybrid_training

    leaf_names = {name for name, m in model.named_modules() if name and len(list(m.children())) == 0}
    assignment = {name: "CPU" for name in leaf_names}
    activation_strategies = {}
    if strategy is not None:
        block_leaves = [name for name in leaf_names if name.startswith(block_prefix)]
        assert block_leaves, f"no leaf submodules found under prefix {block_prefix!r}"
        for name in block_leaves:
            activation_strategies[name] = strategy

    plan = DevicePlan(
        assignment_forward=dict(assignment),
        assignment_backward=dict(assignment),
        cut_edges_forward=[],
        cut_edges_backward=[],
        cross_phase_edges=[],
        activation_strategies=activation_strategies,
    )
    torch.manual_seed(1)
    model_copy = type(model)(model.config)
    model_copy.load_state_dict(model.state_dict())
    model_copy.train()

    result = run_hybrid_training(
        model=model_copy,
        input_data={"input_ids": input_ids},
        target_data=target,
        plan=plan,
        activation_strategies=activation_strategies,
        steps=1,
        lr=0.0,  # no update -- just checking forward/backward correctness, not training dynamics
        gpu_id=0,
        strict_plan=False,
        execution_mode="dag",
    )
    grads = {name: p.grad.detach().clone() for name, p in model_copy.named_parameters() if p.grad is not None}
    return result, grads


@pytest.mark.parametrize("strategy", ["recompute", "checkpoint"])
def test_grouped_strategy_matches_retain_baseline_gradients(strategy):
    model = _tiny_gpt2()
    input_ids = torch.randint(0, model.config.vocab_size, (2, 8))
    target = input_ids.clone()

    baseline_result, baseline_grads = _run_with_strategy(model, input_ids, target, strategy=None)
    strategy_result, strategy_grads = _run_with_strategy(model, input_ids, target, strategy=strategy)

    assert baseline_result.status == "ok"
    assert strategy_result.status == "ok"
    assert set(baseline_grads.keys()) == set(strategy_grads.keys())
    for name in baseline_grads:
        assert torch.allclose(baseline_grads[name], strategy_grads[name], atol=1e-5, rtol=1e-4), (
            f"gradient mismatch for {name} under strategy={strategy}"
        )


def test_grouped_strategy_actually_groups_not_per_node():
    """The whole point: the runtime must apply recompute/checkpoint ONCE per
    named layer (a group of ~10-20 ATen nodes for transformer.h.0), not once
    per individual node -- that is exactly the distinction that makes this
    effective (see module docstring)."""
    model = _tiny_gpt2()
    input_ids = torch.randint(0, model.config.vocab_size, (2, 8))
    target = input_ids.clone()

    result, _ = _run_with_strategy(model, input_ids, target, strategy="recompute")
    assert result.status == "ok"
    # recompute_count is incremented once per GROUP execution (see
    # _run_checkpoint_group), not once per ATen node. A single transformer
    # block's attention sub-module splits into a few groups (reshape/split
    # ops carry no layer_name, which breaks grouping continuity there), but
    # that's still a handful of groups, nowhere near one-per-node for a
    # block that decomposes into 20+ ATen ops -- confirming grouping
    # happened at all, which is what this test checks.
    assert all(step.recompute_count <= 5 for step in result.per_step)
    assert any(step.recompute_count >= 1 for step in result.per_step)
    # The group spans multiple distinct LEAF layer names belonging to the
    # same block (ln_1, c_attn, c_proj, ln_2, c_fc, ...) -- confirming the
    # merge-across-leaf-boundary behavior, not just one leaf's own ops.
    assert any(name.startswith("transformer.h.0.") for name in result.recompute_layers)
    assert len({name for name in result.recompute_layers if name.startswith("transformer.h.0.")}) >= 2


@pytest.mark.parametrize("strategy", ["recompute", "checkpoint"])
def test_final_layer_fused_with_loss_matches_baseline_gradients(strategy):
    """Part 2: `lm_head` is a SINGLE call_function node whose only consumer
    is the graph's own output node (confirmed empirically -- see
    _detect_checkpoint_groups' final-node detection). The main grouping
    loop requires >1 node, so lm_head alone would never be grouped by Part
    1; this is the deferred-final-node + loss-fusion path (Part 2,
    _DeferredCheckpointSegment) being exercised specifically."""
    model = _tiny_gpt2()
    input_ids = torch.randint(0, model.config.vocab_size, (2, 8))
    target = input_ids.clone()

    baseline_result, baseline_grads = _run_with_strategy(model, input_ids, target, strategy=None)
    strategy_result, strategy_grads = _run_with_strategy(
        model, input_ids, target, strategy=strategy, block_prefix="lm_head"
    )

    assert baseline_result.status == "ok"
    assert strategy_result.status == "ok"
    assert set(baseline_grads.keys()) == set(strategy_grads.keys())
    for name in baseline_grads:
        assert torch.allclose(baseline_grads[name], strategy_grads[name], atol=1e-5, rtol=1e-4), (
            f"gradient mismatch for {name} under strategy={strategy} (lm_head fused with loss)"
        )

    reported_layers = (
        strategy_result.recompute_layers if strategy == "recompute" else strategy_result.checkpoint_layers
    )
    assert "lm_head" in reported_layers
    count_attr = "recompute_count" if strategy == "recompute" else "checkpoint_count"
    assert all(getattr(step, count_attr) == 1 for step in strategy_result.per_step)


def test_vision_call_module_path_unaffected():
    """Explicit regression guarantee for the user's stated concern: the new
    grouping logic must be a complete no-op for fx.symbolic_trace
    (call_module) graphs. _detect_checkpoint_groups returns immediately
    whenever supports_activation_strategies is True (the vision case), so
    _checkpoint_group_start stays empty and run_node's new checks never
    match -- this test exercises that path directly with a real
    checkpoint-strategy vision-style model and checks it behaves exactly
    like the existing, already-passing tests in tests/test_hybrid_executor.py."""
    import torch.nn as nn

    from runtime.device_plan import DevicePlan
    from runtime.hybrid_executor import run_hybrid_training

    class TinyConvNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.ReLU(), nn.Conv2d(4, 4, 3, padding=1))

        def forward(self, x):
            return self.net(x)

    model = TinyConvNet()
    x = torch.randn(2, 3, 8, 8)
    plan = DevicePlan(
        assignment_forward={"net.0": "CPU", "net.1": "CPU", "net.2": "CPU"},
        assignment_backward={"net.0": "CPU", "net.1": "CPU", "net.2": "CPU"},
        cut_edges_forward=[],
        cut_edges_backward=[],
        cross_phase_edges=[],
        activation_strategies={"net.2": "checkpoint"},
    )
    result = run_hybrid_training(
        model=model, input_data=x, plan=plan,
        activation_strategies=plan.activation_strategies, steps=1, lr=0.01, gpu_id=0,
    )
    assert result.status == "ok"
    assert "net.2" in result.checkpoint_layers
    assert all(step.checkpoint_count == 1 for step in result.per_step)
