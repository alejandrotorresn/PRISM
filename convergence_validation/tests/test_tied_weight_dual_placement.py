"""Correctness test for `_run_operator_with_dual_placement` on a shared
(tied) weight tensor, the exact pattern GPT-2's `lm_head.weight` /
`transformer.wte.weight` exhibits.

Why this test exists
---------------------
`src/runtime/hybrid_executor.py::_run_operator_with_dual_placement` returns
gradients through the normal `torch.autograd.Function` contract instead of
manually accumulating onto `.grad` (see that function's docstring for why:
manual accumulation would race with or double-count a tied weight's other,
unwrapped usage site). This test is the actual correctness check for that
design decision: it is not enough for the dual-placement path to "not
crash" — the gradient on the shared weight must match, within floating-point
tolerance, what a fully single-device (no dual placement at all) run
produces for the exact same computation and inputs.

This lives in `convergence_validation/tests/`, not `tests/`, following the
project's existing convention that new, non-thesis-critical test coverage
for this investigation goes here (see `convergence_validation/README.md`).
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from runtime.hybrid_executor import _run_operator_with_dual_placement  # noqa: E402


def _shared_weight_scenario(
    use_dual_placement_for_second_usage: bool,
    forward_device: torch.device,
    backward_device: torch.device,
) -> torch.Tensor:
    """Mirrors GPT-2's tied embedding/lm_head pattern: one shared weight used
    twice (once as an "embedding"-style lookup via matmul, once as a
    "lm_head"-style projection), summed into a scalar loss.

    When `use_dual_placement_for_second_usage` is True, the second usage goes
    through `_run_operator_with_dual_placement` with forward on
    `forward_device` and the backward recompute on `backward_device` — the
    exact situation `lm_head`'s `linear` node is in. When False, both usages
    run as plain, unwrapped PyTorch ops on `forward_device` only (the
    reference/baseline computation).

    Returns the shared weight's `.grad` after `.backward()`.
    """
    torch.manual_seed(0)
    weight = torch.randn(16, 8, device=forward_device, requires_grad=True)
    x1 = torch.randn(4, 8, device=forward_device)
    x2 = torch.randn(4, 8, device=forward_device)

    # First usage: plain, unwrapped (stands in for GPT-2's `embedding` node,
    # which is never wrapped even when it also wants dual placement in the
    # real plan — see fix/02.../README.md's "transformer.wte also wants dual
    # placement" note; here we isolate the effect of wrapping ONLY the
    # second usage, matching the asymmetric case the design review stress-tested).
    out1 = F.linear(x1, weight)

    if use_dual_placement_for_second_usage:
        out2 = _run_operator_with_dual_placement(
            target=torch.ops.aten.linear.default,
            args=(x2, weight),
            kwargs={},
            forward_device=forward_device,
            backward_device=backward_device,
            layer_name="test_shared_weight_layer",
        )
    else:
        out2 = F.linear(x2, weight)

    loss = out1.sum() + out2.sum()
    loss.backward()
    assert weight.grad is not None
    return weight.grad.detach().clone()


def test_shared_weight_gradient_matches_baseline_cpu_only() -> None:
    """CPU-only sanity check: dual placement with forward_device ==
    backward_device == cpu should be mathematically a no-op wrapper around
    the same computation, so the gradient must match exactly."""
    device = torch.device("cpu")
    baseline_grad = _shared_weight_scenario(False, device, device)
    wrapped_grad = _shared_weight_scenario(True, device, device)
    assert torch.allclose(baseline_grad, wrapped_grad, atol=1e-6)


def test_shared_weight_gradient_matches_baseline_cross_device() -> None:
    """The real scenario: forward on CPU, backward recompute on GPU — the
    exact placement distilgpt2's lm_head gets in regime B. The tied weight's
    gradient must still match a fully single-device baseline within the same
    tolerance _compare_tensors uses internally for the forward/recompute
    consistency check itself."""
    if not torch.cuda.is_available():
        import pytest

        pytest.skip("CUDA not available on this machine")

    forward_device = torch.device("cpu")
    backward_device = torch.device("cuda")
    baseline_grad = _shared_weight_scenario(False, forward_device, forward_device)
    wrapped_grad = _shared_weight_scenario(True, forward_device, backward_device)
    assert torch.allclose(baseline_grad, wrapped_grad, rtol=0.05, atol=0.15)


def test_non_tensor_and_non_differentiable_args_pass_through() -> None:
    """layer_norm-style signature: a non-Tensor arg (normalized_shape) in the
    middle of the argument list must reach the op unchanged, and a
    non-differentiable Tensor arg (bias with requires_grad=False) must still
    be moved to the correct device per call without being treated as a
    gradient-producing input."""
    device = torch.device("cpu")
    x = torch.randn(4, 8, device=device, requires_grad=True)
    weight = torch.ones(8, device=device, requires_grad=True)
    bias = torch.zeros(8, device=device, requires_grad=False)

    out = _run_operator_with_dual_placement(
        target=torch.ops.aten.layer_norm.default,
        args=(x, [8], weight, bias),
        kwargs={},
        forward_device=device,
        backward_device=device,
        layer_name="test_layer_norm",
    )
    out.sum().backward()
    assert x.grad is not None
    assert weight.grad is not None
    assert bias.grad is None  # never required grad; must not have been forced into one
