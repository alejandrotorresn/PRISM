from __future__ import annotations

import torch

from src.ilp.advanced_terms import ActivationStrategy
from src.runtime.device_plan import DevicePlan, plan_requests_gpu
from src.runtime.hybrid_executor import run_hybrid_training


class ToyMLP(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(4, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def test_hybrid_executor_runs_cpu_plan() -> None:
    model = ToyMLP()
    inp = torch.randn(3, 4)
    plan = DevicePlan(
        assignment_forward={"net.0": "CPU", "net.1": "CPU", "net.2": "CPU"},
        assignment_backward={"net.0": "CPU", "net.1": "CPU", "net.2": "CPU"},
        cut_edges_forward=[],
        cut_edges_backward=[],
        cross_phase_edges=[],
    )

    result = run_hybrid_training(
        model=model,
        input_data=inp,
        plan=plan,
        steps=2,
        lr=0.01,
        strict_plan=True,
    )

    assert result.status == "ok"
    assert result.steps == 2
    assert len(result.per_step) == 2
    assert result.total_transfer_events == 0
    assert result.avg_power_w >= 0.0
    assert result.total_energy_j >= 0.0
    assert result.recompute_layers == []
    assert result.checkpoint_layers == []
    assert result.backward_relocation_layers == []
    assert result.prefetch_layers == []
    assert result.total_prefetch_events == 0
    assert result.total_prefetch_mb == 0.0
    assert all(step.recompute_count == 0 for step in result.per_step)
    assert all(step.checkpoint_count == 0 for step in result.per_step)
    assert all(step.backward_relocation_count == 0 for step in result.per_step)
    assert all(step.prefetch_count == 0 for step in result.per_step)
    assert all(step.prefetch_total_mb == 0.0 for step in result.per_step)


def test_hybrid_executor_warns_when_gpu_unavailable() -> None:
    model = ToyMLP()
    inp = torch.randn(3, 4)
    plan = DevicePlan(
        assignment_forward={"net.0": "GPU", "net.1": "CPU", "net.2": "CPU"},
        assignment_backward={"net.0": "GPU", "net.1": "CPU", "net.2": "CPU"},
        cut_edges_forward=[],
        cut_edges_backward=[],
        cross_phase_edges=[],
    )

    result = run_hybrid_training(
        model=model,
        input_data=inp,
        plan=plan,
        steps=1,
        lr=0.01,
        strict_plan=True,
    )

    assert result.status == "ok"
    assert result.energy_source in {"nvml", "rapl"}
    if not torch.cuda.is_available():
        assert any("CUDA is unavailable" in w for w in result.warnings)


def test_plan_requests_gpu_detection() -> None:
    cpu_plan = DevicePlan(
        assignment_forward={"net.0": "CPU", "net.1": "CPU"},
        assignment_backward={"net.0": "CPU", "net.1": "CPU"},
        cut_edges_forward=[],
        cut_edges_backward=[],
        cross_phase_edges=[],
    )
    gpu_plan = DevicePlan(
        assignment_forward={"net.0": "GPU", "net.1": "CPU"},
        assignment_backward={"net.0": "CPU", "net.1": "CPU"},
        cut_edges_forward=[],
        cut_edges_backward=[],
        cross_phase_edges=[("net.0", "net.0")],
    )

    assert plan_requests_gpu(cpu_plan) is False
    assert plan_requests_gpu(gpu_plan) is True


def test_hybrid_executor_supports_recompute_strategy() -> None:
    model = ToyMLP()
    inp = torch.randn(3, 4)
    plan = DevicePlan(
        assignment_forward={"net.0": "CPU", "net.1": "CPU", "net.2": "CPU"},
        assignment_backward={"net.0": "CPU", "net.1": "CPU", "net.2": "CPU"},
        cut_edges_forward=[],
        cut_edges_backward=[],
        cross_phase_edges=[],
    )

    result = run_hybrid_training(
        model=model,
        input_data=inp,
        plan=plan,
        activation_strategies={"net.0": ActivationStrategy("net.0", recompute=True)},
        steps=2,
        lr=0.01,
        strict_plan=True,
    )

    assert result.status == "ok"
    assert result.recompute_layers == ["net.0"]
    assert all(step.recompute_count == 1 for step in result.per_step)


def test_hybrid_executor_supports_checkpoint_strategy() -> None:
    model = ToyMLP()
    inp = torch.randn(3, 4)
    plan = DevicePlan(
        assignment_forward={"net.0": "CPU", "net.1": "CPU", "net.2": "CPU"},
        assignment_backward={"net.0": "CPU", "net.1": "CPU", "net.2": "CPU"},
        cut_edges_forward=[],
        cut_edges_backward=[],
        cross_phase_edges=[],
    )

    result = run_hybrid_training(
        model=model,
        input_data=inp,
        plan=plan,
        activation_strategies={"net.1": "checkpoint"},
        steps=1,
        lr=0.01,
        strict_plan=True,
    )

    assert result.status == "ok"
    assert result.checkpoint_layers == ["net.1"]
    assert result.unsupported_checkpoint_layers == []
    assert all(step.checkpoint_count == 1 for step in result.per_step)
    assert not any("runtime currently supports only recompute" in warning for warning in result.warnings)


def test_hybrid_executor_warns_when_backward_assignment_differs() -> None:
    model = ToyMLP()
    inp = torch.randn(3, 4)
    plan = DevicePlan(
        assignment_forward={"net.0": "CPU", "net.1": "CPU", "net.2": "CPU"},
        assignment_backward={"net.0": "GPU", "net.1": "CPU", "net.2": "CPU"},
        cut_edges_forward=[],
        cut_edges_backward=[],
        cross_phase_edges=[("net.0", "net.0")],
    )

    result = run_hybrid_training(
        model=model,
        input_data=inp,
        plan=plan,
        steps=1,
        lr=0.01,
        strict_plan=True,
    )

    assert result.status == "ok"
    if torch.cuda.is_available():
        assert result.backward_relocation_layers == ["net.0"]
        assert all(step.backward_relocation_count == 1 for step in result.per_step)
    else:
        assert result.backward_relocation_layers == []
        assert any("cannot be materialized" in warning for warning in result.warnings)


def test_hybrid_executor_prefetch_policy_activation() -> None:
    model = ToyMLP()
    inp = torch.randn(3, 4)
    plan = DevicePlan(
        assignment_forward={"net.0": "GPU", "net.1": "CPU", "net.2": "CPU"},
        assignment_backward={"net.0": "GPU", "net.1": "CPU", "net.2": "CPU"},
        cut_edges_forward=[("net.0", "net.1")],
        cut_edges_backward=[],
        cross_phase_edges=[],
    )

    result = run_hybrid_training(
        model=model,
        input_data=inp,
        plan=plan,
        steps=1,
        lr=0.01,
        strict_plan=True,
        enable_async_transfer=True,
        enable_prefetch=True,
    )

    assert result.status == "ok"
    if torch.cuda.is_available():
        assert result.total_prefetch_events >= 1
        assert result.total_prefetch_mb >= 0.0
        assert len(result.prefetch_layers) >= 1
        assert any(step.prefetch_count >= 1 for step in result.per_step)
    else:
        assert result.total_prefetch_events == 0
        assert result.prefetch_layers == []
