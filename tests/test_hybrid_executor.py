from __future__ import annotations

import torch

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
        assignment={"net.0": "CPU", "net.1": "CPU", "net.2": "CPU"},
        cut_edges=[],
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


def test_hybrid_executor_warns_when_gpu_unavailable() -> None:
    model = ToyMLP()
    inp = torch.randn(3, 4)
    plan = DevicePlan(
        assignment={"net.0": "GPU", "net.1": "CPU", "net.2": "CPU"},
        cut_edges=[],
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
        assignment={"net.0": "CPU", "net.1": "CPU"},
        cut_edges=[],
    )
    gpu_plan = DevicePlan(
        assignment={"net.0": "GPU", "net.1": "CPU"},
        cut_edges=[],
    )

    assert plan_requests_gpu(cpu_plan) is False
    assert plan_requests_gpu(gpu_plan) is True
