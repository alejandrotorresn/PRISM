from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn

from .device_plan import DevicePlan, validate_plan_coverage

try:
    from core.energy import EnergyMonitor
except ModuleNotFoundError:
    import sys
    from pathlib import Path

    src_dir = str(Path(__file__).resolve().parents[1])
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    from core.energy import EnergyMonitor


@dataclass
class TransferEvent:
    layer: str
    src_device: str
    dst_device: str
    size_mb: float
    time_ms: float


@dataclass
class StepTrace:
    step: int
    total_ms: float
    forward_ms: float
    backward_ms: float
    optimizer_ms: float
    transfer_count: int
    transfer_total_mb: float
    peak_gpu_mem_mb: float


@dataclass
class HybridExecutionResult:
    status: str
    steps: int
    avg_step_ms: float
    avg_power_w: float
    total_energy_j: float
    energy_source: str
    total_transfer_mb: float
    total_transfer_events: int
    peak_gpu_mem_mb: float
    warnings: List[str]
    per_step: List[StepTrace]

    def to_dict(self) -> Dict[str, object]:
        return {
            "status": self.status,
            "steps": self.steps,
            "avg_step_ms": self.avg_step_ms,
            "avg_power_w": self.avg_power_w,
            "total_energy_j": self.total_energy_j,
            "energy_source": self.energy_source,
            "total_transfer_mb": self.total_transfer_mb,
            "total_transfer_events": self.total_transfer_events,
            "peak_gpu_mem_mb": self.peak_gpu_mem_mb,
            "warnings": self.warnings,
            "per_step": [asdict(s) for s in self.per_step],
        }


def _tensor_bytes(x: Any) -> int:
    if isinstance(x, torch.Tensor):
        return int(x.numel() * x.element_size())
    if isinstance(x, (list, tuple)):
        return sum(_tensor_bytes(v) for v in x)
    if isinstance(x, dict):
        return sum(_tensor_bytes(v) for v in x.values())
    return 0


def _to_device(x: Any, device: torch.device) -> Any:
    if isinstance(x, torch.Tensor):
        return x.to(device)
    if isinstance(x, list):
        return [_to_device(v, device) for v in x]
    if isinstance(x, tuple):
        return tuple(_to_device(v, device) for v in x)
    if isinstance(x, dict):
        return {k: _to_device(v, device) for k, v in x.items()}
    return x


def _current_device(x: Any) -> torch.device | None:
    if isinstance(x, torch.Tensor):
        return x.device
    if isinstance(x, (list, tuple)):
        for v in x:
            d = _current_device(v)
            if d is not None:
                return d
    if isinstance(x, dict):
        for v in x.values():
            d = _current_device(v)
            if d is not None:
                return d
    return None


def _compute_loss(out: Any) -> torch.Tensor:
    if hasattr(out, "loss") and out.loss is not None:
        return out.loss
    if hasattr(out, "logits"):
        return out.logits.sum()
    if isinstance(out, torch.Tensor):
        return out.sum()
    if isinstance(out, (tuple, list)) and out and isinstance(out[0], torch.Tensor):
        return out[0].sum()
    raise ValueError("Unsupported model output for loss computation")


def _ordered_sequential_layers(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    if hasattr(model, "net") and isinstance(model.net, nn.Sequential):
        return [(f"net.{idx}", mod) for idx, mod in enumerate(model.net)]

    leaves: List[Tuple[str, nn.Module]] = []
    for name, module in model.named_modules():
        if name and len(list(module.children())) == 0:
            leaves.append((name, module))
    if not leaves:
        raise ValueError("Model has no leaf modules to execute")
    return leaves


def run_hybrid_training(
    model: nn.Module,
    input_data: Any,
    plan: DevicePlan,
    steps: int = 5,
    lr: float = 0.01,
    gpu_id: int = 0,
    strict_plan: bool = True,
    enable_rapl: bool = False,
    energy_sample_interval: float = 0.05,
) -> HybridExecutionResult:
    if steps <= 0:
        raise ValueError(f"steps must be > 0, got {steps}")

    warnings = validate_plan_coverage(model=model, plan=plan, strict=strict_plan)
    if not torch.cuda.is_available() and any(v == "GPU" for v in plan.assignment.values()):
        warnings.append("GPU layers requested but CUDA is unavailable; those layers run on CPU")

    layer_order = _ordered_sequential_layers(model)

    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    step_traces: List[StepTrace] = []
    total_transfer_bytes = 0
    total_transfer_events = 0
    global_peak_gpu_mem_mb = 0.0

    uses_gpu = torch.cuda.is_available() and any(str(v).upper() == "GPU" for v in plan.assignment.values())
    energy_device_type = "cuda" if uses_gpu else "cpu"
    energy_source = "nvml" if energy_device_type == "cuda" else "rapl"
    energy_monitor = EnergyMonitor(
        device_type=energy_device_type,
        gpu_id=gpu_id,
        sample_interval=energy_sample_interval,
        enable_rapl=enable_rapl,
    )
    run_start = time.perf_counter()
    energy_monitor.start()

    try:
        for step_idx in range(steps):
            step_start = time.perf_counter()
            optimizer.zero_grad()

            x = input_data
            forward_start = time.perf_counter()
            step_transfer_bytes = 0
            step_transfer_events = 0

            for layer_name, layer in layer_order:
                target_device = plan.resolve_torch_device(layer_name=layer_name, gpu_id=gpu_id)
                layer.to(target_device)

                src_device = _current_device(x)
                if src_device is not None and src_device != target_device:
                    transfer_start = time.perf_counter()
                    payload_bytes = _tensor_bytes(x)
                    x = _to_device(x, target_device)
                    if target_device.type == "cuda":
                        torch.cuda.synchronize(gpu_id)
                    transfer_ms = (time.perf_counter() - transfer_start) * 1000.0
                    _ = TransferEvent(
                        layer=layer_name,
                        src_device=str(src_device),
                        dst_device=str(target_device),
                        size_mb=payload_bytes / (1024**2),
                        time_ms=transfer_ms,
                    )
                    step_transfer_bytes += payload_bytes
                    step_transfer_events += 1

                x = layer(x)

            if isinstance(x, torch.Tensor) and x.device.type == "cuda":
                torch.cuda.synchronize(gpu_id)
            forward_ms = (time.perf_counter() - forward_start) * 1000.0

            backward_start = time.perf_counter()
            loss = _compute_loss(x)
            loss.backward()
            if torch.cuda.is_available():
                torch.cuda.synchronize(gpu_id)
            backward_ms = (time.perf_counter() - backward_start) * 1000.0

            opt_start = time.perf_counter()
            optimizer.step()
            if torch.cuda.is_available():
                torch.cuda.synchronize(gpu_id)
            optimizer_ms = (time.perf_counter() - opt_start) * 1000.0

            total_ms = (time.perf_counter() - step_start) * 1000.0

            peak_gpu_mem_mb = 0.0
            if torch.cuda.is_available():
                peak_gpu_mem_mb = torch.cuda.max_memory_allocated(gpu_id) / (1024**2)
                global_peak_gpu_mem_mb = max(global_peak_gpu_mem_mb, peak_gpu_mem_mb)

            step_trace = StepTrace(
                step=step_idx,
                total_ms=total_ms,
                forward_ms=forward_ms,
                backward_ms=backward_ms,
                optimizer_ms=optimizer_ms,
                transfer_count=step_transfer_events,
                transfer_total_mb=step_transfer_bytes / (1024**2),
                peak_gpu_mem_mb=peak_gpu_mem_mb,
            )
            step_traces.append(step_trace)

            total_transfer_bytes += step_transfer_bytes
            total_transfer_events += step_transfer_events
    finally:
        run_duration_sec = max(0.0, time.perf_counter() - run_start)
        energy_monitor.stop()

    avg_power_w = float(energy_monitor.get_avg_power())
    total_energy_j = float(avg_power_w * run_duration_sec) if avg_power_w > 0 else 0.0
    if total_energy_j <= 0.0:
        warnings.append(
            "Observed energy could not be measured (power monitor unavailable or zero readings)"
        )

    avg_step_ms = sum(s.total_ms for s in step_traces) / len(step_traces)

    return HybridExecutionResult(
        status="ok",
        steps=steps,
        avg_step_ms=avg_step_ms,
        avg_power_w=avg_power_w,
        total_energy_j=total_energy_j,
        energy_source=energy_source,
        total_transfer_mb=total_transfer_bytes / (1024**2),
        total_transfer_events=total_transfer_events,
        peak_gpu_mem_mb=global_peak_gpu_mem_mb,
        warnings=warnings,
        per_step=step_traces,
    )
