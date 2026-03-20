from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Tuple
from typing import cast

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as activation_checkpoint

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
    prefetch_count: int
    prefetch_total_mb: float
    peak_gpu_mem_mb: float
    recompute_count: int
    checkpoint_count: int
    backward_relocation_count: int


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
    total_prefetch_mb: float
    total_prefetch_events: int
    peak_gpu_mem_mb: float
    prefetch_layers: List[str]
    recompute_layers: List[str]
    checkpoint_layers: List[str]
    backward_relocation_layers: List[str]
    unsupported_checkpoint_layers: List[str]
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
            "total_prefetch_mb": self.total_prefetch_mb,
            "total_prefetch_events": self.total_prefetch_events,
            "peak_gpu_mem_mb": self.peak_gpu_mem_mb,
            "prefetch_layers": self.prefetch_layers,
            "recompute_layers": self.recompute_layers,
            "checkpoint_layers": self.checkpoint_layers,
            "backward_relocation_layers": self.backward_relocation_layers,
            "unsupported_checkpoint_layers": self.unsupported_checkpoint_layers,
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


def _activation_strategy_name(strategy: Any) -> str:
    if strategy is None:
        return "retain"
    if isinstance(strategy, str):
        return strategy.lower()
    if getattr(strategy, "recompute", False):
        return "recompute"
    if getattr(strategy, "checkpoint", False):
        return "checkpoint"
    if getattr(strategy, "retain", False):
        return "retain"
    return "retain"


def _saved_tensor_cpu_offload_context() -> Any:
    def pack_hook(tensor: torch.Tensor) -> Tuple[str, torch.Tensor]:
        return (str(tensor.device), tensor.detach().to("cpu"))

    def unpack_hook(packed: Tuple[str, torch.Tensor]) -> torch.Tensor:
        device_label, tensor = packed
        return tensor.to(torch.device(device_label))

    return torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook)


def _run_layer_with_dual_placement(
    layer: nn.Module,
    layer_name: str,
    input_tensor: torch.Tensor,
    forward_device: torch.device,
    backward_device: torch.device,
) -> Any:
    class _DualPlacementFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x: torch.Tensor) -> torch.Tensor:
            layer.to(forward_device)
            x_forward = x.to(forward_device)
            output = layer(x_forward)

            ctx.layer = layer
            ctx.layer_name = layer_name
            ctx.forward_device = forward_device
            ctx.backward_device = backward_device
            ctx.input_device = x.device
            ctx.save_for_backward(x.detach().to("cpu"))
            return output

        @staticmethod
        def backward(ctx, *grad_outputs: Any) -> Any:
            grad_output = grad_outputs[0]
            (saved_input_cpu,) = ctx.saved_tensors
            layer_local = ctx.layer
            layer_local.to(ctx.backward_device)

            recompute_input = saved_input_cpu.to(ctx.backward_device).detach().requires_grad_(True)
            params = tuple(param for param in layer_local.parameters() if param.requires_grad)

            with torch.enable_grad():
                recompute_output = layer_local(recompute_input)

            grad_output_local = grad_output.to(ctx.backward_device)
            grads = torch.autograd.grad(
                recompute_output,
                (recompute_input,) + params,
                grad_outputs=grad_output_local,
                allow_unused=True,
            )

            grad_input = grads[0]
            for param, grad in zip(params, grads[1:]):
                if grad is None:
                    continue
                if param.grad is None:
                    param.grad = grad.detach()
                else:
                    param.grad = param.grad + grad.detach()

            if grad_input is None:
                grad_input = torch.zeros_like(saved_input_cpu, device=ctx.input_device)
            else:
                grad_input = grad_input.to(ctx.input_device)
            return (grad_input,)

    return _DualPlacementFn.apply(input_tensor)


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
    activation_strategies: Dict[str, Any] | None = None,
    steps: int = 5,
    lr: float = 0.01,
    gpu_id: int = 0,
    strict_plan: bool = True,
    enable_rapl: bool = False,
    energy_sample_interval: float = 0.05,
    enable_async_transfer: bool = False,
    enable_prefetch: bool = False,
) -> HybridExecutionResult:
    if steps <= 0:
        raise ValueError(f"steps must be > 0, got {steps}")

    warnings = validate_plan_coverage(model=model, plan=plan, strict=strict_plan)
    if not torch.cuda.is_available() and (
        any(v == "GPU" for v in plan.assignment_forward.values())
        or any(v == "GPU" for v in plan.assignment_backward.values())
    ):
        warnings.append("GPU layers requested but CUDA is unavailable; those layers run on CPU")

    layer_order = _ordered_sequential_layers(model)

    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    transfer_stream = torch.cuda.Stream(device=gpu_id) if (enable_async_transfer and torch.cuda.is_available()) else None

    step_traces: List[StepTrace] = []
    total_transfer_bytes = 0
    total_transfer_events = 0
    total_prefetch_bytes = 0
    total_prefetch_events = 0
    global_peak_gpu_mem_mb = 0.0
    prefetch_layers_used: set[str] = set()
    recompute_layers_used: set[str] = set()
    checkpoint_layers_used: set[str] = set()
    backward_relocation_layers_used: set[str] = set()
    unsupported_checkpoint_layers: set[str] = set()

    uses_gpu = torch.cuda.is_available() and (
        any(str(v).upper() == "GPU" for v in plan.assignment_forward.values())
        or any(str(v).upper() == "GPU" for v in plan.assignment_backward.values())
    )
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
            step_prefetch_bytes = 0
            step_prefetch_events = 0
            step_recompute_count = 0
            step_checkpoint_count = 0
            step_backward_relocation_count = 0
            prefetched_inputs: Dict[str, Any] = {}

            for layer_idx, (layer_name, layer) in enumerate(layer_order):
                target_device = plan.resolve_torch_device(layer_name=layer_name, gpu_id=gpu_id, phase="forward")
                backward_device = plan.resolve_torch_device(layer_name=layer_name, gpu_id=gpu_id, phase="backward")
                layer.to(target_device)
                strategy_name = _activation_strategy_name(
                    None if activation_strategies is None else activation_strategies.get(layer_name)
                )

                used_prefetched_input = layer_name in prefetched_inputs
                if used_prefetched_input:
                    x = prefetched_inputs.pop(layer_name)
                    if transfer_stream is not None:
                        dst_dev = _current_device(x)
                        if dst_dev is not None and dst_dev.type == "cuda":
                            torch.cuda.current_stream(gpu_id).wait_stream(transfer_stream)

                src_device = _current_device(x)
                if (not used_prefetched_input) and src_device is not None and src_device != target_device:
                    transfer_start = time.perf_counter()
                    payload_bytes = _tensor_bytes(x)
                    if transfer_stream is not None and target_device.type == "cuda":
                        with torch.cuda.stream(transfer_stream):
                            x = _to_device(x, target_device)
                        torch.cuda.current_stream(gpu_id).wait_stream(transfer_stream)
                        torch.cuda.synchronize(gpu_id)
                    else:
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

                wants_dual_runtime = (
                    plan.assignment_forward.get(layer_name) != plan.assignment_backward.get(layer_name)
                )
                can_materialize_dual = (
                    wants_dual_runtime
                    and isinstance(x, torch.Tensor)
                    and target_device != backward_device
                )

                if wants_dual_runtime and not can_materialize_dual:
                    warnings.append(
                        f"Layer '{layer_name}' requests different backward placement but it cannot be materialized in the current runtime/device context; using forward placement for backward"
                    )

                if can_materialize_dual:
                    x = _run_layer_with_dual_placement(
                        layer=layer,
                        layer_name=layer_name,
                        input_tensor=cast(torch.Tensor, x),
                        forward_device=target_device,
                        backward_device=backward_device,
                    )
                    backward_relocation_layers_used.add(layer_name)
                    step_backward_relocation_count += 1
                elif strategy_name == "recompute":
                    if isinstance(x, torch.Tensor):
                        x = activation_checkpoint(layer, x, use_reentrant=False)
                        recompute_layers_used.add(layer_name)
                        step_recompute_count += 1
                    else:
                        warnings.append(
                            f"Layer '{layer_name}' requested recompute but runtime input is not a Tensor; falling back to retain"
                        )
                        x = layer(x)
                elif strategy_name == "checkpoint":
                    checkpoint_ctx = _saved_tensor_cpu_offload_context()
                    with checkpoint_ctx:
                        x = layer(x)
                    checkpoint_layers_used.add(layer_name)
                    step_checkpoint_count += 1
                else:
                    x = layer(x)

                # Prefetch policy (look-ahead): proactively transfer current output to
                # the next layer forward device when devices differ.
                if enable_prefetch and (layer_idx + 1) < len(layer_order):
                    next_layer_name, _next_layer = layer_order[layer_idx + 1]
                    next_target_device = plan.resolve_torch_device(
                        layer_name=next_layer_name,
                        gpu_id=gpu_id,
                        phase="forward",
                    )
                    current_output_device = _current_device(x)
                    if current_output_device is not None and current_output_device != next_target_device:
                        prefetch_start = time.perf_counter()
                        payload_bytes = _tensor_bytes(x)

                        if transfer_stream is not None and next_target_device.type == "cuda":
                            with torch.cuda.stream(transfer_stream):
                                prefetched_inputs[next_layer_name] = _to_device(x, next_target_device)
                        else:
                            prefetched_inputs[next_layer_name] = _to_device(x, next_target_device)

                        prefetch_ms = (time.perf_counter() - prefetch_start) * 1000.0
                        _ = TransferEvent(
                            layer=f"prefetch:{next_layer_name}",
                            src_device=str(current_output_device),
                            dst_device=str(next_target_device),
                            size_mb=payload_bytes / (1024**2),
                            time_ms=prefetch_ms,
                        )

                        prefetch_layers_used.add(next_layer_name)
                        step_prefetch_bytes += payload_bytes
                        step_prefetch_events += 1

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
                prefetch_count=step_prefetch_events,
                prefetch_total_mb=step_prefetch_bytes / (1024**2),
                peak_gpu_mem_mb=peak_gpu_mem_mb,
                recompute_count=step_recompute_count,
                checkpoint_count=step_checkpoint_count,
                backward_relocation_count=step_backward_relocation_count,
            )
            step_traces.append(step_trace)

            total_transfer_bytes += step_transfer_bytes
            total_transfer_events += step_transfer_events
            total_prefetch_bytes += step_prefetch_bytes
            total_prefetch_events += step_prefetch_events
    finally:
        run_duration_sec = max(0.0, time.perf_counter() - run_start)
        energy_monitor.stop()

    avg_power_w = float(energy_monitor.get_avg_power())
    total_energy_j = float(avg_power_w * run_duration_sec) if avg_power_w > 0 else 0.0
    if total_energy_j <= 0.0:
        warnings.append(
            "Observed energy could not be measured (power monitor unavailable or zero readings)"
        )

    if unsupported_checkpoint_layers:
        warnings.append(
            "Checkpoint strategy requested but runtime currently supports only recompute; falling back to retain for: "
            f"{sorted(unsupported_checkpoint_layers)}"
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
        total_prefetch_mb=total_prefetch_bytes / (1024**2),
        total_prefetch_events=total_prefetch_events,
        peak_gpu_mem_mb=global_peak_gpu_mem_mb,
        prefetch_layers=sorted(prefetch_layers_used),
        recompute_layers=sorted(recompute_layers_used),
        checkpoint_layers=sorted(checkpoint_layers_used),
        backward_relocation_layers=sorted(backward_relocation_layers_used),
        unsupported_checkpoint_layers=sorted(unsupported_checkpoint_layers),
        warnings=warnings,
        per_step=step_traces,
    )
