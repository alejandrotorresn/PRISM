from __future__ import annotations

import inspect
import math
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple
from typing import cast

import torch
import torch.fx as fx
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as activation_checkpoint

try:
    from core.loss_utils import compute_training_objective
except ModuleNotFoundError:
    import sys
    from pathlib import Path

    src_dir = str(Path(__file__).resolve().parents[1])
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    from core.loss_utils import compute_training_objective

try:
    from core.decoder_export_backend import try_export_decoder_only_trace
except ModuleNotFoundError:
    import sys
    from pathlib import Path

    src_dir = str(Path(__file__).resolve().parents[1])
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    from core.decoder_export_backend import try_export_decoder_only_trace

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
    loss_value: float
    task_metric_value: float | None = None


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
    initial_loss: float
    final_loss: float
    min_loss: float
    loss_delta: float
    prefetch_layers: List[str]
    recompute_layers: List[str]
    checkpoint_layers: List[str]
    backward_relocation_layers: List[str]
    unsupported_checkpoint_layers: List[str]
    warnings: List[str]
    per_step: List[StepTrace]
    quality_metric_name: str | None = None
    initial_quality_metric: float | None = None
    final_quality_metric: float | None = None
    best_quality_metric: float | None = None
    quality_metric_delta: float | None = None

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
            "initial_loss": self.initial_loss,
            "final_loss": self.final_loss,
            "min_loss": self.min_loss,
            "loss_delta": self.loss_delta,
            "prefetch_layers": self.prefetch_layers,
            "recompute_layers": self.recompute_layers,
            "checkpoint_layers": self.checkpoint_layers,
            "backward_relocation_layers": self.backward_relocation_layers,
            "unsupported_checkpoint_layers": self.unsupported_checkpoint_layers,
            "warnings": self.warnings,
            "quality_metric_name": self.quality_metric_name,
            "initial_quality_metric": self.initial_quality_metric,
            "final_quality_metric": self.final_quality_metric,
            "best_quality_metric": self.best_quality_metric,
            "quality_metric_delta": self.quality_metric_delta,
            "per_step": [asdict(s) for s in self.per_step],
        }


class HybridExecutionUnsupportedError(RuntimeError):
    pass


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


def _to_device_non_blocking(x: Any, device: torch.device) -> Any:
    if isinstance(x, torch.Tensor):
        return x.to(device, non_blocking=True)
    if isinstance(x, list):
        return [_to_device_non_blocking(v, device) for v in x]
    if isinstance(x, tuple):
        return tuple(_to_device_non_blocking(v, device) for v in x)
    if isinstance(x, dict):
        return {k: _to_device_non_blocking(v, device) for k, v in x.items()}
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


def _all_tensor_devices(x: Any) -> List[torch.device]:
    if isinstance(x, torch.Tensor):
        return [x.device]
    if isinstance(x, (list, tuple)):
        devices: List[torch.device] = []
        for value in x:
            devices.extend(_all_tensor_devices(value))
        return devices
    if isinstance(x, dict):
        devices = []
        for value in x.values():
            devices.extend(_all_tensor_devices(value))
        return devices
    return []


def _compute_loss(out: Any) -> torch.Tensor:
    loss, _, _ = compute_training_objective(out)
    return loss


def _compute_objective(
    out: Any,
    target_data: torch.Tensor | None = None,
) -> tuple[torch.Tensor, str | None, float | None]:
    return compute_training_objective(out, target=target_data)


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


def _try_huggingface_symbolic_trace(model: nn.Module) -> Optional[fx.GraphModule]:
    try:
        from transformers import PreTrainedModel
        from transformers.utils.fx import symbolic_trace as hf_symbolic_trace
    except Exception:
        return None

    if not isinstance(model, PreTrainedModel):
        return None

    return hf_symbolic_trace(model)


def _try_symbolic_trace(model: nn.Module) -> Optional[fx.GraphModule]:
    was_training = model.training
    config = getattr(model, "config", None)
    original_use_cache = getattr(config, "use_cache", None) if config is not None else None
    try:
        return fx.symbolic_trace(model)
    except Exception:
        try:
            return _try_huggingface_symbolic_trace(model)
        except Exception:
            return None
    finally:
        model.train(was_training)
        if config is not None and original_use_cache is not None:
            config.use_cache = original_use_cache


def _prepare_fx_run_args(graph_module: fx.GraphModule, input_data: Any) -> Tuple[Any, ...]:
    if isinstance(input_data, tuple):
        return input_data
    if not isinstance(input_data, dict):
        return (input_data,)

    placeholder_nodes = [node for node in graph_module.graph.nodes if node.op == "placeholder"]
    args: List[Any] = []
    missing: List[str] = []
    for node in placeholder_nodes:
        if node.name in input_data:
            args.append(input_data[node.name])
            continue

        default_value = node.args[0] if node.args else inspect._empty
        if default_value is inspect._empty:
            missing.append(node.name)
            continue
        args.append(default_value)

    if missing:
        raise HybridExecutionUnsupportedError(
            f"Structured FX input is missing placeholder values required by the traced graph: {missing}"
        )
    return tuple(args)


class _DeviceAwareFXInterpreter(fx.Interpreter):
    def __init__(
        self,
        module: fx.GraphModule,
        plan: DevicePlan,
        gpu_id: int,
        transfer_metrics: Dict[str, float],
        activation_strategies: Dict[str, Any] | None,
        warnings: List[str],
        runtime_features: Dict[str, Any],
        enable_async_transfer: bool,
        enable_prefetch: bool,
        transfer_stream: torch.cuda.Stream | None,
        node_layer_names: Dict[str, str] | None = None,
        supports_activation_strategies: bool = True,
    ):
        super().__init__(module)
        self._plan = plan
        self._gpu_id = gpu_id
        self._transfer_metrics = transfer_metrics
        self._activation_strategies = activation_strategies or {}
        self._warnings = warnings
        self._runtime_features = runtime_features
        self._enable_async_transfer = enable_async_transfer
        self._enable_prefetch = enable_prefetch
        self._transfer_stream = transfer_stream
        self._node_layer_names = node_layer_names or {}
        self._supports_activation_strategies = supports_activation_strategies
        self._prefetched_inputs: Dict[str, torch.Tensor] = {}
        self._active_node: fx.Node | None = None
        self._attr_cache: Dict[Tuple[str, str], torch.Tensor] = {}
        self._warned_messages: set[str] = set()

    def _warn_once(self, message: str) -> None:
        if message in self._warned_messages:
            return
        self._warned_messages.add(message)
        self._warnings.append(message)

    def _layer_name_for_node(self, node: fx.Node | None) -> str | None:
        if node is None:
            return None
        mapped = self._node_layer_names.get(node.name)
        if mapped:
            return mapped
        if node.op == "call_module":
            return str(node.target)
        return None

    def _move_inputs_to_device(
        self,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
        target_device: torch.device,
    ) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        devices = _all_tensor_devices(args) + _all_tensor_devices(kwargs)
        if not devices:
            return args, kwargs
        if all(device == target_device for device in devices):
            return args, kwargs

        if self._transfer_stream is not None and self._enable_async_transfer and target_device.type == "cuda":
            with torch.cuda.stream(self._transfer_stream):
                moved_args = _to_device_non_blocking(args, target_device)
                moved_kwargs = _to_device_non_blocking(kwargs, target_device)
            torch.cuda.current_stream(self._gpu_id).wait_stream(self._transfer_stream)
        else:
            moved_args = _to_device(args, target_device)
            moved_kwargs = _to_device(kwargs, target_device)

        self._transfer_metrics["bytes"] += float(_tensor_bytes(args) + _tensor_bytes(kwargs))
        self._transfer_metrics["events"] += 1.0
        return moved_args, moved_kwargs

    def _record_unsupported_activation(self, layer_name: str) -> None:
        strategy_name = _activation_strategy_name(self._activation_strategies.get(layer_name))
        if strategy_name == "retain":
            return
        self._runtime_features["unsupported_activation_layers"].add(layer_name)
        self._warn_once(
            f"Layer '{layer_name}' requested activation strategy '{strategy_name}' but decoder-only export DAG executes at operator granularity; falling back to retain"
        )

    def _align_operator_inputs(self, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        devices = _all_tensor_devices(args) + _all_tensor_devices(kwargs)
        if len(devices) <= 1:
            return args, kwargs

        target_device = next((device for device in devices if device.type == "cuda"), devices[0])
        if all(device == target_device for device in devices):
            return args, kwargs

        if self._transfer_stream is not None and self._enable_async_transfer and target_device.type == "cuda":
            with torch.cuda.stream(self._transfer_stream):
                moved_args = _to_device_non_blocking(args, target_device)
                moved_kwargs = _to_device_non_blocking(kwargs, target_device)
            torch.cuda.current_stream(self._gpu_id).wait_stream(self._transfer_stream)
        else:
            moved_args = _to_device(args, target_device)
            moved_kwargs = _to_device(kwargs, target_device)

        self._transfer_metrics["bytes"] += float(_tensor_bytes(args) + _tensor_bytes(kwargs))
        self._transfer_metrics["events"] += 1.0
        return moved_args, moved_kwargs

    def get_attr(self, target: fx.node.Target, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        value = super().get_attr(target, args, kwargs)
        if not isinstance(value, torch.Tensor):
            return value

        layer_name = self._layer_name_for_node(self._active_node)
        if layer_name is None:
            return value

        forward_device = self._plan.resolve_torch_device(layer_name=layer_name, gpu_id=self._gpu_id, phase="forward")
        if value.device == forward_device:
            return value

        cache_key = (str(target), str(forward_device))
        cached = self._attr_cache.get(cache_key)
        if cached is not None:
            return cached

        moved = value.to(forward_device)
        self._transfer_metrics["bytes"] += float(_tensor_bytes(value))
        self._transfer_metrics["events"] += 1.0
        self._attr_cache[cache_key] = moved
        return moved

    def run_node(self, n: fx.Node) -> Any:
        self._active_node = n
        try:
            return super().run_node(n)
        finally:
            self._active_node = None

    def call_module(self, target: fx.node.Target, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        layer_name = self._layer_name_for_node(self._active_node) or str(target)
        forward_device = self._plan.resolve_torch_device(layer_name=layer_name, gpu_id=self._gpu_id, phase="forward")
        backward_device = self._plan.resolve_torch_device(layer_name=layer_name, gpu_id=self._gpu_id, phase="backward")

        src_dev = _current_device(args)
        payload_bytes = _tensor_bytes(args) + _tensor_bytes(kwargs)

        current_node = self._active_node
        node_name = current_node.name if current_node is not None else layer_name

        used_prefetched_input = (
            node_name in self._prefetched_inputs
            and len(args) == 1
            and isinstance(args[0], torch.Tensor)
            and len(kwargs) == 0
        )
        if used_prefetched_input:
            moved_args = (self._prefetched_inputs.pop(node_name),)
            moved_kwargs = kwargs
            if self._transfer_stream is not None and moved_args[0].device.type == "cuda":
                torch.cuda.current_stream(self._gpu_id).wait_stream(self._transfer_stream)
        else:
            if self._transfer_stream is not None and self._enable_async_transfer and forward_device.type == "cuda":
                with torch.cuda.stream(self._transfer_stream):
                    moved_args = _to_device_non_blocking(args, forward_device)
                    moved_kwargs = _to_device_non_blocking(kwargs, forward_device)
                torch.cuda.current_stream(self._gpu_id).wait_stream(self._transfer_stream)
            else:
                moved_args = _to_device(args, forward_device)
                moved_kwargs = _to_device(kwargs, forward_device)

        if src_dev is not None and src_dev != forward_device:
            self._transfer_metrics["bytes"] += float(payload_bytes)
            self._transfer_metrics["events"] += 1.0

        submod = self.fetch_attr(target)
        submod.to(forward_device)

        strategy_name = _activation_strategy_name(self._activation_strategies.get(layer_name))
        if strategy_name == "recompute":
            self._runtime_features["recompute_layers"].add(layer_name)
        elif strategy_name == "checkpoint":
            self._runtime_features["checkpoint_layers"].add(layer_name)

        wants_dual_runtime = (
            self._plan.assignment_forward.get(layer_name) != self._plan.assignment_backward.get(layer_name)
        )
        can_materialize_dual = (
            wants_dual_runtime
            and len(moved_args) == 1
            and isinstance(moved_args[0], torch.Tensor)
            and len(moved_kwargs) == 0
            and forward_device != backward_device
        )

        if wants_dual_runtime and not can_materialize_dual:
            self._warnings.append(
                f"Layer '{layer_name}' requests different backward placement but DAG runtime cannot materialize it for this call signature; using forward placement"
            )

        if can_materialize_dual:
            self._runtime_features["backward_relocation_layers"].add(layer_name)
            self._runtime_features["backward_relocation_count"] += 1
            return _run_layer_with_dual_placement(
                layer=submod,
                layer_name=layer_name,
                input_tensor=cast(torch.Tensor, moved_args[0]),
                forward_device=forward_device,
                backward_device=backward_device,
            )

        if strategy_name == "recompute":
            if len(moved_args) == 1 and isinstance(moved_args[0], torch.Tensor) and len(moved_kwargs) == 0:
                self._runtime_features["recompute_count"] += 1
                return activation_checkpoint(submod, moved_args[0], use_reentrant=False)
            self._warnings.append(
                f"Layer '{layer_name}' requested recompute but DAG runtime supports only single-Tensor positional input; falling back to retain"
            )
            self._runtime_features["unsupported_activation_layers"].add(layer_name)

        if strategy_name == "checkpoint":
            if len(moved_args) == 1 and isinstance(moved_args[0], torch.Tensor) and len(moved_kwargs) == 0:
                self._runtime_features["checkpoint_count"] += 1
                checkpoint_ctx = _saved_tensor_cpu_offload_context()
                with checkpoint_ctx:
                    return submod(moved_args[0])
            self._warnings.append(
                f"Layer '{layer_name}' requested checkpoint but DAG runtime supports only single-Tensor positional input; falling back to retain"
            )
            self._runtime_features["unsupported_activation_layers"].add(layer_name)

        out = submod(*moved_args, **moved_kwargs)

        if self._enable_prefetch and current_node is not None and isinstance(out, torch.Tensor):
            for user_node in current_node.users:
                if user_node.op != "call_module":
                    continue
                user_name = user_node.name
                user_target = str(user_node.target)
                user_device = self._plan.resolve_torch_device(
                    layer_name=user_target,
                    gpu_id=self._gpu_id,
                    phase="forward",
                )
                out_device = out.device
                if out_device == user_device:
                    continue
                if self._transfer_stream is not None and self._enable_async_transfer and user_device.type == "cuda":
                    with torch.cuda.stream(self._transfer_stream):
                        prefetched = out.to(user_device, non_blocking=True)
                else:
                    prefetched = out.to(user_device)
                self._prefetched_inputs[user_name] = prefetched
                self._runtime_features["prefetch_layers"].add(user_target)
                self._runtime_features["prefetch_count"] += 1
                self._runtime_features["prefetch_bytes"] += float(_tensor_bytes(out))

        return out

    def call_function(self, target: fx.node.Target, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        if not self._supports_activation_strategies and str(target) == "aten._assert_tensor_metadata.default":
            return None

        layer_name = self._layer_name_for_node(self._active_node)
        if layer_name is None:
            moved_args, moved_kwargs = self._align_operator_inputs(args, kwargs)
            return target(*moved_args, **moved_kwargs)

        forward_device = self._plan.resolve_torch_device(layer_name=layer_name, gpu_id=self._gpu_id, phase="forward")
        backward_device = self._plan.resolve_torch_device(layer_name=layer_name, gpu_id=self._gpu_id, phase="backward")
        if not self._supports_activation_strategies:
            self._record_unsupported_activation(layer_name)
        if forward_device != backward_device:
            self._warn_once(
                f"Layer '{layer_name}' requests different backward placement but decoder-only export DAG cannot materialize operator-level dual placement; using forward placement"
            )
        moved_args, moved_kwargs = self._move_inputs_to_device(args, kwargs, forward_device)
        return target(*moved_args, **moved_kwargs)

    def call_method(self, target: fx.node.Target, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        layer_name = self._layer_name_for_node(self._active_node)
        if layer_name is None:
            moved_args, moved_kwargs = self._align_operator_inputs(args, kwargs)
            self_obj, *method_args = moved_args
            return getattr(self_obj, target)(*method_args, **moved_kwargs)

        forward_device = self._plan.resolve_torch_device(layer_name=layer_name, gpu_id=self._gpu_id, phase="forward")
        backward_device = self._plan.resolve_torch_device(layer_name=layer_name, gpu_id=self._gpu_id, phase="backward")
        if not self._supports_activation_strategies:
            self._record_unsupported_activation(layer_name)
        if forward_device != backward_device:
            self._warn_once(
                f"Layer '{layer_name}' requests different backward placement but decoder-only export DAG cannot materialize operator-level dual placement; using forward placement"
            )
        moved_args, moved_kwargs = self._move_inputs_to_device(args, kwargs, forward_device)
        self_obj, *method_args = moved_args
        return getattr(self_obj, target)(*method_args, **moved_kwargs)


def _run_hybrid_training_dag(
    model: nn.Module,
    input_data: Any,
    target_data: torch.Tensor | None,
    plan: DevicePlan,
    activation_strategies: Dict[str, Any] | None,
    steps: int,
    lr: float,
    gpu_id: int,
    enable_rapl: bool,
    energy_sample_interval: float,
    enable_async_transfer: bool,
    enable_prefetch: bool,
    warnings: List[str],
    traced_module: Optional[fx.GraphModule] = None,
    node_layer_names: Dict[str, str] | None = None,
    supports_activation_strategies: bool = True,
) -> HybridExecutionResult:
    graph_module = traced_module or _try_symbolic_trace(model)
    if graph_module is None:
        raise RuntimeError("DAG execution requested but symbolic tracing failed for this model")

    model.train()
    try:
        graph_module.train()
    except NotImplementedError:
        warnings.append("Trace backend does not support GraphModule.train(); executing with the exported module as-is")
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    step_traces: List[StepTrace] = []
    total_transfer_bytes = 0.0
    total_transfer_events = 0.0
    total_prefetch_bytes = 0.0
    total_prefetch_events = 0
    global_peak_gpu_mem_mb = 0.0
    recompute_layers_used: set[str] = set()
    checkpoint_layers_used: set[str] = set()
    backward_relocation_layers_used: set[str] = set()
    prefetch_layers_used: set[str] = set()
    unsupported_activation_layers: set[str] = set()
    quality_metric_name_used: str | None = None

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
    transfer_stream = torch.cuda.Stream(device=gpu_id) if (enable_async_transfer and torch.cuda.is_available()) else None
    try:
        for step_idx in range(steps):
            step_start = time.perf_counter()
            optimizer.zero_grad()

            transfer_metrics = {"bytes": 0.0, "events": 0.0}
            runtime_features = {
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
            interp = _DeviceAwareFXInterpreter(
                graph_module,
                plan,
                gpu_id,
                transfer_metrics,
                activation_strategies,
                warnings,
                runtime_features,
                enable_async_transfer,
                enable_prefetch,
                transfer_stream,
                node_layer_names=node_layer_names,
                supports_activation_strategies=supports_activation_strategies,
            )

            forward_start = time.perf_counter()
            out = interp.run(*_prepare_fx_run_args(graph_module, input_data))

            if torch.cuda.is_available():
                torch.cuda.synchronize(gpu_id)
            forward_ms = (time.perf_counter() - forward_start) * 1000.0

            backward_start = time.perf_counter()
            loss, quality_metric_name, quality_metric_value = _compute_objective(out, target_data=target_data)
            if quality_metric_name_used is None and quality_metric_name is not None:
                quality_metric_name_used = quality_metric_name
            loss_value = float(loss.detach().cpu().item())
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

            step_transfer_bytes = float(transfer_metrics["bytes"])
            step_transfer_events = int(transfer_metrics["events"])
            step_prefetch_bytes = float(runtime_features["prefetch_bytes"])
            step_prefetch_events = int(runtime_features["prefetch_count"])

            step_traces.append(
                StepTrace(
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
                    recompute_count=int(runtime_features["recompute_count"]),
                    checkpoint_count=int(runtime_features["checkpoint_count"]),
                    backward_relocation_count=int(runtime_features["backward_relocation_count"]),
                    loss_value=loss_value,
                    task_metric_value=quality_metric_value,
                )
            )

            recompute_layers_used.update(cast(set[str], runtime_features["recompute_layers"]))
            checkpoint_layers_used.update(cast(set[str], runtime_features["checkpoint_layers"]))
            backward_relocation_layers_used.update(cast(set[str], runtime_features["backward_relocation_layers"]))
            prefetch_layers_used.update(cast(set[str], runtime_features["prefetch_layers"]))
            unsupported_activation_layers.update(cast(set[str], runtime_features["unsupported_activation_layers"]))

            total_prefetch_bytes += step_prefetch_bytes
            total_prefetch_events += step_prefetch_events

            total_transfer_bytes += step_transfer_bytes
            total_transfer_events += float(step_transfer_events)
    finally:
        run_duration_sec = max(0.0, time.perf_counter() - run_start)
        energy_monitor.stop()

    avg_power_w = float(energy_monitor.get_avg_power())
    total_energy_j = float(avg_power_w * run_duration_sec) if avg_power_w > 0 else 0.0
    if total_energy_j <= 0.0:
        warnings.append("Observed energy could not be measured (power monitor unavailable or zero readings)")

    avg_step_ms = sum(s.total_ms for s in step_traces) / len(step_traces)
    loss_values = [s.loss_value for s in step_traces]
    initial_loss = float(loss_values[0]) if loss_values else 0.0
    final_loss = float(loss_values[-1]) if loss_values else 0.0
    min_loss = float(min(loss_values)) if loss_values else 0.0
    loss_delta = float(final_loss - initial_loss) if loss_values else 0.0
    quality_values = [float(s.task_metric_value) for s in step_traces if s.task_metric_value is not None and math.isfinite(float(s.task_metric_value))]
    initial_quality_metric = float(quality_values[0]) if quality_values else None
    final_quality_metric = float(quality_values[-1]) if quality_values else None
    best_quality_metric = float(max(quality_values)) if quality_values else None
    quality_metric_delta = (
        float(final_quality_metric - initial_quality_metric)
        if initial_quality_metric is not None and final_quality_metric is not None
        else None
    )

    if unsupported_activation_layers:
        warnings.append(
            "DAG runtime could not apply some activation strategies due to unsupported call signatures; falling back to retain for: "
            f"{sorted(unsupported_activation_layers)}"
        )

    return HybridExecutionResult(
        status="ok",
        steps=steps,
        avg_step_ms=avg_step_ms,
        avg_power_w=avg_power_w,
        total_energy_j=total_energy_j,
        energy_source=energy_source,
        total_transfer_mb=total_transfer_bytes / (1024**2),
        total_transfer_events=int(total_transfer_events),
        total_prefetch_mb=total_prefetch_bytes / (1024**2),
        total_prefetch_events=total_prefetch_events,
        peak_gpu_mem_mb=global_peak_gpu_mem_mb,
        initial_loss=initial_loss,
        final_loss=final_loss,
        min_loss=min_loss,
        loss_delta=loss_delta,
        prefetch_layers=sorted(prefetch_layers_used),
        recompute_layers=sorted(recompute_layers_used),
        checkpoint_layers=sorted(checkpoint_layers_used),
        backward_relocation_layers=sorted(backward_relocation_layers_used),
        unsupported_checkpoint_layers=sorted(unsupported_activation_layers),
        warnings=warnings,
        per_step=step_traces,
        quality_metric_name=quality_metric_name_used,
        initial_quality_metric=initial_quality_metric,
        final_quality_metric=final_quality_metric,
        best_quality_metric=best_quality_metric,
        quality_metric_delta=quality_metric_delta,
    )


def run_hybrid_training(
    model: nn.Module,
    input_data: Any,
    plan: DevicePlan,
    target_data: torch.Tensor | None = None,
    activation_strategies: Dict[str, Any] | None = None,
    steps: int = 5,
    lr: float = 0.01,
    gpu_id: int = 0,
    strict_plan: bool = True,
    enable_rapl: bool = False,
    energy_sample_interval: float = 0.05,
    enable_async_transfer: bool = False,
    enable_prefetch: bool = False,
    execution_mode: str = "linear",
) -> HybridExecutionResult:
    if steps <= 0:
        raise ValueError(f"steps must be > 0, got {steps}")
    if execution_mode not in {"auto", "linear", "dag"}:
        raise ValueError(f"execution_mode must be one of auto|linear|dag, got: {execution_mode}")

    traced_module = _try_symbolic_trace(model)
    export_trace_ctx = None if traced_module is not None else try_export_decoder_only_trace(model, input_data)
    coverage_layer_names = None if export_trace_ctx is None else set(export_trace_ctx.node_layer_names.values())
    warnings = validate_plan_coverage(
        model=model,
        plan=plan,
        strict=strict_plan,
        model_layer_names=coverage_layer_names,
    )
    if not torch.cuda.is_available() and (
        any(v == "GPU" for v in plan.assignment_forward.values())
        or any(v == "GPU" for v in plan.assignment_backward.values())
    ):
        warnings.append("GPU layers requested but CUDA is unavailable; those layers run on CPU")

    if execution_mode == "dag" and traced_module is None and export_trace_ctx is None:
        raise HybridExecutionUnsupportedError(
            "execution_mode=dag requires FX symbolic tracing support for this model"
        )

    active_trace = traced_module if traced_module is not None else (None if export_trace_ctx is None else export_trace_ctx.graph_module)
    active_node_layer_names = None if export_trace_ctx is None else export_trace_ctx.node_layer_names
    active_supports_activation_strategies = export_trace_ctx is None

    if execution_mode == "dag":
        return _run_hybrid_training_dag(
            model=model,
            input_data=input_data,
            target_data=target_data,
            plan=plan,
            activation_strategies=activation_strategies,
            steps=steps,
            lr=lr,
            gpu_id=gpu_id,
            enable_rapl=enable_rapl,
            energy_sample_interval=energy_sample_interval,
            enable_async_transfer=enable_async_transfer,
            enable_prefetch=enable_prefetch,
            warnings=warnings,
            traced_module=active_trace,
            node_layer_names=active_node_layer_names,
            supports_activation_strategies=active_supports_activation_strategies,
        )

    if execution_mode == "auto":
        can_use_dag = active_trace is not None
        if can_use_dag:
            return _run_hybrid_training_dag(
                model=model,
                input_data=input_data,
                target_data=target_data,
                plan=plan,
                activation_strategies=activation_strategies,
                steps=steps,
                lr=lr,
                gpu_id=gpu_id,
                enable_rapl=enable_rapl,
                energy_sample_interval=energy_sample_interval,
                enable_async_transfer=enable_async_transfer,
                enable_prefetch=enable_prefetch,
                warnings=warnings,
                traced_module=active_trace,
                node_layer_names=active_node_layer_names,
                supports_activation_strategies=active_supports_activation_strategies,
            )
        if active_trace is None:
            if not isinstance(input_data, torch.Tensor):
                raise HybridExecutionUnsupportedError(
                    "Auto execution mode requires FX graph tracing for structured inputs, but tracing failed for this model; "
                    "linear fallback only supports single-tensor leaf-chain execution"
                )
            warnings.append("Auto execution mode fell back to linear runtime (FX tracing unavailable)")

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
    quality_metric_name_used: str | None = None

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
            loss, quality_metric_name, quality_metric_value = _compute_objective(x, target_data=target_data)
            if quality_metric_name_used is None and quality_metric_name is not None:
                quality_metric_name_used = quality_metric_name
            loss_value = float(loss.detach().cpu().item())
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
                loss_value=loss_value,
                task_metric_value=quality_metric_value,
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
    loss_values = [s.loss_value for s in step_traces]
    initial_loss = float(loss_values[0]) if loss_values else 0.0
    final_loss = float(loss_values[-1]) if loss_values else 0.0
    min_loss = float(min(loss_values)) if loss_values else 0.0
    loss_delta = float(final_loss - initial_loss) if loss_values else 0.0
    quality_values = [float(s.task_metric_value) for s in step_traces if s.task_metric_value is not None and math.isfinite(float(s.task_metric_value))]
    initial_quality_metric = float(quality_values[0]) if quality_values else None
    final_quality_metric = float(quality_values[-1]) if quality_values else None
    best_quality_metric = float(max(quality_values)) if quality_values else None
    quality_metric_delta = (
        float(final_quality_metric - initial_quality_metric)
        if initial_quality_metric is not None and final_quality_metric is not None
        else None
    )

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
        initial_loss=initial_loss,
        final_loss=final_loss,
        min_loss=min_loss,
        loss_delta=loss_delta,
        prefetch_layers=sorted(prefetch_layers_used),
        recompute_layers=sorted(recompute_layers_used),
        checkpoint_layers=sorted(checkpoint_layers_used),
        backward_relocation_layers=sorted(backward_relocation_layers_used),
        unsupported_checkpoint_layers=sorted(unsupported_checkpoint_layers),
        warnings=warnings,
        per_step=step_traces,
        quality_metric_name=quality_metric_name_used,
        initial_quality_metric=initial_quality_metric,
        final_quality_metric=final_quality_metric,
        best_quality_metric=best_quality_metric,
        quality_metric_delta=quality_metric_delta,
    )
