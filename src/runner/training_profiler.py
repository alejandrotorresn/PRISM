import logging
import math
import os
import time
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn

from core.constants import (
    BACKWARD_FACTOR,
    MEASURE_STEPS,
    OPTIMIZER_OVERHEAD_FACTOR,
    OPTIMIZER_OVERHEAD_MAP,
    WARMUP_STEPS,
)
from core.energy import EnergyMonitor
from core.loss_utils import compute_stable_surrogate_loss
from core.graph_extractor import export_graph_artifacts
from core.io_artifacts import cleanup_artifacts, write_csv_rows, write_json_dict
from core.metrics import estimate_flops, get_tensor_size_recursive
from core.precision_policy import run_cpu_fp16_model_preflight
from core.system import get_hardware_metadata

logger = logging.getLogger(__name__)


def _is_oom_runtime_error(ex: RuntimeError) -> bool:
    msg = str(ex).lower()
    oom_tokens = [
        "out of memory",
        "cuda out of memory",
        "cublas_status_alloc_failed",
        "cuda error: out of memory",
        "hip out of memory",
    ]
    return any(token in msg for token in oom_tokens)


def _new_layer_stat(module: nn.Module) -> Dict[str, Any]:
    return {
        "type": module.__class__.__name__,
        "time_ms_accum": 0.0,
        "dispatch_ms_accum": 0.0,
        "bwd_time_ms_accum": 0.0,
        "bwd_dispatch_ms_accum": 0.0,
        "mem_mb": 0.0,
        "count": 0,
        "bwd_count": 0,
        "output_bytes": 0,
        "grad_output_bytes": 0,
        "params_mb": 0.0,
        "flops": 0.0,
    }


def _piecewise_transfer_ms(
    tensor_mb: float,
    alpha_ms: float,
    beta_nominal_mb_per_ms: float,
    beta_congested_mb_per_ms: float,
    congestion_knee_mb: float,
) -> float:
    beta_nominal = max(float(beta_nominal_mb_per_ms), 1e-6)
    beta_congested = max(float(beta_congested_mb_per_ms), 1e-6)
    knee = max(float(congestion_knee_mb), 0.0)
    size_mb = max(float(tensor_mb), 0.0)
    if size_mb <= knee:
        return float(alpha_ms + (size_mb / beta_nominal))
    return float(alpha_ms + (knee / beta_nominal) + ((size_mb - knee) / beta_congested))


def _build_branch_pressure_maps(graph_edges: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    outgoing_volume_mb: Dict[str, float] = {}
    incoming_volume_mb: Dict[str, float] = {}

    for edge in graph_edges:
        producer = str(edge.get("producer_name", ""))
        consumer = str(edge.get("consumer_name", ""))
        tensor_mb = float(edge.get("tensor_mb", 0.0) or 0.0)

        outgoing_volume_mb[producer] = outgoing_volume_mb.get(producer, 0.0) + tensor_mb
        incoming_volume_mb[consumer] = incoming_volume_mb.get(consumer, 0.0) + tensor_mb

    return {
        "outgoing_volume_mb": outgoing_volume_mb,
        "incoming_volume_mb": incoming_volume_mb,
    }


class TrainingProfiler:
    def __init__(self, model: nn.Module, model_name: str, args):
        self.model = model
        self.model_name = model_name
        self.args = args
        self.layer_stats = {}
        self.hooks = []
        self._last_opt_step_ms = 0.0
        self._last_opt_step_count = 0
        self._partial_csv_path = None
        self._partial_json_path = None
        self._fwd_tstarts = {}
        self._bwd_tstarts = {}
        self._use_backward_hooks = True
        self._active_optimizer = None
        self._oom_retry_events = 0
        self._oom_retry_last_micro_batch = None
        self.has_gpu = torch.cuda.is_available() and not args.no_gpu
        self.gpu_id = args.gpu_id if self.has_gpu else 0
        self._disable_inplace_modules_for_backward_hooks()
        if self.has_gpu:
            self.model.to(f"cuda:{self.gpu_id}")
        else:
            self.model.to("cpu")

    def _disable_inplace_modules_for_backward_hooks(self) -> None:
        changed = 0
        for module in self.model.modules():
            if hasattr(module, "inplace"):
                try:
                    if bool(getattr(module, "inplace")):
                        setattr(module, "inplace", False)
                        changed += 1
                except Exception:
                    continue
        if changed > 0:
            logger.info(
                "Disabled inplace execution on %d modules to ensure backward-hook compatibility.",
                changed,
            )

    def _get_leaf_modules(self) -> Iterator[Tuple[str, nn.Module]]:
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:
                yield name, module

    def _compute_loss(self, out: Any) -> torch.Tensor:
        return compute_stable_surrogate_loss(out)

    def _register_hooks(self, device_type: str):
        for h in self.hooks:
            h.remove()
        self.hooks = []
        self._fwd_tstarts = {}
        self._bwd_tstarts = {}

        def pre_hook(name):
            def hook(module, inp):
                self._fwd_tstarts[name] = {"cpu": time.perf_counter()}
                if device_type == "cuda":
                    start = torch.cuda.Event(enable_timing=True)
                    start.record()
                    self._fwd_tstarts[name]["gpu"] = start

            return hook

        def post_hook(name):
            def hook(module, inp, out):
                cpu_end = time.perf_counter()

                if name not in self.layer_stats:
                    self.layer_stats[name] = _new_layer_stat(module)
                s = self.layer_stats[name]

                if s["count"] == 0:
                    try:
                        params_bytes = sum(p.numel() * p.element_size() for p in module.parameters(recurse=False))
                        s["params_mb"] = params_bytes / (1024**2)
                    except Exception:
                        pass

                    try:
                        s["output_bytes"] = get_tensor_size_recursive(out)
                    except Exception:
                        pass

                    try:
                        s["flops"] = estimate_flops(module, inp, out)
                    except Exception:
                        pass

                kernel_ms = 0.0
                if device_type == "cuda":
                    end = torch.cuda.Event(enable_timing=True)
                    end.record()
                    torch.cuda.synchronize()
                    kernel_ms = self._fwd_tstarts[name]["gpu"].elapsed_time(end)
                    s["mem_mb"] = max(s["mem_mb"], torch.cuda.memory_allocated(self.gpu_id) / (1024**2))
                else:
                    kernel_ms = (cpu_end - self._fwd_tstarts[name]["cpu"]) * 1000.0

                wall_ms = (cpu_end - self._fwd_tstarts[name]["cpu"]) * 1000.0
                dispatch_ms = max(0.0, wall_ms - kernel_ms)

                s["time_ms_accum"] += kernel_ms
                s["dispatch_ms_accum"] += dispatch_ms
                s["count"] += 1

            return hook

        def backward_pre_hook(name):
            def hook(module, grad_output):
                self._bwd_tstarts[name] = {"cpu": time.perf_counter()}
                if device_type == "cuda":
                    start = torch.cuda.Event(enable_timing=True)
                    start.record()
                    self._bwd_tstarts[name]["gpu"] = start

            return hook

        def backward_post_hook(name):
            def hook(module, grad_input, grad_output):
                start_info = self._bwd_tstarts.get(name)
                if start_info is None:
                    return

                cpu_end = time.perf_counter()
                if name not in self.layer_stats:
                    self.layer_stats[name] = _new_layer_stat(module)
                s = self.layer_stats[name]

                if s["bwd_count"] == 0:
                    try:
                        s["grad_output_bytes"] = get_tensor_size_recursive(grad_output)
                    except Exception:
                        pass

                if device_type == "cuda":
                    end = torch.cuda.Event(enable_timing=True)
                    end.record()
                    torch.cuda.synchronize()
                    kernel_ms = start_info["gpu"].elapsed_time(end)
                else:
                    kernel_ms = (cpu_end - start_info["cpu"]) * 1000.0

                wall_ms = (cpu_end - start_info["cpu"]) * 1000.0
                dispatch_ms = max(0.0, wall_ms - kernel_ms)

                s["bwd_time_ms_accum"] += kernel_ms
                s["bwd_dispatch_ms_accum"] += dispatch_ms
                s["bwd_count"] += 1

            return hook

        for name, module in self._get_leaf_modules():
            self.hooks.append(module.register_forward_pre_hook(pre_hook(name)))
            self.hooks.append(module.register_forward_hook(post_hook(name)))
            if self._use_backward_hooks:
                self.hooks.append(module.register_full_backward_pre_hook(backward_pre_hook(name)))
                self.hooks.append(module.register_full_backward_hook(backward_post_hook(name)))

    def _build_optimizer(self, params, opt_name: str, lr: float, momentum: float):
        if opt_name == "SGD":
            return torch.optim.SGD(params, lr=lr)
        if opt_name == "SGD_momentum":
            return torch.optim.SGD(params, lr=lr, momentum=momentum)
        if opt_name == "Adam":
            return torch.optim.Adam(params, lr=lr)
        if opt_name == "AdamW":
            return torch.optim.AdamW(params, lr=lr)
        if opt_name == "RMSprop":
            return torch.optim.RMSprop(params, lr=lr, momentum=momentum)
        if opt_name == "Adagrad":
            return torch.optim.Adagrad(params, lr=lr)
        if opt_name == "Adadelta":
            return torch.optim.Adadelta(params, lr=lr)
        return torch.optim.SGD(params, lr=lr)

    def _infer_batch_size(self, input_data: Any) -> int:
        if isinstance(input_data, torch.Tensor):
            if input_data.dim() == 0:
                raise ValueError("Cannot infer batch size from scalar tensor input")
            return int(input_data.size(0))

        if isinstance(input_data, dict):
            candidates: List[int] = []
            for value in input_data.values():
                if isinstance(value, torch.Tensor) and value.dim() > 0:
                    candidates.append(int(value.size(0)))
            if not candidates:
                raise ValueError("Cannot infer batch size from dict input without batched tensors")
            return min(candidates)

        raise TypeError(f"Unsupported input_data type for batch inference: {type(input_data)}")

    def _slice_batch(self, input_data: Any, start: int, end: int) -> Any:
        if isinstance(input_data, torch.Tensor):
            return input_data[start:end]

        if isinstance(input_data, dict):
            out: Dict[str, Any] = {}
            for key, value in input_data.items():
                if isinstance(value, torch.Tensor) and value.dim() > 0 and value.size(0) >= end:
                    out[key] = value[start:end]
                else:
                    out[key] = value
            return out

        raise TypeError(f"Unsupported input_data type for batch slicing: {type(input_data)}")

    def _run_step(self, inp: Any, device: str) -> float:
        if isinstance(inp, dict):
            out = self.model(**inp)
        else:
            out = self.model(inp)

        loss = self._compute_loss(out)
        loss.backward()

        t0_opt = time.perf_counter()
        self._active_optimizer.step()
        if device == "cuda":
            torch.cuda.synchronize()
        return (time.perf_counter() - t0_opt) * 1000.0

    def _run_step_with_oom_fallback(self, inp: Any, device: str) -> float:
        oom_retry_enabled = bool(getattr(self.args, "oom_retry_enabled", True))
        target_batch = self._infer_batch_size(inp)
        min_micro_batch = max(1, int(getattr(self.args, "oom_retry_min_batch", 1)))
        backoff = max(2, int(getattr(self.args, "oom_retry_backoff", 2)))

        micro_batch = target_batch
        while True:
            try:
                self._active_optimizer.zero_grad(set_to_none=True)

                if micro_batch >= target_batch:
                    return self._run_step(inp=inp, device=device)

                chunks = int(math.ceil(target_batch / float(micro_batch)))
                for start in range(0, target_batch, micro_batch):
                    end = min(start + micro_batch, target_batch)
                    chunk_inp = self._slice_batch(inp, start, end)
                    if isinstance(chunk_inp, dict):
                        out = self.model(**chunk_inp)
                    else:
                        out = self.model(chunk_inp)

                    # Keep gradient magnitude aligned with full-batch average loss semantics.
                    loss = self._compute_loss(out) / float(chunks)
                    loss.backward()

                t0_opt = time.perf_counter()
                self._active_optimizer.step()
                if device == "cuda":
                    torch.cuda.synchronize()
                return (time.perf_counter() - t0_opt) * 1000.0

            except RuntimeError as ex:
                is_oom = _is_oom_runtime_error(ex)
                can_retry = (
                    oom_retry_enabled
                    and device == "cuda"
                    and is_oom
                    and micro_batch > min_micro_batch
                )
                if not can_retry:
                    raise

                next_micro = max(min_micro_batch, micro_batch // backoff)
                if next_micro == micro_batch and micro_batch > min_micro_batch:
                    next_micro = micro_batch - 1

                self._oom_retry_events += 1
                self._oom_retry_last_micro_batch = next_micro
                logger.warning(
                    "OOM during profiling step for model=%s batch=%d; retrying with micro_batch=%d",
                    self.model_name,
                    target_batch,
                    next_micro,
                )
                if device == "cuda":
                    torch.cuda.empty_cache()
                micro_batch = next_micro

    def _run_epoch(
        self,
        input_data: Any,
        device: str,
        steps: int,
        allow_backward_hook_fallback: bool = True,
    ) -> Tuple[Optional[float], float]:
        device_str = f"cuda:{self.gpu_id}" if device == "cuda" else "cpu"
        self.model.to(device_str)
        self.model.train()

        if isinstance(input_data, dict):
            inp = {k: v.to(device_str) for k, v in input_data.items()}
        else:
            inp = input_data.to(device_str)

        opt_name = getattr(self.args, "optimizer", "SGD")
        lr = getattr(self.args, "lr", 0.01)
        momentum = getattr(self.args, "momentum", 0.9)
        params = self.model.parameters()
        opt = self._build_optimizer(params=params, opt_name=opt_name, lr=lr, momentum=momentum)
        self._active_optimizer = opt

        self.layer_stats = {}
        self._register_hooks(device)

        monitor = EnergyMonitor(device_type=device, gpu_id=self.gpu_id, enable_rapl=self.args.rapl)
        monitor.start()
        time.sleep(0.05)

        total_start = time.perf_counter()
        opt_step_accum_ms = 0.0
        opt_step_count = 0

        try:
            for _ in range(steps):
                opt_step_ms = self._run_step_with_oom_fallback(inp=inp, device=device)
                opt_step_accum_ms += opt_step_ms
                opt_step_count += 1

        except RuntimeError as ex:
            msg = str(ex)
            if _is_oom_runtime_error(ex):
                self._active_optimizer = None
                raise RuntimeError(
                    "Profiling failed after OOM retries were exhausted. "
                    "Reduce batch size or tune OOM retry parameters "
                    "(--oom_retry_min_batch / --oom_retry_backoff)."
                ) from ex
            hook_inplace_conflict = (
                "BackwardHookFunctionBackward" in msg
                and "view" in msg
                and "inplace" in msg
            )
            if allow_backward_hook_fallback and self._use_backward_hooks and hook_inplace_conflict:
                logger.warning(
                    "Detected autograd incompatibility between backward hooks and in-place ops; "
                    "retrying epoch with backward hooks disabled for model=%s on device=%s.",
                    self.model_name,
                    device,
                )
                self._use_backward_hooks = False
                self.layer_stats = {}
                self._active_optimizer = None
                return self._run_epoch(
                    input_data=input_data,
                    device=device,
                    steps=steps,
                    allow_backward_hook_fallback=False,
                )
            raise

        finally:
            monitor.stop()
            for h in self.hooks:
                h.remove()
            self.hooks = []

        total_duration_sec = time.perf_counter() - total_start
        avg_power = monitor.get_avg_power()
        total_energy_j = (avg_power * total_duration_sec) if avg_power > 0 else None

        self._last_opt_step_ms = opt_step_accum_ms
        self._last_opt_step_count = opt_step_count
        self._active_optimizer = None

        return total_energy_j, total_duration_sec

    def _profile_device_phase(
        self,
        input_data: Any,
        device: str,
        measure: int,
    ) -> Tuple[Optional[float], float, Dict[str, Dict[str, Any]], float]:
        total_energy, run_time_sec = self._run_epoch(input_data, device, measure)
        layer_stats = self.layer_stats.copy()
        self.layer_stats = {}
        measured_peak_tflops = self._measure_peak_flops(device)
        return total_energy, run_time_sec, layer_stats, measured_peak_tflops

    def _merge_layer_stats(
        self,
        cpu_layer_stats: Dict[str, Dict[str, Any]],
        gpu_layer_stats: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        merged_layer_stats: Dict[str, Dict[str, Any]] = {}
        for layer_name, layer_values in cpu_layer_stats.items():
            merged_layer_stats[layer_name] = dict(layer_values)
        for layer_name, layer_values in gpu_layer_stats.items():
            if layer_name not in merged_layer_stats:
                merged_layer_stats[layer_name] = {}
            merged_layer_stats[layer_name].update(layer_values)
        return merged_layer_stats

    def _compute_phase_overheads(
        self,
        gpu_layer_stats: Dict[str, Dict[str, Any]],
        cpu_layer_stats: Dict[str, Dict[str, Any]],
        gpu_run_time_sec: float,
        cpu_run_time_sec: float,
        measure: int,
    ) -> Dict[str, float]:
        g_fwd_layers_ms = sum((gpu_layer_stats[l].get("time_ms_accum", 0) / measure) for l in gpu_layer_stats)
        g_bwd_layers_ms = sum((gpu_layer_stats[l].get("bwd_time_ms_accum", 0) / measure) for l in gpu_layer_stats)
        c_fwd_layers_ms = sum((cpu_layer_stats[l].get("time_ms_accum", 0) / measure) for l in cpu_layer_stats)
        c_bwd_layers_ms = sum((cpu_layer_stats[l].get("bwd_time_ms_accum", 0) / measure) for l in cpu_layer_stats)

        g_total_layers_ms = (g_fwd_layers_ms + g_bwd_layers_ms) or 1.0
        c_total_layers_ms = (c_fwd_layers_ms + c_bwd_layers_ms) or 1.0

        avg_step_time_gpu_ms = (gpu_run_time_sec * 1000.0) / measure
        avg_step_time_cpu_ms = (cpu_run_time_sec * 1000.0) / measure

        framework_overhead_gpu_ms = max(0.0, avg_step_time_gpu_ms - g_total_layers_ms)
        framework_overhead_cpu_ms = max(0.0, avg_step_time_cpu_ms - c_total_layers_ms)

        return {
            "g_total_layers_ms": g_total_layers_ms,
            "c_total_layers_ms": c_total_layers_ms,
            "g_fwd_layers_ms": g_fwd_layers_ms,
            "g_bwd_layers_ms": g_bwd_layers_ms,
            "c_fwd_layers_ms": c_fwd_layers_ms,
            "c_bwd_layers_ms": c_bwd_layers_ms,
            "avg_step_time_gpu_ms": avg_step_time_gpu_ms,
            "avg_step_time_cpu_ms": avg_step_time_cpu_ms,
            "framework_overhead_gpu_ms": framework_overhead_gpu_ms,
            "framework_overhead_cpu_ms": framework_overhead_cpu_ms,
            "framework_overhead_ratio_gpu": (framework_overhead_gpu_ms / avg_step_time_gpu_ms) if avg_step_time_gpu_ms > 0 else 0.0,
            "framework_overhead_ratio_cpu": (framework_overhead_cpu_ms / avg_step_time_cpu_ms) if avg_step_time_cpu_ms > 0 else 0.0,
        }

    def _export_graph_artifacts_safe(
        self,
        input_data: Any,
        cpu_layer_stats: Dict[str, Dict[str, Any]],
        gpu_layer_stats: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        try:
            self.model.to("cpu")
            merged_layer_stats = self._merge_layer_stats(
                cpu_layer_stats=cpu_layer_stats,
                gpu_layer_stats=gpu_layer_stats,
            )
            return export_graph_artifacts(
                model=self.model,
                model_name=self.model_name,
                output_dir=self.args.output_dir,
                input_data=input_data,
                layer_stats=merged_layer_stats,
                include_records=True,
                allow_fallback_graph=getattr(self.args, "allow_fallback_graph", False),
            )
        except Exception as e:
            if not getattr(self.args, "allow_fallback_graph", False):
                raise RuntimeError(
                    "Graph artifact export failed under strict structured-trace policy. "
                    "Re-run with --allow_fallback_graph only for explicit diagnostics."
                ) from e
            logger.warning(f"Graph artifact export failed (non-fatal): {e}")
            return {}

    def _measure_pci_and_overlap(self) -> Dict[str, float]:
        if not self.has_gpu:
            return {}
        logger.info("--> Calibrating PCIe Bandwidth & Overlap Ratio (Sigma)...")

        fallback = {
            "pci_bw_mb_s": 12.0,
            "t_comm_ms_base": 0.0,
            "pci_alpha_ms": 0.05,
            "overlap_ratio_sigma": 0.0,
            "t_comp_ms_base": 0.0,
            "t_overlap_ms": 0.0,
            "overlap_calibration_source": "fallback",
        }

        try:
            dev = f"cuda:{self.gpu_id}"
            size_mb = 256
            numel = int(size_mb * 1024**2 / 4)
            h_tensor = torch.randn(numel).pin_memory()

            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = h_tensor.to(dev, non_blocking=True)
            torch.cuda.synchronize()
            t_comm = (time.perf_counter() - start) * 1000.0

            s_transfer = torch.cuda.Stream()
            a = torch.randn(4096, 4096, device=dev)

            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = torch.mm(a, a)
            torch.cuda.synchronize()
            t_comp = (time.perf_counter() - start) * 1000.0

            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = torch.mm(a, a)
            with torch.cuda.stream(s_transfer):
                _ = h_tensor.to(dev, non_blocking=True)
            torch.cuda.synchronize()
            t_overlap = (time.perf_counter() - start) * 1000.0

            sigma = 1.0 - (max(0, t_overlap - max(t_comm, t_comp)) / (t_comm + 1e-6))
            alpha_est = 0.05
            beta_est = size_mb / (t_comm / 1000.0)

            return {
                "pci_bw_mb_s": beta_est,
                "t_comm_ms_base": t_comm,
                "pci_alpha_ms": alpha_est,
                "overlap_ratio_sigma": max(0.0, min(1.0, sigma)),
                "t_comp_ms_base": t_comp,
                "t_overlap_ms": t_overlap,
                "overlap_calibration_source": "measured",
            }
        except Exception as e:
            logger.warning(
                "PCIe/overlap calibration failed; using neutral fallback transfer parameters. Error: %s",
                e,
            )
            return fallback

    def _measure_pci_bandwidth_detailed(self) -> Dict[str, float]:
        if not self.has_gpu:
            return {}
        logger.info("--> Calibrating Detailed PCIe (H2D vs D2H)...")
        fallback = {
            "alpha_h2d": 0.05,
            "beta_h2d": 12.0,
            "beta_h2d_congested": 12.0,
            "congestion_knee_h2d_mb": 128.0,
            "alpha_d2h": 0.05,
            "beta_d2h": 12.0,
            "beta_d2h_congested": 12.0,
            "congestion_knee_d2h_mb": 128.0,
            "pci_detailed_calibration_source": "fallback",
        }

        try:
            results = {}
            sizes_mb = [8.0, 32.0, 128.0, 256.0]
            dev = f"cuda:{self.gpu_id}"

            for direction in ["h2d", "d2h"]:
                times = []
                for sz in sizes_mb:
                    numel = int(sz * 1024**2 / 4)
                    if direction == "h2d":
                        src = torch.randn(numel).pin_memory()
                        dst_dev = dev
                    else:
                        src = torch.randn(numel, device=dev)
                        dst_dev = "cpu"

                    _ = src.to(dst_dev, non_blocking=True)
                    torch.cuda.synchronize()

                    start = time.perf_counter()
                    _ = src.to(dst_dev, non_blocking=True)
                    torch.cuda.synchronize()
                    times.append((time.perf_counter() - start) * 1000.0)

                if times[1] > times[0]:
                    beta = (sizes_mb[1] - sizes_mb[0]) / (times[1] - times[0])
                    alpha = max(0.0, times[0] - (sizes_mb[0] / beta))
                else:
                    beta = 12.0
                    alpha = 0.05

                if times[-1] > times[-2]:
                    beta_congested = (sizes_mb[-1] - sizes_mb[-2]) / (times[-1] - times[-2])
                    beta_congested = min(beta, beta_congested)
                else:
                    beta_congested = beta

                results[f"alpha_{direction}"] = alpha
                results[f"beta_{direction}"] = beta
                results[f"beta_{direction}_congested"] = beta_congested
                results[f"congestion_knee_{direction}_mb"] = sizes_mb[-2]

            results["pci_detailed_calibration_source"] = "measured"
            return results
        except Exception as e:
            logger.warning(
                "Detailed PCIe calibration failed; using neutral fallback bandwidth parameters. Error: %s",
                e,
            )
            return fallback

    def _measure_peak_flops(self, device: str) -> float:
        logger.info(f"--> Benchmarking Empirical {device.upper()} TFLOPS...")
        N = 8192 if device == "cuda" else 2048
        dev_str = f"cuda:{self.gpu_id}" if device == "cuda" else "cpu"

        try:
            a = torch.randn(N, N, device=dev_str)
            b = torch.randn(N, N, device=dev_str)
            for _ in range(3):
                torch.mm(a, b)
            if device == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()
            ITER = 5
            for _ in range(ITER):
                torch.mm(a, b)
            if device == "cuda":
                torch.cuda.synchronize()
            dur = (time.perf_counter() - start) / ITER

            tflops = (2 * N**3 / 1e12) / dur
            logger.info(f"    Peak {device.upper()}: {tflops:.2f} TFLOPS")
            return tflops
        except Exception as e:
            logger.warning(f"Failed to measure TFLOPS on {device}: {e}")
            return 0.0

    def _save_gpu_partial_results(
        self,
        gpu_layer_stats: Dict[str, Dict[str, Any]],
        gpu_total_energy: Optional[float],
        gpu_run_time_sec: float,
        measured_gpu_peak_tflops: float,
        measure: int,
    ) -> None:
        if not gpu_layer_stats:
            return

        os.makedirs(self.args.output_dir, exist_ok=True)

        g_total_layers_ms = sum(
            ((gpu_layer_stats[l].get("time_ms_accum", 0) + gpu_layer_stats[l].get("bwd_time_ms_accum", 0)) / measure)
            for l in gpu_layer_stats
        ) or 1.0
        avg_step_time_gpu_ms = (gpu_run_time_sec * 1000.0) / measure if measure > 0 else 0.0
        energy_avg_step_gpu = (gpu_total_energy / measure) if gpu_total_energy else 0.0

        opt_name = getattr(self.args, "optimizer", "SGD")
        opt_factor_used = OPTIMIZER_OVERHEAD_MAP.get(opt_name, OPTIMIZER_OVERHEAD_FACTOR)
        rows = []

        for name in sorted(gpu_layer_stats.keys()):
            g_s = gpu_layer_stats.get(name, {})
            t_fwd_gpu = g_s.get("time_ms_accum", 0) / max(1, g_s.get("count", 1))
            t_bwd_gpu = (
                g_s.get("bwd_time_ms_accum", 0) / max(1, g_s.get("bwd_count", 1))
                if g_s.get("bwd_count", 0) > 0
                else t_fwd_gpu * BACKWARD_FACTOR
            )
            gpu_energy_per_ms = (energy_avg_step_gpu / g_total_layers_ms) if g_total_layers_ms > 0 else 0.0
            gpu_layer_energy_j = gpu_energy_per_ms * t_fwd_gpu
            gpu_bwd_energy_j = gpu_energy_per_ms * t_bwd_gpu
            flops = g_s.get("flops", 0.0)

            tflops = 0.0
            eff_ratio = 0.0
            if t_fwd_gpu > 0:
                tflops = (flops / 1e12) / (t_fwd_gpu / 1000.0)
                if measured_gpu_peak_tflops > 0:
                    eff_ratio = tflops / measured_gpu_peak_tflops

            layer_work_tflops = flops / 1e12
            dispatch_ms = g_s.get("dispatch_ms_accum", 0) / max(1, g_s.get("count", 1))
            params_mb = g_s.get("params_mb", 0.0)

            rows.append({
                "layer": name,
                "model": self.model_name,
                "batch_size": getattr(self.args, "batch_size", None),
                "run_id": getattr(self.args, "run_id", "run_001"),
                "seed": getattr(self.args, "seed", 42),
                "type": g_s.get("type", "Unknown"),
                "params_mb": params_mb,
                "grads_mb": params_mb,
                "optimizer_states_mb": params_mb * opt_factor_used,
                "activations_mb": g_s.get("output_bytes", 0) / (1024**2),
                "theoretical_flops": flops,
                "tflops": tflops,
                "efficiency_ratio": eff_ratio,
                "gpu_fwd_time_ms": t_fwd_gpu,
                "gpu_bwd_time_ms": t_bwd_gpu,
                "gpu_fwd_energy_j": gpu_layer_energy_j,
                "gpu_bwd_energy_j": gpu_bwd_energy_j,
                "gpu_mem_peak_mb": g_s.get("mem_mb", 0),
                "layer_j_per_tflop_gpu": (gpu_layer_energy_j / layer_work_tflops)
                if (layer_work_tflops > 0 and gpu_layer_energy_j > 0)
                else 0.0,
                "dispatch_overhead_ratio": dispatch_ms / t_fwd_gpu if t_fwd_gpu > 0 else 0,
                "cpu_fwd_time_ms": None,
                "cpu_bwd_time_ms": None,
                "cpu_fwd_energy_j": None,
                "cpu_bwd_energy_j": None,
                "cpu_mem_mb": None,
                "layer_j_per_tflop_cpu": None,
                "transfer_h2d_ms": None,
                "transfer_d2h_ms": None,
                "remat_penalty_ms": t_fwd_gpu,
                "precision_requested": self.args.precision,
                "cpu_precision_executed": getattr(self.args, "cpu_precision_executed", "unknown"),
                "gpu_precision_executed": getattr(self.args, "gpu_precision_executed", "unknown"),
                "run_executed": True,
                "skip_unsupported_precision": False,
                "skip_reason": "",
                "optimizer": opt_name,
                "opt_step_time_ms": getattr(self, "_last_opt_step_ms", 0.0),
            })

        partial_csv_path = os.path.join(self.args.output_dir, f"{self.model_name}_metrics_gpu_partial.csv")
        write_csv_rows(partial_csv_path, rows)

        partial_meta = get_hardware_metadata()
        partial_meta.update({
            "model": self.model_name,
            "run_id": getattr(self.args, "run_id", "run_001"),
            "seed": getattr(self.args, "seed", 42),
            "phase": "gpu_partial",
            "layers_profiled_count": len(gpu_layer_stats),
            "precision_mode": self.args.precision,
            "cpu_precision_executed": getattr(self.args, "cpu_precision_executed", "unknown"),
            "gpu_precision_executed": getattr(self.args, "gpu_precision_executed", "unknown"),
            "cpu_fp16_supported": getattr(self.args, "cpu_fp16_supported", None),
            "cpu_fp16_isa_avx512": getattr(self.args, "cpu_fp16_isa_avx512", None),
            "cpu_fp16_smoke_test_ok": getattr(self.args, "cpu_fp16_smoke_test_ok", None),
            "cpu_fp16_model_smoke_ok": getattr(self.args, "cpu_fp16_model_smoke_ok", None),
            "cpu_fp16_model_smoke_reason": getattr(self.args, "cpu_fp16_model_smoke_reason", None),
            "cpu_fp16_support_reason": getattr(self.args, "cpu_fp16_support_reason", None),
            "gpu_total_layer_time_ms": g_total_layers_ms,
            "gpu_step_time_ms": avg_step_time_gpu_ms,
            "energy_avg_per_step_gpu_j": (gpu_total_energy / measure) if gpu_total_energy else None,
            "energy_total_gpu_j": gpu_total_energy,
            "measured_peak_tflops_gpu": measured_gpu_peak_tflops,
        })

        partial_json_path = os.path.join(self.args.output_dir, f"{self.model_name}_meta_gpu_partial.json")
        write_json_dict(partial_json_path, partial_meta)

        self._partial_csv_path = partial_csv_path
        self._partial_json_path = partial_json_path

        logger.info(f"Saved early GPU partial artifacts: {partial_csv_path}, {partial_json_path}")

    def _cleanup_partial_artifacts(self) -> None:
        if getattr(self.args, "keep_partial_artifacts", False):
            return
        cleanup_artifacts([self._partial_csv_path, self._partial_json_path])

    def _save_skip_artifacts(self, reason: str, skip_status: str = "skipped_unsupported_precision") -> None:
        os.makedirs(self.args.output_dir, exist_ok=True)
        skip_unsupported_precision = skip_status == "skipped_unsupported_precision"

        row = {
            "layer": "__profiling_skipped__",
            "model": self.model_name,
            "batch_size": getattr(self.args, "batch_size", None),
            "run_id": getattr(self.args, "run_id", "run_001"),
            "seed": getattr(self.args, "seed", 42),
            "type": "NA",
            "params_mb": 0.0,
            "grads_mb": 0.0,
            "optimizer_states_mb": 0.0,
            "activations_mb": 0.0,
            "theoretical_flops": 0.0,
            "tflops": 0.0,
            "efficiency_ratio": 0.0,
            "gpu_fwd_time_ms": 0.0,
            "gpu_bwd_time_ms": 0.0,
            "gpu_fwd_energy_j": 0.0,
            "gpu_bwd_energy_j": 0.0,
            "gpu_mem_peak_mb": 0.0,
            "layer_j_per_tflop_gpu": None,
            "dispatch_overhead_ratio": 0.0,
            "cpu_fwd_time_ms": 0.0,
            "cpu_bwd_time_ms": 0.0,
            "cpu_fwd_energy_j": 0.0,
            "cpu_bwd_energy_j": 0.0,
            "cpu_mem_mb": 0.0,
            "layer_j_per_tflop_cpu": None,
            "transfer_h2d_ms": 0.0,
            "transfer_d2h_ms": 0.0,
            "remat_penalty_ms": 0.0,
            "precision_requested": self.args.precision,
            "cpu_precision_executed": getattr(self.args, "cpu_precision_executed", "unknown"),
            "gpu_precision_executed": getattr(self.args, "gpu_precision_executed", "unknown"),
            "run_executed": False,
            "skip_unsupported_precision": skip_unsupported_precision,
            "skip_reason": reason,
            "optimizer": getattr(self.args, "optimizer", "SGD"),
            "opt_step_time_ms": 0.0,
        }

        csv_path = os.path.join(self.args.output_dir, f"{self.model_name}_metrics.csv")
        write_csv_rows(csv_path, [row])

        meta = get_hardware_metadata()
        meta.update({
            "model": self.model_name,
            "run_id": getattr(self.args, "run_id", "run_001"),
            "seed": getattr(self.args, "seed", 42),
            "precision_mode": self.args.precision,
            "input_source": getattr(self.args, "input_source", "synthetic"),
            "target_source": getattr(self.args, "target_source", None),
            "datasets_root": getattr(self.args, "datasets_root", None),
            "dataset_name": getattr(self.args, "dataset_name", None),
            "dataset_split": getattr(self.args, "dataset_split", None),
            "dataset_path": getattr(self.args, "dataset_path", None),
            "execution_status": skip_status,
            "execution_skip_reason": reason,
            "run_executed": False,
            "skip_unsupported_precision": skip_unsupported_precision,
            "cpu_precision_executed": getattr(self.args, "cpu_precision_executed", "unknown"),
            "gpu_precision_executed": getattr(self.args, "gpu_precision_executed", "unknown"),
            "cpu_instruction_flags": getattr(self.args, "cpu_instruction_flags", []),
            "cpu_isa_probe": getattr(self.args, "cpu_isa_probe", {}),
        })

        json_path = os.path.join(self.args.output_dir, f"{self.model_name}_meta.json")
        write_json_dict(json_path, meta)

        logger.warning(f"Profiling skipped ({skip_status}). Artifacts saved: {csv_path}, {json_path}. Reason: {reason}")

    def _build_edge_transfer_costs(
        self,
        graph_edges: Optional[Any],
        pci_stats: Dict[str, float],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, float], Dict[str, float]]:
        edge_rows: List[Dict[str, Any]] = []
        incoming_h2d_by_layer: Dict[str, float] = {}
        outgoing_d2h_by_layer: Dict[str, float] = {}

        if not graph_edges:
            return edge_rows, incoming_h2d_by_layer, outgoing_d2h_by_layer

        pressure_maps = _build_branch_pressure_maps(graph_edges)
        outgoing_volume_mb = pressure_maps["outgoing_volume_mb"]
        incoming_volume_mb = pressure_maps["incoming_volume_mb"]

        alpha_h2d = float(pci_stats.get("alpha_h2d", 0.05))
        beta_h2d = max(float(pci_stats.get("beta_h2d", 12.0)), 1e-6)
        beta_h2d_congested = max(float(pci_stats.get("beta_h2d_congested", beta_h2d)), 1e-6)
        congestion_knee_h2d_mb = float(pci_stats.get("congestion_knee_h2d_mb", 128.0))
        alpha_d2h = float(pci_stats.get("alpha_d2h", 0.05))
        beta_d2h = max(float(pci_stats.get("beta_d2h", 12.0)), 1e-6)
        beta_d2h_congested = max(float(pci_stats.get("beta_d2h_congested", beta_d2h)), 1e-6)
        congestion_knee_d2h_mb = float(pci_stats.get("congestion_knee_d2h_mb", 128.0))
        sigma_overlap = float(pci_stats.get("overlap_ratio_sigma", 0.0))

        slowdown_margin_h2d = max(0.0, (beta_h2d / beta_h2d_congested) - 1.0)
        slowdown_margin_d2h = max(0.0, (beta_d2h / beta_d2h_congested) - 1.0)

        for idx, e in enumerate(graph_edges):
            tensor_mb = float(e.get("tensor_mb", 0.0) or 0.0)
            producer_name = str(e.get("producer_name", ""))
            consumer_name = str(e.get("consumer_name", ""))

            h2d_ms_piecewise = _piecewise_transfer_ms(
                tensor_mb=tensor_mb,
                alpha_ms=alpha_h2d,
                beta_nominal_mb_per_ms=beta_h2d,
                beta_congested_mb_per_ms=beta_h2d_congested,
                congestion_knee_mb=congestion_knee_h2d_mb,
            )
            d2h_ms_piecewise = _piecewise_transfer_ms(
                tensor_mb=tensor_mb,
                alpha_ms=alpha_d2h,
                beta_nominal_mb_per_ms=beta_d2h,
                beta_congested_mb_per_ms=beta_d2h_congested,
                congestion_knee_mb=congestion_knee_d2h_mb,
            )

            producer_pressure = max(0.0, outgoing_volume_mb.get(producer_name, 0.0) - tensor_mb)
            consumer_pressure = max(0.0, incoming_volume_mb.get(consumer_name, 0.0) - tensor_mb)
            branch_pressure = 0.5 * (
                (producer_pressure / max(congestion_knee_h2d_mb, 1e-6))
                + (consumer_pressure / max(congestion_knee_d2h_mb, 1e-6))
            )

            h2d_ms_raw = h2d_ms_piecewise * (1.0 + (slowdown_margin_h2d * branch_pressure))
            d2h_ms_raw = d2h_ms_piecewise * (1.0 + (slowdown_margin_d2h * branch_pressure))

            # Overlap-aware approximation: sigma in [0,1] attenuates transfer penalty.
            overlap_factor = max(0.0, min(1.0, 1.0 - (0.5 * sigma_overlap)))
            h2d_ms_eff = h2d_ms_raw * overlap_factor
            d2h_ms_eff = d2h_ms_raw * overlap_factor

            incoming_h2d_by_layer[consumer_name] = incoming_h2d_by_layer.get(consumer_name, 0.0) + h2d_ms_eff
            outgoing_d2h_by_layer[producer_name] = outgoing_d2h_by_layer.get(producer_name, 0.0) + d2h_ms_eff

            edge_rows.append({
                "edge_id": f"e{idx}",
                "src_id": e.get("src_id", ""),
                "dst_id": e.get("dst_id", ""),
                "producer_name": producer_name,
                "consumer_name": consumer_name,
                "tensor_mb": tensor_mb,
                "branch_pressure": branch_pressure,
                "transfer_h2d_ms_piecewise": h2d_ms_piecewise,
                "transfer_d2h_ms_piecewise": d2h_ms_piecewise,
                "transfer_h2d_ms_raw": h2d_ms_raw,
                "transfer_d2h_ms_raw": d2h_ms_raw,
                "transfer_h2d_ms": h2d_ms_eff,
                "transfer_d2h_ms": d2h_ms_eff,
                "transfer_sym_ms": 0.5 * (h2d_ms_eff + d2h_ms_eff),
                "alpha_h2d_ms": alpha_h2d,
                "beta_h2d_mb_s": beta_h2d,
                "beta_h2d_congested_mb_s": beta_h2d_congested,
                "congestion_knee_h2d_mb": congestion_knee_h2d_mb,
                "alpha_d2h_ms": alpha_d2h,
                "beta_d2h_mb_s": beta_d2h,
                "beta_d2h_congested_mb_s": beta_d2h_congested,
                "congestion_knee_d2h_mb": congestion_knee_d2h_mb,
                "sigma_overlap": sigma_overlap,
            })

        return edge_rows, incoming_h2d_by_layer, outgoing_d2h_by_layer

    def run_profiling(self, input_data: Any):
        logger.info(f"Starting Profiling Run for: {self.model_name}")

        if getattr(self.args, "skip_cpu", False) and not self.has_gpu:
            reason = "--skip_cpu requested but GPU is unavailable or disabled; no execution target available"
            self._save_skip_artifacts(reason, skip_status="skipped_invalid_configuration")
            return

        if getattr(self.args, "abort_profiling_due_to_isa", False):
            reason = getattr(self.args, "abort_profiling_reason", "unsupported precision ISA")
            self._save_skip_artifacts(reason, skip_status="skipped_unsupported_precision")
            return

        warmup = int(getattr(self.args, "warmup", WARMUP_STEPS))
        measure = int(getattr(self.args, "measure", MEASURE_STEPS))

        self._run_epoch(input_data, "cuda" if self.has_gpu else "cpu", warmup)

        gpu_total_energy, gpu_run_time_sec = 0.0, 0.0
        gpu_layer_stats = {}
        measured_gpu_peak_tflops = 0.0

        if self.has_gpu:
            logger.info("--> Profiling GPU Execution...")
            (
                gpu_total_energy,
                gpu_run_time_sec,
                gpu_layer_stats,
                measured_gpu_peak_tflops,
            ) = self._profile_device_phase(input_data=input_data, device="cuda", measure=measure)
            self._save_gpu_partial_results(
                gpu_layer_stats=gpu_layer_stats,
                gpu_total_energy=gpu_total_energy,
                gpu_run_time_sec=gpu_run_time_sec,
                measured_gpu_peak_tflops=measured_gpu_peak_tflops,
                measure=measure,
            )

        cpu_total_energy, cpu_run_time_sec = None, 0.0
        cpu_layer_stats = {}
        measured_cpu_peak_tflops = 0.0

        if self.args.precision == "fp16" and not getattr(self.args, "skip_cpu", False):
            model_preflight = run_cpu_fp16_model_preflight(self.model, input_data)
            self.args.cpu_fp16_model_smoke_ok = model_preflight["ok"]
            self.args.cpu_fp16_model_smoke_reason = model_preflight["reason"]
            if not model_preflight["ok"]:
                logger.warning(
                    "CPU FP16 model preflight failed. Skipping CPU profiling. "
                    f"Reason: {model_preflight['reason']}"
                )

        if self.args.precision == "fp16" and self.args.cpu_fp16_model_smoke_ok is False:
            self.args.cpu_precision_executed = "fp16_requested_model_preflight_failed"

        skip_cpu_profile = getattr(self.args, "skip_cpu", False) or (
            self.args.precision == "fp16" and getattr(self.args, "cpu_fp16_model_smoke_ok", None) is False
        )

        if skip_cpu_profile:
            if getattr(self.args, "skip_cpu", False):
                logger.warning("Skipping CPU profiling: --skip_cpu requested by user.")
            else:
                logger.warning("Skipping CPU profiling: CPU FP16 model preflight failed and FP32 fallback is disabled.")
        else:
            logger.info("--> Profiling CPU Execution...")
            (
                cpu_total_energy,
                cpu_run_time_sec,
                cpu_layer_stats,
                measured_cpu_peak_tflops,
            ) = self._profile_device_phase(input_data=input_data, device="cpu", measure=measure)

        self.args.execution_status = "completed"
        self.args.abort_profiling_reason = ""

        gpu_peak_mb = 0.0
        if self.has_gpu:
            try:
                with torch.cuda.device(self.gpu_id):
                    torch.cuda.synchronize()
                    gpu_peak_mb = torch.cuda.max_memory_allocated(self.gpu_id) / (1024**2)
            except Exception:
                pass

        overlap_stats = self._measure_pci_and_overlap()
        pci_detailed = self._measure_pci_bandwidth_detailed()
        pci_stats = {**overlap_stats, **pci_detailed}
        transfer_calibration_source = "measured"
        if (
            overlap_stats.get("overlap_calibration_source") != "measured"
            or pci_detailed.get("pci_detailed_calibration_source") != "measured"
        ):
            transfer_calibration_source = "fallback"

        all_layers = sorted(set(gpu_layer_stats.keys()) | set(cpu_layer_stats.keys()))
        if not all_layers:
            logger.warning("No layers profiled on either device!")

        phase_overheads = self._compute_phase_overheads(
            gpu_layer_stats=gpu_layer_stats,
            cpu_layer_stats=cpu_layer_stats,
            gpu_run_time_sec=gpu_run_time_sec,
            cpu_run_time_sec=cpu_run_time_sec,
            measure=measure,
        )
        g_total_layers_ms = phase_overheads["g_total_layers_ms"]
        c_total_layers_ms = phase_overheads["c_total_layers_ms"]
        avg_step_time_gpu_ms = phase_overheads["avg_step_time_gpu_ms"]
        avg_step_time_cpu_ms = phase_overheads["avg_step_time_cpu_ms"]
        framework_overhead_gpu_ms = phase_overheads["framework_overhead_gpu_ms"]
        framework_overhead_cpu_ms = phase_overheads["framework_overhead_cpu_ms"]
        framework_overhead_ratio_gpu = phase_overheads["framework_overhead_ratio_gpu"]
        framework_overhead_ratio_cpu = phase_overheads["framework_overhead_ratio_cpu"]

        rows = []
        framework_overhead_vector = []
        energy_dist_vector = []

        opt_name = getattr(self.args, "optimizer", "SGD")
        opt_factor_used = OPTIMIZER_OVERHEAD_MAP.get(opt_name, OPTIMIZER_OVERHEAD_FACTOR)

        total_model_flops = 0.0

        graph_info = self._export_graph_artifacts_safe(
            input_data=input_data,
            cpu_layer_stats=cpu_layer_stats,
            gpu_layer_stats=gpu_layer_stats,
        )

        edge_transfer_rows, incoming_h2d_by_layer, outgoing_d2h_by_layer = self._build_edge_transfer_costs(
            graph_edges=graph_info.get("edges") if graph_info else None,
            pci_stats=pci_stats,
        )

        transfer_edges_path = None
        if edge_transfer_rows:
            transfer_edges_path = os.path.join(self.args.output_dir, f"{self.model_name}_transfer_edges.csv")
            write_csv_rows(transfer_edges_path, edge_transfer_rows)

        for name in all_layers:
            c_s = cpu_layer_stats.get(name, {})
            g_s = gpu_layer_stats.get(name, {})

            t_fwd_gpu = g_s.get("time_ms_accum", 0) / max(1, g_s.get("count", 1))
            t_fwd_cpu = c_s.get("time_ms_accum", 0) / max(1, c_s.get("count", 1))

            disp_ms = g_s.get("dispatch_ms_accum", 0) / max(1, g_s.get("count", 1))
            framework_overhead_vector.append({"layer": name, "dispatch_overhead_ms": disp_ms})

            t_bwd_gpu = (
                g_s.get("bwd_time_ms_accum", 0) / max(1, g_s.get("bwd_count", 1))
                if g_s.get("bwd_count", 0) > 0
                else t_fwd_gpu * BACKWARD_FACTOR
            )
            t_bwd_cpu = (
                c_s.get("bwd_time_ms_accum", 0) / max(1, c_s.get("bwd_count", 1))
                if c_s.get("bwd_count", 0) > 0
                else t_fwd_cpu * BACKWARD_FACTOR
            )

            gpu_share = ((t_fwd_gpu + t_bwd_gpu) / g_total_layers_ms) if g_total_layers_ms > 0 else 0
            cpu_share = ((t_fwd_cpu + t_bwd_cpu) / c_total_layers_ms) if c_total_layers_ms > 0 else 0
            energy_dist_vector.append({"layer": name, "gpu_share": gpu_share, "cpu_share": cpu_share})

            energy_avg_step_gpu = (gpu_total_energy / measure) if gpu_total_energy else 0.0
            gpu_energy_per_ms = (energy_avg_step_gpu / g_total_layers_ms) if g_total_layers_ms > 0 else 0.0
            gpu_layer_energy_j = gpu_energy_per_ms * t_fwd_gpu
            gpu_bwd_energy_j = gpu_energy_per_ms * t_bwd_gpu
            energy_avg_step_cpu = (cpu_total_energy / measure) if cpu_total_energy is not None else None
            cpu_energy_per_ms = (energy_avg_step_cpu / c_total_layers_ms) if (energy_avg_step_cpu is not None and c_total_layers_ms > 0) else 0.0
            cpu_layer_energy_j = cpu_energy_per_ms * t_fwd_cpu
            cpu_bwd_energy_j = cpu_energy_per_ms * t_bwd_cpu

            act_bytes = c_s.get("output_bytes", g_s.get("output_bytes", 0))
            act_mb = act_bytes / (1024**2)
            params_mb = g_s.get("params_mb", c_s.get("params_mb", 0.0))

            flops = g_s.get("flops", 0.0)
            total_model_flops += flops

            tflops = 0.0
            eff_ratio = 0.0
            if t_fwd_gpu > 0:
                tflops = (flops / 1e12) / (t_fwd_gpu / 1000.0)
                if measured_gpu_peak_tflops > 0:
                    eff_ratio = tflops / measured_gpu_peak_tflops

            layer_j_per_tflop_gpu = 0.0
            layer_work_tflops = flops / 1e12
            if layer_work_tflops > 0 and gpu_layer_energy_j > 0:
                layer_j_per_tflop_gpu = gpu_layer_energy_j / layer_work_tflops

            alpha_h2d = pci_stats.get("alpha_h2d", 0.05)
            beta_h2d = max(pci_stats.get("beta_h2d", 12.0), 1e-6)
            alpha_d2h = pci_stats.get("alpha_d2h", 0.05)
            beta_d2h = max(pci_stats.get("beta_d2h", 12.0), 1e-6)

            transfer_h2d_ms_legacy = alpha_h2d + (params_mb / beta_h2d)
            transfer_d2h_ms_legacy = alpha_d2h + (act_mb / beta_d2h)

            transfer_h2d_ms = incoming_h2d_by_layer.get(name, transfer_h2d_ms_legacy)
            transfer_d2h_ms = outgoing_d2h_by_layer.get(name, transfer_d2h_ms_legacy)

            rows.append({
                "layer": name,
                "model": self.model_name,
                "batch_size": getattr(self.args, "batch_size", None),
                "run_id": getattr(self.args, "run_id", "run_001"),
                "seed": getattr(self.args, "seed", 42),
                "type": g_s.get("type") or c_s.get("type", "Unknown"),
                "params_mb": params_mb,
                "grads_mb": params_mb,
                "optimizer_states_mb": params_mb * opt_factor_used,
                "activations_mb": act_mb,
                "theoretical_flops": flops,
                "tflops": tflops,
                "efficiency_ratio": eff_ratio,
                "gpu_fwd_time_ms": t_fwd_gpu,
                "gpu_bwd_time_ms": t_bwd_gpu,
                "gpu_fwd_energy_j": gpu_layer_energy_j,
                "gpu_bwd_energy_j": gpu_bwd_energy_j,
                "gpu_mem_peak_mb": g_s.get("mem_mb", 0),
                "layer_j_per_tflop_gpu": layer_j_per_tflop_gpu,
                "dispatch_overhead_ratio": disp_ms / t_fwd_gpu if t_fwd_gpu > 0 else 0,
                "cpu_fwd_time_ms": t_fwd_cpu,
                "cpu_bwd_time_ms": t_bwd_cpu,
                "cpu_fwd_energy_j": cpu_layer_energy_j,
                "cpu_bwd_energy_j": cpu_bwd_energy_j,
                "cpu_mem_mb": act_mb,
                "layer_j_per_tflop_cpu": (cpu_layer_energy_j / layer_work_tflops)
                if (layer_work_tflops > 0 and cpu_layer_energy_j > 0)
                else None,
                "transfer_h2d_ms": transfer_h2d_ms,
                "transfer_d2h_ms": transfer_d2h_ms,
                "transfer_h2d_ms_legacy": transfer_h2d_ms_legacy,
                "transfer_d2h_ms_legacy": transfer_d2h_ms_legacy,
                "transfer_edge_aware_total_ms": transfer_h2d_ms + transfer_d2h_ms,
                "remat_penalty_ms": t_fwd_gpu,
                "precision_requested": self.args.precision,
                "cpu_precision_executed": self.args.cpu_precision_executed,
                "gpu_precision_executed": self.args.gpu_precision_executed,
                "run_executed": True,
                "skip_unsupported_precision": False,
                "skip_reason": "",
                "optimizer": opt_name,
                "opt_step_time_ms": getattr(self, "_last_opt_step_ms", 0.0),
            })

        csv_path = os.path.join(self.args.output_dir, f"{self.model_name}_metrics.csv")
        write_csv_rows(csv_path, rows)

        meta = get_hardware_metadata()
        meta.update({
            "model": self.model_name,
            "run_id": getattr(self.args, "run_id", "run_001"),
            "seed": getattr(self.args, "seed", 42),
            "input_source": getattr(self.args, "input_source", "synthetic"),
            "target_source": getattr(self.args, "target_source", None),
            "datasets_root": getattr(self.args, "datasets_root", None),
            "dataset_name": getattr(self.args, "dataset_name", None),
            "dataset_split": getattr(self.args, "dataset_split", None),
            "dataset_path": getattr(self.args, "dataset_path", None),
            "layers_profiled_count": len(all_layers),
            "graph_nodes_count": graph_info.get("nodes_count"),
            "graph_edges_count": graph_info.get("edges_count"),
            "graph_trace_source": graph_info.get("trace_source"),
            "graph_nodes_path": graph_info.get("nodes_path"),
            "graph_edges_path": graph_info.get("edges_path"),
            "transfer_edges_count": len(edge_transfer_rows),
            "transfer_edges_path": transfer_edges_path,
            "precision_mode": self.args.precision,
            "gpu_total_layer_time_ms": g_total_layers_ms,
            "cpu_total_layer_time_ms": c_total_layers_ms,
            "gpu_step_time_ms": avg_step_time_gpu_ms,
            "cpu_step_time_ms": avg_step_time_cpu_ms,
            "framework_overhead_gpu_ms": framework_overhead_gpu_ms,
            "framework_overhead_cpu_ms": framework_overhead_cpu_ms,
            "framework_overhead_ratio_gpu": framework_overhead_ratio_gpu,
            "framework_overhead_ratio_cpu": framework_overhead_ratio_cpu,
            "framework_overhead_vector": framework_overhead_vector,
            "energy_avg_per_step_gpu_j": (gpu_total_energy / measure) if gpu_total_energy else None,
            "energy_avg_per_step_cpu_j": (cpu_total_energy / measure) if cpu_total_energy else None,
            "energy_total_gpu_j": gpu_total_energy,
            "energy_total_cpu_j": cpu_total_energy,
            "energy_distribution_vector": energy_dist_vector,
            "gpu_mem_peak_mb_global": gpu_peak_mb,
            "gpu_mem_reserved_mb_global": 0,
            "cpu_uss_mb_global": 0,
            "cpu_pss_mb_global": 0,
            "params_mb_total": sum(r["params_mb"] for r in rows),
            "grads_mb_total": sum(r["grads_mb"] for r in rows),
            "activations_mb_total": sum(r["activations_mb"] for r in rows),
            "optimizer_state_mb_factor_fallback": OPTIMIZER_OVERHEAD_FACTOR,
            "optimizer_state_mb_factor_used": opt_factor_used,
            "transfer_alpha_h2d": pci_stats.get("alpha_h2d", 0),
            "transfer_beta_h2d": pci_stats.get("beta_h2d", 12.0),
            "transfer_alpha_d2h": pci_stats.get("alpha_d2h", 0),
            "transfer_beta_d2h": pci_stats.get("beta_d2h", 12.0),
            "transfer_beta_h2d_congested": pci_stats.get("beta_h2d_congested", pci_stats.get("beta_h2d", 12.0)),
            "transfer_beta_d2h_congested": pci_stats.get("beta_d2h_congested", pci_stats.get("beta_d2h", 12.0)),
            "transfer_congestion_knee_h2d_mb": pci_stats.get("congestion_knee_h2d_mb", 128.0),
            "transfer_congestion_knee_d2h_mb": pci_stats.get("congestion_knee_d2h_mb", 128.0),
            "transfer_model_version": "alpha_beta_piecewise_branch_pressure_v1",
            "pcie_stats_raw": pci_stats,
            "transfer_calibration_source": transfer_calibration_source,
            "transfer_calibration_fallback": transfer_calibration_source != "measured",
            "measured_peak_tflops_gpu": measured_gpu_peak_tflops,
            "measured_peak_tflops_cpu": measured_cpu_peak_tflops,
            "efficiency_ratio_avg": 0.0,
            "efficiency_ratio_vector": [],
            "avg_tflops_per_layer": 0.0,
            "weighted_avg_tflops_per_layer": 0.0,
            "energy_efficiency_j_per_tflop_gpu": 0.0,
            "energy_efficiency_j_per_tflop_cpu": 0.0,
            "cpu_precision_executed": getattr(self.args, "cpu_precision_executed", "unknown"),
            "gpu_precision_executed": getattr(self.args, "gpu_precision_executed", "unknown"),
            "optimizer_step_time_total_ms": getattr(self, "_last_opt_step_ms", 0.0),
            "optimizer_step_time_avg_ms": (getattr(self, "_last_opt_step_ms", 0.0) / max(1, getattr(self, "_last_opt_step_count", 1))),
            "optimizer_used": opt_name,
            "optimizer_lr": getattr(self.args, "lr", 0.01),
            "optimizer_momentum": getattr(self.args, "momentum", None),
            "cpu_fp16_supported": getattr(self.args, "cpu_fp16_supported", None),
            "cpu_fp16_isa_avx512": getattr(self.args, "cpu_fp16_isa_avx512", None),
            "cpu_fp16_smoke_test_ok": getattr(self.args, "cpu_fp16_smoke_test_ok", None),
            "cpu_fp16_model_smoke_ok": getattr(self.args, "cpu_fp16_model_smoke_ok", None),
            "cpu_fp16_model_smoke_reason": getattr(self.args, "cpu_fp16_model_smoke_reason", None),
            "cpu_fp16_support_reason": getattr(self.args, "cpu_fp16_support_reason", None),
            "cpu_instruction_flags": getattr(self.args, "cpu_instruction_flags", []),
            "cpu_isa_probe": getattr(self.args, "cpu_isa_probe", {}),
            "execution_status": getattr(self.args, "execution_status", "completed") or "completed",
            "execution_skip_reason": getattr(self.args, "abort_profiling_reason", ""),
            "oom_retry_enabled": bool(getattr(self.args, "oom_retry_enabled", True)),
            "oom_retry_min_batch": int(getattr(self.args, "oom_retry_min_batch", 1)),
            "oom_retry_backoff": int(getattr(self.args, "oom_retry_backoff", 2)),
            "oom_retry_triggered": self._oom_retry_events > 0,
            "oom_retry_events": int(self._oom_retry_events),
            "oom_retry_last_micro_batch": self._oom_retry_last_micro_batch,
            "run_executed": True,
            "skip_unsupported_precision": False,
            "total_model_flops": total_model_flops,
            "total_model_flops_per_step": total_model_flops / measure,
        })

        json_path = os.path.join(self.args.output_dir, f"{self.model_name}_meta.json")
        write_json_dict(json_path, meta)

        self._cleanup_partial_artifacts()

        logger.info(f"Profiling Complete. Data saved to {self.args.output_dir}")
