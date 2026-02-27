import logging
import os
import time
from typing import Any, Dict, Iterator, Optional, Tuple

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
from core.io_artifacts import cleanup_artifacts, write_csv_rows, write_json_dict
from core.metrics import estimate_flops, get_tensor_size_recursive
from core.precision_policy import run_cpu_fp16_model_preflight
from core.system import get_hardware_metadata

logger = logging.getLogger(__name__)


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
        self.has_gpu = torch.cuda.is_available() and not args.no_gpu
        self.gpu_id = args.gpu_id if self.has_gpu else 0
        if self.has_gpu:
            self.model.to(f"cuda:{self.gpu_id}")
        else:
            self.model.to("cpu")

    def _get_leaf_modules(self) -> Iterator[Tuple[str, nn.Module]]:
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:
                yield name, module

    def _compute_loss(self, out: Any) -> torch.Tensor:
        if hasattr(out, "loss") and out.loss is not None:
            return out.loss
        if hasattr(out, "logits"):
            return out.logits.sum()
        if isinstance(out, torch.Tensor):
            return out.sum()
        if isinstance(out, (tuple, list)) and isinstance(out[0], torch.Tensor):
            return out[0].sum()
        return torch.tensor(0.0, requires_grad=True)

    def _register_hooks(self, device_type: str):
        for h in self.hooks:
            h.remove()
        self.hooks = []
        self._tstarts = {}

        def pre_hook(name):
            def hook(module, inp):
                self._tstarts[name] = {"cpu": time.perf_counter()}
                if device_type == "cuda":
                    start = torch.cuda.Event(enable_timing=True)
                    start.record()
                    self._tstarts[name]["gpu"] = start

            return hook

        def post_hook(name):
            def hook(module, inp, out):
                cpu_end = time.perf_counter()

                if name not in self.layer_stats:
                    self.layer_stats[name] = {
                        "type": module.__class__.__name__,
                        "time_ms_accum": 0.0,
                        "dispatch_ms_accum": 0.0,
                        "mem_mb": 0.0,
                        "count": 0,
                        "output_bytes": 0,
                        "params_mb": 0.0,
                        "flops": 0.0,
                    }
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
                    kernel_ms = self._tstarts[name]["gpu"].elapsed_time(end)
                    s["mem_mb"] = max(s["mem_mb"], torch.cuda.memory_allocated(self.gpu_id) / (1024**2))
                else:
                    kernel_ms = (cpu_end - self._tstarts[name]["cpu"]) * 1000.0

                wall_ms = (cpu_end - self._tstarts[name]["cpu"]) * 1000.0
                dispatch_ms = max(0.0, wall_ms - kernel_ms)

                s["time_ms_accum"] += kernel_ms
                s["dispatch_ms_accum"] += dispatch_ms
                s["count"] += 1

            return hook

        for name, module in self._get_leaf_modules():
            self.hooks.append(module.register_forward_pre_hook(pre_hook(name)))
            self.hooks.append(module.register_forward_hook(post_hook(name)))

    def _run_epoch(self, input_data: Any, device: str, steps: int) -> Tuple[Optional[float], float]:
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

        if opt_name == "SGD":
            opt = torch.optim.SGD(params, lr=lr)
        elif opt_name == "SGD_momentum":
            opt = torch.optim.SGD(params, lr=lr, momentum=momentum)
        elif opt_name == "Adam":
            opt = torch.optim.Adam(params, lr=lr)
        elif opt_name == "AdamW":
            opt = torch.optim.AdamW(params, lr=lr)
        elif opt_name == "RMSprop":
            opt = torch.optim.RMSprop(params, lr=lr, momentum=momentum)
        elif opt_name == "Adagrad":
            opt = torch.optim.Adagrad(params, lr=lr)
        elif opt_name == "Adadelta":
            opt = torch.optim.Adadelta(params, lr=lr)
        else:
            opt = torch.optim.SGD(params, lr=lr)

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
                opt.zero_grad()
                if isinstance(inp, dict):
                    out = self.model(**inp)
                else:
                    out = self.model(inp)

                loss = self._compute_loss(out)
                loss.backward()

                t0_opt = time.perf_counter()
                opt.step()
                if device == "cuda":
                    torch.cuda.synchronize()
                opt_step_ms = (time.perf_counter() - t0_opt) * 1000.0
                opt_step_accum_ms += opt_step_ms
                opt_step_count += 1

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

        return total_energy_j, total_duration_sec

    def _measure_pci_and_overlap(self) -> Dict[str, float]:
        if not self.has_gpu:
            return {}
        logger.info("--> Calibrating PCIe Bandwidth & Overlap Ratio (Sigma)...")

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
        }

    def _measure_pci_bandwidth_detailed(self) -> Dict[str, float]:
        if not self.has_gpu:
            return {}
        logger.info("--> Calibrating Detailed PCIe (H2D vs D2H)...")
        results = {}
        sizes_mb = [10.0, 100.0]
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
                beta = 10.0
                alpha = 0.05

            results[f"alpha_{direction}"] = alpha
            results[f"beta_{direction}"] = beta

        return results

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

        g_total_layers_ms = sum((gpu_layer_stats[l].get("time_ms_accum", 0) / measure) for l in gpu_layer_stats) or 1.0
        avg_step_time_gpu_ms = (gpu_run_time_sec * 1000.0) / measure if measure > 0 else 0.0
        energy_avg_step_gpu = (gpu_total_energy / measure) if gpu_total_energy else 0.0

        opt_name = getattr(self.args, "optimizer", "SGD")
        opt_factor_used = OPTIMIZER_OVERHEAD_MAP.get(opt_name, OPTIMIZER_OVERHEAD_FACTOR)
        rows = []

        for name in sorted(gpu_layer_stats.keys()):
            g_s = gpu_layer_stats.get(name, {})
            t_fwd_gpu = g_s.get("time_ms_accum", 0) / max(1, g_s.get("count", 1))
            gpu_share = (t_fwd_gpu / g_total_layers_ms) if g_total_layers_ms > 0 else 0.0
            gpu_layer_energy_j = energy_avg_step_gpu * gpu_share
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
                "type": g_s.get("type", "Unknown"),
                "params_mb": params_mb,
                "grads_mb": params_mb,
                "optimizer_states_mb": params_mb * opt_factor_used,
                "activations_mb": g_s.get("output_bytes", 0) / (1024**2),
                "theoretical_flops": flops,
                "tflops": tflops,
                "efficiency_ratio": eff_ratio,
                "gpu_fwd_time_ms": t_fwd_gpu,
                "gpu_bwd_time_ms": t_fwd_gpu * BACKWARD_FACTOR,
                "gpu_fwd_energy_j": gpu_layer_energy_j,
                "gpu_bwd_energy_j": gpu_layer_energy_j * BACKWARD_FACTOR,
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

    def _save_skip_artifacts(self, reason: str) -> None:
        os.makedirs(self.args.output_dir, exist_ok=True)

        row = {
            "layer": "__profiling_skipped__",
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
            "skip_unsupported_precision": True,
            "skip_reason": reason,
            "optimizer": getattr(self.args, "optimizer", "SGD"),
            "opt_step_time_ms": 0.0,
        }

        csv_path = os.path.join(self.args.output_dir, f"{self.model_name}_metrics.csv")
        write_csv_rows(csv_path, [row])

        meta = get_hardware_metadata()
        meta.update({
            "model": self.model_name,
            "precision_mode": self.args.precision,
            "execution_status": "skipped_unsupported_precision",
            "execution_skip_reason": reason,
            "run_executed": False,
            "skip_unsupported_precision": True,
            "cpu_precision_executed": getattr(self.args, "cpu_precision_executed", "unknown"),
            "gpu_precision_executed": getattr(self.args, "gpu_precision_executed", "unknown"),
            "cpu_instruction_flags": getattr(self.args, "cpu_instruction_flags", []),
            "cpu_isa_probe": getattr(self.args, "cpu_isa_probe", {}),
        })

        json_path = os.path.join(self.args.output_dir, f"{self.model_name}_meta.json")
        write_json_dict(json_path, meta)

        logger.warning(
            "Profiling skipped due to unsupported precision ISA. "
            f"Artifacts saved: {csv_path}, {json_path}. Reason: {reason}"
        )

    def run_profiling(self, input_data: Any):
        logger.info(f"Starting Profiling Run for: {self.model_name}")

        if getattr(self.args, "abort_profiling_due_to_isa", False):
            reason = getattr(self.args, "abort_profiling_reason", "unsupported precision ISA")
            self._save_skip_artifacts(reason)
            return

        warmup = int(getattr(self.args, "warmup", WARMUP_STEPS))
        measure = int(getattr(self.args, "measure", MEASURE_STEPS))

        self._run_epoch(input_data, "cuda" if self.has_gpu else "cpu", warmup)

        gpu_total_energy, gpu_run_time_sec = 0.0, 0.0
        gpu_layer_stats = {}
        measured_gpu_peak_tflops = 0.0

        if self.has_gpu:
            logger.info("--> Profiling GPU Execution...")
            gpu_total_energy, gpu_run_time_sec = self._run_epoch(input_data, "cuda", measure)
            gpu_layer_stats = self.layer_stats.copy()
            self.layer_stats = {}
            measured_gpu_peak_tflops = self._measure_peak_flops("cuda")
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
            logger.warning("Skipping CPU profiling: CPU FP16 model preflight failed and FP32 fallback is disabled.")
        else:
            logger.info("--> Profiling CPU Execution...")
            cpu_total_energy, cpu_run_time_sec = self._run_epoch(input_data, "cpu", measure)
            cpu_layer_stats = self.layer_stats.copy()
            self.layer_stats = {}
            measured_cpu_peak_tflops = self._measure_peak_flops("cpu")

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

        all_layers = sorted(set(gpu_layer_stats.keys()) | set(cpu_layer_stats.keys()))
        if not all_layers:
            logger.warning("No layers profiled on either device!")

        g_total_layers_ms = sum((gpu_layer_stats[l].get("time_ms_accum", 0) / measure) for l in gpu_layer_stats) or 1.0
        c_total_layers_ms = sum((cpu_layer_stats[l].get("time_ms_accum", 0) / measure) for l in cpu_layer_stats) or 1.0

        avg_step_time_gpu_ms = (gpu_run_time_sec * 1000.0) / measure
        avg_step_time_cpu_ms = (cpu_run_time_sec * 1000.0) / measure

        framework_overhead_gpu_ms = max(0.0, avg_step_time_gpu_ms - g_total_layers_ms)
        framework_overhead_cpu_ms = max(0.0, avg_step_time_cpu_ms - c_total_layers_ms)
        framework_overhead_ratio_gpu = framework_overhead_gpu_ms / avg_step_time_gpu_ms if avg_step_time_gpu_ms > 0 else 0.0
        framework_overhead_ratio_cpu = framework_overhead_cpu_ms / avg_step_time_cpu_ms if avg_step_time_cpu_ms > 0 else 0.0

        rows = []
        framework_overhead_vector = []
        energy_dist_vector = []

        opt_name = getattr(self.args, "optimizer", "SGD")
        opt_factor_used = OPTIMIZER_OVERHEAD_MAP.get(opt_name, OPTIMIZER_OVERHEAD_FACTOR)

        total_model_flops = 0.0

        for name in all_layers:
            c_s = cpu_layer_stats.get(name, {})
            g_s = gpu_layer_stats.get(name, {})

            t_fwd_gpu = g_s.get("time_ms_accum", 0) / max(1, g_s.get("count", 1))
            t_fwd_cpu = c_s.get("time_ms_accum", 0) / max(1, c_s.get("count", 1))

            disp_ms = g_s.get("dispatch_ms_accum", 0) / max(1, g_s.get("count", 1))
            framework_overhead_vector.append({"layer": name, "dispatch_overhead_ms": disp_ms})

            gpu_share = (t_fwd_gpu / g_total_layers_ms) if g_total_layers_ms > 0 else 0
            cpu_share = (t_fwd_cpu / c_total_layers_ms) if c_total_layers_ms > 0 else 0
            energy_dist_vector.append({"layer": name, "gpu_share": gpu_share, "cpu_share": cpu_share})

            energy_avg_step_gpu = (gpu_total_energy / measure) if gpu_total_energy else 0.0
            gpu_layer_energy_j = energy_avg_step_gpu * gpu_share
            energy_avg_step_cpu = (cpu_total_energy / measure) if cpu_total_energy is not None else None
            cpu_layer_energy_j = (energy_avg_step_cpu * cpu_share) if energy_avg_step_cpu is not None else 0.0

            act_mb = c_s.get("output_bytes", 0) / (1024**2)
            params_mb = g_s.get("params_mb", 0.0)

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
            beta_h2d = pci_stats.get("beta_h2d", 12.0)
            alpha_d2h = pci_stats.get("alpha_d2h", 0.05)
            beta_d2h = pci_stats.get("beta_d2h", 12.0)

            rows.append({
                "layer": name,
                "type": g_s.get("type") or c_s.get("type", "Unknown"),
                "params_mb": params_mb,
                "grads_mb": params_mb,
                "optimizer_states_mb": params_mb * opt_factor_used,
                "activations_mb": act_mb,
                "theoretical_flops": flops,
                "tflops": tflops,
                "efficiency_ratio": eff_ratio,
                "gpu_fwd_time_ms": t_fwd_gpu,
                "gpu_bwd_time_ms": t_fwd_gpu * BACKWARD_FACTOR,
                "gpu_fwd_energy_j": gpu_layer_energy_j,
                "gpu_bwd_energy_j": gpu_layer_energy_j * BACKWARD_FACTOR,
                "gpu_mem_peak_mb": g_s.get("mem_mb", 0),
                "layer_j_per_tflop_gpu": layer_j_per_tflop_gpu,
                "dispatch_overhead_ratio": disp_ms / t_fwd_gpu if t_fwd_gpu > 0 else 0,
                "cpu_fwd_time_ms": t_fwd_cpu,
                "cpu_bwd_time_ms": t_fwd_cpu * BACKWARD_FACTOR,
                "cpu_fwd_energy_j": cpu_layer_energy_j,
                "cpu_bwd_energy_j": cpu_layer_energy_j * BACKWARD_FACTOR,
                "cpu_mem_mb": act_mb,
                "layer_j_per_tflop_cpu": (cpu_layer_energy_j / layer_work_tflops)
                if (layer_work_tflops > 0 and cpu_layer_energy_j > 0)
                else None,
                "transfer_h2d_ms": alpha_h2d + (params_mb / beta_h2d),
                "transfer_d2h_ms": alpha_d2h + (act_mb / beta_d2h),
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
            "layers_profiled_count": len(all_layers),
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
            "pcie_stats_raw": pci_stats,
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
            "execution_status": getattr(self.args, "execution_status", "completed"),
            "execution_skip_reason": getattr(self.args, "abort_profiling_reason", ""),
            "run_executed": True,
            "skip_unsupported_precision": False,
            "total_model_flops": total_model_flops,
            "total_model_flops_per_step": total_model_flops / measure,
        })

        json_path = os.path.join(self.args.output_dir, f"{self.model_name}_meta.json")
        write_json_dict(json_path, meta)

        self._cleanup_partial_artifacts()

        logger.info(f"Profiling Complete. Data saved to {self.args.output_dir}")
