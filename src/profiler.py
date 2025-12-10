"""
Advanced Hybrid Profiler for Deep Learning Training (PhD Thesis).

This tool characterizes neural network architectures to generate cost metrics
required for the Integer Linear Programming (ILP) optimization model defined
in Chapter 3 of the Thesis.

METHODOLOGY OVERVIEW:
---------------------
1. Granularity: 
   Profiling is performed at the "Leaf Module" level (atomic layers like Conv2d, Linear).
   Container modules (Sequential, Bottleneck) are ignored to avoid double-counting.

2. Timing Strategy (Sum of Latencies vs. Step Time):
   - We measure T_start and T_end for *each layer* using:
     a) CUDA Events (for GPU): Captures pure kernel execution time, excluding Python overhead.
     b) perf_counter (for CPU): Captures wall-clock execution time.
   - 'gpu_total_layer_time_ms' in metadata is the SUM of these individual layer latencies.
   - NOTE: This sum differs from the global End-to-End Step Time. The difference represents 
     "Framework Overhead" (Python dispatch, kernel launch latency).

3. Energy Attribution:
   - Global energy is measured via hardware sensors (NVML/RAPL) over the entire measurement window.
   - This total energy is distributed to individual layers proportional to their 
     contribution to the total *computational time*.
     Formula: E_layer = E_total_measured * (T_layer / Sum(T_layers))
   - Resulting energy distribution vectors are normalized to sum exactly to 1.0 per device.

4. Empirical Efficiency (TFLOPS):
   - Instead of using theoretical datasheet values, we run a micro-benchmark (GEMM) 
     at runtime to measure the effective Peak TFLOPS.
   - NOTE (CPU): On CPU, the GEMM benchmark is provided as a reference only. CPUs are often 
     memory-bandwidth bound rather than compute-bound for deep learning workloads.
   - Weighted Average TFLOPS: We report a weighted average (Total FLOPs / Total Time) 
     to provide a representative metric of sustained performance.

5. Memory Constraints:
   - gpu_mem_peak_mb: This metric is a snapshot of the global CUDA allocator state at 
     the end of each layer. It serves as a **conservative global proxy**, not a strictly 
     isolated per-layer cost. It inherently captures fragmentation and buffered memory 
     from previous operations, ensuring safety in ILP constraints.

6. Framework Overhead Vector:
   - For GPU execution, 'Dispatch Overhead' is calculated as:
     (CPU Wall Time - GPU Kernel Time).
   - This explicitly quantifies the non-overlapping time the CPU spends preparing 
     tasks (Python/PyTorch overhead) versus the device executing them.

LIMITATIONS (Thesis Section 3.2):
    - CPU Energy: Returns 'NaN' (None) if Intel RAPL interface (/sys/class/powercap) is unavailable.
      Consequently, 'layer_j_per_tflop_cpu' and 'energy_efficiency_cpu' will also be None.
    - Backward Estimation: We rely on the heuristic T_bwd = 2.0 * T_fwd, standard in 
      hybrid offloading literature (e.g., vDNN, Checkmate).
"""

import os
import argparse
import time
import threading
import logging
import psutil
import platform
import random
import json
import subprocess
import re
from typing import Dict, Tuple, Iterator, Any, Optional, List, Union

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import pynvml

# --- Conditional Imports for Hardware Specifics ---
try:
    import pyRAPL
    PYRAPL_AVAILABLE = True
except ImportError:
    PYRAPL_AVAILABLE = False

from torchvision.models import (
    resnet50, resnet152,
    ResNet50_Weights, ResNet152_Weights,
    vit_b_16, ViT_B_16_Weights
)
from transformers import BertModel, GPT2Model

# ========================================================================
# SELECT DTYPE
# ========================================================================
def _select_torch_dtype(precision: str) -> torch.dtype:
    if precision == "fp16":
        return torch.float16
    elif precision == "bf16":
        return torch.bfloat16
    else:
        return torch.float32
    
# ========================================================================
# PARSE CPU INFO
# ========================================================================
def _parse_proc_cpuinfo_model_name() -> Optional[str]:
    """Extrae 'model name' de /proc/cpuinfo de forma robusta (Linux).
    Evita confundir 'model' (numérico) con 'model name' (cadena).
    """
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                # Buscar exactamente "model name" (ignorar mayúsc/minúsc)
                m = re.match(r'^\s*model name\s*:\s*(.+)$', line, flags=re.IGNORECASE)
                if m:
                    val = m.group(1).strip()
                    if val:
                        return val
        return None
    except Exception:
        return None

def _parse_lscpu_model_name() -> Optional[str]:
    """Ejecuta lscpu y parsea 'Model name' si está disponible."""
    try:
        out = subprocess.check_output(["lscpu"], stderr=subprocess.DEVNULL, text=True)
        for line in out.splitlines():
            m = re.match(r'^\s*Model name\s*:\s*(.+)$', line, flags=re.IGNORECASE)
            if m:
                val = m.group(1).strip()
                if val:
                    return val
        return None
    except Exception:
        return None

def _read_dmi_product_name() -> Optional[str]:
    """Lee /sys/devices/virtual/dmi/id/product_name y product_version si existen."""
    try:
        parts = []
        for path in ("/sys/devices/virtual/dmi/id/product_name", "/sys/devices/virtual/dmi/id/product_version"):
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    v = f.read().strip()
                    if v:
                        parts.append(v)
            except Exception:
                continue
        if parts:
            return " ".join(parts)
        return None
    except Exception:
        return None

def _parse_wmic_cpu_name() -> Optional[str]:
    """Intento en Windows: 'wmic cpu get Name'."""
    try:
        out = subprocess.check_output(["wmic", "cpu", "get", "Name"], stderr=subprocess.DEVNULL, text=True)
        lines = [l.strip() for l in out.splitlines() if l.strip()]
        if len(lines) >= 2:
            return lines[1]
        return None
    except Exception:
        return None


# ========================================================================
# CONFIGURATION & CONSTANTS
# ========================================================================
logging.basicConfig(
    level=logging.INFO, 
    format='[%(levelname)s] %(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

WARMUP_STEPS = 5
MEASURE_STEPS = 15
OUTPUT_DIR = "data"

# Heuristics for the Optimization Model
BACKWARD_FACTOR = 2.0  # Cost multiplier for Backward Pass relative to Forward
OPTIMIZER_OVERHEAD_FACTOR = 2.0 # Memory multiplier for Optimizer States (e.g., Adam m, v)

# ========================================================================
# UTILITY: REPRODUCIBILITY & HARDWARE CHECKS
# ========================================================================
def set_determinism(seed: int = 42):
    """
    Enforce strict determinism for reproducible scientific profiling.
    Crucial for valid A/B testing between CPU and GPU execution paths.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception as e:
        logger.warning(f"Warning: Could not force strict deterministic algorithms: {e}")

def cpu_supports_bf16() -> bool:
    """Checks for AVX512_BF16 instruction set support on Linux."""
    try:
        if platform.system() != "Linux": return False
        with open("/proc/cpuinfo", "r") as f:
            flags = f.read()
        return "avx512_bf16" in flags
    except Exception:
        return False

def get_hardware_metadata() -> Dict[str, Any]:
    """Captures hardware identifiers for the experimental setup section."""
    meta = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python_version": platform.python_version(),
        "torch_version": getattr(torch, "__version__", "unknown"),
        "os": platform.platform(),
        "cpu_model": "",
        "gpu_name": "None",
        "gpu_driver": "None",
        "rapl_available": PYRAPL_AVAILABLE
    }

    # --- Robust CPU model detection ---
    cpu_model = None
    source = None

    # 1) /proc/cpuinfo (Linux) - buscar 'model name' exactamente
    if platform.system() == "Linux":
        cpu_model = _parse_proc_cpuinfo_model_name()
        if cpu_model:
            source = "/proc/cpuinfo (model name)"

    # 2) lscpu (Linux)
    if not cpu_model and platform.system() == "Linux":
        cpu_model = _parse_lscpu_model_name()
        if cpu_model:
            source = "lscpu (Model name)"

    # 3) DMI product name (común en portátiles/servidores)
    if not cpu_model and platform.system() == "Linux":
        cpu_model = _read_dmi_product_name()
        if cpu_model:
            source = "/sys/devices/virtual/dmi/id/product_name"

    # 4) Windows WMIC
    if not cpu_model and platform.system() == "Windows":
        cpu_model = _parse_wmic_cpu_name()
        if cpu_model:
            source = "wmic cpu get Name"

    # 5) platform.processor() / uname() (fallback)
    if not cpu_model:
        try:
            p = platform.processor()
            if p and p.strip() and p.strip().lower() not in ("", "x86_64", "amd64", "i386"):
                cpu_model = p.strip()
                source = "platform.processor()"
        except Exception:
            cpu_model = None

    if not cpu_model:
        try:
            uname = platform.uname()
            candidate = getattr(uname, "processor", None) or getattr(uname, "machine", None)
            if candidate and str(candidate).strip().lower() not in ("", "x86_64", "amd64", "i386"):
                cpu_model = str(candidate).strip()
                source = "platform.uname()"
        except Exception:
            cpu_model = None

    # 6) py-cpuinfo (opcional)
    if not cpu_model:
        try:
            import cpuinfo  # type: ignore
            info = cpuinfo.get_cpu_info()
            cpu_model = info.get("brand_raw") or info.get("brand") or info.get("arch_string_raw")
            if cpu_model:
                cpu_model = str(cpu_model).strip()
                source = "py-cpuinfo"
        except Exception:
            cpu_model = None

    if not cpu_model:
        cpu_model = "Unknown CPU"
        source = "fallback"

    meta["cpu_model"] = cpu_model

    # Registrar la fuente usada (útil para auditoría)
    try:
        logger.debug(f"get_hardware_metadata: cpu_model source = {source}; cpu_model = {cpu_model}")
    except Exception:
        pass

    # --- GPU metadata via NVML ---
    if torch.cuda.is_available():
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())
            try:
                name = pynvml.nvmlDeviceGetName(handle)
                # nvml puede devolver bytes en algunas bindings; normalizar a str
                if isinstance(name, bytes):
                    name = name.decode("utf-8", errors="ignore")
                meta["gpu_name"] = str(name)
            except Exception:
                meta["gpu_name"] = "Unknown GPU"

            try:
                driver = pynvml.nvmlSystemGetDriverVersion()
                if isinstance(driver, bytes):
                    driver = driver.decode("utf-8", errors="ignore")
                meta["gpu_driver"] = str(driver)
            except Exception:
                meta["gpu_driver"] = "Unknown Driver"
        except Exception as e:
            logger.warning(f"Failed to query NVML in get_hardware_metadata: {e}")
        finally:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
    return meta

def get_tensor_size_recursive(data: Any) -> int:
    """Recursively calculates the payload size (bytes) of complex outputs."""
    size = 0
    if isinstance(data, torch.Tensor):
        size += data.numel() * data.element_size()
    elif isinstance(data, (tuple, list)):
        for item in data:
            size += get_tensor_size_recursive(item)
    elif isinstance(data, dict):
        for v in data.values():
            size += get_tensor_size_recursive(v)
    elif hasattr(data, 'to_tuple'): # HuggingFace ModelOutput
        size += get_tensor_size_recursive(data.to_tuple())
    return int(size)

# ========================================================================
# UTILITY: FLOPs ESTIMATION & MICRO-BENCHMARKING
# ========================================================================
def _numel(t: Any) -> int:
    return t.numel() if hasattr(t, 'numel') else 0

def estimate_flops(module: nn.Module, inputs: Any, output: Any) -> float:
    """Estimates theoretical FLOPs based on layer geometry."""
    in_t = inputs[0] if isinstance(inputs, (tuple, list)) and len(inputs) > 0 else inputs
    if not isinstance(in_t, torch.Tensor): return 0.0

    # 1. Conv2d
    if isinstance(module, nn.Conv2d) and isinstance(output, torch.Tensor):
        Cin = module.in_channels
        Cout = module.out_channels
        Kx, Ky = module.kernel_size if isinstance(module.kernel_size, tuple) else (module.kernel_size, module.kernel_size)
        Hout, Wout = output.shape[2], output.shape[3]
        effective_cin = max(Cin // module.groups, 1)
        return 2.0 * Cout * Hout * Wout * (effective_cin * Kx * Ky)
    
    # 2. Linear
    if isinstance(module, nn.Linear):
        in_f = module.in_features
        out_f = module.out_features
        positions = int(torch.tensor(in_t.shape[:-1]).prod().item()) if in_t is not None else 1
        return 2.0 * positions * in_f * out_f
    
    # 3. Activations / Norms
    if isinstance(module, (nn.ReLU, nn.GELU)): return float(_numel(in_t))
    if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)): return 5.0 * float(_numel(in_t))
    
    # 4. Attention
    name = module.__class__.__name__.lower()
    if "attention" in name and "multi" in name:
        B = in_t.shape[0]
        S = in_t.shape[1] if in_t.ndim >= 3 else 1
        d = in_t.shape[-1]
        H = getattr(module, "num_attention_heads", getattr(module, "n_heads", 1))
        return 4.0 * B * S * (d * d) + 2.0 * B * H * (S * S) * (d // H)

    return 0.0

# ========================================================================
# CUSTOM MODELS
# ========================================================================
class SimpleMLP(nn.Module):
    """Pure MLP for control experiments."""
    def __init__(self, input_dim=784, hidden_dims=(512, 256), output_dim=10):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# ========================================================================
# ENERGY MONITOR
# ========================================================================
class EnergyMonitor(threading.Thread):
    _rapl_lock = threading.Lock()
    _rapl_initialized = False

    def __init__(self, device_type: str = 'cuda', gpu_id: int = 0, sample_interval: float = 0.05):
        super().__init__()
        self.device_type = device_type
        self.gpu_id = gpu_id
        self.interval = sample_interval
        self.stop_event = threading.Event()
        self.readings = []
        self.avg_power = 0.0
        self.nvml_handle = None
        self.cpu_meter = None
        self._init_sensors()

    def _init_sensors(self):
        if self.device_type == 'cuda' and torch.cuda.is_available():
            try:
                pynvml.nvmlInit()
                self.nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_id)
            except Exception as e:
                logger.warning(f"NVML Init failed: {e}. GPU Energy = 0.")

        if self.device_type == 'cpu' and PYRAPL_AVAILABLE:
            import pyRAPL 
            with self._rapl_lock:
                try:
                    if not self._rapl_initialized:
                        pyRAPL.setup()
                        EnergyMonitor._rapl_initialized = True
                    self.cpu_meter = pyRAPL.Measurement('cpu_meter')
                except Exception as e:
                    logger.warning(f"PyRAPL Init failed: {e}. CPU Energy = None.")

    def run(self):
        if self.cpu_meter:
            with self._rapl_lock:
                self.cpu_meter.begin()
        while not self.stop_event.is_set():
            if self.device_type == 'cuda' and self.nvml_handle:
                try:
                    p_mw = pynvml.nvmlDeviceGetPowerUsage(self.nvml_handle)
                    self.readings.append(p_mw / 1000.0)
                except:
                    self.readings.append(0.0)
            time.sleep(self.interval)
        if self.cpu_meter:
            with self._rapl_lock:
                self.cpu_meter.end()

    def stop(self):
        self.stop_event.set()
        self.join()
        if self.device_type == 'cuda' and self.readings:
            self.avg_power = sum(self.readings) / len(self.readings)
            try:
                pynvml.nvmlShutdown()
            except Exception as e:
                logger.debug(f"NVML shutdown skipped: {e}")
        elif self.device_type == 'cpu' and self.cpu_meter:
            if self.cpu_meter.result:
                dur = self.cpu_meter.result.duration
                pkg = self.cpu_meter.result.pkg
                if dur is not None and pkg is not None:
                    d_val = dur[0] if isinstance(dur, list) else dur
                    e_val = pkg[0] if isinstance(pkg, list) else pkg
                    if d_val and e_val:
                        self.avg_power = (e_val / 1e6) / (d_val / 1e6) if d_val > 0 else 0.0
                    else:
                        self.avg_power = 0.0
                else:
                    self.avg_power = 0.0
            else:
                self.avg_power = 0.0


    def get_avg_power(self) -> float:
        return self.avg_power

# ========================================================================
# PROFILER LOGIC
# ========================================================================
class TrainingProfiler:
    def __init__(self, model: nn.Module, model_name: str, args):
        self.model = model
        self.model_name = model_name
        self.args = args
        self.layer_stats = {}
        self.hooks = []
        self._tstarts = {} 
        
        self.has_gpu = torch.cuda.is_available() and not args.no_gpu
        self.gpu_id = args.gpu_id if self.has_gpu else 0
        if self.has_gpu:
            self.model.to(f"cuda:{self.gpu_id}")

    def _get_leaf_modules(self) -> Iterator[Tuple[str, nn.Module]]:
        """Identify atomic operations for the ILP nodes."""
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:
                yield name, module

    def _compute_loss(self, out: Any) -> torch.Tensor:
        """Robust loss calculation."""
        try:
            current_dev = next(self.model.parameters()).device
        except StopIteration:
            current_dev = torch.device("cpu")

        if hasattr(out, "loss") and out.loss is not None: return out.loss
        if hasattr(out, "last_hidden_state"): return out.last_hidden_state.sum()
        if hasattr(out, "logits"): return out.logits.sum()

        if isinstance(out, dict):
            valid = [v.sum() for v in out.values() if isinstance(v, torch.Tensor)]
            if valid: return sum(valid, torch.tensor(0.0, device=current_dev))
        
        if isinstance(out, (tuple, list)):
            valid = [v.sum() for v in out if isinstance(v, torch.Tensor)]
            if valid: return sum(valid, torch.tensor(0.0, device=current_dev))

        if isinstance(out, torch.Tensor): return out.sum()
        return torch.tensor(0.0, device=current_dev, requires_grad=True)

    def _register_hooks(self, device_type: str):
        """
        Instruments the model using Pre/Post hooks for accurate layer timing.
        Captures both Device Time (CUDA Event) and Dispatch Time (CPU Wall).
        """
        for h in self.hooks: h.remove()
        self.hooks = []
        self._tstarts = {}

        def pre_hook_fn(layer_name):
            def pre(module, inputs):
                # Always capture CPU Wall Clock Start (for Dispatch Overhead)
                self._tstarts[layer_name] = {}
                self._tstarts[layer_name]['cpu_start'] = time.perf_counter()
                
                if device_type == 'cuda':
                    start = torch.cuda.Event(enable_timing=True)
                    self._tstarts[layer_name]['cuda_start'] = start
                    start.record()
            return pre

        def post_hook_fn(layer_name):
            def post(module, inputs, output):
                # Initialize Stats entry
                if layer_name not in self.layer_stats:
                    self.layer_stats[layer_name] = {
                        "type": module.__class__.__name__,
                        "params_mb": 0.0,
                        "output_bytes": 0,
                        "time_ms_accum": 0.0,
                        "dispatch_ms_accum": 0.0, # New: CPU Wall time per layer
                        "count": 0,
                        "gpu_mem_peak_mb": 0.0,
                        "flops": 0.0
                    }
                s = self.layer_stats[layer_name]

                try:
                    params_bytes = sum(p.numel() * p.element_size() for p in module.parameters(recurse=False))
                    s["params_mb"] = params_bytes / (1024**2)
                except: pass

                s["output_bytes"] = max(s["output_bytes"], get_tensor_size_recursive(output))

                if device_type == 'cuda':
                    # Global snapshot - conservative global proxy, not isolated
                    s["gpu_mem_peak_mb"] = max(s["gpu_mem_peak_mb"], 
                        torch.cuda.max_memory_allocated(self.gpu_id) / (1024**2))

                try:
                    s["flops"] = max(s["flops"], estimate_flops(module, inputs, output))
                except: pass

                # --- Timing ---
                time_ms = 0.0
                if device_type == 'cuda':
                    end = torch.cuda.Event(enable_timing=True)
                    end.record()
                    torch.cuda.synchronize() # Barrier for accurate per-layer measurement
                    start = self._tstarts.get(layer_name, {}).get('cuda_start', None)
                    if start is not None:
                        time_ms = start.elapsed_time(end)
                else:
                    # For CPU, Execution Time == Wall Clock Time
                    t0 = self._tstarts.get(layer_name, {}).get('cpu_start', None)
                    if t0 is not None:
                        time_ms = (time.perf_counter() - t0) * 1000.0

                # --- Dispatch Overhead (Wall - Kernel) ---
                # NOTE: Only valid when synchronization is enforced (as above).
                # Dispatch Overhead = (Total CPU Wall Time) - (GPU Kernel Time)
                t_now = time.perf_counter()
                t0_cpu = self._tstarts.get(layer_name, {}).get('cpu_start', t_now)
                wall_ms = (t_now - t0_cpu) * 1000.0
                
                # Avoid negative overhead if timing drift occurs
                dispatch_ms = max(0.0, wall_ms - time_ms)

                s["time_ms_accum"] += float(time_ms)
                s["dispatch_ms_accum"] += float(dispatch_ms)
                s["count"] += 1
            return post

        for name, module in self._get_leaf_modules():
            self.hooks.append(module.register_forward_pre_hook(pre_hook_fn(name)))
            self.hooks.append(module.register_forward_hook(post_hook_fn(name)))

    def _run_epoch(self, input_data: Any, device: str, steps: int) -> Tuple[Optional[float], float]:
        """Runs the training loop. Returns: (total_energy_joules, total_time_seconds)"""
        device_str = f"cuda:{self.gpu_id}" if device == "cuda" else "cpu"
        self.model.to(device_str)
        self.model.train()
        
        if isinstance(input_data, dict):
            inp = {k: v.to(device_str) for k, v in input_data.items()}
        else:
            inp = input_data.to(device_str)

        opt = torch.optim.SGD(self.model.parameters(), lr=0.01)
        self.layer_stats = {} 
        self._register_hooks(device)

        monitor = EnergyMonitor(device_type=device, gpu_id=self.gpu_id)
        monitor.start()
        
        total_start = time.perf_counter()
        
        try:
            for _ in range(steps):
                opt.zero_grad()
                if device == "cuda": torch.cuda.reset_peak_memory_stats(self.gpu_id)

                if isinstance(inp, dict): out = self.model(**inp)
                else: out = self.model(inp)
                
                loss = self._compute_loss(out)
                loss.backward()
                opt.step()

                if device == "cuda": torch.cuda.synchronize()
        finally:
            monitor.stop()
            for h in self.hooks: h.remove()
            self.hooks = []

        total_duration_sec = time.perf_counter() - total_start
        
        avg_power = monitor.get_avg_power()
        
        if device == "cpu" and not PYRAPL_AVAILABLE:
            return None, total_duration_sec
        
        total_energy_j = avg_power * total_duration_sec
        return total_energy_j, total_duration_sec

    def _measure_peak_flops(self, device: str) -> float:
        """
        Micro-benchmark to estimate Empirical Peak TFLOPS for the current precision.
        """
        logger.info(f"--> Benchmarking Empirical Peak {device.upper()} TFLOPS...")
        
        N = 8192 
        dtype = torch.float32
        if self.args.precision == "fp16": dtype = torch.float16
        elif self.args.precision == "bf16": dtype = torch.bfloat16
        
        device_str = f"{device}:{self.gpu_id}" if device == 'cuda' else 'cpu'
        
        try:
            a = torch.randn(N, N, device=device_str, dtype=dtype)
            b = torch.randn(N, N, device=device_str, dtype=dtype)
        except RuntimeError: 
            N = 4096
            a = torch.randn(N, N, device=device_str, dtype=dtype)
            b = torch.randn(N, N, device=device_str, dtype=dtype)

        for _ in range(3): _ = torch.mm(a, b)
        if device == 'cuda': torch.cuda.synchronize()

        iterations = 5
        
        if device == 'cuda':
            start_evt = torch.cuda.Event(enable_timing=True)
            end_evt = torch.cuda.Event(enable_timing=True)
            start_evt.record()
            for _ in range(iterations): _ = torch.mm(a, b)
            end_evt.record()
            torch.cuda.synchronize()
            duration_s = start_evt.elapsed_time(end_evt) / 1000.0
        else:
            t0 = time.perf_counter()
            for _ in range(iterations): _ = torch.mm(a, b)
            duration_s = time.perf_counter() - t0

        if duration_s <= 0: return 1.0

        total_ops = iterations * 2 * (N**3)
        tflops = (total_ops / 1e12) / duration_s
        
        logger.info(f"    Measured Peak: {tflops:.2f} TFLOPS")
        return tflops

    def _measure_pci_bandwidth(self) -> Dict[str, float]:
        """Calibrates Linear Transfer Model (Alpha-Beta)."""
        if not self.has_gpu: return {}
        logger.info("--> Calibrating PCIe Transfer Model (H2D & D2H)...")
        results = {}
        sizes_mb = [1.0, 100.0]
        device_str = f"cuda:{self.gpu_id}"

        for direction in ['h2d', 'd2h']:
            times = []
            for sz in sizes_mb:
                numel = int(sz * 1024**2 / 4)
                # Use pinned memory for CPU tensors to simulate realistic Dataloader behavior
                if direction == 'h2d':
                    t_tensor = torch.randn(numel).pin_memory() 
                else:
                    t_tensor = torch.randn(numel) # Device tensor created below
                
                if direction == 'd2h': t_tensor = t_tensor.to(device_str)
                
                # Warmup
                if direction == 'h2d': _ = t_tensor.to(device_str, non_blocking=True)
                else: _ = t_tensor.to('cpu', non_blocking=True)
                torch.cuda.synchronize()
                
                start = time.perf_counter()
                if direction == 'h2d': _ = t_tensor.to(device_str, non_blocking=True)
                else: _ = t_tensor.to('cpu', non_blocking=True)
                torch.cuda.synchronize()
                times.append((time.perf_counter() - start) * 1000.0)
            
            delta_t = times[1] - times[0]
            delta_sz = sizes_mb[1] - sizes_mb[0]
            beta = delta_sz / delta_t if delta_t > 0 else 1.0
            alpha = max(0.0, times[0] - (sizes_mb[0] / beta))
            results[f"pci_alpha_{direction}_ms"] = alpha
            results[f"pci_beta_{direction}_mb_ms"] = beta
        return results

    def run(self, input_data: Any):
        logger.info(f"Starting Profiling Run for: {self.model_name}")
        self._run_epoch(input_data, "cuda" if self.has_gpu else "cpu", self.args.warmup)

        # --- GPU Profiling ---
        gpu_total_energy, gpu_run_time_sec = 0.0, 0.0
        gpu_layer_stats = {}
        measured_gpu_peak_tflops = 0.0
        
        if self.has_gpu:
            logger.info("--> Profiling GPU Execution...")
            gpu_total_energy, gpu_run_time_sec = self._run_epoch(input_data, "cuda", self.args.measure)
            gpu_layer_stats = self.layer_stats.copy()
            measured_gpu_peak_tflops = self._measure_peak_flops("cuda")
        
        # --- CPU Profiling ---
        logger.info("--> Profiling CPU Execution...")
        cpu_total_energy, cpu_run_time_sec = self._run_epoch(input_data, "cpu", self.args.measure)
        cpu_layer_stats = self.layer_stats.copy()
        
        # Measure CPU Peak TFLOPS (Reference)
        measured_cpu_peak_tflops = self._measure_peak_flops("cpu")

        gpu_peak_mb = 0.0
        if self.has_gpu:
            torch.cuda.synchronize()
            gpu_peak_mb = torch.cuda.max_memory_allocated(self.gpu_id) / (1024**2)

        transfers = self._measure_pci_bandwidth()
        all_layers = list(gpu_layer_stats.keys()) if self.has_gpu else list(cpu_layer_stats.keys())
        
        g_total_layers_ms = sum(
            (gpu_layer_stats[l].get("time_ms_accum", 0) / max(1, gpu_layer_stats[l].get("count", 1)))
            for l in gpu_layer_stats
        ) or 1.0
        
        c_total_layers_ms = sum(
            (cpu_layer_stats[l].get("time_ms_accum", 0) / max(1, cpu_layer_stats[l].get("count", 1)))
            for l in cpu_layer_stats
        ) or 1.0

        avg_step_time_gpu_ms = (gpu_run_time_sec * 1000.0) / self.args.measure
        avg_step_time_cpu_ms = (cpu_run_time_sec * 1000.0) / self.args.measure
        
        # Ratios: Overhead
        framework_overhead_gpu_ms = max(0.0, avg_step_time_gpu_ms - g_total_layers_ms)
        framework_overhead_ratio_gpu = framework_overhead_gpu_ms / avg_step_time_gpu_ms if avg_step_time_gpu_ms > 0 else 0.0
        
        framework_overhead_cpu_ms = max(0.0, avg_step_time_cpu_ms - c_total_layers_ms)
        framework_overhead_ratio_cpu = framework_overhead_cpu_ms / avg_step_time_cpu_ms if avg_step_time_cpu_ms > 0 else 0.0

        rows = []
        efficiency_list = [] 
        tflops_list = [] 
        weighted_tflops_numerator = 0.0 
        weighted_tflops_denominator = 0.0 
        energy_dist_vector = [] 
        framework_overhead_vector = [] # Per layer dispatch times
        
        total_model_flops = 0.0

        for layer in all_layers:
            g_stat = gpu_layer_stats.get(layer, {})
            c_stat = cpu_layer_stats.get(layer, {})
            
            g_t_fwd = g_stat.get("time_ms_accum", 0) / max(1, g_stat.get("count", 1))
            c_t_fwd = c_stat.get("time_ms_accum", 0) / max(1, c_stat.get("count", 1))
            
            # Dispatch Time (CPU Wall time per layer)
            g_dispatch = g_stat.get("dispatch_ms_accum", 0) / max(1, g_stat.get("count", 1))
            dispatch_overhead_ratio = g_dispatch / g_t_fwd if g_t_fwd > 0 else 0.0
            
            # Normalized Energy Shares (Sum = 1.0)
            g_share = g_t_fwd / g_total_layers_ms
            c_share = c_t_fwd / c_total_layers_ms
            
            energy_dist_vector.append({
                "layer": layer,
                "gpu_share": g_share,
                "cpu_share": c_share
            })
            
            framework_overhead_vector.append({
                "layer": layer,
                "dispatch_overhead_ms": g_dispatch
            })
            
            out_mb = g_stat.get("output_bytes", c_stat.get("output_bytes", 0)) / (1024**2)
            
            alpha_h2d = transfers.get("pci_alpha_h2d_ms", 0.05)
            beta_h2d  = transfers.get("pci_beta_h2d_mb_ms", 12.0)
            alpha_d2h = transfers.get("pci_alpha_d2h_ms", 0.05)
            beta_d2h  = transfers.get("pci_beta_d2h_mb_ms", 12.0)
            
            flops = g_stat.get("flops", c_stat.get("flops", 0.0))
            count = max(1, g_stat.get("count", c_stat.get("count", 1)))
            # Accumulate Total FLOPs per step (Ops/Pass * Count)
            total_model_flops += (flops * count)
            
            params_mb = g_stat.get("params_mb", 0.0)
            # Calculated Optimizer States MB
            optimizer_states_mb = params_mb * OPTIMIZER_OVERHEAD_FACTOR

            g_e_avg_step = (gpu_total_energy / self.args.measure) if gpu_total_energy else 0.0
            g_e_fwd = g_e_avg_step * g_share

            c_e_avg_step = (cpu_total_energy / self.args.measure) if cpu_total_energy is not None else None
            c_e_fwd = c_e_avg_step * c_share if c_e_avg_step is not None else None

            # TFLOPS & Empirical Efficiency
            tflops = 0.0
            efficiency_ratio = 0.0
            
            # Energy Efficiency (J/TFLOP) - Per Layer
            # Work done in this layer step = flops * count
            layer_work_tflops = (flops * count) / 1e12
            layer_j_per_tflop_gpu = 0.0
            layer_j_per_tflop_cpu = None

            if g_t_fwd > 0:
                # TFLOPS rate: (FLOPs / Time)
                tflops = (flops / 1e12) / (g_t_fwd / 1000.0)
                if measured_gpu_peak_tflops > 0:
                    efficiency_ratio = tflops / measured_gpu_peak_tflops
                
                weighted_tflops_numerator += (flops / 1e12)
                weighted_tflops_denominator += (g_t_fwd / 1000.0)
                
                # J/TFLOP (Energy / Work)
                if layer_work_tflops > 0 and g_e_fwd > 0:
                    layer_j_per_tflop_gpu = g_e_fwd / layer_work_tflops

            if c_e_fwd is not None and layer_work_tflops > 0:
                 layer_j_per_tflop_cpu = c_e_fwd / layer_work_tflops
            
            efficiency_list.append({
                "layer": layer,
                "efficiency_ratio": efficiency_ratio,
                "tflops": tflops,
                "layer_j_per_tflop_gpu": layer_j_per_tflop_gpu,
                "layer_j_per_tflop_cpu": layer_j_per_tflop_cpu,
                "params_mb": params_mb,
                "optimizer_states_mb": optimizer_states_mb,
                "output_mb": out_mb,
                "dispatch_overhead_ratio": dispatch_overhead_ratio
            })

            tflops_list.append(tflops)

            rows.append({
                "layer": layer,
                "type": g_stat.get("type", c_stat.get("type", "Unknown")),
                "params_mb": params_mb,
                "grads_mb": params_mb,
                "optimizer_states_mb": optimizer_states_mb, # New: Explicit optimizer memory
                "theoretical_flops": flops,
                "tflops": tflops, 
                "efficiency_ratio": efficiency_ratio, 
                "activations_mb": out_mb, 
                "gpu_fwd_time_ms": g_t_fwd,
                "gpu_bwd_time_ms": g_t_fwd * BACKWARD_FACTOR,
                "gpu_fwd_energy_j": g_e_fwd,
                "gpu_bwd_energy_j": g_e_fwd * BACKWARD_FACTOR if g_e_fwd is not None else 0.0,
                "gpu_mem_peak_mb": g_stat.get("gpu_mem_peak_mb", 0.0),
                "layer_j_per_tflop_gpu": layer_j_per_tflop_gpu, 
                "dispatch_overhead_ratio": dispatch_overhead_ratio, # New: Per-layer Python overhead ratio
                # CPU
                "cpu_fwd_time_ms": c_t_fwd,
                "cpu_bwd_time_ms": c_t_fwd * BACKWARD_FACTOR,
                "cpu_fwd_energy_j": c_e_fwd,
                "cpu_bwd_energy_j": c_e_fwd * BACKWARD_FACTOR if c_e_fwd is not None else None,
                "cpu_mem_mb": out_mb,
                "layer_j_per_tflop_cpu": layer_j_per_tflop_cpu,
                # Transfers
                "transfer_h2d_ms": alpha_h2d + (out_mb / beta_h2d),
                "transfer_d2h_ms": alpha_d2h + (out_mb / beta_d2h),
                # Meta
                "precision_requested": self.args.precision,
                "cpu_precision_executed": getattr(self.args, "cpu_precision_executed", self.args.precision),
                "gpu_precision_executed": getattr(self.args, "gpu_precision_executed", self.args.precision)
            })

        os.makedirs(self.args.output_dir, exist_ok=True)
        pd.DataFrame(rows).to_csv(os.path.join(self.args.output_dir, f"{self.model_name}_metrics.csv"), index=False)

        # --- Metadata Collection ---
        proc = psutil.Process()
        try:
            mem_info = proc.memory_full_info()
            uss_mb = mem_info.uss / (1024**2)
            pss_mb = getattr(mem_info, "pss", 0) / (1024**2)
        except:
            uss_mb = 0.0
            pss_mb = 0.0
            
        stats_source = gpu_layer_stats if self.has_gpu else cpu_layer_stats
        params_total = sum(s.get("params_mb", 0.0) for s in stats_source.values())
        grads_total = sum(row['grads_mb'] for row in rows)
        activations_total = sum(row['activations_mb'] for row in rows)
        gpu_reserved_mb = torch.cuda.memory_reserved(self.gpu_id) / (1024**2) if self.has_gpu else 0
        
        # Global Averages & Metrics
        eff_values = [e['efficiency_ratio'] for e in efficiency_list]
        eff_avg = sum(eff_values) / len(eff_values) if eff_values else 0.0
        simple_tflops_avg = sum(tflops_list) / len(tflops_list) if tflops_list else 0.0
        weighted_tflops_avg = weighted_tflops_numerator / weighted_tflops_denominator if weighted_tflops_denominator > 0 else 0.0
        
        # Global Energy Efficiency (J/TFLOP)
        energy_efficiency_gpu = 0.0
        energy_efficiency_cpu = None
        
        # Work = total_model_flops (per step) * measures
        total_work_tflops = (total_model_flops * self.args.measure) / 1e12
        
        if total_work_tflops > 0:
            if gpu_total_energy is not None and gpu_total_energy > 0:
                energy_efficiency_gpu = gpu_total_energy / total_work_tflops
            else:
                energy_efficiency_gpu = None

            if cpu_total_energy is not None and cpu_total_energy > 0:
                energy_efficiency_cpu = cpu_total_energy / total_work_tflops
            else:
                energy_efficiency_cpu = None

        meta = get_hardware_metadata()
        meta.update({
            "model": self.model_name,
            "layers_profiled_count": len(all_layers),
            "precision_mode": self.args.precision, 
            # Timings & Overhead
            "gpu_total_layer_time_ms": g_total_layers_ms, 
            "cpu_total_layer_time_ms": c_total_layers_ms,
            "gpu_step_time_ms": avg_step_time_gpu_ms,
            "cpu_step_time_ms": avg_step_time_cpu_ms,
            "framework_overhead_gpu_ms": framework_overhead_gpu_ms,
            "framework_overhead_cpu_ms": framework_overhead_cpu_ms,
            "framework_overhead_ratio_gpu": framework_overhead_ratio_gpu,
            "framework_overhead_ratio_cpu": framework_overhead_ratio_cpu,
            "framework_overhead_vector": framework_overhead_vector, # New: Dispatch overhead per layer
            # Energy
            "energy_avg_per_step_gpu_j": (gpu_total_energy / self.args.measure) if gpu_total_energy else 0.0,
            "energy_avg_per_step_cpu_j": (cpu_total_energy / self.args.measure) if cpu_total_energy is not None else None,
            "energy_total_gpu_j": gpu_total_energy,
            "energy_total_cpu_j": cpu_total_energy,
            "energy_distribution_vector": energy_dist_vector, 
            # Memory
            "gpu_mem_peak_mb_global": gpu_peak_mb,
            "gpu_mem_reserved_mb_global": gpu_reserved_mb, 
            "cpu_uss_mb_global": uss_mb,
            "cpu_pss_mb_global": pss_mb,
            "params_mb_total": params_total,
            "grads_mb_total": grads_total,
            "activations_mb_total": activations_total,
            "optimizer_state_mb_factor": OPTIMIZER_OVERHEAD_FACTOR,
            # Transfer
            "transfer_alpha_h2d": transfers.get("pci_alpha_h2d_ms", 0),
            "transfer_beta_h2d": transfers.get("pci_beta_h2d_mb_ms", 1),
            "transfer_alpha_d2h": transfers.get("pci_alpha_d2h_ms", 0),
            "transfer_beta_d2h": transfers.get("pci_beta_d2h_mb_ms", 1),
            "pcie_stats_raw": transfers,
            # Hardware & Efficiency
            "measured_peak_tflops_gpu": measured_gpu_peak_tflops,
            "measured_peak_tflops_cpu": measured_cpu_peak_tflops,
            "efficiency_ratio_avg": eff_avg,
            "efficiency_ratio_vector": efficiency_list, # Confirmed vector
            "avg_tflops_per_layer": simple_tflops_avg, 
            "weighted_avg_tflops_per_layer": weighted_tflops_avg, 
            "energy_efficiency_j_per_tflop_gpu": energy_efficiency_gpu,
            "energy_efficiency_j_per_tflop_cpu": energy_efficiency_cpu,
            "cpu_precision": getattr(self.args, "cpu_precision_executed", "unknown"),
            "gpu_precision": getattr(self.args, "gpu_precision_executed", "unknown")
        })
        with open(os.path.join(self.args.output_dir, f"{self.model_name}_meta.json"), 'w') as f:
            json.dump(meta, f, indent=4)
        logger.info(f"DONE. Meta saved. Eff Avg={eff_avg:.3f}, GPU Overhead={framework_overhead_gpu_ms:.2f}ms")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        choices=["resnet50", "resnet152", "vit", "bert", "gpt2", "mlp"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp32")
    parser.add_argument("--warmup", type=int, default=WARMUP_STEPS)
    parser.add_argument("--measure", type=int, default=MEASURE_STEPS)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--no-gpu", action="store_true")
    args = parser.parse_args()

    set_determinism(42)

    # --- Precision Fallback Logic ---
    if args.precision == "bf16":
        if not cpu_supports_bf16():
            logger.warning("CPU lacks AVX512_BF16 support. Falling back CPU execution to FP32.")
            args.cpu_precision_executed = "fp32_fallback"
        else:
            args.cpu_precision_executed = "bf16"
    else:
        args.cpu_precision_executed = args.precision

    args.gpu_precision_executed = args.precision if torch.cuda.is_available() else "none"

    # --- Select dtype ---
    torch_dtype = _select_torch_dtype(args.precision)

    # --- Model Factory ---
    model: nn.Module
    inp: Any

    if args.model == "resnet50":
        model = resnet50(weights=ResNet50_Weights.DEFAULT)
        inp = torch.randn(args.batch_size, 3, 224, 224)
        if args.precision == "fp16":
            model = model.half()
        elif args.precision == "bf16":
            model = model.to(torch_dtype)

    elif args.model == "resnet152":
        model = resnet152(weights=ResNet152_Weights.DEFAULT)
        inp = torch.randn(args.batch_size, 3, 224, 224)
        if args.precision == "fp16":
            model = model.half()
        elif args.precision == "bf16":
            model = model.to(torch_dtype)

    elif args.model == "vit":
        model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        inp = torch.randn(args.batch_size, 3, 224, 224)
        if args.precision == "fp16":
            model = model.half()
        elif args.precision == "bf16":
            model = model.to(torch_dtype)

    elif args.model == "bert":
        # HuggingFace: set dtype at construction
        model = BertModel.from_pretrained("bert-base-uncased", torch_dtype=torch_dtype)
        inp = {
            "input_ids": torch.randint(0, 1000, (args.batch_size, 128)),
            "attention_mask": torch.ones(args.batch_size, 128),
        }

    elif args.model == "gpt2":
        model = GPT2Model.from_pretrained("gpt2", torch_dtype=torch_dtype)
        inp = {"input_ids": torch.randint(0, 1000, (args.batch_size, 128))}

    elif args.model == "mlp":
        model = SimpleMLP()
        inp = torch.randn(args.batch_size, 784)
        if args.precision == "fp16":
            model = model.half()
        elif args.precision == "bf16":
            model = model.to(torch_dtype)

    else:
        raise ValueError(f"Unknown model: {args.model}")

    # --- Cast Precision for inputs ---
    if args.precision == "fp16":
        if isinstance(inp, torch.Tensor):
            inp = inp.half()
        elif isinstance(inp, dict):
            inp = {k: (v.half() if v.is_floating_point() else v) for k, v in inp.items()}

    elif args.precision == "bf16":
        if isinstance(inp, torch.Tensor):
            inp = inp.to(torch_dtype)
        elif isinstance(inp, dict):
            inp = {k: (v.to(torch_dtype) if v.is_floating_point() else v) for k, v in inp.items()}

    # --- Run Profiler ---
    TrainingProfiler(model, args.model, args).run(inp)


if __name__ == "__main__":
    main()