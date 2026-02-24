"""
Advanced Hybrid Profiler for Deep Learning Training (PhD Thesis).

This tool characterizes neural network architectures to generate cost metrics
required by the Integer Linear Programming (ILP) optimization model defined
in Chapter 3 of the thesis.

================================================================================
DATA DICTIONARY (OUTPUTS)
================================================================================

1. METRICS CSV ({model_name}_metrics.csv): Per-layer execution metrics.
--------------------------------------------------------------------------------
| Column                  | Description                                            |
|-------------------------|--------------------------------------------------------|
| layer                   | Name of the leaf module (e.g., conv1, fc)              |
| type                    | PyTorch class name (e.g., Conv2d, Linear, ReLU)        |
| params_mb               | Static parameter memory size (MB)                      |
| grads_mb                | Gradient memory size (approx. equal to params) (MB)    |
| optimizer_states_mb     | Optimizer state memory (params_mb * factor) (MB)       |
| activations_mb          | Output tensor size (activations) (MB)                  |
| theoretical_flops       | Calculated FLOPs based on layer geometry               |
| tflops                  | Effective throughput (TeraFLOPS)                       |
| efficiency_ratio        | Hardware Utilization (tflops / measured_peak_tflops)   |
| gpu_fwd_time_ms         | GPU Kernel time for Forward pass (CUDA Events) (ms)    |
| gpu_bwd_time_ms         | GPU Kernel time for Backward pass (Heuristic) (ms)     |
| gpu_fwd_energy_j        | Energy consumed during Forward (Joules)                |
| gpu_bwd_energy_j        | Energy consumed during Backward (Joules)               |
| gpu_mem_peak_mb         | GPU Memory Peak Snapshot (Global Proxy) (MB)           |
| layer_j_per_tflop_gpu   | Energy Efficiency: Joules per TFLOP (GPU)              |
| dispatch_overhead_ratio | CPU Dispatch Overhead / GPU Kernel Time                |
| cpu_fwd_time_ms         | CPU Wall time for Forward pass (ms)                    |
| cpu_bwd_time_ms         | CPU Wall time for Backward pass (ms)                   |
| cpu_fwd_energy_j        | Energy consumed by CPU (RAPL) during Forward (Joules)  |
| cpu_bwd_energy_j        | Energy consumed by CPU (RAPL) during Backward (Joules) |
| transfer_h2d_ms         | Estimated Host->Device transfer time (alpha+beta) (ms) |
| transfer_d2h_ms         | Estimated Device->Host transfer time (alpha+beta) (ms) |
| remat_penalty_ms        | [NEW] Cost to recompute layer (for Checkpointing) (ms) |

2. METADATA JSON ({model_name}_meta.json): Global environment & summary stats.
--------------------------------------------------------------------------------
| Key                     | Description                                            |
|-------------------------|--------------------------------------------------------|
| pcie_stats_raw          | Calibration data for PCIe bus (Alpha/Beta/Sigma)      |
| measured_peak_tflops_*  | Empirical peak TFLOPS measured via GEMM benchmark      |
| energy_total_*_j        | Total energy consumed during the profiling window      |
| overlap_ratio_sigma     | [NEW] Streaming concurrency ratio (0.0 - 1.0)          |
| framework_overhead_*    | Time spent in Python dispatch vs Kernel execution      |
| cpu_precision_executed  | Effective precision executed on CPU (with status tags) |
| gpu_precision_executed  | Effective precision executed on GPU                    |
| cpu_fp16_supported      | Functional CPU FP16 support result (True/False)        |
| cpu_fp16_isa_avx512     | Whether AVX512_FP16 ISA flag is detected               |
| cpu_fp16_smoke_test_ok  | Result of runtime FP16 CPU smoke test (torch.mm)       |
| cpu_fp16_model_smoke_ok | Result of CPU FP16 mini training-step preflight        |
| cpu_fp16_model_smoke_reason | Diagnostic reason from CPU FP16 model preflight   |
| cpu_fp16_support_reason | Human-readable reason/diagnostic for FP16 support      |
================================================================================

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
     contribution to the total *computational time*.\
   - Formula: E_layer = E_total_measured * (T_layer / Sum(T_layers))
   - Resulting energy distribution vectors are normalized to sum exactly to 1.0 per device.

4. Empirical Efficiency (TFLOPS):
   - Instead of using theoretical datasheet values, we run a micro-benchmark (GEMM)
     at runtime to measure the effective Peak TFLOPS.
   - NOTE (CPU): On CPU, the GEMM benchmark is provided as a reference only.
     CPUs are often memory-bandwidth bound rather than compute-bound for deep learning workloads.
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

7. Advanced Features (New in Thesis):
   - Overlap Ratio (Sigma): Measures potential for overlapping PCIe transfers with GPU compute.
   - Rematerialization Penalty: Explicitly measures time to recompute layers for Checkpointing.

LIMITATIONS (Thesis Section 3.2):
    - CPU Energy: Returns 'NaN' (None) if Intel RAPL interface (/sys/class/powercap) is unavailable.
      Consequently, 'layer_j_per_tflop_cpu' and 'energy_efficiency_cpu' will also be None.
        - CPU FP16 Support: The profiler validates functional FP16 support on CPU and records this
            in metadata for traceability. If unsupported, it logs a warning and continues execution
            without forcing a fallback to FP32.
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
import re
from typing import Dict, Tuple, Iterator, Any, Optional, List, Union
import atexit

import torch
import torch.nn as nn
import torch.optim as optim
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
# CONFIGURATION & CONSTANTS
# ========================================================================
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s - %(message)s'
)
logger = logging.getLogger(__name__)

WARMUP_STEPS = 5
MEASURE_STEPS = 15
OUTPUT_DIR = "data"
BACKWARD_FACTOR = 2.0
OPTIMIZER_OVERHEAD_FACTOR = 2.0

OPTIMIZER_OVERHEAD_MAP = {
    "SGD": 0.0,
    "SGD_momentum": 1.0,
    "Adam": 2.0,
    "AdamW": 2.0,
    "RMSprop": 1.0,
    "Adagrad": 1.0,
    "Adadelta": 2.0
}

# ========================================================================
# UTILITY: HARDWARE & DETERMINISM
# ========================================================================
def set_determinism(seed: int = 42):
    """Enforce reproducibility for scientific profiling."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(seed)

def configure_cpu_runtime(force_threads: int = 0) -> None:
    """Configure CPU threading and denormal handling for stable HPC behavior.
    
    Args:
        force_threads: If > 0, override auto-detection and use this thread count.
                      Useful for overriding SLURM/HPC CPU affinity restrictions.
    """
    env_threads = os.getenv("OMP_NUM_THREADS")
    target_threads = None
    source = None

    # Priority: force_threads > OMP_NUM_THREADS > cpu_affinity > physical_cores
    if force_threads > 0:
        target_threads = force_threads
        source = "user_forced"
    elif env_threads and env_threads.isdigit() and int(env_threads) > 0:
        target_threads = int(env_threads)
        source = "OMP_NUM_THREADS"
    else:
        try:
            affinity = psutil.Process().cpu_affinity()
            affinity_count = len(affinity) if affinity is not None else 0
        except Exception:
            affinity_count = 0

        physical_count = psutil.cpu_count(logical=False) or psutil.cpu_count(logical=True) or 1
        target_threads = affinity_count if affinity_count > 0 else physical_count
        source = "cpu_affinity" if affinity_count > 0 else "physical_cores"

    target_threads = max(1, int(target_threads))

    try:
        torch.set_num_threads(target_threads)
    except Exception as e:
        logger.warning(f"Failed to set torch intra-op threads: {e}")

    try:
        torch.set_num_interop_threads(max(1, min(4, target_threads)))
    except Exception as e:
        logger.warning(f"Failed to set torch inter-op threads: {e}")

    try:
        torch.set_flush_denormal(True)
    except Exception as e:
        logger.warning(f"Failed to enable flush denormals: {e}")

    logger.info(
        f"CPU runtime configured: threads={target_threads} "
        f"(source={source}), interop={max(1, min(4, target_threads))}, flush_denormal=True"
    )

def cpu_supports_bf16() -> bool:
    """Check for AVX512_BF16 support on Linux."""
    try:
        if platform.system() != "Linux": return False
        with open("/proc/cpuinfo", "r") as f:
            return "avx512_bf16" in f.read()
    except: return False

def get_cpu_fp16_support_info() -> Dict[str, Any]:
    """Validate functional CPU FP16 support and provide traceable diagnostics."""
    info = {
        "supported": False,
        "isa_avx512_fp16": False,
        "smoke_test_ok": False,
        "reason": "unknown"
    }

    try:
        if platform.system() == "Linux":
            with open("/proc/cpuinfo", "r") as f:
                cpuinfo = f.read().lower()
                info["isa_avx512_fp16"] = "avx512_fp16" in cpuinfo
    except Exception:
        info["isa_avx512_fp16"] = False

    try:
        a = torch.randn((64, 64), dtype=torch.float16, device="cpu")
        b = torch.randn((64, 64), dtype=torch.float16, device="cpu")
        _ = torch.mm(a, b)
        info["smoke_test_ok"] = True
    except Exception as e:
        info["smoke_test_ok"] = False
        info["reason"] = f"fp16 cpu smoke test failed: {e}"

    if info["smoke_test_ok"]:
        info["supported"] = True
        if info["isa_avx512_fp16"]:
            info["reason"] = "functional support validated; avx512_fp16 detected"
        else:
            info["reason"] = "functional support validated; avx512_fp16 not detected (possible emulation/slower path)"
    elif info["reason"] == "unknown":
        info["reason"] = "functional support not available"

    return info

def _extract_loss_for_preflight(out: Any) -> torch.Tensor:
    """Extract a scalar loss tensor from heterogeneous model outputs."""
    if hasattr(out, "loss") and out.loss is not None:
        return out.loss
    if hasattr(out, "logits"):
        return out.logits.sum()
    if isinstance(out, torch.Tensor):
        return out.sum()
    if isinstance(out, (tuple, list)) and len(out) > 0 and isinstance(out[0], torch.Tensor):
        return out[0].sum()
    return torch.tensor(0.0, requires_grad=True)

def _build_mini_input_for_cpu_fp16(input_data: Any) -> Any:
    """Build a minimal batch (size=1) and cast floating tensors to FP16 on CPU."""
    def _slice_first(x: torch.Tensor) -> torch.Tensor:
        if x.ndim > 0 and x.shape[0] > 1:
            return x[:1].clone()
        return x.clone()

    def _to_cpu_fp16_if_float(x: torch.Tensor) -> torch.Tensor:
        x_cpu = x.to("cpu")
        if torch.is_floating_point(x_cpu):
            return x_cpu.to(dtype=torch.float16)
        return x_cpu

    if isinstance(input_data, dict):
        mini = {}
        for key, value in input_data.items():
            if isinstance(value, torch.Tensor):
                mini[key] = _to_cpu_fp16_if_float(_slice_first(value))
            else:
                mini[key] = value
        return mini

    if isinstance(input_data, torch.Tensor):
        return _to_cpu_fp16_if_float(_slice_first(input_data))

    return input_data

def run_cpu_fp16_model_preflight(model: nn.Module, input_data: Any, timeout_safety_factor: float = 2.5) -> Dict[str, Any]:
    """
    CPU FP16 full training-step preflight with adaptive timeout based on forward time.
    
    Validates that the model can execute forward+backward+step in FP16 on CPU without blocking.
    Uses adaptive timeout: measures forward pass, then allows backward time = forward_time * BACKWARD_FACTOR * safety_factor.
    
    Args:
        model: Neural network model to validate
        input_data: Sample input data
        timeout_safety_factor: Multiplier for timeout calculation (default 2.5×forward×BACKWARD_FACTOR)
                              Accounts for overhead and hardware variability
    
    Result fields:
    - ok: True if training step completed within adaptive timeout, False if timeout or exception
    - reason: Human-readable diagnostic (includes measured forward time if available)
    
    Adaptive timeout behavior:
    - Measures forward pass time
    - Sets backward timeout = forward_time * BACKWARD_FACTOR * timeout_safety_factor
    - If backward blocks beyond timeout (missing AVX512_FP16, etc.), returns False with diagnostic
    - Prevents invalid data (guessed backward time) from reaching ILP
    """
    result = {"ok": False, "reason": "unknown"}
    execution_result = {
        "forward_completed": False,
        "forward_time_ms": 0.0,
        "backward_completed": False,
        "exception": None
    }
    
    def _run_training_step():
        """Inner function to run training step, measures and sets flags on completion."""
        try:
            model.train()
            model.to("cpu")
            model.to(dtype=torch.float16)

            mini_inp = _build_mini_input_for_cpu_fp16(input_data)
            opt = torch.optim.SGD(model.parameters(), lr=1e-6)
            opt.zero_grad(set_to_none=True)

            # --- FORWARD PASS with measurement ---
            t_forward_start = time.perf_counter()
            if isinstance(mini_inp, dict):
                out = model(**mini_inp)
            else:
                out = model(mini_inp)
            t_forward_end = time.perf_counter()
            execution_result["forward_time_ms"] = (t_forward_end - t_forward_start) * 1000.0
            execution_result["forward_completed"] = True

            # Extract loss
            loss = _extract_loss_for_preflight(out)
            
            # --- BACKWARD PASS (monitored for blocking) ---
            loss.backward()
            
            # --- OPTIMIZER STEP ---
            opt.step()

            execution_result["backward_completed"] = True
        except Exception as e:
            execution_result["exception"] = e

    # Run preflight in a thread
    preflight_thread = threading.Thread(target=_run_training_step, daemon=True)
    preflight_thread.start()

    # PHASE 1: Wait for forward pass to complete (with generous timeout of 60s)
    # This allows us to measure forward_time_ms for adaptive backward timeout
    preflight_thread.join(timeout=60.0)
    
    # PHASE 2: Calculate backward timeout based on measured forward time
    # timeout = forward_time * BACKWARD_FACTOR * safety_factor, with minimum of 10s
    if execution_result["forward_completed"]:
        forward_time_sec = execution_result["forward_time_ms"] / 1000.0
        backward_timeout = max(
            10.0,  # Minimum 10s for backward even if forward is very fast
            forward_time_sec * BACKWARD_FACTOR * timeout_safety_factor
        )
        logger.debug(
            f"FP16 preflight forward time: {execution_result['forward_time_ms']:.2f}ms, "
            f"calculated backward timeout: {backward_timeout:.2f}s "
            f"(formula: {forward_time_sec:.4f}s × {BACKWARD_FACTOR} × {timeout_safety_factor})"
        )
    else:
        # Forward timed out or failed - use minimum timeout
        backward_timeout = 10.0
        logger.warning(
            f"FP16 preflight forward pass did not complete within 60s timeout; "
            f"using minimum backward timeout of {backward_timeout:.2f}s"
        )

    # PHASE 3: Wait for backward+optimizer to complete (or timeout)
    # Thread has already completed forward, now waiting on backward+step
    preflight_thread.join(timeout=backward_timeout)

    # Analyze results
    if execution_result["exception"] is not None:
        # Exception occurred during training step (forward, backward, or optimizer)
        result["ok"] = False
        result["reason"] = f"cpu fp16 training-step preflight failed with exception: {execution_result['exception']}"
    elif execution_result["backward_completed"]:
        # Full training step (forward + backward + optimizer step) completed successfully
        result["ok"] = True
        result["reason"] = (
            f"cpu fp16 training-step preflight succeeded "
            f"(forward={execution_result['forward_time_ms']:.2f}ms, "
            f"backward allowed {backward_timeout:.2f}s timeout)"
        )
    elif execution_result["forward_completed"]:
        # Forward passed but backward timed out
        result["ok"] = False
        result["reason"] = (
            f"cpu fp16 backward pass blocked after {backward_timeout:.2f}s timeout "
            f"(forward took {execution_result['forward_time_ms']:.2f}ms, "
            f"calculated timeout=forward×{BACKWARD_FACTOR}×{timeout_safety_factor}={backward_timeout:.2f}s); "
            f"likely missing AVX512_FP16 ISA flag, insufficient CPU resources, or model too complex for CPU FP16"
        )
    else:
        # Forward didn't complete within 60s initial timeout
        result["ok"] = False
        result["reason"] = (
            f"cpu fp16 forward pass timeout after 60s; "
            f"model layers too large for CPU FP16 on this hardware or severe resource limitation"
        )

    return result

def get_cpu_model():
    cpu_model = platform.processor()
    if not cpu_model or cpu_model == 'x86_64' :
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if "model name" in line:
                        cpu_model = line.strip().split(":")[1].strip()
                        break
        except Exception:
            cpu_model = "Unknown"
    return cpu_model

def get_hardware_metadata() -> Dict[str, Any]:
    """Capture environment details for reproducibility."""
    meta = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "torch_version": torch.__version__,
        "os": platform.platform(),
        "cpu_model": get_cpu_model(),
        "gpu_name": "None",
        "gpu_driver": "None",
        "rapl_available": PYRAPL_AVAILABLE
    }
    if torch.cuda.is_available():
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            meta["gpu_name"] = pynvml.nvmlDeviceGetName(handle)
            meta["gpu_driver"] = pynvml.nvmlSystemGetDriverVersion()
        except Exception as e:
            logger.warning(f"NVML Init failed in metadata check: {e}")
    return meta

def get_tensor_size_recursive(data: Any) -> int:
    """Recursively calculates the payload size (bytes) of complex outputs."""
    size = 0
    try:
        if data is None:
            return 0
        if isinstance(data, torch.Tensor):
            size += data.numel() * data.element_size()
        elif isinstance(data, (tuple, list)):
            for item in data:
                size += get_tensor_size_recursive(item)
        elif isinstance(data, dict):
            for v in data.values():
                size += get_tensor_size_recursive(v)
        elif hasattr(data, 'to_tuple'):
            size += get_tensor_size_recursive(data.to_tuple())
    except Exception:
        pass
    return int(size)

# ========================================================================
# UTILITY: FLOPs ESTIMATION & MICRO-BENCHMARKING
# ========================================================================
def _numel(t: Any) -> int:
    """Helper: Get total elements in tensor."""
    return t.numel() if hasattr(t, 'numel') else 0

def estimate_flops(module: nn.Module, inputs: Any, output: Any) -> float:
    """
    Estimates theoretical FLOPs based on layer geometry.
    Handles non-standard layers gracefully without logging warnings.
    """
    try:
        in_t = inputs[0] if isinstance(inputs, (tuple, list)) and len(inputs) > 0 else inputs
        if not isinstance(in_t, torch.Tensor):
            return 0.0

        # Conv2d layer
        if isinstance(module, nn.Conv2d) and isinstance(output, torch.Tensor):
            try:
                Cin = module.in_channels
                Cout = module.out_channels
                Kx, Ky = module.kernel_size if isinstance(module.kernel_size, tuple) else (module.kernel_size, module.kernel_size)
                Hout, Wout = output.shape[2], output.shape[3]
                return 2.0 * Cout * Hout * Wout * (Cin // module.groups * Kx * Ky)
            except (IndexError, AttributeError, RuntimeError):
                return 0.0

        # Linear layer
        if isinstance(module, nn.Linear):
            try:
                in_f = module.in_features
                out_f = module.out_features
                positions = int(torch.tensor(in_t.shape[:-1]).prod().item()) if in_t is not None else 1
                return 2.0 * positions * in_f * out_f
            except (IndexError, AttributeError, RuntimeError, ValueError):
                return 0.0

        # Activation functions (ReLU, GELU)
        if isinstance(module, (nn.ReLU, nn.GELU)):
            try:
                return float(_numel(in_t))
            except (AttributeError, RuntimeError):
                return 0.0

        # Normalization layers (BatchNorm, LayerNorm)
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
            try:
                return 5.0 * float(_numel(in_t))
            except (AttributeError, RuntimeError):
                return 0.0

        # Attention layers (custom heuristic)
        module_name = module.__class__.__name__.lower()
        if "attention" in module_name and "multi" in module_name:
            try:
                B = in_t.shape[0]
                S = in_t.shape[1] if in_t.ndim >= 3 else 1
                d = in_t.shape[-1]
                return 4.0 * B * S * (d * d) + 2.0 * B * (S * S) * d
            except (IndexError, AttributeError, RuntimeError):
                return 0.0

        # Non-standard or unrecognized layer: return 0.0 silently
        return 0.0

    except Exception:
        # Fallback: any unexpected error returns 0.0 without logging
        return 0.0

class SimpleMLP(nn.Module):
    """Simple MLP for control experiments."""
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
    def forward(self, x): return self.net(x)

# ========================================================================
# ENERGY MONITOR (NVML/RAPL)
# ========================================================================
class EnergyMonitor(threading.Thread):
    """Background thread to sample power usage from NVML (GPU) and RAPL (CPU)."""
    def __init__(self, device_type: str = 'cuda', gpu_id: int = 0, sample_interval: float = 0.05, enable_rapl: bool = False):
        super().__init__()
        self.device_type = device_type
        self.gpu_id = gpu_id
        self.interval = sample_interval
        self.enable_rapl = enable_rapl
        self.stop_event = threading.Event()
        self.readings = []
        self.avg_power = 0.0
        self.nvml_handle = None
        self.cpu_meter = None
        self.daemon = True
        self._init_sensors()

    def _init_sensors(self):
        # NVML Init
        if self.device_type == 'cuda' and torch.cuda.is_available():
            try:
                pynvml.nvmlInit()
                self.nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_id)
            except Exception as e:
                logger.warning(f"NVML Init failed: {e}")
                self.nvml_handle = None
        
        # RAPL Init
        if self.device_type == 'cpu' and PYRAPL_AVAILABLE and self.enable_rapl:
            try:
                pyRAPL.setup() # type: ignore
                self.cpu_meter = pyRAPL.Measurement('cpu_meter') # type: ignore
            except Exception as e:
                logger.warning(f"pyRAPL Init failed: {e}")
                self.cpu_meter = None

    def run(self):
        if self.cpu_meter: 
            try: self.cpu_meter.begin()
            except: pass
            
        while not self.stop_event.is_set():
            # Sample GPU
            if self.device_type == 'cuda' and self.nvml_handle:
                try:
                    p_mw = pynvml.nvmlDeviceGetPowerUsage(self.nvml_handle)
                    self.readings.append(p_mw / 1000.0) # Convert mW to W
                except: 
                    self.readings.append(0.0)
            time.sleep(self.interval)
            
        if self.cpu_meter: 
            try: self.cpu_meter.end()
            except: pass

    def stop(self):
        self.stop_event.set()
        self.join()
        
        # GPU Average Calculation
        if self.device_type == 'cuda' and self.readings:
            self.avg_power = sum(self.readings) / len(self.readings)
            try: pynvml.nvmlShutdown()
            except: pass
            
        # CPU Average Calculation
        elif self.device_type == 'cpu' and self.cpu_meter and self.cpu_meter.result:
            res = self.cpu_meter.result
            # Power (W) = Energy (J) / Time (s)
            # pyRAPL returns energy in microJoules and duration in microSeconds
            if res.duration > 0 and res.pkg is not None and len(res.pkg) > 0:
                self.avg_power = (res.pkg[0] / 1e6) / (res.duration / 1e6)
            else:
                self.avg_power = 0.0

    def get_avg_power(self) -> float:
        return float(self.avg_power) if self.avg_power else 0.0

# ========================================================================
# ADVANCED PROFILER LOGIC
# ========================================================================
class TrainingProfiler:
    def __init__(self, model: nn.Module, model_name: str, args):
        self.model = model
        self.model_name = model_name
        self.args = args
        self.layer_stats = {}
        self.hooks = []
        self._last_opt_step_ms = 0.0
        self._last_opt_step_count = 0
        # Check GPU availability respecting the no-gpu flag
        self.has_gpu = torch.cuda.is_available() and not args.no_gpu
        self.gpu_id = args.gpu_id if self.has_gpu else 0
        if self.has_gpu: 
            self.model.to(f"cuda:{self.gpu_id}")
        else:
            self.model.to("cpu")

    def _get_leaf_modules(self) -> Iterator[Tuple[str, nn.Module]]:
        """Identify atomic modules to avoid double counting."""
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:
                yield name, module

    def _compute_loss(self, out: Any) -> torch.Tensor:
        """Robustly extracts or computes a scalar loss tensor."""
        if hasattr(out, "loss") and out.loss is not None: return out.loss
        if hasattr(out, "logits"): return out.logits.sum()
        if isinstance(out, torch.Tensor): return out.sum()
        if isinstance(out, (tuple, list)) and isinstance(out[0], torch.Tensor): return out[0].sum()
        return torch.tensor(0.0, requires_grad=True)

    def _register_hooks(self, device_type: str):
        """
        Registers PyTorch hooks to measure execution time per layer.
        Distinguishes between CPU Wall Time and GPU Kernel Time.
        """
        for h in self.hooks: h.remove()
        self.hooks = []
        self._tstarts = {}

        def pre_hook(name):
            def hook(module, inp):
                # Always track CPU start for overhead calc
                self._tstarts[name] = {"cpu": time.perf_counter()}
                
                if device_type == 'cuda':
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
                        "flops": 0.0
                    }
                s = self.layer_stats[name]

                if s["count"] == 0:
                    try:
                        params_bytes = sum(p.numel() * p.element_size() for p in module.parameters(recurse=False))
                        s["params_mb"] = params_bytes / (1024**2)
                    except:
                        pass

                    try:
                        s["output_bytes"] = get_tensor_size_recursive(out)
                    except:
                        pass

                    try:
                        s["flops"] = estimate_flops(module, inp, out)
                    except:
                        pass

                # --- Timing ---
                kernel_ms = 0.0
                if device_type == 'cuda':
                    end = torch.cuda.Event(enable_timing=True)
                    end.record()
                    torch.cuda.synchronize()
                    # GPU Time via CUDA Events
                    kernel_ms = self._tstarts[name]["gpu"].elapsed_time(end)
                    # Snapshot Memory
                    s["mem_mb"] = max(s["mem_mb"], torch.cuda.memory_allocated(self.gpu_id) / (1024**2))
                else:
                    # CPU Time via Perf Counter
                    kernel_ms = (cpu_end - self._tstarts[name]["cpu"]) * 1000.0
                    # CPU Memory Proxy (Output size) could be added here if needed

                # Dispatch Overhead = Wall Time - Kernel Time
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
        """Runs the training steps (Forward + Backward + Optimizer Step)."""
        device_str = f"cuda:{self.gpu_id}" if device == "cuda" else "cpu"
        self.model.to(device_str)
        self.model.train()

        if isinstance(input_data, dict):
            inp = {k: v.to(device_str) for k, v in input_data.items()}
        else:
            inp = input_data.to(device_str)

        # Enhanced Optimizer Selection from User Update
        opt_name = getattr(self.args, "optimizer", "SGD")
        lr = getattr(self.args, "lr", 0.01)
        momentum = getattr(self.args, "momentum", 0.9)
        params = self.model.parameters()
        
        if opt_name == "SGD": opt = torch.optim.SGD(params, lr=lr)
        elif opt_name == "SGD_momentum": opt = torch.optim.SGD(params, lr=lr, momentum=momentum)
        elif opt_name == "Adam": opt = torch.optim.Adam(params, lr=lr)
        elif opt_name == "AdamW": opt = torch.optim.AdamW(params, lr=lr)
        elif opt_name == "RMSprop": opt = torch.optim.RMSprop(params, lr=lr, momentum=momentum)
        elif opt_name == "Adagrad": opt = torch.optim.Adagrad(params, lr=lr)
        elif opt_name == "Adadelta": opt = torch.optim.Adadelta(params, lr=lr)
        else: opt = torch.optim.SGD(params, lr=lr)

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
                if isinstance(inp, dict): out = self.model(**inp)
                else: out = self.model(inp)

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
            for h in self.hooks: h.remove()
            self.hooks = []

        total_duration_sec = time.perf_counter() - total_start
        avg_power = monitor.get_avg_power()
        total_energy_j = (avg_power * total_duration_sec) if avg_power > 0 else None

        self._last_opt_step_ms = opt_step_accum_ms
        self._last_opt_step_count = opt_step_count

        return total_energy_j, total_duration_sec

    def _measure_pci_and_overlap(self) -> Dict[str, float]:
        """
        [Advanced Feature]
        Calibrates PCIe bandwidth and measures the Overlap Ratio (Sigma).
        Sigma = 1.0 means perfect overlap capability.
        """
        if not self.has_gpu: return {}
        logger.info("--> Calibrating PCIe Bandwidth & Overlap Ratio (Sigma)...")
        
        dev = f"cuda:{self.gpu_id}"
        size_mb = 256
        numel = int(size_mb * 1024**2 / 4)
        h_tensor = torch.randn(numel).pin_memory()
        
        # 1. Pure Transfer Time (Baseline)
        torch.cuda.synchronize()
        start = time.perf_counter()
        _ = h_tensor.to(dev, non_blocking=True)
        torch.cuda.synchronize()
        t_comm = (time.perf_counter() - start) * 1000.0
        
        # 2. Overlapped Execution (Streaming)
        s_transfer = torch.cuda.Stream()
        # Heavy GEMM to saturate Compute
        a = torch.randn(4096, 4096, device=dev)
        
        # Measure Compute Only
        torch.cuda.synchronize()
        start = time.perf_counter()
        _ = torch.mm(a, a)
        torch.cuda.synchronize()
        t_comp = (time.perf_counter() - start) * 1000.0

        # Measure Concurrent (Compute + Transfer)
        torch.cuda.synchronize()
        start = time.perf_counter()
        # Launch Compute on Default Stream
        _ = torch.mm(a, a)
        # Launch Transfer on Side Stream
        with torch.cuda.stream(s_transfer):
            _ = h_tensor.to(dev, non_blocking=True)
        torch.cuda.synchronize()
        t_overlap = (time.perf_counter() - start) * 1000.0
        
        # Sigma Calculation
        sigma = 1.0 - (max(0, t_overlap - max(t_comm, t_comp)) / (t_comm + 1e-6))
        
        # Fallback Estimation if detailed fails
        alpha_est = 0.05 
        beta_est = size_mb / (t_comm / 1000.0)

        return {
            "pci_bw_mb_s": beta_est,
            "t_comm_ms_base": t_comm,
            "pci_alpha_ms": alpha_est,
            "overlap_ratio_sigma": max(0.0, min(1.0, sigma)),
            "t_comp_ms_base": t_comp,
            "t_overlap_ms": t_overlap
        }

    def _measure_pci_bandwidth_detailed(self) -> Dict[str, float]:
        """
        [Detailed PCIe Calibration]
        Measures H2D and D2H bandwidth separately using 2 data points for alpha/beta.
        """
        if not self.has_gpu: return {}
        logger.info("--> Calibrating Detailed PCIe (H2D vs D2H)...")
        results = {}
        sizes_mb = [10.0, 100.0]
        dev = f"cuda:{self.gpu_id}"

        for direction in ['h2d', 'd2h']:
            times = []
            for sz in sizes_mb:
                numel = int(sz * 1024**2 / 4)
                if direction == 'h2d':
                    src = torch.randn(numel).pin_memory()
                    dst_dev = dev
                else:
                    src = torch.randn(numel, device=dev)
                    dst_dev = 'cpu'
                
                # Warmup
                _ = src.to(dst_dev, non_blocking=True)
                torch.cuda.synchronize()
                
                # Measure
                start = time.perf_counter()
                _ = src.to(dst_dev, non_blocking=True)
                torch.cuda.synchronize()
                times.append((time.perf_counter() - start) * 1000.0)
            
            if times[1] > times[0]:
                beta = (sizes_mb[1] - sizes_mb[0]) / (times[1] - times[0])
                alpha = max(0.0, times[0] - (sizes_mb[0] / beta))
            else:
                beta = 10.0 # Fallback
                alpha = 0.05

            results[f"alpha_{direction}"] = alpha
            results[f"beta_{direction}"] = beta
            
        return results

    def _measure_peak_flops(self, device: str) -> float:
        """
        [Methodology 4] Empirical TFLOPS measurement using GEMM.
        """
        logger.info(f"--> Benchmarking Empirical {device.upper()} TFLOPS...")
        N = 8192 if device == 'cuda' else 2048
        dev_str = f"cuda:{self.gpu_id}" if device == 'cuda' else 'cpu'
        
        try:
            a = torch.randn(N, N, device=dev_str)
            b = torch.randn(N, N, device=dev_str)
            
            # Warmup
            for _ in range(3): torch.mm(a, b)
            if device == 'cuda': torch.cuda.synchronize()
            
            # Measure
            start = time.perf_counter()
            ITER = 5
            for _ in range(ITER): torch.mm(a, b)
            if device == 'cuda': torch.cuda.synchronize()
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
        measure: int
    ) -> None:
        """Persist GPU-only artifacts early to avoid data loss during long CPU profiling."""
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
                "layer_j_per_tflop_gpu": (gpu_layer_energy_j / layer_work_tflops) if (layer_work_tflops > 0 and gpu_layer_energy_j > 0) else 0.0,
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
                "optimizer": opt_name,
                "opt_step_time_ms": getattr(self, "_last_opt_step_ms", 0.0)
            })

        partial_csv_path = os.path.join(self.args.output_dir, f"{self.model_name}_metrics_gpu_partial.csv")
        pd.DataFrame(rows).to_csv(partial_csv_path, index=False)

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
            "measured_peak_tflops_gpu": measured_gpu_peak_tflops
        })

        partial_json_path = os.path.join(self.args.output_dir, f"{self.model_name}_meta_gpu_partial.json")
        with open(partial_json_path, 'w') as f:
            json.dump(partial_meta, f, indent=4)

        logger.info(f"Saved early GPU partial artifacts: {partial_csv_path}, {partial_json_path}")

    def run_profiling(self, input_data: Any):
        """Executes the full profiling sequence."""
        logger.info(f"Starting Profiling Run for: {self.model_name}")
        
        warmup = int(getattr(self.args, "warmup", WARMUP_STEPS))
        measure = int(getattr(self.args, "measure", MEASURE_STEPS))

        # 1. Warmup
        self._run_epoch(input_data, "cuda" if self.has_gpu else "cpu", warmup)

        gpu_total_energy, gpu_run_time_sec = 0.0, 0.0
        gpu_layer_stats = {}
        measured_gpu_peak_tflops = 0.0

        # 2. GPU Profiling
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
                measure=measure
            )

        # 3. CPU Profiling
        cpu_total_energy, cpu_run_time_sec = None, 0.0
        cpu_layer_stats = {}
        measured_cpu_peak_tflops = 0.0

        # CPU FP16 model-level viability preflight (only before CPU profiling)
        # This is moved here to ensure GPU data is already saved before any blocking
        if self.args.precision == "fp16" and not getattr(self.args, "skip_cpu", False):
            model_preflight = run_cpu_fp16_model_preflight(self.model, input_data)
            self.args.cpu_fp16_model_smoke_ok = model_preflight["ok"]
            self.args.cpu_fp16_model_smoke_reason = model_preflight["reason"]
            if not model_preflight["ok"]:
                logger.warning(
                    "CPU FP16 model preflight failed. Skipping CPU profiling. "
                    f"Reason: {model_preflight['reason']}"
                )

        # Update precision execution tracking after preflight
        if self.args.precision == "fp16" and self.args.cpu_fp16_model_smoke_ok is False:
            self.args.cpu_precision_executed = "fp16_requested_model_preflight_failed"

        # Determine if we should skip CPU profiling
        skip_cpu_profile = (
            getattr(self.args, "skip_cpu", False) or  # User requested skip
            (
                self.args.precision == "fp16" and
                getattr(self.args, "cpu_fp16_model_smoke_ok", None) is False
            )
        )


        if skip_cpu_profile:
            logger.warning(
                "Skipping CPU profiling: CPU FP16 model preflight failed and FP32 fallback is disabled."
            )
        else:
            logger.info("--> Profiling CPU Execution...")
            cpu_total_energy, cpu_run_time_sec = self._run_epoch(input_data, "cpu", measure)
            cpu_layer_stats = self.layer_stats.copy()
            self.layer_stats = {}
            measured_cpu_peak_tflops = self._measure_peak_flops("cpu")

        # 4. Global Memory Snapshot
        gpu_peak_mb = 0.0
        if self.has_gpu:
            try:
                with torch.cuda.device(self.gpu_id):
                    torch.cuda.synchronize()
                    gpu_peak_mb = torch.cuda.max_memory_allocated(self.gpu_id) / (1024**2)
            except: pass

        # 5. Calibration (PCIe)
        overlap_stats = self._measure_pci_and_overlap()
        pci_detailed = self._measure_pci_bandwidth_detailed()
        pci_stats = {**overlap_stats, **pci_detailed}
        
        # 6. Metric Compilation
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
            # Fallback to compiled stats
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
                "layer_j_per_tflop_cpu": (cpu_layer_energy_j / layer_work_tflops) if (layer_work_tflops > 0 and cpu_layer_energy_j > 0) else None,
                "transfer_h2d_ms": alpha_h2d + (params_mb / beta_h2d),
                "transfer_d2h_ms": alpha_d2h + (act_mb / beta_d2h),
                "remat_penalty_ms": t_fwd_gpu, 
                "precision_requested": self.args.precision,
                "cpu_precision_executed": self.args.cpu_precision_executed,
                "gpu_precision_executed": self.args.gpu_precision_executed,
                "optimizer": opt_name,
                "opt_step_time_ms": getattr(self, "_last_opt_step_ms", 0.0)
            })

        # Save CSV
        os.makedirs(self.args.output_dir, exist_ok=True)
        csv_path = os.path.join(self.args.output_dir, f"{self.model_name}_metrics.csv")
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        
        # Save Metadata
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
            "params_mb_total": sum(r['params_mb'] for r in rows),
            "grads_mb_total": sum(r['grads_mb'] for r in rows),
            "activations_mb_total": sum(r['activations_mb'] for r in rows),
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
            "total_model_flops": total_model_flops,
            "total_model_flops_per_step": total_model_flops / measure
        })
        
        json_path = os.path.join(self.args.output_dir, f"{self.model_name}_meta.json")
        with open(json_path, 'w') as f:
            json.dump(meta, f, indent=4)

        logger.info(f"Profiling Complete. Data saved to {self.args.output_dir}")

# ========================================================================
# MAIN ENTRY
# ========================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Advanced Profiler for Deep Learning Training")
    parser.add_argument("--model", type=str, required=True, choices=["resnet50", "resnet152", "vit_b16", "bert_base", "gpt2_small", "simple_mlp"], help="Model architecture to profile")
    parser.add_argument("--precision", type=str, default="fp32", choices=["fp32", "fp16", "bf16"], help="Precision mode for profiling")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for input data")
    parser.add_argument("--warmup", type=int, default=WARMUP_STEPS, help="Number of warmup steps")
    parser.add_argument("--measure", type=int, default=MEASURE_STEPS, help="Number of measurement steps")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR, help="Directory to save profiling data")
    parser.add_argument("--no_gpu", action='store_true', help="Disable GPU profiling even if available")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use for profiling")
    parser.add_argument("--rapl", action='store_true', help="Enable RAPL energy measurement on CPU (Linux only)")
    parser.add_argument("--input_size", type=int, default=224, help="Input size (for vision models)")
    parser.add_argument("--seq_length", type=int, default=128, help="Sequence length (for NLP models)")
    parser.add_argument("--optimizer", type=str, default="SGD", choices=list(OPTIMIZER_OVERHEAD_MAP.keys()), help="Optimizer type for overhead estimation")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate (for metadata)")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum (for metadata)")
    parser.add_argument("--skip_cpu", action='store_true', help="Skip CPU profiling entirely")
    parser.add_argument("--num_threads", type=int, default=0, help="Force CPU thread count (0 = auto-detect)")

    args = parser.parse_args()

    configure_cpu_runtime(force_threads=args.num_threads)
    set_determinism()

    args.cpu_fp16_supported = None
    args.cpu_fp16_isa_avx512 = None
    args.cpu_fp16_smoke_test_ok = None
    args.cpu_fp16_model_smoke_ok = None
    args.cpu_fp16_model_smoke_reason = None
    args.cpu_fp16_support_reason = None

    # Precision Setup
    torch_dtype = torch.float32
    if args.precision == "fp16":
        fp16_info = get_cpu_fp16_support_info()
        args.cpu_fp16_supported = fp16_info["supported"]
        args.cpu_fp16_isa_avx512 = fp16_info["isa_avx512_fp16"]
        args.cpu_fp16_smoke_test_ok = fp16_info["smoke_test_ok"]
        args.cpu_fp16_support_reason = fp16_info["reason"]

        if not fp16_info["supported"]:
            logger.warning(
                "CPU FP16 requested but functional support is not available. "
                "Continuing without FP32 fallback. "
                f"Reason: {fp16_info['reason']}"
            )
        elif not fp16_info["isa_avx512_fp16"]:
            logger.warning(
                "CPU FP16 is functionally available, but AVX512_FP16 was not detected. "
                "Execution may use a slower path."
            )

        torch_dtype = torch.float16
    elif args.precision == "bf16":
        if cpu_supports_bf16():
            torch_dtype = torch.bfloat16
        else:
            logger.warning("CPU does not support BF16. Falling back to FP32.")
            torch_dtype = torch.float32

    # Model Selection
    logger.info(f"Initializing {args.model} with batch size {args.batch_size}...")

    if args.model == "resnet50":
        weights = ResNet50_Weights.DEFAULT
        model = resnet50(weights=weights).to(dtype=torch_dtype)
        inp = torch.randn((args.batch_size, 3, args.input_size, args.input_size), dtype=torch_dtype)
    elif args.model == "resnet152":
        weights = ResNet152_Weights.DEFAULT
        model = resnet152(weights=weights).to(dtype=torch_dtype)
        inp = torch.randn((args.batch_size, 3, args.input_size, args.input_size), dtype=torch_dtype)
    elif args.model == "vit_b16":
        weights = ViT_B_16_Weights.DEFAULT
        model = vit_b_16(weights=weights).to(dtype=torch_dtype)
        inp = torch.randn((args.batch_size, 3, args.input_size, args.input_size), dtype=torch_dtype)
    elif args.model == "bert_base":
        model = BertModel.from_pretrained("bert-base-uncased")
        inp = torch.randint(0, 1000, (args.batch_size, args.seq_length), dtype=torch.long)
    elif args.model == "gpt2_small":
        model = GPT2Model.from_pretrained("gpt2")
        inp = torch.randint(0, 1000, (args.batch_size, args.seq_length), dtype=torch.long)
    elif args.model == "simple_mlp":
        model = SimpleMLP().to(dtype=torch_dtype)
        inp = torch.randn((args.batch_size, 784), dtype=torch_dtype)
    else:
        raise ValueError(f"Unsupported model: {args.model}")
    
    # Apply Precision
    if args.precision in ["fp16", "bf16"] and args.model not in ["bert_base", "gpt2_small"]:
        # HF models handle precision via dtype in from_pretrained usually, or .to()
        if isinstance(inp, torch.Tensor):
            inp = inp.to(dtype=torch_dtype)
            # For NLP inputs (int64 token IDs), no cast needed

    # Precision Execution Tracking (Initial - before preflight)
    # Note: Final tracking with preflight result will be set inside run_profiling()
    if args.precision == "bf16" and not cpu_supports_bf16():
        args.cpu_precision_executed = "fp32_fallback"
    elif args.precision == "fp16" and args.cpu_fp16_supported is False:
        args.cpu_precision_executed = "fp16_requested_no_cpu_support"
    else:
        args.cpu_precision_executed = args.precision
    args.gpu_precision_executed = args.precision

    TrainingProfiler(model, args.model, args).run_profiling(inp)