You’re absolutely right, Luis — the English manual should be a **direct translation of the Spanish one**, with the same length, structure, and detail. Let me give you the **full consolidated English version**, section by section, including the Data Dictionary in proper table format.

---

# User Manual – Advanced Hybrid Profiler

## 1. Purpose of the Script
This profiler characterizes deep learning architectures and generates metrics for integration into an **Integer Linear Programming (ILP)** model. It produces, per layer and per device (CPU/GPU), measurements of time, FLOPs, energy, memory, and transfer, along with global metadata that ensures reproducibility and traceability. Instrumentation is safe and idempotent (NVML/RAPL), determinism is enforced, and a GEMM benchmark is used to estimate empirical peak TFLOPS.

---

## 2. Prerequisites

### Hardware
- NVIDIA GPU with CUDA support for kernel metrics and GPU energy (NVML).  
- Linux CPU with `/sys/class/powercap` access for RAPL energy measurement (optional).  
- For bf16 on CPU: AVX512-BF16 support; otherwise automatic fallback to fp32.  

### Software
- Python ≥ 3.9; PyTorch ≥ 2.0; TorchVision; Transformers (HuggingFace).  
- Pandas, NumPy, psutil.  
- pynvml (`nvidia-ml-py`).  
- pyRAPL (optional, Linux). Requires read permissions on `/sys/class/powercap/intel-rapl:0/energy_uj`.  

Typical installation:
```bash
conda install pytorch torchvision -c pytorch
pip install transformers pandas psutil nvidia-ml-py pyRAPL
```

---

## 3. Execution

### Basic command
```bash
python src/profiler.py --model resnet50
```

### Available arguments
- `--model`: resnet50, resnet152, vit, bert, gpt2, mlp  
- `--batch-size`: batch size (default: 8)  
- `--seq-len`: sequence length for NLP (default: 128; applies to HuggingFace models)  
- `--precision`: fp32, fp16, bf16  
- `--warmup`: warmup iterations (default: 5)  
- `--measure`: measurement iterations (default: 15)  
- `--output-dir`: output directory for CSV/JSON (default: data)  
- `--gpu-id`: GPU index (default: 0)  
- `--no-gpu`: force CPU-only execution  
- `--rapl`: enable CPU energy measurement via pyRAPL  
- `--nvml-sample-interval`: NVML sampling interval in seconds (default: 0.05)  
- `--gpu-gemm-n`, `--cpu-gemm-n`: GEMM size N for TFLOPS benchmark (default: 8192 / 2048)  

Examples:
```bash
# GPU + CPU, bf16 on CPU with support check, CPU energy measurement
python src/profiler.py --model bert --batch-size 16 --precision bf16 --rapl

# CPU only, fp32
python src/profiler.py --model mlp --no-gpu --precision fp32
```

---

## 4. Internal Workflow

- Determinism: global seeds and PyTorch flags; fallback if deterministic algorithms cannot be enforced.  
- Factory and casting: model creation and synthetic data; floating tensors converted to fp16/bf16 as requested. For bf16 on CPU, AVX512-BF16 support is checked; otherwise fp32 is used.  
- Hooks per layer (leaf-only): pre/post hooks measure:  
  - GPU kernel time (CUDA Events) and CPU wall-clock time.  
  - Framework overhead: `dispatch_ms = max(0, wall_ms - kernel_ms)`.  
  - Peak memory (global CUDA allocator proxy) and output size (payload for PCIe).  
  - Theoretical FLOPs by geometry (Conv, Linear, activations, norm; heuristic for attention).  
- PCIe calibration: estimates α (fixed latency) and β (MB/ms) for H2D/D2H using pinned memory.  
- Energy:  
  - GPU: NVML periodic reads, average power (W) and total energy; idempotent shutdown.  
  - CPU: pyRAPL (optional) computes average power from µJ energy and µs duration; if unavailable, CPU energy is reported as None in metadata and 0.0 in CSV columns.  
- TFLOPS benchmark: GEMM with dtype according to requested precision and safe fallbacks; empirical peak reported per device.  
- Output: per-layer CSV and global JSON metadata.  

---

## 5. Output and Data Schema

### Per-layer CSV
Location: `data/{model_name}_metrics.csv`

### Global JSON metadata
Location: `data/{model_name}_meta.json`

---

## 6. Data Dictionary

### Per-layer CSV (`{model_name}_metrics.csv`)

| Column | Description |
|--------|-------------|
| **layer** | Leaf module name (e.g., `conv1`, `fc`) |
| **type** | Layer type (`Conv2d`, `Linear`, `ReLU`, etc.) |
| **params_mb** | Parameter size in MB |
| **grads_mb** | Gradient size in MB (≈ params_mb) |
| **optimizer_states_mb** | Optimizer state size in MB (`params_mb × factor`) |
| **theoretical_flops** | Theoretical FLOPs (forward) |
| **tflops** | Effective performance in TFLOPS |
| **efficiency_ratio** | Efficiency ratio = tflops / measured peak |
| **activations_mb** | Output activations size in MB |
| **gpu_fwd_time_ms** | GPU forward time (ms, CUDA Events) |
| **gpu_bwd_time_ms** | GPU backward time (ms, heuristic = 2× forward) |
| **gpu_fwd_energy_j** | GPU forward energy (J, proportional to time) |
| **gpu_bwd_energy_j** | GPU backward energy (J, heuristic = 2× forward) |
| **gpu_mem_peak_mb** | Peak GPU memory (global proxy, MB) |
| **layer_j_per_tflop_gpu** | Energy per TFLOP in GPU (J/TFLOP) |
| **dispatch_overhead_ratio** | GPU framework overhead ratio = dispatch_ms / fwd_ms |
| **cpu_fwd_time_ms** | CPU forward time (ms, wall‑clock) |
| **cpu_bwd_time_ms** | CPU backward time (ms, heuristic = 2× forward) |
| **cpu_fwd_energy_j** | CPU forward energy (J, proportional to time; 0.0 if RAPL unavailable) |
| **cpu_bwd_energy_j** | CPU backward energy (J, heuristic = 2× forward; 0.0 if RAPL unavailable) |
| **cpu_mem_mb** | CPU memory proxy (activations MB) |
| **layer_j_per_tflop_cpu** | Energy per TFLOP in CPU (J/TFLOP; None if RAPL unavailable) |
| **transfer_h2d_ms** | Estimated Host→Device transfer time (α + MB/β); uses `params_mb` as payload proxy |
| **transfer_d2h_ms** | Estimated Device→Host transfer time (α + MB/β); uses `activations_mb` as payload proxy |
| **precision_requested** | Requested precision (`fp32`, `fp16`, `bf16`) |
| **cpu_precision_executed** | Effective CPU precision (e.g., `fp32_fallback`) |
| **gpu_precision_executed** | Effective GPU precision |
| **optimizer** | Optimizer used (`SGD`, `Adam`, etc.) |
| **opt_step_time_ms** | Average optimizer step time (ms) |

---

### Global JSON metadata (`{model_name}_meta.json`)

| Key | Description |
|-----|-------------|
| **timestamp** | Execution date/time |
| **python_version** | Python version |
| **torch_version** | PyTorch version |
| **os** | Operating system |
| **cpu_model** | Detected CPU model |
| **gpu_name** | Detected GPU name |
| **gpu_driver** | GPU driver version |
| **rapl_available** | Boolean: RAPL availability |
| **nvml_status** | NVML state (`initialized`, `last_error`) |
| **model** | Profiled model name |
| **layers_profiled_count** | Number of leaf layers profiled |
| **precision_mode** | Requested precision |
| **cpu_precision**, **gpu_precision** | Effective CPU/GPU precision |
| **gpu_total_layer_time_ms** | Sum of GPU forward times across layers |
| **cpu_total_layer_time_ms** | Sum of CPU forward times across layers |
| **gpu_step_time_ms**, **cpu_step_time_ms** | Average step time GPU/CPU |
| **framework_overhead_gpu_ms**, **framework_overhead_cpu_ms** | Global overhead GPU/CPU |
| **framework_overhead_ratio_gpu**, **framework_overhead_ratio_cpu** | Global overhead ratios |
| **framework_overhead_vector** | Per‑layer overhead vector (GPU semantics) |
| **energy_total_gpu_j**, **energy_total_cpu_j** | Total GPU/CPU energy (CPU may be None) |
| **energy_avg_per_step_gpu_j**, **energy_avg_per_step_cpu_j** | Average energy per step |
| **energy_distribution_vector** | Energy distribution per layer (normalized shares) |
| **gpu_mem_peak_mb_global** | Global GPU peak memory |
| **gpu_mem_reserved_mb_global** | Global GPU reserved memory
| **total_model_flops_per_step** | Total FLOPs per training step (forward) divided by `measure` |
| **optimizer_step_time_total_ms**, **optimizer_step_time_avg_ms** | Total and average `optimizer.step()` time over measured iterations |