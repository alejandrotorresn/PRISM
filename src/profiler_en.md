# User Manual – Advanced Hybrid Profiler

## Purpose of the Script
This profiler characterizes deep learning architectures and generates metrics for integration into an Integer Linear Programming (ILP) model. It produces, per layer and per device (CPU/GPU), measurements of time, FLOPs, energy, memory, and transfer, plus global metadata for reproducibility. Instrumentation uses NVML/pyRAPL safely, enforces determinism, and runs a GEMM benchmark to estimate empirical peak TFLOPS.

---

## Prerequisites

### Hardware
- NVIDIA GPU with CUDA support for kernel metrics and GPU energy (NVML).
- Linux CPU with `/sys/class/powercap` access for RAPL energy (optional).
- For fp16/bf16 on CPU: accelerated ISA is probed first. fp16 requires AVX512-FP16; bf16 requires AVX512-BF16 or AMX_BF16+AMX_TILE. If accelerated support is missing, profiling is skipped and reported in CSV/JSON.

### Software
- Python ≥ 3.9; PyTorch ≥ 2.0; TorchVision; Transformers (HuggingFace).
- pandas, NumPy, psutil.
- pynvml (`nvidia-ml-py`).
- pyRAPL (optional, Linux) with read access to `/sys/class/powercap/intel-rapl:0/energy_uj`.

Typical installation:
```bash
conda install pytorch torchvision -c pytorch
pip install transformers pandas psutil nvidia-ml-py pyRAPL
```

---

## Execution

### Basic command
```bash
python src/profiler.py --model resnet50
```

### Available arguments
- `--model`: resnet50, resnet152, vit_b16, bert_base, gpt2_small, simple_mlp
- `--precision`: fp32, fp16, bf16
- `--batch_size`: batch size (default: 8)
- `--warmup`: warmup iterations (default: 5)
- `--measure`: measurement iterations (default: 15)
- `--output_dir`: output directory for CSV/JSON (default: data)
- `--no_gpu`: force CPU-only execution
- `--gpu_id`: GPU index (default: 0)
- `--rapl`: enable CPU energy measurement via pyRAPL
- `--input_size`: input size for vision models (default: 224)
- `--seq_length`: sequence length for NLP (default: 128)
- `--optimizer`: SGD, SGD_momentum, Adam, AdamW, RMSprop, Adagrad, Adadelta
- `--lr`: learning rate for metadata (default: 0.01)
- `--momentum`: momentum (default: 0.9; applies where relevant)

Examples:
```bash
# GPU + CPU, bf16 on CPU with support check, CPU energy measurement
python src/profiler.py --model bert_base --batch_size 16 --precision bf16 --rapl

# CPU only, fp32
python src/profiler.py --model simple_mlp --no_gpu --precision fp32
```

- Quick smoke test (CPU-only, outputs to data/test):
```bash
python src/profiler.py --model simple_mlp --no_gpu --precision fp32 --warmup 1 --measure 2 --output_dir data/test
```

---

## Internal Workflow

- Determinism: global seeds and PyTorch flags; fallback if deterministic algorithms cannot be forced.
- Factory and casting: model creation and synthetic data; before execution, a CPU ISA precision policy is evaluated. If the requested precision has no accelerated path, training/profiling is not executed and skip-status artifacts are saved.
- Per-layer hooks (leaf-only): pre/post hooks measure GPU kernel time (CUDA Events) or CPU wall-clock, dispatch overhead (`dispatch_ms = max(0, wall_ms - kernel_ms)`), peak memory (global proxy), output size (PCIe payload), and theoretical FLOPs by geometry (Conv, Linear, activations, norm; attention heuristic).
- PCIe calibration: estimates α/β for H2D and D2H; parameters are used as the H2D payload proxy and activations as the D2H payload proxy.
- Energy: NVML for GPU; optional pyRAPL for CPU. If RAPL is unavailable or fails, CPU energy is `None` in metadata and `0.0` in the per-layer CSV.
- TFLOPS benchmark: GEMM with dtype per requested precision; reports empirical peak per device.
- Output: per-layer CSV and global JSON metadata.

---

## Output and Data Schema

### Per-layer CSV
Location: `data/{model_name}_metrics.csv`

### Global JSON metadata
Location: `data/{model_name}_meta.json`

---

## Data Dictionary

### Per-layer CSV (`{model_name}_metrics.csv`)

| Column | Description |
|--------|-------------|
| **layer** | Leaf module name (e.g., `conv1`, `fc`) |
| **type** | Layer type (`Conv2d`, `Linear`, `ReLU`, etc.) |
| **params_mb** | Parameter size in MB |
| **grads_mb** | Gradient size in MB (≈ params_mb) |
| **optimizer_states_mb** | Optimizer state size in MB (`params_mb × factor`) |
| **activations_mb** | Output activations size in MB |
| **theoretical_flops** | Theoretical FLOPs (forward) |
| **tflops** | Effective performance in TFLOPS |
| **efficiency_ratio** | tflops / measured peak |
| **gpu_fwd_time_ms** | GPU forward time (ms, CUDA Events) |
| **gpu_bwd_time_ms** | GPU backward time (ms, heuristic = 2× forward) |
| **gpu_fwd_energy_j** | GPU forward energy (J, proportional to time) |
| **gpu_bwd_energy_j** | GPU backward energy (J, heuristic = 2× forward) |
| **gpu_mem_peak_mb** | Peak GPU memory (global proxy, MB) |
| **layer_j_per_tflop_gpu** | Energy per TFLOP on GPU (J/TFLOP) |
| **dispatch_overhead_ratio** | GPU framework overhead = dispatch_ms / fwd_ms |
| **cpu_fwd_time_ms** | CPU forward time (ms, wall-clock) |
| **cpu_bwd_time_ms** | CPU backward time (ms, heuristic = 2× forward) |
| **cpu_fwd_energy_j** | CPU forward energy (J; 0.0 if RAPL unavailable) |
| **cpu_bwd_energy_j** | CPU backward energy (J; 0.0 if RAPL unavailable) |
| **cpu_mem_mb** | CPU memory proxy (activations MB) |
| **layer_j_per_tflop_cpu** | Energy per TFLOP on CPU (J/TFLOP; None if RAPL unavailable) |
| **transfer_h2d_ms** | Estimated Host→Device time (α + params_mb / β) |
| **transfer_d2h_ms** | Estimated Device→Host time (α + activations_mb / β) |
| **remat_penalty_ms** | Rematerialization penalty (≈ GPU forward time) |
| **precision_requested** | Requested precision (`fp32`, `fp16`, `bf16`) |
| **cpu_precision_executed** | Effective CPU precision (includes unsupported states, e.g., `fp16_requested_isa_unsupported`) |
| **gpu_precision_executed** | Effective GPU precision |
| **run_executed** | Boolean: `true` if profiling executed, `false` if skipped |
| **skip_unsupported_precision** | Boolean: `true` when skipped due to missing accelerated ISA |
| **skip_reason** | Detailed skip reason |
| **optimizer** | Optimizer used |
| **opt_step_time_ms** | Accumulated `optimizer.step()` time within the measurement window (ms) |

---

### Global JSON metadata (`{model_name}_meta.json`)

| Key | Description |
|-----|-------------|
| **timestamp** | Execution date/time |
| **torch_version** | PyTorch version |
| **os** | Operating system |
| **cpu_model** | Detected CPU model |
| **gpu_name** | Detected GPU name |
| **gpu_driver** | GPU driver version |
| **rapl_available** | Boolean: RAPL availability |
| **model** | Profiled model name |
| **layers_profiled_count** | Number of leaf layers profiled |
| **precision_mode** | Requested precision |
| **execution_status** | Execution status (`completed` or `skipped_unsupported_precision`) |
| **execution_skip_reason** | Skip reason (if applicable) |
| **cpu_instruction_flags** | Detected CPU ISA flags (`/proc/cpuinfo`) |
| **cpu_isa_probe** | Structured ISA probe result used for precision decisions |
| **cpu_precision**, **gpu_precision** | Effective CPU/GPU precision |
| **gpu_total_layer_time_ms** | Sum of GPU forward times across layers |
| **cpu_total_layer_time_ms** | Sum of CPU forward times across layers |
| **gpu_step_time_ms**, **cpu_step_time_ms** | Average step time GPU/CPU |
| **framework_overhead_gpu_ms**, **framework_overhead_cpu_ms** | Global overhead GPU/CPU |
| **framework_overhead_ratio_gpu**, **framework_overhead_ratio_cpu** | Global overhead ratios |
| **framework_overhead_vector** | Per-layer overhead vector (GPU semantics) |
| **energy_total_gpu_j**, **energy_total_cpu_j** | Total GPU/CPU energy (CPU may be None) |
| **energy_avg_per_step_gpu_j**, **energy_avg_per_step_cpu_j** | Average energy per step |
| **energy_distribution_vector** | Energy distribution per layer (normalized shares) |
| **gpu_mem_peak_mb_global** | Global GPU peak memory |
| **gpu_mem_reserved_mb_global** | Global GPU reserved memory |
| **cpu_uss_mb_global**, **cpu_pss_mb_global** | Global CPU USS/PSS |
| **params_mb_total**, **grads_mb_total**, **activations_mb_total** | Totals of params, grads, activations |
| **optimizer_state_mb_factor_fallback**, **optimizer_state_mb_factor_used** | Fallback and used factor for optimizer state size |
| **transfer_alpha_h2d**, **transfer_beta_h2d** | PCIe α/β for H2D |
| **transfer_alpha_d2h**, **transfer_beta_d2h** | PCIe α/β for D2H |
| **pcie_stats_raw** | Raw PCIe calibration results |
| **measured_peak_tflops_gpu**, **measured_peak_tflops_cpu** | Empirical peak TFLOPS GPU/CPU |
| **efficiency_ratio_avg**, **efficiency_ratio_vector** | Average and per-layer efficiency ratios |
| **avg_tflops_per_layer**, **weighted_avg_tflops_per_layer** | Simple and weighted TFLOPS averages |
| **energy_efficiency_j_per_tflop_gpu**, **energy_efficiency_j_per_tflop_cpu** | Global energy efficiency (J/TFLOP) |
| **optimizer_used**, **optimizer_lr**, **optimizer_momentum** | Optimizer and parameters |
| **optimizer_step_time_total_ms**, **optimizer_step_time_avg_ms** | Total and average `optimizer.step()` time |
| **total_model_flops**, **total_model_flops_per_step** | Total FLOPs accumulated and per step (forward divided by `measure`) |