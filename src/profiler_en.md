# User Manual – Advanced Hybrid Profiler

## 1. Purpose of the Script
This profiler characterizes deep learning architectures and generates metrics required for optimization via **Integer Linear Programming (ILP)**.  
It captures per-layer:

- **FLOPs** (computational complexity)  
- **Memory footprint**  
  - **GPU VRAM**: global peak memory snapshot via CUDA (conservative proxy, includes fragmentation and buffers)  
  - **CPU memory**: approximated using activations as proxy and process RSS/USS delta  
  - **Optimizer states**: explicitly calculated as `params_mb × factor` (e.g., Adam m,v)  
- **Transfer payload**: exact output tensor size (critical for PCIe modeling)  
- **Execution time**  
  - **GPU kernel time** (CUDA Events)  
  - **CPU wall-clock time** (perf_counter)  
  - **Framework overhead**: per-layer vector with `dispatch_overhead_ms` and `dispatch_overhead_ratio`  
- **Energy consumption**  
  - GPU via NVML  
  - CPU via RAPL (if available)  
  - Energy distribution per layer proportional to compute time, normalized to 1.0 per device  
- **Empirical efficiency**  
  - TFLOPS per layer and global weighted average  
  - Efficiency ratio relative to measured peak  
  - Energy per TFLOP (`layer_j_per_tflop_gpu`, `layer_j_per_tflop_cpu`)  

---

## 2. Prerequisites

### Hardware
- NVIDIA GPU with CUDA support (for GPU metrics and NVML)  
- CPU with access to `/sys/class/powercap` (Linux) for RAPL energy measurement  
- For **bf16 on CPU**, AVX512-BF16 support required; otherwise automatic fallback to fp32  

### Software
- Python ≥ 3.9  
- PyTorch ≥ 2.0  
- TorchVision  
- Transformers (HuggingFace)  
- Pandas, NumPy, psutil  
- pynvml (`nvidia-ml-py`)  
- pyRAPL (optional, Linux only): sudo chmod a+r /sys/class/powercap/intel-rapl:0/energy_uj 

Typical installation:
```bash
conda install pytorch torchvision -c pytorch
pip install transformers pandas psutil nvidia-ml-py pyRAPL
```

---

## 3. Running the Script

### Basic command
```bash
python src/profiler.py --model resnet50
```

### Available arguments
- **`--model`**: architecture to profile. Options:  
  - `resnet50`, `resnet152`, `vit`, `bert`, `gpt2`, `mlp`
- **`--batch-size`**: batch size (default: 4)  
- **`--seq-len`**: sequence length (for NLP models, default: 128)  
- **`--precision`**: numeric precision. Options: `fp32`, `fp16`, `bf16`  
- **`--warmup`**: warmup iterations (default: 5)  
- **`--measure`**: measurement iterations (default: 15)  
- **`--output-dir`**: output directory for CSV (default: `data`)  
- **`--gpu-id`**: GPU index (default: 0)  
- **`--no-gpu`**: force CPU-only execution  

Example:
```bash
python src/profiler.py --model bert --batch-size 16 --seq-len 256 --precision bf16
```

---

## 4. Internal Workflow

1. **Determinism**: seeds and flags set for reproducibility  
2. **Factory**: builds the model and synthetic input data  
3. **Precision handling**:  
   - Converts model and inputs to fp16/bf16 if requested  
   - Only floating-point tensors are converted (e.g., `attention_mask`), never `input_ids`  
   - If CPU does not support bf16, automatic fallback to fp32 is applied  
4. **Profiling loops**:  
   - Runs training iterations on GPU and CPU separately  
   - Captures time, memory, and energy metrics  
   - Estimates FLOPs and transfer payloads per layer  
   - Captures per-layer framework overhead (`dispatch_overhead_ms`, `dispatch_overhead_ratio`)  
5. **PCIe transfer calibration**: runtime calibration of α and β parameters for H2D/D2H transfers using pinned memory  
6. **CSV/JSON output**: saves per-layer metrics and global metadata  

---

## 5. Output

### CSV file
Generated at `data/{model_name}_metrics.csv`.  
Main columns:

- **layer**, **type**  
- **params_mb**, **grads_mb**, **optimizer_states_mb**  
- **theoretical_flops**, **tflops**, **efficiency_ratio**  
- **activations_mb**  
- **gpu_fwd_time_ms**, **gpu_bwd_time_ms**, **gpu_fwd_energy_j**, **gpu_bwd_energy_j**  
- **gpu_mem_peak_mb** (global proxy)  
- **cpu_fwd_time_ms**, **cpu_bwd_time_ms**, **cpu_fwd_energy_j**, **cpu_bwd_energy_j**  
- **cpu_mem_mb** (proxy activations)  
- **layer_j_per_tflop_gpu**, **layer_j_per_tflop_cpu**  
- **transfer_h2d_ms**, **transfer_d2h_ms**  
- **dispatch_overhead_ms**, **dispatch_overhead_ratio**  
- **precision_requested**, **cpu_precision_executed**, **gpu_precision_executed**

### JSON metadata
Includes:
- Hardware/software info (`get_hardware_metadata()`)  
- Layer count, precision used  
- Total times and global overhead  
- Total energy and average per step  
- Normalized energy distribution vectors  
- Global memory (GPU reserved, CPU USS/PSS)  
- PCIe α and β parameters  
- Simple and weighted average TFLOPS  
- Global energy efficiency (J/TFLOP)  
- Integrity ratios (e.g., sum of per-layer energy vs total)  

---

## 6. Example CSV Output (Commented)

```
layer,type,params_mb,grads_mb,optimizer_states_mb,theoretical_flops,tflops,efficiency_ratio,
activations_mb,gpu_fwd_time_ms,gpu_bwd_time_ms,gpu_fwd_energy_j,gpu_bwd_energy_j,
gpu_mem_peak_mb,layer_j_per_tflop_gpu,dispatch_overhead_ms,dispatch_overhead_ratio,
cpu_fwd_time_ms,cpu_bwd_time_ms,cpu_fwd_energy_j,cpu_bwd_energy_j,cpu_mem_mb,layer_j_per_tflop_cpu,
transfer_h2d_ms,transfer_d2h_ms,precision_requested,cpu_precision_executed,gpu_precision_executed

conv1,Conv2d,0.0179,0.0179,0.0358,236027904,1.03,0.45,
6.125,1.03,2.06,0.024,0.048,
420.48,0.023,0.12,0.11,
143.48,286.96,10.20,20.40,1.416,0.071,
0.92,0.92,fp16,fp16,fp16
```

---

## 7. Limitations

- **CPU energy**: requires `/sys/class/powercap`; otherwise reported as `NaN`  
- **GPU memory per layer**: global proxy, not strictly isolated  
- **CPU memory**: approximated using activations as proxy  
- **Backward pass**: estimated as `T_bwd = 2 × T_fwd` unless backward hooks are enabled  
- **PCIe transfers**: α and β are hardware-specific, calibrated at runtime  

---

## 8. Best Practices

- Always run in a controlled environment (same GPU/CPU) for comparability  
- Document any fallback (e.g., bf16→fp32 on CPU)  
- Use batch sizes and sequence lengths representative of real workloads  
- Save CSV outputs together with hardware metadata (`get_hardware_metadata()`)  
- Validate integrity: sum of per-layer energy ≈ total energy, weighted TFLOPS consistent  

---




# ILP Correspondence Table – Profiler Metrics to Optimization Variables

| **Profiler Metric (CSV/JSON)** | **ILP Variable / Constraint** | **Interpretation in ILP Model** |
|--------------------------------|-------------------------------|---------------------------------|
| **layer** / **type** | Node identifier | Each profiled layer is a node in the ILP graph. |
| **params_mb** | \(M^{params}_i\) | Parameter memory requirement for layer \(i\). |
| **grads_mb** | \(M^{grads}_i\) | Gradient memory requirement for layer \(i\). |
| **optimizer_states_mb** | \(M^{opt}_i\) | Optimizer state memory (e.g., Adam m,v) for layer \(i\). |
| **activations_mb** | \(M^{act}_i\) | Activation persistence memory for layer \(i\). |
| **gpu_mem_peak_mb** | \(M^{gpu}_{peak}\) | Global GPU memory constraint (conservative proxy). |
| **cpu_mem_mb** | \(M^{cpu}_i\) | CPU memory proxy for layer \(i\). |
| **theoretical_flops** | \(F_i\) | Computational work (FLOPs) for layer \(i\). |
| **tflops** | \(Perf^{gpu}_i\) | Achieved TFLOPS rate for layer \(i\) on GPU. |
| **efficiency_ratio** | \(Eff^{gpu}_i\) | Efficiency ratio relative to measured GPU peak TFLOPS. |
| **layer_j_per_tflop_gpu** | \(E^{gpu}_i\) | Energy cost per TFLOP for layer \(i\) on GPU. |
| **layer_j_per_tflop_cpu** | \(E^{cpu}_i\) | Energy cost per TFLOP for layer \(i\) on CPU. |
| **gpu_fwd_time_ms**, **gpu_bwd_time_ms** | \(T^{gpu}_{fwd,i}, T^{gpu}_{bwd,i}\) | Forward/backward execution time for layer \(i\) on GPU. |
| **cpu_fwd_time_ms**, **cpu_bwd_time_ms** | \(T^{cpu}_{fwd,i}, T^{cpu}_{bwd,i}\) | Forward/backward execution time for layer \(i\) on CPU. |
| **gpu_fwd_energy_j**, **gpu_bwd_energy_j** | \(En^{gpu}_{fwd,i}, En^{gpu}_{bwd,i}\) | Forward/backward energy cost for layer \(i\) on GPU. |
| **cpu_fwd_energy_j**, **cpu_bwd_energy_j** | \(En^{cpu}_{fwd,i}, En^{cpu}_{bwd,i}\) | Forward/backward energy cost for layer \(i\) on CPU. |
| **transfer_h2d_ms**, **transfer_d2h_ms** | \(T^{h2d}_i, T^{d2h}_i\) | Transfer time for activations of layer \(i\) (PCIe α–β model). |
| **dispatch_overhead_ms** | \(O^{dispatch}_i\) | Framework overhead per layer (Python/PyTorch dispatch). |
| **dispatch_overhead_ratio** | \(O^{ratio}_i\) | Overhead ratio relative to kernel time for layer \(i\). |
| **gpu_total_layer_time_ms**, **cpu_total_layer_time_ms** | \(\sum T^{gpu}_i, \sum T^{cpu}_i\) | Total profiled kernel times across all layers. |
| **gpu_step_time_ms**, **cpu_step_time_ms** | \(T^{gpu}_{step}, T^{cpu}_{step}\) | Average end-to-end step time per device. |
| **framework_overhead_gpu_ms**, **framework_overhead_cpu_ms** | \(O^{gpu}_{global}, O^{cpu}_{global}\) | Global framework overhead per device. |
| **framework_overhead_ratio_gpu**, **framework_overhead_ratio_cpu** | \(O^{gpu}_{ratio}, O^{cpu}_{ratio}\) | Ratio of overhead to total step time per device. |
| **energy_total_gpu_j**, **energy_total_cpu_j** | \(En^{gpu}_{total}, En^{cpu}_{total}\) | Total energy consumption per device. |
| **energy_distribution_vector** | \(\alpha^{gpu}_i, \alpha^{cpu}_i\) | Normalized energy shares per layer (sum = 1.0 per device). |
| **weighted_tflops_avg** | \(Perf^{avg}_{weighted}\) | Sustained performance metric (Total FLOPs / Total Time). |
| **energy_efficiency_gpu**, **energy_efficiency_cpu** | \(En^{gpu}_{eff}, En^{cpu}_{eff}\) | Global energy efficiency (J/TFLOP). |
| **precision_requested**, **cpu_precision_executed**, **gpu_precision_executed** | Precision constraints | Actual numeric precision used in execution. |

---

## Notes for ILP Integration
- **Node constraints:** Each layer \(i\) has memory, time, energy, and transfer costs.  
- **Device assignment variables:** ILP decides whether layer \(i\) runs on GPU or CPU.  
- **Global constraints:**  
  - GPU memory ≤ \(M^{gpu}_{peak}\)  
  - CPU memory ≤ process limits  
  - Energy budget ≤ \(En^{device}_{total}\)  
- **Objective functions:**  
  - Minimize total step time (including overhead and transfers).  
  - Minimize energy consumption (weighted by J/TFLOP).  
  - Balance efficiency ratio across devices.  

---
