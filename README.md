[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1%2B-red)](https://pytorch.org)

# Advanced Hybrid Profiler for Deep Learning Training

Comprehensive profiling framework for characterizing neural network architectures to generate cost metrics required by Integer Linear Programming (ILP) optimization models in GPU-CPU system research.

## Overview

This profiler measures per-layer execution characteristics of deep learning models across CPU and GPU devices:

- **Execution Time**: Forward, backward, and total latency per layer
- **Energy**: GPU energy (NVML), CPU energy (RAPL), and per-layer attribution
- **Compute**: Theoretical FLOPs, measured TFLOPS, and hardware efficiency
- **Memory**: Parameter, activation, gradient, and optimizer state sizes
- **Transfers**: PCIe bandwidth modeling (α-β calibration)
- **Precision**: FP32, FP16, BF16 with ISA-aware execution policy (skip + report if unsupported)
- **Framework Overhead**: CPU dispatch vs. kernel execution time

**Output**: Per-layer CSV metrics + global JSON metadata for ILP model parameterization.

---

## Quick Start

### Installation (30 seconds)

```bash
# Clone repository
git clone <repo-url>
cd Final\ Thesis\ Code

# Option A: Conda (Recommended)
conda env create -f config/environment.yml
conda activate thesis_env

# Option B: Pip + Manual PyTorch
pip install -r config/requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Profile Your First Model (2 minutes)

```bash
# GPU-only profiling (independent of CPU precision ISA support)
python src/profiler.py \
  --model vit_b16 \
  --precision fp32 \
  --batch_size 32 \
  --skip_cpu \
  --warmup 3 \
  --measure 10
```

Output: `data/vit_b16_metrics.csv` + `data/vit_b16_meta.json`

### Verify Installation

```bash
python validation/validate_code.py
python validation/validate_all_models.py --preflight-scope fast
python validation/validate_zombie_fix.py
bash validation/run_unit_tests.sh
```

---

## Supported Models (6 Total)

| Category | Models |
|----------|--------|
| **Vision** | ResNet50, ResNet152, ViT-B/16 |
| **NLP** | BERT-base, GPT2-small |
| **Baseline** | SimpleMLP |

---

## Key Features

### ✅ Multi-Device Profiling
- GPU profiling with NVML energy monitoring
- CPU profiling with RAPL energy (Linux)
- Unified metrics across devices

### ✅ Advanced Timeout Handling
- **Phase 1**: Forward pass timeout (60 seconds)
- **Phase 2**: Adaptive backward timeout
- **Phase 3**: Final completion wait
- Prevents zombie threads on slow CPU FP16

### ✅ Zombie Thread Prevention
- `--skip_cpu` flag: Skip CPU profiling entirely
- `--num_threads N`: Override SLURM single-core limitation
- Preflight moved inside run_profiling() for safety

### ✅ Precision Support
- FP32, FP16, BF16 with CPU ISA probe (AVX512_FP16, AVX512_BF16, AMX_BF16/AMX_TILE)
- Unsupported accelerated precision is skipped (no emulated fallback training)
- Per-layer precision tracking
- Explicit execution status reporting in CSV/JSON artifacts

### ✅ Comprehensive Output
- **CSV**: Per-layer execution and energy metrics
- **JSON**: Hardware metadata, energy totals, calibration data

---

## Project Structure

```
.
├── src/                          # Source code
│   ├── profiler.py              # CLI entrypoint / orchestrator
│   ├── core/                    # Shared core modules
│   ├── models/                  # Model factory
│   ├── runner/                  # Runtime profiling pipeline
│   ├── profiler_en.md           # English documentation
│   └── profiler_es.md           # Spanish documentation
│
├── config/                       # Configuration
│   ├── environment.yml          # Conda environment
│   └── requirements.txt          # Pip dependencies
│
├── scripts/                      # Automation
│   ├── run_experiments.sh       # Full grid search
│   └── launch_grid5k.sh         # HPC submission
│
├── data/                         # Storage
│   ├── results/                 # Generated metrics
│   └── test-*                   # Test fixtures
│
├── docs/                         # Documentation
│   ├── README.md                # Quick start
│   ├── PROJECT_STRUCTURE.md     # Detailed reference
│   ├── TESTING_VALIDATION_MAP.md # Validation runbook
│   └── documentation.md         # Technical methodology
│
├── validation/                   # Validation scripts
├── tests/                        # Unit tests
├── logs/                         # Execution logs
└── README.md                     # This file

```

For detailed structure explanation, see [docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md)

---

## Usage Examples

### Profile Single Model

```bash
# Minimal (CPU-only, no GPU)
python src/profiler.py --model simple_mlp --precision fp32

# Full (GPU + CPU with energy)
python src/profiler.py \
  --model resnet50 \
  --precision fp32 \
  --batch_size 64 \
  --rapl \
  --num_threads 16

# GPU-only (fast, no CPU FP16 blocking)
python src/profiler.py \
  --model vit_b16 \
  --precision fp16 \
  --skip_cpu
```

### Run Full Experiment Grid

```bash
# Edit configuration in scripts/run_experiments.sh:
# - MODELS: which models to profile
# - BATCH_SIZES: batch sizes to test
# - PRECISIONS: fp32, fp16, bf16
# - OPTIMIZERS: SGD, Adam, AdamW, etc.

bash scripts/run_experiments.sh
# Results: data/{hostname}/results/{model}/{optimizer}/{precision}/batch_{N}/
```

### Run Fast Smoke Validation (Script Mode)

```bash
# One quick end-to-end check (1 model × 1 optimizer × 1 precision × 1 batch)
SMOKE_MODE=true \
USE_SKIP_CPU=true \
FORCE_THREADS=4 \
PYTHON_CMD=.venv/bin/python \
bash scripts/run_experiments.sh
```

Useful environment variables for `scripts/run_experiments.sh`:
- `SMOKE_MODE=true|false`: Enables minimal campaign for quick sanity checks.
- `USE_SKIP_CPU=true|false`: Enables GPU-only profiling mode.
- `FORCE_THREADS=N`: Forces CPU thread count passed to `--num_threads`.
- `PYTHON_CMD=/path/to/python`: Selects interpreter (useful for `.venv`).
- `BASE_OUTPUT_DIR=...` and `LOG_DIR=...`: Override output and logs directories.
- `WARMUP=N` and `MEASURE=N`: Override profiling iterations globally.

#### `run_experiments.sh` Troubleshooting

- `exit code: 127`: Python executable path is invalid. Set `PYTHON_CMD` to a valid interpreter (for example `.venv/bin/python`).
- Frequent OOM errors: Reduce `BATCH_SIZES` in the script or use a smaller subset while validating.
- `USE_SKIP_CPU=true` with no GPU: the script auto-disables this mode and logs a warning; either enable GPU or run CPU profiling.
- Slow execution on shared/HPC nodes: set `FORCE_THREADS` to a suitable value for your allocation and reduce `MEASURE` during test runs.
- Missing results for different batches: each run now writes to `.../batch_{N}/`; verify that path when inspecting outputs.

### Monitor Execution

```bash
# Terminal 1: Watch logs
tail -f logs/experiments_*.txt

# Terminal 2: Check output directory
watch -n 5 'ls -R data/*/results'
```

---

## Command-Line Interface

### Essential Arguments

```bash
--model {name}              # resnet50, resnet152, vit_b16, bert_base, gpt2_small, simple_mlp
--batch_size N              # Batch size (default: 32)
--precision {mode}          # fp32, fp16, bf16 (default: fp32)
--warmup N                  # Warmup iterations (default: 5)
--measure N                 # Measurement iterations (default: 15)
```

### Optional Arguments

```bash
--optimizer {name}          # SGD, Adam, AdamW, RMSprop, Adagrad, Adadelta
--output_dir path           # Output directory (default: data)
--rapl                      # Enable CPU RAPL energy (Linux only)
--no_gpu                    # CPU-only profiling
--gpu_id N                  # GPU device ID (default: 0)
--input_size N              # Vision model input size (default: 224)
--seq_length N              # NLP model sequence length (default: 128)
--keep_partial_artifacts    # Keep intermediate *_gpu_partial.csv/json after successful final save
```

Note on outputs:
- During long runs, the profiler may create temporary `*_metrics_gpu_partial.csv` and `*_meta_gpu_partial.json` as safety checkpoints.
- By default, these partial files are removed automatically once final artifacts are saved successfully.
- Use `--keep_partial_artifacts` to keep them for debugging/auditing.

### Zombie Thread Fix Arguments (NEW)

```bash
--skip_cpu                  # Skip CPU profiling entirely
--num_threads N             # Force N CPU threads (override SLURM)
```

---

## Output Format

### CSV: Per-Layer Metrics
```
layer,type,params_mb,flops,gpu_fwd_time_ms,cpu_fwd_time_ms,gpu_fwd_energy_j,...
conv2d,Conv2d,2.4,3.7e9,125.3,425.7,42.5,...
linear,Linear,0.8,2.1e8,15.2,45.2,5.1,...

# New status columns:
# run_executed, skip_unsupported_precision, skip_reason
```

### JSON: Global Metadata
```json
{
  "model": "resnet50",
  "precision_requested": "fp32",
  "cpu_precision_executed": "fp32",
  "gpu_precision_executed": "fp32",
  "energy_total_gpu_j": 1250.3,
  "energy_total_cpu_j": 425.1,
  "measured_peak_tflops_gpu": 85.2,
  "pcie_alpha_h2d_us": 2.1,
  "pcie_beta_h2d_gbps": 45.3
}
```

See [docs/documentation.md](docs/documentation.md) for complete schema.

---

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Run reported as skipped | Requested FP16/BF16 lacks accelerated CPU ISA support | Check `skip_reason` in CSV/JSON, then use `--precision fp32` or supported hardware |
| OOM (Out of Memory) | Batch size too large | Reduce `--batch_size` |
| RAPL: Permission denied | RAPL requires read access to sysfs | Run with sudo or omit `--rapl` |
| GPU: ImportError `pynvml` module | NVML package/driver mismatch | Install `nvidia-ml-py` and verify NVIDIA driver/CUDA |
| Slow CPU profiling | SLURM/HPC single-core allocation | Use `--num_threads {physical_cores}` |

For more, see [docs/README.md#troubleshooting](docs/README.md#troubleshooting)

---

## Recent Improvements

### Zombie Thread Fix (v1.0)
Fixed critical issue where CPU FP16 preflight could block GPU profiling:
- **Problem**: Slow/blocked CPU FP16 paths in unsupported ISA scenarios
- **Solution**: ISA-aware precision policy (`skip` + report), plus `--skip_cpu` and `--num_threads`
- Details are integrated in [docs/documentation.md](docs/documentation.md) and [docs/TESTING_VALIDATION_MAP.md](docs/TESTING_VALIDATION_MAP.md)

### Two-Phase Timeout (Previous)
Adaptive timeout mechanism for CPU profiling:
- Phase 1: Fixed 60s timeout for forward pass
- Phase 2: Adaptive timeout for backward pass
- Phase 3: Final wait for completion

---

## System Requirements

- **Python**: 3.10 or higher
- **PyTorch**: 2.1.0 or higher
- **CUDA**: 12.1+ (for GPU profiling)
- **GPU**: NVIDIA GPU with NVML support (optional, but recommended)
- **CPU Energy**: Linux with RAPL support (optional)

---

## Documentation

| Document | Purpose |
|----------|---------|
| [docs/README.md](docs/README.md) | Quick start guide |
| [docs/QUICK_START.sh](docs/QUICK_START.sh) | Interactive quick reference |
| [docs/documentation.md](docs/documentation.md) | Technical documentation & data schema |
| [docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md) | Detailed project reference |
| [docs/TESTING_VALIDATION_MAP.md](docs/TESTING_VALIDATION_MAP.md) | Validation strategy and runbook |

---

## Citation

If you use this profiler in research, please cite:

```bibtex
@misc{torres2026hybridprofiler,
  title={Advanced Hybrid Profiler for Deep Learning Training},
  author={Torres, Luis Alejandro},
  year={2026},
  howpublished={\url{https://github.com/alejandrotorresn/Final-Thesis-Code}},
  note={PhD Thesis Code}
}
```

---

## Related Work

- **ILP Optimization**: See thesis Chapter 3 for optimization model
- **NVML**: NVIDIA Management Library (GPU monitoring)
- **RAPL**: Intel Running Average Power Limit (CPU energy)
- **PyTorch Profiling**: torch.profiler as baseline

---

## Development

### Running Tests
```bash
python validation/validate_code.py        # Syntax checks
python validation/validate_all_models.py --preflight-scope fast  # Model validation
python validation/validate_zombie_fix.py   # Zombie thread fix validation
bash validation/comprehensive_check.sh    # Full suite
```

### Contributing
For contributions or bug reports, please open an issue or pull request on GitHub.

---

## License

This project is part of an academic PhD thesis. For usage rights, contact the author.

---

## Contact

**Author**: Luis Alejandro Torres  
**Email**: [Your Email]  
**GitHub**: [Your GitHub Profile]  

For questions about the profiler, refer to the documentation or submit an issue.

---

## Acknowledgments

- PhD Thesis Advisor: [Advisor Name]
- Research Group: [Group Name]
- Funding: [Funding Source]

---

*Last Updated*: February 24, 2026  
*Project Status*: ✅ Production Ready  
*Version*: 1.0 (Post-Zombie-Thread-Fix & Reorganization)
