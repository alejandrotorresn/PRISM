# Advanced Hybrid Profiler - Project Structure

## Overview
This project implements an advanced profiler for deep learning models to generate cost metrics for Integer Linear Programming (ILP) optimization. The code has been reorganized for clarity and maintainability.

---

## Directory Layout

```
.
├── src/                          # Core source code
│   ├── profiler.py              # Main profiler implementation
│   ├── profiler_en.md           # English documentation (embedded)
│   ├── profiler_es.md           # Spanish documentation (embedded)
│   └── profiler_old.py          # Legacy/backup version
│
├── config/                       # Configuration files
│   ├── environment.yml          # Conda environment specification
│   └── requirements.txt          # Python pip dependencies
│
├── scripts/                      # Execution scripts
│   ├── run_experiments.sh       # Main experiment sweep script
│   └── launch_grid5k.sh         # HPC/Grid5k submission script
│
├── data/                         # Data storage
│   ├── results/                 # Generated profiling results (created at runtime)
│   │   ├── {model}/
│   │   │   ├── {optimizer}/
│   │   │   │   └── {precision}/  # CSV/JSON metrics
│   │   │   └── ...
│   │   └── ...
│   ├── test-ci/                 # CI test data
│   ├── test-ci-simple/          # Simplified CI test data
│   └── test-ci-vit/             # Vision Transformer test data
│
├── logs/                         # Runtime logs
│   └── experiments_*.txt        # Timestamped experiment logs
│
├── tests/                        # Unit/integration tests
│   └── test_timeout_validation.py # Timeout mechanism testing
│
├── validation/                   # Validation & verification scripts
│   ├── validate_code.py         # Quick syntax/import checks
│   ├── validate_all_models.py   # Model loading validation
│   ├── validate_zombie_fix.py   # Zombie thread fix validation
│   ├── comprehensive_check.sh   # Full validation suite
│   └── VALIDATION_SUMMARY.sh    # Validation results summary
│
├── docs/                         # Documentation
│   ├── README.md                # Project overview & quick start
│   ├── documentation.md         # Detailed technical documentation
│   ├── FINAL_VALIDATION_REPORT.md      # Test results summary
│   ├── CODE_REVIEW_FINAL_REPORT.md     # Code review findings
│   ├── MODEL_VALIDATION_REPORT.md      # Model-specific validation
│   └── ZOMBIE_THREAD_FIX_SUMMARY.md    # Zombie thread issue & fixes
│
├── .gitignore                    # Git ignore patterns
├── .git/                         # Git repository metadata
└── .venv/                        # Python virtual environment

```

---

## File Descriptions

### Core Source Code (`src/`)

#### `profiler.py` (Main Application - 1455 lines)
The complete profiler implementation for characterizing DL models:
- **Key Classes**:
  - `TrainingProfiler`: Main profiler orchestrating all measurements
  - `SimpleMLP`: Basic 3-layer MLP for testing

- **Key Functions**:
  - `configure_cpu_runtime(force_threads=0)`: CPU thread configuration (NEW: supports SLURM override)
  - `run_cpu_fp16_model_preflight()`: FP16 support verification (MOVED: now inside run_profiling())
  - `get_cpu_fp16_support_info()`: CPU FP16 capability detection
  - `get_hardware_metadata()`: System & hardware info collection

- **Supported Models** (6 total):
  - Vision: ResNet50, ResNet152, ViT-B/16
  - NLP: BERT-base, GPT2-small
  - Baseline: SimpleMLP

- **Key Features**:
  - Multi-precision support: FP32, FP16, BF16
  - GPU energy via NVML, CPU energy via RAPL
  - PCIe bandwidth calibration
  - Framework overhead quantification
  - Timeout enforcement (3-phase: forward, backward, wait)

- **Recent Changes** (Zombie Thread Fix):
  - Added `--skip_cpu` flag to skip CPU profiling
  - Added `--num_threads N` to override SLURM CPU affinity
  - Moved preflight inside `run_profiling()` after GPU data saved
  - Updated `configure_cpu_runtime()` signature to accept `force_threads` parameter

#### `profiler_en.md` / `profiler_es.md`
Inline documentation within profiler.py extracted to separate files.

#### `profiler_old.py`
Legacy version for reference/rollback.

---

### Configuration (`config/`)

#### `environment.yml`
Conda environment specification (recommended for installation):
```bash
conda env create -f config/environment.yml
conda activate thesis_env
```

**Key packages**:
- PyTorch 2.1.0+ with CUDA support
- torchvision 0.16.0+
- transformers 4.40.0+
- pandas, numpy, psutil
- pynvml (GPU monitoring)
- pyRAPL (CPU energy measurement, Linux only)

#### `requirements.txt`
Pip-installable dependencies (alternative to conda).
**Note**: PyTorch must be installed separately via pytorch.org due to CUDA version specificity.

---

### Execution Scripts (`scripts/`)

#### `run_experiments.sh` (Updated with Zombie Thread Fixes)
Master experiment execution script implementing grid search:

**Grid Search Parameters**:
- Models: resnet50, resnet152, vit_b16, bert_base, gpt2_small, simple_mlp
- Batch Sizes: 8, 16, 32, 64, 128, 256
- Precisions: fp32, fp16, bf16
- Optimizers: SGD, SGD_momentum, Adam, AdamW, RMSprop, Adagrad, Adadelta

**New Zombie Thread Fix Flags** (Lines ~51-52):
```bash
USE_SKIP_CPU=false        # Set to 'true' for GPU-only mode
FORCE_THREADS=0           # 0=auto-detect, >0=override SLURM
```

**Usage**:
```bash
# Full grid search (6 models × 7 optimizers × 3 precisions × 6 batches = 756 experiments)
bash scripts/run_experiments.sh

# Quick test (GPU-only, skip slow CPU FP16 profiling)
# Edit run_experiments.sh: USE_SKIP_CPU=true, MODELS=("vit_b16"), BATCH_SIZES=(32)
bash scripts/run_experiments.sh
```

**Output Structure**:
```
data/results/{model}/{optimizer}/{precision}/
├── {model}_metrics.csv          # Per-layer execution metrics
├── {model}_meta.json            # Hardware & summary metadata
├── {model}_metrics_gpu_partial.csv  # GPU-only metrics (saved early)
└── {model}_meta_gpu_partial.json    # GPU-only metadata
```

#### `launch_grid5k.sh`
HPC job submission script for Grid'5000 (French supercomputer).
Submits experiment batches via OAR scheduler.

---

### Data (`data/`)

#### `data/results/` (Created at Runtime)
Output directory where profiler generates metrics:
- Structure: `{model_name}/{optimizer}/{precision}/{model_name}_metrics.csv`
- Each CSV contains per-layer profiling results for ILP model

#### `data/test-*` Directories
Pre-computed test data for validation/CI pipelines.

---

### Logs (`logs/`)

#### `experiments_YYYYMMDD_HHMMSS.txt`
Timestamped logs from `run_experiments.sh`:
- Captures all profiler stdout/stderr
- Useful for debugging OOM, precision issues, or hardware constraints

---

### Testing & Validation (`tests/` & `validation/`)

#### `tests/test_timeout_validation.py`
Unit tests for timeout mechanism (two-phase timeout fix):
- Tests forward timeout calculation
- Tests backward timeout adaptation
- Tests threading behavior

#### `validation/validate_code.py`
Quick syntax & import checks:
```bash
python validation/validate_code.py
```

#### `validation/validate_all_models.py`
Comprehensive model validation:
```bash
python validation/validate_all_models.py
```

#### `validation/validate_zombie_fix.py`
Validates zombie thread fix implementation:
```bash
python validation/validate_zombie_fix.py
```
Checks that:
- `--skip_cpu` and `--num_threads` arguments exist
- `configure_cpu_runtime(force_threads)` signature correct
- Preflight moved inside `run_profiling()`
- All 7/7 validation checks pass

#### `validation/comprehensive_check.sh`
Full bash validation suite testing all model-precision-batch combinations.

#### `validation/VALIDATION_SUMMARY.sh`
Quick reference for validation test results.

---

### Documentation (`docs/`)

#### `README.md`
Quick start guide:
- Installation instructions
- Usage examples
- Expected outputs

#### `documentation.md`
Detailed technical documentation:
- Methodology explanation
- Data dictionary (CSV/JSON schema)
- Profiling strategy details
- Advanced features

#### `FINAL_VALIDATION_REPORT.md`
Summary of all validation tests (60/60 checks passed):
- Model loading validation
- Precision handling verification
- CPU FP16 preflight testing
- Metadata completeness checks

#### `CODE_REVIEW_FINAL_REPORT.md`
Detailed code review of timeout mechanism fix:
- Race condition identification
- Two-phase timeout implementation
- Diagnostic improvements
- Edge case handling

#### `MODEL_VALIDATION_REPORT.md`
Per-model validation analysis:
- Model-specific constraints
- Supported precision combinations
- Memory requirements
- Timing benchmarks

#### `ZOMBIE_THREAD_FIX_SUMMARY.md`
Complete description of zombie thread issue & fixes:
- Problem diagnosis (preflight blocking GPU profiling)
- Root cause analysis
- Three solutions implemented:
  1. Arguments: `--skip_cpu`, `--num_threads`
  2. Function signature: `configure_cpu_runtime(force_threads)`
  3. Code placement: Move preflight inside run_profiling()
- Mitigation strategies
- Recommended usage patterns

---

## Quick Start Guide

### 1. Setup Environment
```bash
# Option A: Conda (Recommended)
conda env create -f config/environment.yml
conda activate thesis_env

# Option B: Pip
pip install -r config/requirements.txt
# Then separately: https://pytorch.org/get-started/locally/
```

### 2. Run Quick Validation
```bash
# Check all systems operational
python validation/validate_code.py
python validation/validate_all_models.py
python validation/validate_zombie_fix.py
```

### 3. Profile Single Model (GPU-only, fast)
```bash
python src/profiler.py \
  --model vit_b16 \
  --precision fp16 \
  --skip_cpu \
  --num_threads 16 \
  --batch_size 32
```

### 4. Run Full Experiment Campaign
```bash
# Edit scripts/run_experiments.sh as needed for grid search parameters
bash scripts/run_experiments.sh
# Results in: data/results/{model}/{optimizer}/{precision}/
```

### 5. Monitor Progress
```bash
# In another terminal, watch logs
tail -f logs/experiments_*.txt
```

---

## Key Design Decisions

### File Organization Rationale

| Folder | Purpose | Why Separate |
|--------|---------|--------------|
| `config/` | Environment & dependencies | Version control friendly, easy updates |
| `scripts/` | Experiment automation | Executable bash, clear separation from tests |
| `tests/` | Unit tests | Traditional Python test location |
| `validation/` | Validation/CI scripts | Distinct from unit tests (validation ≠ testing) |
| `docs/` | Documentation | Easy to publish/reference |
| `data/` | Runtime results | Git-ignored, large files, test fixtures |
| `logs/` | Execution logs | Git-ignored, timestamped, debugging |

### Code Structure in `profiler.py`

```
profiler.py (1455 lines)
├── Docstring (data dictionary, methodology)
├── Imports & Configuration (lines 114-180)
├── Constants & Helpers (lines 181-300)
│   ├── CPU/Hardware detection
│   ├── FP16 support verification
│   └── HW telemetry via NVML/RAPL
├── TrainingProfiler Class (lines 300-1300)
│   ├── __init__
│   ├── Forward/backward passes
│   ├── Energy measurement
│   ├── PCIe calibration
│   └── run_profiling() method
├── SimpleMLP Model (utility)
└── __main__ section (lines 1300-1455)
    ├── Argument parsing (with zombie thread fix flags)
    ├── Model initialization
    └── TrainingProfiler invocation
```

---

## Common Tasks

### Add a New Model
1. Add constructor in `__main__` (line ~1380)
2. Add to model list in `run_experiments.sh` (line ~22)
3. Update `args.model` documentation

### Change Profiling Step Count
Edit in `run_experiments.sh`:
```bash
WARMUP=3    # Warmup iterations (was 5)
MEASURE=10  # Measurement iterations (was 15)
```

### Skip CPU Due to Slow FP16 Emulation
Edit `run_experiments.sh`:
```bash
USE_SKIP_CPU=true   # Skip all CPU profiling
FORCE_THREADS=16    # Or: force threads if CPU profiling enabled
```

### Submit HPC Job (Grid'5000)
```bash
bash scripts/launch_grid5k.sh
```
(Requires Grid'5000 account)

---

## Troubleshooting

| Issue | Root Cause | Solution |
|-------|-----------|----------|
| ImportError: No module named 'torch' | PyTorch not installed | Install via pytorch.org or conda |
| NVIDIA_SMILIB_ERROR | CUDA/driver mismatch | Reinstall PyTorch for your CUDA version |
| Permission denied: /sys/class/powercap/ | RAPL requires sudo | Run with `--rapl` only if you have permissions |
| Process hangs @ batch X | ViT-B16 FP16 without AVX512 | Use `--skip_cpu` or `--precision fp32` |
| OOM (Out of Memory) | Batch size too large | Reduce batch size or increase GPU memory |
| Energy readings as NaN | RAPL disabled or unavailable | Use `--rapl` only on Linux with RAPL support |

---

## References

- **Thesis**: PhD thesis Chapter 3 (ILP model definition)
- **Documentation**: See [docs/documentation.md](docs/documentation.md)
- **Validation Report**: [docs/FINAL_VALIDATION_REPORT.md](docs/FINAL_VALIDATION_REPORT.md)
- **Zombie Thread Fix**: [docs/ZOMBIE_THREAD_FIX_SUMMARY.md](docs/ZOMBIE_THREAD_FIX_SUMMARY.md)

---

## Contact & Support

For issues or questions:
1. Check validation reports in `docs/`
2. Review logs in `logs/experiments_*.txt`
3. Run validation scripts in `validation/`

---

*Last Updated*: February 23, 2026
*Version*: 1.0 (Post-Zombie-Thread-Fix)
