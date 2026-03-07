# Quick Start Guide

**For general project overview and setup, see [../README.md](../README.md) in the repository root.**

This guide is the single operational entrypoint for using and validating the profiler.

## 1) Installation

### Option A: Conda (recommended)
```bash
conda env create -f config/environment.yml
conda activate thesis_env
```

### Option B: Pip + virtualenv
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r config/requirements.txt
# Install PyTorch separately for your CUDA version:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## 2) Validation (recommended order)

```bash
bash validation/run_unit_tests.sh
python validation/validate_code.py
python validation/validate_zombie_fix.py
bash validation/comprehensive_check.sh
```

Optional all-model validation:
```bash
# Fast default preflight
python validation/validate_all_models.py --preflight-scope fast

# Exhaustive (slow on some CPUs)
python validation/validate_all_models.py --preflight-scope all
```

## 3) Basic Usage

### Minimal smoke run
```bash
python src/profiler.py \
  --model simple_mlp \
  --precision fp32 \
  --warmup 1 \
  --measure 2 \
  --no_gpu
```

### GPU-focused run
```bash
python src/profiler.py \
  --model vit_b16 \
  --precision fp32 \
  --batch_size 32 \
  --skip_cpu
```

### Full experiment grid
```bash
bash scripts/run_experiments.sh
```

### Fast script smoke mode
```bash
SMOKE_MODE=true \
USE_SKIP_CPU=true \
FORCE_THREADS=4 \
PYTHON_CMD=.venv/bin/python \
bash scripts/run_experiments.sh
```

Useful script environment variables:
- `SMOKE_MODE=true|false`: runs a minimal campaign (1 model × 1 optimizer × 1 precision × 1 batch).
- `USE_SKIP_CPU=true|false`: enables GPU-only profiling mode.
- `FORCE_THREADS=N`: passes CPU thread override to `--num_threads`.
- `PYTHON_CMD=/path/to/python`: selects the interpreter used by the script.
- `BASE_OUTPUT_DIR=...`, `LOG_DIR=...`, `WARMUP=N`, `MEASURE=N`: override default paths and run lengths.

## 4) Outputs

- CSV metrics: `data/{model}_metrics.csv`
- JSON metadata: `data/{model}_meta.json`
- Grid execution output tree: `data/results/{model}/{optimizer}/{precision}/batch_{N}/`

Both artifacts are designed to feed ILP parameterization workflows.

## 5) Troubleshooting

| Issue | Cause | Action |
|------|------|------|
| Run skipped | Unsupported accelerated ISA for requested precision | Use `--precision fp32` or `--skip_cpu` |
| Slow CPU profiling on HPC | CPU affinity/core limitation | Use `--num_threads N` |
| RAPL permission error | Missing access to `/sys/class/powercap` | Omit `--rapl` or grant read access |
| CUDA OOM | Batch too large | Reduce `--batch_size` |
| `exit code: 127` in `run_experiments.sh` | Invalid Python executable path | Set `PYTHON_CMD` (for example `.venv/bin/python`) |
| `USE_SKIP_CPU=true` with no GPU | No valid execution target for GPU-only mode | Script auto-disables skip mode; run with GPU or keep CPU enabled |

## Documentation Index

| Document | Purpose |
|----------|---------|
| [../README.md](../README.md) | Project overview |
| [documentation.md](documentation.md) | Full technical methodology and schema |
| [TESTING_VALIDATION_MAP.md](TESTING_VALIDATION_MAP.md) | Validation strategy and runbook |
| [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) | Architecture and folder map |

---

*Last Updated*: March 1, 2026
