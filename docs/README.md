# Documentation Index

**Official project name:** PRISM  
**Expansion:** Partitioning and Routing via ILP for Scalable Memory  
**Technical subtitle:** A Hybrid CPU-GPU Training Optimization Framework Guided by Profiling and ILP

This file is the canonical entrypoint to the documentation tree. The repository root [../README.md](../README.md) explains the project at a high level; this index tells you where each kind of knowledge now lives after the cleanup of docs, logs, and reports.

## 1) Working environment

### Conda (official path)
```bash
conda env create -f config/environment.yml
conda activate prism_env
```

### Pip + virtualenv
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r config/requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## 2) Recommended validation order

```bash
conda activate prism_env
bash validation/run_unit_tests.sh
python validation/validate_code.py
python validation/validate_all_models.py --preflight-scope fast
python validation/validate_zombie_fix.py
bash validation/comprehensive_check.sh
```

For a slower exhaustive model preflight:
```bash
python validation/validate_all_models.py --preflight-scope all
```

## 3) Fast operational path

### Minimal CPU smoke run
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

### Full profiling campaign
```bash
bash scripts/run_experiments.sh
```

### Reduced thesis workflow
```bash
bash scripts/run_thesis_smoke_workflow.sh
```

### Thesis mode profiles

- `doctoral_minimal` and `doctoral_full`: strict official profiles for core doctoral evidence.
- `doctoral_diagnostic`: fallback-enabled diagnostic profile for operational debugging only.
- `quick_smoke`: fast smoke validation profile.
- `custom`: explicit manual overrides.

Example:

```bash
PROFILE=doctoral_diagnostic RUN_HYBRID=true bash scripts/run_thesis_mode.sh
```

### Fast script preflight
```bash
conda activate prism_env
SMOKE_MODE=true \
USE_SKIP_CPU=true \
FORCE_THREADS=4 \
bash scripts/run_experiments.sh
```

## 4) Documentation tree

The documentation now uses four blocks so the top level stays readable.

| Area | Contents | Use |
|------|----------|-----|
| [architecture/](architecture) | Repository map and full technical references | Understand system design and pipeline semantics |
| [operations/](operations) | Runbooks, launch profiles, tutorials, quick command helper | Execute campaigns and recover failures |
| [thesis/](thesis) | Monographic chapters and thesis writing map | Consume the project as doctoral material |
| [archive/](archive) | Legacy office/editor support material | Keep historical context out of the canonical path |

## 5) Canonical reading order

| Step | Document | Role |
|------|----------|------|
| 1 | [../README.md](../README.md) | Global overview and quick project orientation |
| 2 | [architecture/PROJECT_STRUCTURE.md](architecture/PROJECT_STRUCTURE.md) | Repository cartography and responsibility split |
| 3 | [architecture/GLOBAL_PROJECT_DOCUMENTATION_ES.md](architecture/GLOBAL_PROJECT_DOCUMENTATION_ES.md) | Canonical technical reference in academic Spanish |
| 4 | [architecture/GLOBAL_PROJECT_DOCUMENTATION.md](architecture/GLOBAL_PROJECT_DOCUMENTATION.md) | English technical reference |
| 5 | [operations/PROTOCOLO_VALIDACION_MULTISERVIDOR_ES.md](operations/PROTOCOLO_VALIDACION_MULTISERVIDOR_ES.md) | Go/No-Go operational protocol |
| 6 | [operations/SERVER_LAUNCH_PROFILES.md](operations/SERVER_LAUNCH_PROFILES.md) | Server-class launch profiles |
| 7 | [operations/MULTI_NODE_ILP_RUNBOOK.md](operations/MULTI_NODE_ILP_RUNBOOK.md) | Multi-host ILP workflow |
| 8 | [thesis/CAPITULO_TESIS_PROFILING_ES.md](thesis/CAPITULO_TESIS_PROFILING_ES.md) | Profiling chapter |
| 9 | [thesis/CAPITULO_TESIS_ILP_ES.md](thesis/CAPITULO_TESIS_ILP_ES.md) | ILP chapter |
| 10 | [thesis/schema.md](thesis/schema.md) | Doctoral writing map |

If you only need commands, use [operations/QUICK_START.sh](operations/QUICK_START.sh).

## 6) Outputs and operational folders

- `data/<hostname>/...` stores host-scoped experimental evidence.
- `reports/ilp_results*/` stores canonical ILP and thesis-ready outputs.
- `reports/diagnostics/` stores ad hoc audits and profiling summaries outside the canonical report path.
- `logs/experiments/`, `logs/thesis_mode/`, `logs/thesis_smoke_workflow/`, and `logs/grid5k/` isolate logs by workflow.
- `logs/archive/` keeps historical traces out of the main logs view.

## 7) Useful runtime overrides

- `SMOKE_MODE=true|false`: minimal profiling grid.
- `MODELS_CSV=...`, `BATCH_SIZES_CSV=...`, `PRECISIONS_CSV=...`, `OPTIMIZERS_CSV=...`: override campaign axes.
- `USE_SKIP_CPU=true|false`: GPU-only profiling mode.
- `ENABLE_RAPL=true|false`: toggle `--rapl` when CPU profiling is active.
- `FORCE_THREADS=N`: forward `--num_threads` to the profiler.
- `PYTHON_CMD=/path/to/python`: explicit interpreter for shell workflows.
- `FAIL_FAST=true|false`: abort on the first runtime or aggregation failure.
- `OOM_SKIP_CONFIGS=true|false`: when true, exhausted OOM configurations are marked as skipped and the campaign continues; when false, OOM is treated as a normal failure.
- `DRY_RUN=true|false`: validate and print commands without executing them.
- `BASE_OUTPUT_DIR=...`, `REPORTS_DIR=...`, `LOG_DIR=...`: relocate outputs if the host requires it.

## 8) Troubleshooting

| Issue | Cause | Action |
|------|------|------|
| Run skipped | Unsupported accelerated ISA for requested precision | Use `--precision fp32` or `--skip_cpu` |
| Slow CPU profiling on HPC | CPU affinity/core limitation | Use `--num_threads N` |
| RAPL permission error | Missing access to `/sys/class/powercap` | Omit `--rapl` or grant read access |
| CUDA OOM | Batch too large | Reduce `--batch_size` |
| `exit code: 127` in `run_experiments.sh` | Invalid Python executable path | Set `PYTHON_CMD` |
| Missing expected artifacts | Pipeline failed in an intermediate stage | Check the workflow-specific subdirectory under `logs/` |

## 9) Consolidation policy

- `README.md` and `docs/README.md` remain the only entrypoints a new reader should need.
- `architecture/` holds the canonical explanation of the system.
- `operations/` holds execution knowledge and operator guidance.
- `thesis/` holds monographic material rather than runbooks.
- `archive/` is intentionally non-canonical and should not grow unless a file must be preserved for historical reasons.

---

*Last Updated*: June 1, 2026
