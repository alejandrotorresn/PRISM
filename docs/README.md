# Quick Start Guide

**For general project overview and setup, see [../README.md](../README.md) in the repository root.**

This file is the entrypoint to the consolidated documentation set. It keeps only the operational quick start and the index of canonical documents.

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

### Full thesis smoke workflow (real machine, reduced scope)
```bash
# End-to-end: profiling (FP32 reduced grid) -> aggregation -> ILP -> pareto -> plots -> LaTeX
bash scripts/run_thesis_smoke_workflow.sh
```

This script is intended for real operational verification with a small campaign (few models, optimizers, and batch sizes), while still exercising the complete thesis pipeline.

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
- `MODELS_CSV=a,b,c`: overrides the default model grid without editing the script.
- `BATCH_SIZES_CSV=8,16,32`: overrides batch sizes without editing the script.
- `PRECISIONS_CSV=fp32,fp16`: overrides the precision grid without editing the script.
- `OPTIMIZERS_CSV=SGD,AdamW`: overrides the optimizer grid without editing the script.
- `USE_SKIP_CPU=true|false`: enables GPU-only profiling mode.
- `ENABLE_RAPL=true|false`: controls whether `--rapl` is passed when CPU profiling is enabled.
- `FORCE_THREADS=N`: passes CPU thread override to `--num_threads`.
- `PYTHON_CMD=/path/to/python`: selects the interpreter used by the script.
- `FAIL_FAST=true|false`: aborts the campaign on the first runtime or aggregation failure.
- `DRY_RUN=true|false`: validates the campaign and prints commands without executing runs.
- `BASE_OUTPUT_DIR=...`, `LOG_DIR=...`, `WARMUP=N`, `MEASURE=N`: override default paths and run lengths.

Recommended production launch profiles for heterogeneous servers are documented in:
- [SERVER_LAUNCH_PROFILES.md](SERVER_LAUNCH_PROFILES.md)

## 4) Outputs

- Host-scoped root created automatically on each profiling run: `data/{hostname}/`
- CSV metrics: `data/{hostname}/.../{model}_metrics.csv`
- JSON metadata: `data/{hostname}/.../{model}_meta.json`
- Grid execution output tree: `data/{hostname}/results/{model}/{optimizer}/{precision}/batch_{N}/`

Both artifacts are designed to feed ILP parameterization workflows.

## 5) Multi-Node ILP (Hardware-Aware)

When profiling is executed across multiple cluster nodes, each node can produce different costs (CPU/GPU/bus/memory). The ILP pipeline now supports merging multiple hardware profiles before solving:

```bash
python validation/run_ilp_partition.py \
  --config_dirs "data/nodeA/results/simple_mlp/SGD/fp32/batch_8,data/nodeB/results/simple_mlp/SGD/fp32/batch_8" \
  --model simple_mlp \
  --hw_aggregate max \
  --hw_dispersion_k 0.0
```

Hardware aggregation options:
- `--hw_aggregate max`: conservative envelope (worst-case across nodes).
- `--hw_aggregate mean --hw_dispersion_k K`: robust mean plus variability term (`mean + K*std`).

Shell wrappers also support this mode via `CONFIG_DIRS`:
- `scripts/run_ilp_partition.sh`
- `scripts/run_ilp_pareto_sweep.sh`
- `scripts/discover_ilp_config_dirs.sh` (autodiscovery + optional execution)

Note: autodiscovery requires ILP-ready folders (stats + graph + transfer artifacts).

Recommended: follow the full no-memory workflow in:
- [MULTI_NODE_ILP_RUNBOOK.md](MULTI_NODE_ILP_RUNBOOK.md)

## 6) Troubleshooting

| Issue | Cause | Action |
|------|------|------|
| Run skipped | Unsupported accelerated ISA for requested precision | Use `--precision fp32` or `--skip_cpu` |
| Slow CPU profiling on HPC | CPU affinity/core limitation | Use `--num_threads N` |
| RAPL permission error | Missing access to `/sys/class/powercap` | Omit `--rapl` or grant read access |
| CUDA OOM | Batch too large | Reduce `--batch_size` |
| `exit code: 127` in `run_experiments.sh` | Invalid Python executable path | Set `PYTHON_CMD` (for example `.venv/bin/python`) |
| `USE_SKIP_CPU=true` with no GPU | No valid execution target for GPU-only mode | Script auto-disables skip mode; run with GPU or keep CPU enabled |

## Canonical Documentation Index

| Document | Purpose |
|----------|---------|
| [../README.md](../README.md) | Project overview |
| [GLOBAL_PROJECT_DOCUMENTATION.md](GLOBAL_PROJECT_DOCUMENTATION.md) | Full end-to-end technical reference, including validation and thesis-mode orchestration |
| [GLOBAL_PROJECT_DOCUMENTATION_ES.md](GLOBAL_PROJECT_DOCUMENTATION_ES.md) | Version academica en espanol de la referencia tecnica integral |
| [CAPITULO_TESIS_PROFILING_ES.md](CAPITULO_TESIS_PROFILING_ES.md) | Capitulo monografico doctoral en espanol: metodologia de profiling |
| [CAPITULO_TESIS_ILP_ES.md](CAPITULO_TESIS_ILP_ES.md) | Capitulo monografico doctoral en espanol: formulacion y analisis ILP |
| [PLAN_IMPLEMENTACION_FASES_ES.md](PLAN_IMPLEMENTACION_FASES_ES.md) | Hoja de ruta doctoral y estado de cierre por fases |
| [PROTOCOLO_VALIDACION_MULTISERVIDOR_ES.md](PROTOCOLO_VALIDACION_MULTISERVIDOR_ES.md) | Protocolo operativo y metodologico para construir el dataset multi-hardware y cerrar la validacion doctoral |
| [SERVER_LAUNCH_PROFILES.md](SERVER_LAUNCH_PROFILES.md) | Recommended launch configurations by server type |
| [MULTI_NODE_ILP_RUNBOOK.md](MULTI_NODE_ILP_RUNBOOK.md) | Step-by-step multi-node ILP workflow |
| [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) | Architecture and folder map |

## Consolidation Policy

- `README.md` y `docs/README.md` cubren entrada rapida y navegacion.
- `GLOBAL_PROJECT_DOCUMENTATION[_ES].md` son la referencia tecnica canónica.
- Los capitulos de tesis en espanol preservan el nivel monografico y no duplican runbooks operativos.
- Los documentos historicos sobreviven solo cuando aportan contexto no sustituible. La trazabilidad historica proyecto-tesis ya fue absorbida por `PLAN_IMPLEMENTACION_FASES_ES.md`.

---

*Last Updated*: March 26, 2026
