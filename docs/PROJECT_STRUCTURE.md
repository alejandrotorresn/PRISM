# Project Structure

This document describes the current architecture and folder responsibilities of the project after consolidation of profiling, ILP, runtime, validation, and thesis-mode orchestration.

## High-Level Layout

```
.
├── config/          # Environment and dependency definitions
├── data/            # Runtime outputs and test fixtures
├── docs/            # Core documentation set
├── logs/            # Runtime logs
├── scripts/         # Experiment orchestration scripts
├── src/             # Profiler + ILP implementation (modular)
├── tests/           # Unit tests
└── validation/      # Validation and audit scripts
```

## Source Architecture (`src/`)

```
src/
├── profiler.py                  # CLI entrypoint and orchestration
├── core/
│   ├── constants.py             # Shared constants
│   ├── decoder_export_backend.py # Export-based graph path for decoder-only models
│   ├── energy.py                # Energy monitor (GPU/CPU)
│   ├── graph_extractor.py       # FX/fallback DAG export (nodes/edges)
│   ├── io_artifacts.py          # CSV/JSON artifact writing
│   ├── loss_utils.py            # Stable training objectives and quality metrics
│   ├── metrics.py               # Metric helpers
│   ├── precision_policy.py      # FP32/FP16/BF16 policy + preflight
│   ├── stats_aggregator.py      # Replicate aggregation (mean/std/percentiles)
│   └── system.py                # Runtime/system configuration
├── data/
│   └── dataset_registry.py      # Dataset-backed batch loading and download registry
├── ilp/
│   ├── data_loader.py           # Loads robust stats + graph/transfer edges
│   ├── model_builder.py         # ILP config validation + objective data
│   ├── solve.py                 # Solver backend (PuLP or exhaustive fallback)
│   └── export_solution.py       # Writes assignment/cut-edge/summary artifacts
├── models/
│   └── factory.py               # Model/input factory with dataset-backed execution policy
└── runner/
    └── training_profiler.py     # Main profiling pipeline + final artifacts
src/runtime/
├── device_plan.py               # Parses dual ILP placement plans
├── hybrid_executor.py           # Real CPU/GPU hybrid runtime
├── plan_representation.py       # Canonical plan model
└── simulator.py                 # Deterministic pre-execution simulator
```

### Supported Models
- Vision: `resnet50`, `resnet152`, `vit_b16`
- NLP: `bert_base`, `gpt2_small`, `distilgpt2`
- Baseline: `simple_mlp`

## Validation Stack

- `tests/test_precision_policy_unit.py`: unit tests for precision policy helpers.
- `tests/test_profiler_gpu_only_precision_policy.py`: GPU-only precision policy behavior.
- `tests/test_timeout_validation.py`: timeout behavior checks.
- `validation/run_unit_tests.sh`: unified test command.
- `validation/validate_code.py`: structural timeout/metadata integrity checks.
- `validation/validate_zombie_fix.py`: skip_cpu/num_threads flow checks.
- `validation/validate_all_models.py`: model loading + configurable preflight scope.
- `validation/run_ilp_partition.py`: execute one ILP partition from artifacts.
- `validation/sweep_ilp_pareto.py`: run ILP over multiple GPU memory budgets.
- `validation/run_ilp_ablation_suite.py`: systematic ILP ablation harness.
- `validation/run_ilp_sensitivity.py`: parameter-sensitivity harness.
- `validation/validate_ilp_pipeline.py`: simulator-backed ILP plan validation.
- `validation/run_hybrid_execution.py`: physical hybrid-runtime execution from ILP plan.
- `validation/aggregate_metrics_stats.py`: aggregate `run_*/*_metrics.csv` into robust stats.
- `validation/comprehensive_check.sh`: full grep-based architecture audit.

## Scripts and Runtime Outputs

- `scripts/run_experiments.sh`: grid execution across models/precisions/optimizers/repeats.
- `scripts/run_thesis_smoke_workflow.sh`: reduced FP32 end-to-end workflow (profiling -> ILP -> reports -> LaTeX).
- `scripts/run_thesis_mode.sh`: full thesis-grade orchestration with datasets and reporting.
- `scripts/launch_grid5k.sh`: HPC job launcher.
- `scripts/download_datasets.py`: dataset preparation for profiling and hybrid runtime.
- `scripts/sanitize_cuda_env.sh`: CUDA environment hygiene for orchestration scripts.
- `scripts/run_ilp_partition.sh`: shell wrapper for single ILP solve.
- `scripts/run_ilp_pareto_sweep.sh`: shell wrapper for ILP Pareto sweep.
- `scripts/discover_ilp_config_dirs.sh`: auto-discovers multi-host config dirs and can execute ILP wrappers.
- `data/.../run_*/{model}_metrics.csv`: per-layer profiling metrics.
- `data/.../run_*/{model}_meta.json`: metadata and execution status.
- `data/.../run_*/{model}_graph_nodes.csv`: DAG node table.
- `data/.../run_*/{model}_graph_edges.csv`: DAG edge table.
- `data/.../run_*/{model}_transfer_edges.csv`: edge-aware transfer costs.
- `data/.../{model}_metrics_stats.csv` or `metrics_stats.csv`: robust aggregated stats.
- `data/.../ilp_solution/`: ILP output (`ilp_assignment.csv`, `ilp_cut_edges.csv`, summary JSON).

## Pipeline Summary

1. `src/profiler.py` parses experiment configuration and precision policy.
2. `src/runner/training_profiler.py` runs profiling and writes per-run artifacts.
3. `src/core/stats_aggregator.py` builds robust `metrics_stats.csv` across replicates.
4. `validation/run_ilp_partition.py` loads stats + graph/transfer artifacts and solves ILP.
5. `src/ilp/export_solution.py` writes final partition artifacts for analysis/reporting.

## Recommended Reading Order

1. [README.md](README.md) (quick start)
2. [GLOBAL_PROJECT_DOCUMENTATION_ES.md](GLOBAL_PROJECT_DOCUMENTATION_ES.md) (canonical technical reference in Spanish)
3. [GLOBAL_PROJECT_DOCUMENTATION.md](GLOBAL_PROJECT_DOCUMENTATION.md) (English technical reference)
4. [PLAN_IMPLEMENTACION_FASES_ES.md](PLAN_IMPLEMENTACION_FASES_ES.md) (doctoral closure roadmap)

---

*Last Updated*: March 26, 2026
