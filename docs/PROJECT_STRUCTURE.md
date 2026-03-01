# Project Structure

This document describes the current architecture and folder responsibilities of the profiler project.

## High-Level Layout

```
.
├── config/          # Environment and dependency definitions
├── data/            # Runtime outputs and test fixtures
├── docs/            # Core documentation set
├── logs/            # Runtime logs
├── scripts/         # Experiment orchestration scripts
├── src/             # Profiler implementation (modular)
├── tests/           # Unit tests
└── validation/      # Validation and audit scripts
```

## Source Architecture (`src/`)

```
src/
├── profiler.py                  # CLI entrypoint and orchestration
├── core/
│   ├── constants.py             # Shared constants
│   ├── energy.py                # Energy monitor (GPU/CPU)
│   ├── io_artifacts.py          # CSV/JSON artifact writing
│   ├── metrics.py               # Metric helpers
│   ├── precision_policy.py      # FP32/FP16/BF16 policy + preflight
│   └── system.py                # Runtime/system configuration
├── models/
│   └── factory.py               # Model/input factory (6 models)
└── runner/
    └── training_profiler.py     # Main profiling execution pipeline
```

### Supported Models
- Vision: `resnet50`, `resnet152`, `vit_b16`
- NLP: `bert_base`, `gpt2_small`
- Baseline: `simple_mlp`

## Validation Stack

- `tests/test_precision_policy_unit.py`: unit tests for precision policy helpers.
- `tests/test_timeout_validation.py`: timeout behavior checks.
- `validation/run_unit_tests.sh`: unified test command.
- `validation/validate_code.py`: structural timeout/metadata integrity checks.
- `validation/validate_zombie_fix.py`: skip_cpu/num_threads flow checks.
- `validation/validate_all_models.py`: model loading + configurable preflight scope.
- `validation/comprehensive_check.sh`: full grep-based architecture audit.

## Scripts and Runtime Outputs

- `scripts/run_experiments.sh`: grid execution across models/precisions/optimizers.
- `scripts/launch_grid5k.sh`: HPC job launcher.
- `data/{model}_metrics.csv`: per-layer metrics for ILP.
- `data/{model}_meta.json`: global metadata and execution status.

## Recommended Reading Order

1. [README.md](README.md) (quick start)
2. [TESTING_VALIDATION_MAP.md](TESTING_VALIDATION_MAP.md) (validation workflow)
3. [documentation.md](documentation.md) (technical methodology)

---

*Last Updated*: March 1, 2026
