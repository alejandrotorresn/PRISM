[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1%2B-red)](https://pytorch.org)

# PRISM

*Partitioning and Resource Intelligence for System Memory*  
*A Hybrid CPU-GPU Training Optimization Framework Guided by Profiling and ILP*

Research code for a complete pipeline that measures deep learning training costs, builds robust Integer Linear Programming partition models, and validates hybrid CPU-GPU execution strategies aimed at reducing GPU VRAM pressure while making the CPU an active participant in training.

## Overview

PRISM is organized as an end-to-end system, not as a standalone profiler. Its core contribution is the connection between empirical evidence and optimization-driven execution:

- layer-wise profiling on CPU and GPU with time, energy, memory, FLOPs, and transfer-aware artifacts
- robust statistical aggregation across replicas and across heterogeneous servers
- ILP-based partitioning under latency, memory, energy, and transfer constraints
- simulation and hybrid runtime validation of the generated plans
- report generation and thesis-ready artifacts

The practical question addressed by PRISM is simple: how to decide which parts of a model should remain on GPU and which can be moved to CPU without treating the CPU as a passive fallback, but rather as an active computational actor in training.

## End-to-End Workflow

1. `src/profiler.py` and `src/runner/training_profiler.py` capture per-layer measurements and structural artifacts.
2. `validation/aggregate_metrics_stats.py` and `src/core/stats_aggregator.py` convert repeated runs into robust coefficients.
3. `validation/run_ilp_partition.py` and `validation/sweep_ilp_pareto.py` solve placement and budget trade-off problems.
4. `src/runtime/` and `validation/run_hybrid_execution.py` validate those plans through simulation or physical hybrid execution.
5. `validation/generate_ilp_report_assets.py`, `validation/export_ilp_tables_latex.py`, `reports/`, and `thesis/` turn the results into analyzable and publishable evidence.

## Quick Start

### Installation

```bash
git clone <repo-url>
cd <repo-folder>

# Option A: Conda
conda env create -f config/environment.yml
conda activate thesis_env

# Option B: Pip + virtualenv
python -m venv .venv
source .venv/bin/activate
pip install -r config/requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Recommended Validation Order

```bash
bash validation/run_unit_tests.sh
python validation/validate_code.py
python validation/validate_all_models.py --preflight-scope fast
python validation/validate_zombie_fix.py
bash validation/comprehensive_check.sh
```

### First Operational Smoke Run

```bash
python scripts/download_datasets.py --models all --datasets_root datasets

python src/profiler.py \
  --model vit_b16 \
  --precision fp32 \
  --batch_size 32 \
  --datasets_root datasets \
  --require_datasets \
  --skip_cpu \
  --warmup 3 \
  --measure 10
```

Expected outputs are written under `data/{hostname}/...`, including `{model}_metrics.csv`, `{model}_meta.json`, graph artifacts, transfer artifacts, and aggregated statistics when the campaign layer is used.

## Main Operational Entry Points

### Profiling campaign

```bash
bash scripts/run_experiments.sh
```

### End-to-end reduced thesis workflow

```bash
bash scripts/run_thesis_smoke_workflow.sh
```

### Single ILP solve from existing artifacts

```bash
python validation/run_ilp_partition.py \
  --config_dir data/<hostname>/results/simple_mlp/SGD/fp32/batch_8 \
  --model simple_mlp
```

### Pareto sweep over GPU budgets

```bash
python validation/sweep_ilp_pareto.py \
  --config_dir data/<hostname>/results/simple_mlp/SGD/fp32/batch_8 \
  --model simple_mlp \
  --gpu_budgets_mb 400,600,800,1000
```

### Server preflight before real collection

```bash
conda activate thesis_env
SMOKE_MODE=true \
DRY_RUN=true \
FAIL_FAST=true \
bash scripts/run_experiments.sh
```

## Supported Models

- Vision: `resnet50`, `resnet152`, `vit_b16`
- NLP: `bert_base`, `gpt2_small`, `distilgpt2`
- Baseline: `simple_mlp`

## Key Outputs

- Per-run metrics: `data/<hostname>/.../{model}_metrics.csv`
- Execution metadata: `data/<hostname>/.../{model}_meta.json`
- Graph representation: `data/<hostname>/.../{model}_graph_nodes.csv` and `{model}_graph_edges.csv`
- Transfer-aware edges: `data/<hostname>/.../{model}_transfer_edges.csv`
- Robust aggregate: `data/<hostname>/.../{model}_metrics_stats.csv`
- ILP solution: `.../ilp_solution/ilp_assignment.csv`, `ilp_cut_edges.csv`, `ilp_solution_summary.json`
- Consolidated report assets: `reports/ilp_results*/`

All production data remains host-scoped under `data/<hostname>/...` so that heterogeneous hardware evidence is never mixed implicitly.

## Repository Map

```text
.
├── config/         # Environment and dependency definitions
├── data/           # Host-scoped experiment outputs and validation fixtures
├── datasets/       # Persisted datasets used by profiling and runtime
├── docs/           # Technical, operational, and thesis-support documentation
├── logs/           # Execution logs
├── reports/        # ILP outputs and thesis-ready report assets
├── scripts/        # Orchestration entrypoints
├── src/            # Profiling, ILP, runtime, and dataset integration code
├── tests/          # Pytest suite
├── thesis/         # LaTeX manuscript and generated PDF artifacts
├── validation/     # Validation, auditing, ILP, and reporting utilities
├── pytest.ini      # Pytest configuration
└── README.md       # This overview
```

For the structural map of responsibilities, see [docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md).

## Documentation Map

The documentation set was reduced so the core references now have distinct responsibilities.

| Document | Role |
|----------|------|
| [docs/README.md](docs/README.md) | Short operational guide |
| [docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md) | Structural map of the repository |
| [docs/GLOBAL_PROJECT_DOCUMENTATION.md](docs/GLOBAL_PROJECT_DOCUMENTATION.md) | Canonical technical reference in English |
| [docs/GLOBAL_PROJECT_DOCUMENTATION_ES.md](docs/GLOBAL_PROJECT_DOCUMENTATION_ES.md) | Canonical technical reference in academic Spanish |
| [docs/PROTOCOLO_VALIDACION_MULTISERVIDOR_ES.md](docs/PROTOCOLO_VALIDACION_MULTISERVIDOR_ES.md) | Master protocol for real multi-server data collection, Go/No-Go criteria, and operational closure |
| [docs/SERVER_LAUNCH_PROFILES.md](docs/SERVER_LAUNCH_PROFILES.md) | Launch profiles by server class |
| [docs/MULTI_NODE_ILP_RUNBOOK.md](docs/MULTI_NODE_ILP_RUNBOOK.md) | Multi-host discovery, merge, and solve workflow |
| [docs/CAPITULO_TESIS_PROFILING_ES.md](docs/CAPITULO_TESIS_PROFILING_ES.md) | Monographic chapter on profiling methodology |
| [docs/CAPITULO_TESIS_ILP_ES.md](docs/CAPITULO_TESIS_ILP_ES.md) | Monographic chapter on ILP formulation and validation |
| [docs/schema.md](docs/schema.md) | Writing map for the doctoral manuscript |
| [docs/QUICK_START.sh](docs/QUICK_START.sh) | Shell helper that prints frequent commands |

## System Requirements

- Python 3.10 or higher
- PyTorch 2.1.0 or higher
- CUDA 12.1 or higher for GPU execution
- NVIDIA GPU with NVML support for GPU energy monitoring
- Linux with RAPL support if CPU energy capture is required

## Citation

If you use PRISM in academic work, cite it as thesis code supporting the doctoral contribution:

```bibtex
@misc{torres2026prism,
  title={PRISM: Partitioning and Resource Intelligence for System Memory},
  author={Torres, Luis Alejandro},
  year={2026},
  howpublished={\url{https://github.com/alejandrotorresn/PRISM}},
  note={Hybrid CPU-GPU training optimization framework guided by profiling and ILP}
}
```

## License

This project is licensed under the MIT License. See the LICENSE file for details.

If you are interested in academic collaboration, please contact the author.

## Contact

Author: Luis Alejandro Torres  
Email: luis.torres@correo.uis.edu.co  
GitHub: @alejandrotorresn

For detailed command semantics, artifact schemas, or deployment guidance, continue from [docs/README.md](docs/README.md).

*Last Updated*: April 11, 2026
