[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)
[![PyTorch 2.5.x](https://img.shields.io/badge/PyTorch-2.5.x-red)](https://pytorch.org)

# PRISM

*Partitioning and Routing via ILP for Scalable Memory*  
*A Hybrid CPU-GPU Training Optimization Framework Guided by Profiling and ILP*

Research code for a complete pipeline that measures deep learning training costs, builds robust Integer Linear Programming partition models, and validates hybrid CPU-GPU execution strategies aimed at reducing GPU VRAM pressure while making the CPU an active participant in training.

## Overview

PRISM is organized as an end-to-end system, not as a standalone profiler. Its core contribution is the connection between empirical evidence and optimization-driven execution:

- layer-wise profiling on CPU and GPU with time, energy, memory, FLOPs, and transfer-aware artifacts
- robust statistical aggregation across replicas and across heterogeneous servers
- ILP-based partitioning under latency, memory, energy, and transfer constraints
- simulation and hybrid runtime validation of the generated plans
- analytical estimation of GPU constraints for OOM (Out-of-Memory) models using structural CPU profiling and empirical TFLOPS
- thesis-ready report generation featuring Top-K bottleneck horizontal charting and clean continuous Pareto curves

The practical question addressed by PRISM is simple: how to decide which parts of a model should remain on GPU and which can be moved to CPU without treating the CPU as a passive fallback, but rather as an active computational actor in training.

## End-to-End Workflow

1. `src/profiler.py` and `src/runner/training_profiler.py` capture per-layer measurements and structural artifacts.
2. `validation/aggregate_metrics_stats.py` and `src/core/stats_aggregator.py` convert repeated runs into robust coefficients.
3. `validation/run_ilp_partition.py` and `validation/sweep_ilp_pareto.py` solve placement and budget trade-off problems.
4. `src/runtime/` and `validation/run_hybrid_execution.py` validate those plans through simulation or physical hybrid execution.
5. `validation/generate_ilp_report_assets.py`, `validation/export_ilp_tables_latex.py`, `reports/`, and `final_thesis/` turn the results into analyzable and publishable evidence.

## Quick Start

### Installation

```bash
git clone <repo-url>
cd <repo-folder>

# Option A: Conda
conda env create -f config/environment.yml
conda activate prism_env

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

### Thesis mode profiles

`scripts/run_thesis_mode.sh` supports the following profiles:

- `doctoral_minimal`: official strict doctoral profile (core evidence).
- `doctoral_full`: official strict doctoral profile with broader campaign scope.
- `doctoral_diagnostic`: fallback-enabled diagnostic profile for troubleshooting only.
- `quick_smoke`: minimal operational smoke path.

### OOM Hard-Crash Handling
For architectures that exceed physical VRAM and cause hard system crashes during empirical profiling, PRISM provides an **Analytical Estimation Regime**:
1. Run the profiler with `--no_gpu` to safely trace the structural execution graph and tensor dimensions on CPU.
2. PRISM automatically projects the missing GPU metrics (latency and energy) based on measured CPU FLOPs and the host's empirically benchmarked GPU TFLOPS.
3. The ILP partitioner seamlessly accepts these projections for robust simulation.
- `custom`: fully explicit manual configuration.

Example:

```bash
PROFILE=doctoral_diagnostic RUN_HYBRID=true bash scripts/run_thesis_mode.sh
```

### Single ILP solve from existing artifacts

```bash
python validation/run_ilp_partition.py \
  --config_dir data/<hostname>/results/simple_mlp/SGD/fp32/batch_8 \
  --model simple_mlp
```

### Grid5000 GPU Discovery

For deploying campaigns on Grid5000, PRISM provides a utility to automatically scan available OAR nodes that have free GPUs, matching specific CPU constraints or cluster locations via SSH proxy jumps:

```bash
python scripts/find_free_gpus.py \
  --username ltorresnino \
  --site lille \
  --cpu-types Intel \
  --proxy-jump access.grid5000.fr
```

### Statistical Significance Validation

To validate that ILP-derived optimizations are statistically significant compared to All-GPU or All-CPU baselines, PRISM automatically calculates a Student's t-test (p-value) for small samples ($n=5$) and the Cohen's d effect size using the maximum coefficient of variation:

```bash
python validation/run_statistical_significance.py \
  --consolidated_csv reports/ilp_results/csv/ilp_pareto_consolidated.csv \
  --output_csv reports/ilp_results/csv/ilp_statistical_significance.csv
```
*(This is seamlessly executed in Step 5 of `run_thesis_mode.sh` and exported directly to LaTeX).*

### Pareto sweep over GPU budgets

```bash
python validation/sweep_ilp_pareto.py \
  --config_dir data/<hostname>/results/simple_mlp/SGD/fp32/batch_8 \
  --model simple_mlp \
  --gpu_budgets_mb 400,600,800,1000
```

### Simulation-vs-Real coupling in hybrid protocol

The Pareto sweep now exports simulator metrics per budget directly in
`{model}_pareto_sweep.csv`, including:

- `sim_time_ms`
- `sim_energy_j`
- `sim_status`

When `scripts/run_thesis_mode.sh` selects `HYBRID_PLAN_SELECTION=pareto_best`, these
simulated references are forwarded to `validation/run_hybrid_execution.py` and written
into `hybrid_execution_protocol.csv` together with:

- `delta_time_sim_vs_real_pct`
- `delta_energy_sim_vs_real_pct`

This closes the methodological loop between simulated ILP evaluation and physical
hybrid execution, allowing explicit quantification of temporal and energetic divergence.

### Grid Audit (Post-Collection)

`scripts/generate_grid_audit.py` is an operational auditing tool designed to evaluate the completeness of the executed grid across all servers. **This script should only be executed after all data from all execution nodes has been collected and consolidated in the `data/` directory.** It generates a comprehensive markdown report (`GRID_AUDIT_SUMMARY.md`) showing the readiness of the data for inclusion in the thesis chapters, but this report itself is not meant to be included in the academic manuscript.

```bash
python scripts/generate_grid_audit.py --input_root data/
```

### Server preflight before real collection

```bash
conda activate prism_env
SMOKE_MODE=true \
DRY_RUN=true \
FAIL_FAST=true \
bash scripts/run_experiments.sh
```

### External server preflight (for Grid5000 or similar)

Before consuming a reservation on a new server, run a strict preflight on a fresh clone and a fresh environment.

```bash
git clone <repo-url>
cd <repo-folder>
conda env create -f config/environment.yml
conda activate prism_env

# Verify CBC executable is present and callable
which cbc && cbc -stop

# Verify PuLP can see a CBC backend
python - <<'PY'
import pulp
print('PuLP:', pulp.__version__)
print('CBC available:', pulp.COIN_CMD().available())
PY
```

If `cbc` is not found, install it in the same environment before launching campaigns:

```bash
conda install -n prism_env -c conda-forge coin-or-cbc
```

This project is fail-fast by design in strict modes. If CBC is missing on the target server, ILP stages can fail and you may lose reserved execution time.

For profiling campaigns, OOM handling is controlled independently from `FAIL_FAST`:

- `OOM_SKIP_CONFIGS=true` (default): when a configuration exhausts OOM retries, the campaign marks that batch configuration as skipped (`oom_skipped_config.json`) and continues.
- `OOM_SKIP_CONFIGS=false`: OOM behaves as a normal failure and `FAIL_FAST=true` aborts immediately.

Example (strict abort also on OOM):

```bash
FAIL_FAST=true OOM_SKIP_CONFIGS=false bash scripts/run_experiments.sh
```

### RAPL permissions and expected warnings

CPU energy capture via RAPL requires kernel and filesystem permissions on the host. On many shared HPC systems, direct access to `/sys/class/powercap/.../energy_uj` is denied to regular users.

1) Check whether RAPL is available and readable for your current user:

```bash
# Use -L because /sys/class/powercap entries are usually symlinks in sysfs.
RAPL_FILES="$(find -L /sys/class/powercap -type f -name energy_uj 2>/dev/null)"
if [ -z "$RAPL_FILES" ]; then
  echo "RAPL not available on this host"
else
  echo "RAPL files detected:" && echo "$RAPL_FILES"
  for f in $RAPL_FILES; do
    [ -r "$f" ] && echo "OK   $f" || echo "NOPE $f"
  done
fi
```

2) If files exist but are not readable, use a temporary chmod fix for the current boot (requires root):

```bash
sudo chmod a+r /sys/class/powercap/intel-rapl:*/energy_uj /sys/class/powercap/intel-rapl:*/intel-rapl:*/energy_uj
sudo chmod a+r /sys/class/powercap/intel-rapl:*/max_energy_range_uj /sys/class/powercap/intel-rapl:*/intel-rapl:*/max_energy_range_uj
```

3) For a permanent solution across reboots, create an udev rule (recommended on Fedora and Rocky Linux):

```bash
sudo tee /etc/udev/rules.d/99-powercap.rules >/dev/null <<'EOF'
SUBSYSTEM=="powercap", RUN+="/bin/chmod 444 /sys/class/powercap/%k/energy_uj /sys/class/powercap/%k/max_energy_range_uj"
EOF

sudo udevadm control --reload-rules
sudo udevadm trigger --subsystem-match=powercap
```

4) Re-run the check command above. If permission is still unavailable and you do not have root control on the node, run with RAPL disabled:

```bash
ENABLE_RAPL=false PROFILE=doctoral_minimal bash scripts/run_thesis_mode.sh
```

Note: ACL-based approaches (for example `setfacl`) can fail on sysfs with `Operation not supported`, because sysfs is a virtual kernel filesystem and may not expose ACL/xattr semantics like a regular disk filesystem.

If RAPL remains unavailable, warnings such as the following are expected and non-fatal:

```text
pyRAPL Init failed: [Errno 13] Permission denied: '/sys/class/powercap/.../energy_uj'
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
├── docs/           # Documentation index plus architecture/, operations/, archive/
├── logs/           # Execution logs grouped by workflow plus archive/
├── reports/        # Canonical ILP outputs, thesis figures, and diagnostics/
├── scripts/        # Orchestration entrypoints
├── src/            # Profiling, ILP, runtime, and dataset integration code
├── tests/          # Pytest suite
├── final_thesis/   # LaTeX manuscript and generated PDF artifacts
├── validation/     # Validation, auditing, ILP, and reporting utilities
├── pytest.ini      # Pytest configuration
└── README.md       # This overview
```

For the structural map of responsibilities, see [docs/architecture/PROJECT_STRUCTURE.md](docs/architecture/PROJECT_STRUCTURE.md).

## Documentation Map

The documentation set was reduced so the core references now have distinct responsibilities.

| Document | Role |
|----------|------|
| [docs/README.md](docs/README.md) | Main documentation index and navigation guide |
| [docs/architecture/PROJECT_STRUCTURE.md](docs/architecture/PROJECT_STRUCTURE.md) | Structural map of the repository |
| [docs/architecture/GLOBAL_PROJECT_DOCUMENTATION.md](docs/architecture/GLOBAL_PROJECT_DOCUMENTATION.md) | Canonical technical reference in English |
| [docs/architecture/GLOBAL_PROJECT_DOCUMENTATION_ES.md](docs/architecture/GLOBAL_PROJECT_DOCUMENTATION_ES.md) | Canonical technical reference in academic Spanish |
| [docs/operations/PROTOCOLO_VALIDACION_MULTISERVIDOR_ES.md](docs/operations/PROTOCOLO_VALIDACION_MULTISERVIDOR_ES.md) | Master protocol for real multi-server data collection, Go/No-Go criteria, and operational closure |
| [docs/operations/SERVER_LAUNCH_PROFILES.md](docs/operations/SERVER_LAUNCH_PROFILES.md) | Launch profiles by server class |
| [docs/operations/MULTI_NODE_ILP_RUNBOOK.md](docs/operations/MULTI_NODE_ILP_RUNBOOK.md) | Multi-host discovery, merge, and solve workflow |
| [docs/operations/QUICK_START.sh](docs/operations/QUICK_START.sh) | Shell helper that prints frequent commands |

## System Requirements

- Python 3.10 or higher
- PyTorch 2.5.x (validated baseline from config/environment.yml)
- CUDA 12.1 or higher for GPU execution
- NVIDIA GPU with NVML support for GPU energy monitoring
- Linux with RAPL support if CPU energy capture is required

## Citation

If you use PRISM in academic work, cite it as thesis code supporting the doctoral contribution:

```bibtex
@misc{torres2026prism,
  title={PRISM: Partitioning and Routing via ILP for Scalable Memory},
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

*Last Updated*: June 4, 2026
