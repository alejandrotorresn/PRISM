# Multi-Node ILP Runbook

This runbook documents the hardware-aware workflow end-to-end so it can be reused without memorizing commands.

## Goal

Combine profiling artifacts from multiple cluster nodes (different CPU/GPU/bus/memory) and solve one robust ILP model.

## 1) Profiling output convention

Each profiling run now writes under a host namespace:

- `data/<hostname>/...`

Typical experiment tree:

- `data/<hostname>/results/<model>/<optimizer>/<precision>/batch_<N>/run_*/`

## 2) Discover matching configs across hosts

Use the helper script to auto-discover all matching `batch_<N>` config folders:

```bash
bash scripts/discover_ilp_config_dirs.sh
```

Defaults used by the script:

- `MODEL=simple_mlp`
- `OPTIMIZER=SGD`
- `PRECISION=fp32`
- `BATCH=8`
- `RESULTS_ROOT=data`

Custom example:

```bash
MODEL=resnet50 OPTIMIZER=AdamW PRECISION=bf16 BATCH=32 bash scripts/discover_ilp_config_dirs.sh
```

The script prints `CONFIG_DIRS="dir1,dir2,..."` ready to use.

Important precondition:

- Discovery only keeps directories with complete ILP artifacts (`metrics_stats` + `graph_edges` + `transfer_edges`).
- If you only have legacy profiler outputs (without graph/transfer artifacts), use explicit `CONFIG_DIR`/`CONFIG_DIRS` wrappers with prepared ILP-ready folders.

## 3) Solve one merged ILP partition

### Option A: let discovery script run it directly

```bash
MODE=partition MODEL=simple_mlp bash scripts/discover_ilp_config_dirs.sh
```

### Option B: explicit wrapper call

```bash
CONFIG_DIRS="data/nodeA/results/simple_mlp/SGD/fp32/batch_8,data/nodeB/results/simple_mlp/SGD/fp32/batch_8" \
MODEL=simple_mlp \
HW_AGGREGATE=max \
HW_DISPERSION_K=0.0 \
bash scripts/run_ilp_partition.sh
```

## 4) Run Pareto sweep over merged hardware profiles

### Option A: direct from discovery

```bash
MODE=pareto MODEL=resnet50 GPU_BUDGETS_MB=400,800,1200 bash scripts/discover_ilp_config_dirs.sh
```

### Option B: explicit wrapper call

```bash
CONFIG_DIRS="data/nodeA/results/resnet50/SGD/fp32/batch_8,data/nodeB/results/resnet50/SGD/fp32/batch_8" \
MODEL=resnet50 \
GPU_BUDGETS_MB=400,800,1200 \
HW_AGGREGATE=max \
HW_DISPERSION_K=0.0 \
bash scripts/run_ilp_pareto_sweep.sh
```

## 5) Aggregation policy (multi-hardware)

Controls how node profiles are merged before solving:

- `HW_AGGREGATE=max`: conservative worst-case envelope across nodes.
- `HW_AGGREGATE=mean` with `HW_DISPERSION_K=K`: robust mean with variability margin (`mean + K*std`).

Recommendation:

- Start with `max` for safety-critical placement.
- Use `mean + K*std` when you want calibrated robustness instead of strict worst-case.

## 6) Files produced

Partition mode:

- `<base_config>/ilp_solution/ilp_assignment.csv`
- `<base_config>/ilp_solution/ilp_cut_edges.csv`
- `<base_config>/ilp_solution/ilp_solution_summary.json`

Pareto mode:

- `<base_config>/<model>_pareto_sweep.csv`
- `<base_config>/<model>_pareto_summary.json`

`<base_config>` is the first directory in `CONFIG_DIRS` unless `OUT_DIR`, `OUT_CSV`, or `OUT_JSON` are set.

## 7) Troubleshooting

- No matches found in discovery:
  - Check `MODEL`, `OPTIMIZER`, `PRECISION`, `BATCH` values.
  - Confirm output layout under `data/<hostname>/results/...`.
- Candidate dirs found but rejected:
  - Ensure each directory contains complete ILP artifacts (`metrics_stats`, `*_graph_edges.csv`, `*_transfer_edges.csv`).
- Node schema mismatch during merge:
  - Ensure all nodes used in `CONFIG_DIRS` come from the same model/optimizer/precision/batch pipeline stage.
- Want a reproducible snapshot of discovered vars:
  - `OUTPUT_ENV_FILE=.ilp_multi_node.env bash scripts/discover_ilp_config_dirs.sh`

---

Last updated: March 12, 2026.
