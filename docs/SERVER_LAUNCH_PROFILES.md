# Server Launch Profiles

This document defines recommended launch configurations for `scripts/run_experiments.sh` across heterogeneous servers.

The goal is to avoid mixing incompatible runtime assumptions across nodes while keeping output layout compatible with the multi-node ILP workflow.

## Core Rule

Use one launch profile per server class.

Do not launch the exact same full grid blindly on every machine if hardware support differs in:

- CUDA availability
- CPU precision ISA support (`fp16`, `bf16`)
- RAPL permissions
- available CPU cores
- usable batch-size range

## Supported Environment Overrides

`scripts/run_experiments.sh` accepts the following comma-separated grid overrides:

- `MODELS_CSV`
- `BATCH_SIZES_CSV`
- `PRECISIONS_CSV`
- `OPTIMIZERS_CSV`

It also accepts operational flags such as:

- `USE_SKIP_CPU=true|false`
- `ENABLE_RAPL=true|false`
- `FORCE_THREADS=N`
- `REPEATS=N`
- `WARMUP=N`
- `MEASURE=N`
- `FAIL_FAST=true|false`
- `DRY_RUN=true|false`
- `BASE_OUTPUT_DIR=...`
- `PYTHON_CMD=...`

Default output root remains host-scoped under:

`data/<hostname>/results/...`

## Profile 0: Preflight Validation

Use this on every server before a real campaign.

```bash
SMOKE_MODE=true \
DRY_RUN=true \
FAIL_FAST=true \
PYTHON_CMD=.venv/bin/python \
bash scripts/run_experiments.sh
```

Purpose:

- validate interpreter and imports
- validate GPU detection
- validate command construction
- validate aggregator preflight

## Profile 1: GPU-Only Multi-Precision

Use this on GPU servers when you want `fp32`, `fp16`, and `bf16` without being blocked by CPU precision limitations.

```bash
MODELS_CSV=simple_mlp,resnet50,resnet152,vit_b16,bert_base,gpt2_small,distilgpt2 \
BATCH_SIZES_CSV=8,16,32,64 \
PRECISIONS_CSV=fp32,fp16,bf16 \
OPTIMIZERS_CSV=SGD,AdamW \
USE_SKIP_CPU=true \
ENABLE_RAPL=false \
FORCE_THREADS=8 \
REPEATS=3 \
WARMUP=3 \
MEASURE=10 \
FAIL_FAST=false \
PYTHON_CMD=.venv/bin/python \
bash scripts/run_experiments.sh
```

## Profile 2: CPU+GPU FP32 Thesis Baseline

Use this on well-instrumented nodes when you want full CPU and GPU profiling with the most portable precision.

```bash
MODELS_CSV=simple_mlp,resnet50,resnet152,vit_b16,bert_base,gpt2_small,distilgpt2 \
BATCH_SIZES_CSV=8,16,32,64 \
PRECISIONS_CSV=fp32 \
OPTIMIZERS_CSV=SGD,AdamW,RMSprop \
USE_SKIP_CPU=false \
ENABLE_RAPL=true \
FORCE_THREADS=16 \
REPEATS=3 \
WARMUP=3 \
MEASURE=10 \
FAIL_FAST=false \
PYTHON_CMD=.venv/bin/python \
bash scripts/run_experiments.sh
```

## Profile 3: CPU+GPU BF16-Qualified Nodes

Use this only on nodes already validated for accelerated BF16 support.

```bash
MODELS_CSV=simple_mlp,resnet50,vit_b16,bert_base \
BATCH_SIZES_CSV=8,16,32 \
PRECISIONS_CSV=fp32,bf16 \
OPTIMIZERS_CSV=SGD,AdamW \
USE_SKIP_CPU=false \
ENABLE_RAPL=true \
FORCE_THREADS=16 \
REPEATS=3 \
WARMUP=3 \
MEASURE=10 \
FAIL_FAST=false \
PYTHON_CMD=.venv/bin/python \
bash scripts/run_experiments.sh
```

Validate such nodes first with:

```bash
.venv/bin/python validation/validate_all_models.py --preflight-scope fast
```

## Profile 4: Conservative Low-Memory GPU Node

Use this for older or smaller GPUs where OOM is likely.

```bash
MODELS_CSV=simple_mlp,resnet50,vit_b16 \
BATCH_SIZES_CSV=8,16 \
PRECISIONS_CSV=fp32,fp16 \
OPTIMIZERS_CSV=SGD \
USE_SKIP_CPU=true \
ENABLE_RAPL=false \
FORCE_THREADS=4 \
REPEATS=2 \
WARMUP=2 \
MEASURE=5 \
FAIL_FAST=false \
PYTHON_CMD=.venv/bin/python \
bash scripts/run_experiments.sh
```

## Profile 5: Real Execution Smoke Test

Run this before a full production launch on each server class.

```bash
MODELS_CSV=simple_mlp \
BATCH_SIZES_CSV=8 \
PRECISIONS_CSV=fp32 \
OPTIMIZERS_CSV=SGD \
USE_SKIP_CPU=true \
ENABLE_RAPL=false \
REPEATS=1 \
WARMUP=1 \
MEASURE=1 \
FAIL_FAST=true \
PYTHON_CMD=.venv/bin/python \
bash scripts/run_experiments.sh
```

### Why Profile 5 uses the same base config

Profile 5 is a canonical operational smoke test, not a performance benchmark.

Use the same base mini-campaign across server classes so pass/fail signals are comparable:

- `MODELS_CSV=simple_mlp`
- `BATCH_SIZES_CSV=8`
- `PRECISIONS_CSV=fp32`
- `OPTIMIZERS_CSV=SGD`
- `REPEATS=1`, `WARMUP=1`, `MEASURE=1`

Primary objective:

- verify end-to-end execution on real hardware
- verify artifact generation and folder schema
- detect environment-specific failures early

Expected artifacts after a successful Profile 5 run:

- `run_*/<model>_metrics.csv`
- `run_*/<model>_meta.json`
- `run_*/<model>_graph_edges.csv`
- `run_*/<model>_transfer_edges.csv`
- `<model>_metrics_stats.csv`

Allowed per-class toggles for Profile 5:

- `USE_SKIP_CPU=true|false`
- `ENABLE_RAPL=true|false`
- `FORCE_THREADS=N`

Keep all other Profile 5 values fixed unless the canonical smoke is infeasible on that class.

## Recommended Deployment Order

1. Run Profile 0 on every server.
2. Run Profile 5 (canonical base config) on one node of each server class.
3. Select one of Profiles 1-4 according to hardware capability.
4. Keep one output root per server hostname.
5. Merge compatible config directories later with the multi-node ILP workflow.

## Post-Run Checks

```bash
tail -f logs/experiments_*.txt
```

```bash
find data -path '*/results/*/batch_*' -maxdepth 8 -type d | sort
```

Each expected config folder should contain:

- `run_*/<model>_metrics.csv`
- `run_*/<model>_meta.json`
- `run_*/<model>_graph_edges.csv`
- `run_*/<model>_transfer_edges.csv`
- `<model>_metrics_stats.csv`

## Multi-Node Merge

Once multiple servers have completed compatible runs, use:

- `scripts/discover_ilp_config_dirs.sh`
- `scripts/run_ilp_partition.sh`
- `scripts/run_ilp_pareto_sweep.sh`

See also:

- [MULTI_NODE_ILP_RUNBOOK.md](MULTI_NODE_ILP_RUNBOOK.md)
- [README.md](README.md)

---

Last updated: March 14, 2026.