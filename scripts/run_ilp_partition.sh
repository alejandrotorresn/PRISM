#!/bin/bash
set -euo pipefail

# Usage:
#   MODEL=simple_mlp CONFIG_DIR=data/test-m3-r2/simple_mlp/SGD/fp32/batch_8 bash scripts/run_ilp_partition.sh

PYTHON_CMD="${PYTHON_CMD:-python}"
MODEL="${MODEL:-simple_mlp}"
CONFIG_DIR="${CONFIG_DIR:-data/test-m3-r2/simple_mlp/SGD/fp32/batch_8}"
K_SIGMA="${K_SIGMA:-1.0}"
W_TIME="${W_TIME:-1.0}"
W_ENERGY="${W_ENERGY:-0.0}"
W_TRANSFER="${W_TRANSFER:-1.0}"
GPU_MEM_BUDGET_MB="${GPU_MEM_BUDGET_MB:-1e18}"
CPU_MEM_BUDGET_MB="${CPU_MEM_BUDGET_MB:-1e18}"
BACKEND="${BACKEND:-auto}"
OUT_DIR="${OUT_DIR:-${CONFIG_DIR}/ilp_solution}"
STRICT_GRAPH_MAPPING="${STRICT_GRAPH_MAPPING:-true}"
STRICT_TRANSFER_MAPPING="${STRICT_TRANSFER_MAPPING:-true}"

STRICT_FLAGS=()
if [ "$STRICT_GRAPH_MAPPING" = true ]; then
  STRICT_FLAGS+=(--strict_graph_mapping)
fi
if [ "$STRICT_TRANSFER_MAPPING" = true ]; then
  STRICT_FLAGS+=(--strict_transfer_mapping)
fi

"$PYTHON_CMD" validation/run_ilp_partition.py \
  --config_dir "$CONFIG_DIR" \
  --model "$MODEL" \
  --k_sigma "$K_SIGMA" \
  --w_time "$W_TIME" \
  --w_energy "$W_ENERGY" \
  --w_transfer "$W_TRANSFER" \
  --gpu_mem_budget_mb "$GPU_MEM_BUDGET_MB" \
  --cpu_mem_budget_mb "$CPU_MEM_BUDGET_MB" \
  --backend "$BACKEND" \
  --output_dir "$OUT_DIR" \
  "${STRICT_FLAGS[@]}"
