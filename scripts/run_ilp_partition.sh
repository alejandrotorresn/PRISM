#!/bin/bash
set -euo pipefail

# Usage:
#   MODEL=simple_mlp CONFIG_DIR=data/test-m3-r2/simple_mlp/SGD/fp32/batch_8 bash scripts/run_ilp_partition.sh

source "$(dirname "$0")/sanitize_cuda_env.sh"
sanitize_cuda_runtime_env

PYTHON_CMD="${PYTHON_CMD:-python}"
MODEL="${MODEL:-simple_mlp}"
CONFIG_DIR="${CONFIG_DIR:-data/test-m3-r2/simple_mlp/SGD/fp32/batch_8}"
CONFIG_DIRS="${CONFIG_DIRS:-}"
K_SIGMA="${K_SIGMA:-1.0}"
W_TIME="${W_TIME:-1.0}"
W_ENERGY="${W_ENERGY:-0.0}"
W_TRANSFER="${W_TRANSFER:-1.0}"
GPU_MEM_BUDGET_MB="${GPU_MEM_BUDGET_MB:-1e18}"
CPU_MEM_BUDGET_MB="${CPU_MEM_BUDGET_MB:-1e18}"
BACKEND="${BACKEND:-auto}"
HW_AGGREGATE="${HW_AGGREGATE:-max}"
HW_DISPERSION_K="${HW_DISPERSION_K:-0.0}"
OUT_DIR="${OUT_DIR:-${CONFIG_DIR}/ilp_solution}"
STRICT_GRAPH_MAPPING="${STRICT_GRAPH_MAPPING:-true}"
STRICT_TRANSFER_MAPPING="${STRICT_TRANSFER_MAPPING:-true}"
ALLOW_LOW_QUALITY_STATS="${ALLOW_LOW_QUALITY_STATS:-false}"
ALLOW_TRANSFER_CALIBRATION_FALLBACK="${ALLOW_TRANSFER_CALIBRATION_FALLBACK:-false}"
ALLOW_FALLBACK_GRAPH_TRACE="${ALLOW_FALLBACK_GRAPH_TRACE:-false}"

STRICT_FLAGS=()
if [ "$STRICT_GRAPH_MAPPING" = true ]; then
  STRICT_FLAGS+=(--strict_graph_mapping)
fi
if [ "$STRICT_TRANSFER_MAPPING" = true ]; then
  STRICT_FLAGS+=(--strict_transfer_mapping)
fi
if [ "$ALLOW_LOW_QUALITY_STATS" = true ]; then
  STRICT_FLAGS+=(--allow_low_quality_stats)
fi
if [ "$ALLOW_TRANSFER_CALIBRATION_FALLBACK" = true ]; then
  STRICT_FLAGS+=(--allow_transfer_calibration_fallback)
fi
if [ "$ALLOW_FALLBACK_GRAPH_TRACE" = true ]; then
  STRICT_FLAGS+=(--allow_fallback_graph_trace)
fi

CONFIG_FLAGS=(--config_dir "$CONFIG_DIR")
if [ -n "$CONFIG_DIRS" ]; then
  CONFIG_FLAGS=(--config_dirs "$CONFIG_DIRS")
fi

"$PYTHON_CMD" validation/run_ilp_partition.py \
  "${CONFIG_FLAGS[@]}" \
  --model "$MODEL" \
  --k_sigma "$K_SIGMA" \
  --w_time "$W_TIME" \
  --w_energy "$W_ENERGY" \
  --w_transfer "$W_TRANSFER" \
  --gpu_mem_budget_mb "$GPU_MEM_BUDGET_MB" \
  --cpu_mem_budget_mb "$CPU_MEM_BUDGET_MB" \
  --backend "$BACKEND" \
  --hw_aggregate "$HW_AGGREGATE" \
  --hw_dispersion_k "$HW_DISPERSION_K" \
  --output_dir "$OUT_DIR" \
  "${STRICT_FLAGS[@]}"
