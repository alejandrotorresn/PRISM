#!/bin/bash
set -euo pipefail

# Usage example:
# PYTHON_CMD=./.venv/bin/python MODEL=resnet50 CONFIG_DIR=data/test-m4/resnet50/SGD/fp32/batch_8 \
# GPU_BUDGETS_MB=400,600,800,1000 CPU_MEM_BUDGET_MB=3000 bash scripts/run_ilp_pareto_sweep.sh

PYTHON_CMD="${PYTHON_CMD:-python}"
MODEL="${MODEL:-resnet50}"
CONFIG_DIR="${CONFIG_DIR:-data/test-m4/resnet50/SGD/fp32/batch_8}"
CONFIG_DIRS="${CONFIG_DIRS:-}"
GPU_BUDGETS_MB="${GPU_BUDGETS_MB:-400,600,800,1000,1200}"
CPU_MEM_BUDGET_MB="${CPU_MEM_BUDGET_MB:-1e18}"
K_SIGMA="${K_SIGMA:-1.0}"
W_TIME="${W_TIME:-1.0}"
W_ENERGY="${W_ENERGY:-0.0}"
W_TRANSFER="${W_TRANSFER:-1.0}"
BACKEND="${BACKEND:-auto}"
HW_AGGREGATE="${HW_AGGREGATE:-max}"
HW_DISPERSION_K="${HW_DISPERSION_K:-0.0}"
OUT_CSV="${OUT_CSV:-${CONFIG_DIR}/${MODEL}_pareto_sweep.csv}"
OUT_JSON="${OUT_JSON:-${CONFIG_DIR}/${MODEL}_pareto_summary.json}"
STRICT_GRAPH_MAPPING="${STRICT_GRAPH_MAPPING:-true}"
STRICT_TRANSFER_MAPPING="${STRICT_TRANSFER_MAPPING:-true}"

STRICT_FLAGS=()
if [ "$STRICT_GRAPH_MAPPING" = true ]; then
  STRICT_FLAGS+=(--strict_graph_mapping)
fi
if [ "$STRICT_TRANSFER_MAPPING" = true ]; then
  STRICT_FLAGS+=(--strict_transfer_mapping)
fi

CONFIG_FLAGS=(--config_dir "$CONFIG_DIR")
if [ -n "$CONFIG_DIRS" ]; then
  CONFIG_FLAGS=(--config_dirs "$CONFIG_DIRS")
fi

"$PYTHON_CMD" validation/sweep_ilp_pareto.py \
  "${CONFIG_FLAGS[@]}" \
  --model "$MODEL" \
  --gpu_budgets_mb "$GPU_BUDGETS_MB" \
  --cpu_mem_budget_mb "$CPU_MEM_BUDGET_MB" \
  --k_sigma "$K_SIGMA" \
  --w_time "$W_TIME" \
  --w_energy "$W_ENERGY" \
  --w_transfer "$W_TRANSFER" \
  --backend "$BACKEND" \
  --hw_aggregate "$HW_AGGREGATE" \
  --hw_dispersion_k "$HW_DISPERSION_K" \
  --output_csv "$OUT_CSV" \
  --output_json "$OUT_JSON" \
  "${STRICT_FLAGS[@]}"
