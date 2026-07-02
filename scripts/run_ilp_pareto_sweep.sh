#!/bin/bash
set -euo pipefail

# Usage example:
# PYTHON_CMD=./.venv/bin/python MODEL=resnet50 CONFIG_DIR=data/test-m4/resnet50/SGD/fp32/batch_8 \
# GPU_BUDGETS_MB=auto CPU_MEM_BUDGET_MB=3000 bash scripts/run_ilp_pareto_sweep.sh

source "$(dirname "$0")/sanitize_cuda_env.sh"
sanitize_cuda_runtime_env

PYTHON_CMD="${PYTHON_CMD:-python}"
MODEL="${MODEL:-resnet50}"
CONFIG_DIR="${CONFIG_DIR:-data/test-m4/resnet50/SGD/fp32/batch_8}"
CONFIG_DIRS="${CONFIG_DIRS:-}"
GPU_BUDGETS_MB="${GPU_BUDGETS_MB:-auto}"
CPU_MEM_BUDGET_MB="${CPU_MEM_BUDGET_MB:-1e18}"

# ---- Resolve GPU_BUDGETS_MB=auto to dynamic values ----
if [ "$GPU_BUDGETS_MB" = "auto" ]; then
  _GPU_VRAM_MB=""
  if command -v nvidia-smi >/dev/null 2>&1; then
    _GPU_VRAM_MB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -d '[:space:]')"
  fi
  if [ -z "$_GPU_VRAM_MB" ] || [ "$_GPU_VRAM_MB" -le 0 ] 2>/dev/null; then
    echo "[ERROR] GPU_BUDGETS_MB=auto but could not detect GPU VRAM via nvidia-smi." >&2
    echo "        Set GPU_BUDGETS_MB explicitly (e.g. GPU_BUDGETS_MB=5000,6000,7000)." >&2
    exit 1
  fi
  _BUDGETS=""
  for _pct in 90 85 80 75 70 50; do
    _val=$(( _GPU_VRAM_MB * _pct / 100 ))
    if [ -z "$_BUDGETS" ]; then _BUDGETS="$_val"; else _BUDGETS="$_BUDGETS,$_val"; fi
  done
  GPU_BUDGETS_MB="$_BUDGETS"
  echo "[INFO] GPU VRAM detected: ${_GPU_VRAM_MB} MiB -> dynamic budgets: $GPU_BUDGETS_MB"
fi

MEMORY_MODEL="${MEMORY_MODEL:-topological}"
PEAK_ACTIVATION_OVERLAP="${PEAK_ACTIVATION_OVERLAP:-0.35}"
K_SIGMA="${K_SIGMA:-1.0}"
W_TIME="${W_TIME:-1.0}"
W_ENERGY="${W_ENERGY:-0.0}"
W_TRANSFER="${W_TRANSFER:-1.0}"
BACKEND="${BACKEND:-auto}"
HW_AGGREGATE="${HW_AGGREGATE:-max}"
HW_DISPERSION_K="${HW_DISPERSION_K:-0.0}"
OUT_CSV="${OUT_CSV:-${CONFIG_DIR}/${MODEL}_pareto_sweep.csv}"
OUT_JSON="${OUT_JSON:-${CONFIG_DIR}/${MODEL}_pareto_summary.json}"
REGIME="${REGIME:-diagnostic}"
ENFORCE_CONVEX_WEIGHTS="${ENFORCE_CONVEX_WEIGHTS:-false}"
STRICT_GRAPH_MAPPING="${STRICT_GRAPH_MAPPING:-true}"
STRICT_TRANSFER_MAPPING="${STRICT_TRANSFER_MAPPING:-true}"
ALLOW_LOW_QUALITY_STATS="${ALLOW_LOW_QUALITY_STATS:-false}"
ALLOW_TRANSFER_CALIBRATION_FALLBACK="${ALLOW_TRANSFER_CALIBRATION_FALLBACK:-false}"
ALLOW_FALLBACK_GRAPH_TRACE="${ALLOW_FALLBACK_GRAPH_TRACE:-false}"
STRICT_METRIC_VALIDITY="${STRICT_METRIC_VALIDITY:-false}"
FAIL_FAST_PROFILE_ERRORS="${FAIL_FAST_PROFILE_ERRORS:-false}"

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
if [ "$STRICT_METRIC_VALIDITY" = true ]; then
  STRICT_FLAGS+=(--strict_metric_validity)
fi
if [ "$FAIL_FAST_PROFILE_ERRORS" = true ]; then
  STRICT_FLAGS+=(--fail_fast_profile_errors)
fi
if [ "$ENFORCE_CONVEX_WEIGHTS" = true ]; then
  STRICT_FLAGS+=(--enforce_convex_weights)
fi

CONFIG_FLAGS=(--config_dir "$CONFIG_DIR")
if [ -n "$CONFIG_DIRS" ]; then
  CONFIG_FLAGS=(--config_dirs "$CONFIG_DIRS")
fi

"$PYTHON_CMD" validation/sweep_ilp_pareto.py \
  "${CONFIG_FLAGS[@]}" \
  --model "$MODEL" \
  --regime "$REGIME" \
  --gpu_budgets_mb "$GPU_BUDGETS_MB" \
  --cpu_mem_budget_mb "$CPU_MEM_BUDGET_MB" \
  --memory_model "$MEMORY_MODEL" \
  --peak_activation_overlap "$PEAK_ACTIVATION_OVERLAP" \
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
