#!/bin/bash
set -euo pipefail

# Usage example:
# PYTHON_CMD=python \
# INPUT_ROOT=data/musa-5/thesis_runs/job_xxx/doctoral_minimal/seed_42 \
# OUTPUT_DIR=reports/ilp_results/grid5k_musa-5_thesis_mode/doctoral_minimal/seed_42/audit \
# EXPECTED_MODELS_CSV=simple_mlp,resnet50,resnet152,vit_b16 \
# EXPECTED_OPTIMIZERS_CSV=SGD,AdamW \
# EXPECTED_PRECISIONS_CSV=fp32 \
# EXPECTED_BATCHES_CSV=8,32,64 \
# EXPECTED_REPEATS=1 \
# LOGS_ROOT=logs \
# bash scripts/audit_experimental_grid.sh

PYTHON_CMD="${PYTHON_CMD:-python}"
INPUT_ROOT="${INPUT_ROOT:-data/test-m4}"
OUTPUT_DIR="${OUTPUT_DIR:-reports/ilp_results/grid_audit}"
EXPECTED_MODELS_CSV="${EXPECTED_MODELS_CSV:-}"
EXPECTED_OPTIMIZERS_CSV="${EXPECTED_OPTIMIZERS_CSV:-}"
EXPECTED_PRECISIONS_CSV="${EXPECTED_PRECISIONS_CSV:-}"
EXPECTED_BATCHES_CSV="${EXPECTED_BATCHES_CSV:-}"
EXPECTED_REPEATS="${EXPECTED_REPEATS:-1}"
LOGS_ROOT="${LOGS_ROOT:-}"
REQUIRE_HYBRID="${REQUIRE_HYBRID:-false}"

ARGS=(
  --input_root "$INPUT_ROOT"
  --output_dir "$OUTPUT_DIR"
  --expected_repeats "$EXPECTED_REPEATS"
)

if [ -n "$EXPECTED_MODELS_CSV" ]; then
  ARGS+=(--expected_models_csv "$EXPECTED_MODELS_CSV")
fi
if [ -n "$EXPECTED_OPTIMIZERS_CSV" ]; then
  ARGS+=(--expected_optimizers_csv "$EXPECTED_OPTIMIZERS_CSV")
fi
if [ -n "$EXPECTED_PRECISIONS_CSV" ]; then
  ARGS+=(--expected_precisions_csv "$EXPECTED_PRECISIONS_CSV")
fi
if [ -n "$EXPECTED_BATCHES_CSV" ]; then
  ARGS+=(--expected_batches_csv "$EXPECTED_BATCHES_CSV")
fi
if [ -n "$LOGS_ROOT" ]; then
  ARGS+=(--logs_root "$LOGS_ROOT")
fi
if [ "$REQUIRE_HYBRID" = true ]; then
  ARGS+=(--require_hybrid)
fi

"$PYTHON_CMD" validation/audit_experimental_grid.py "${ARGS[@]}"