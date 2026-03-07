#!/bin/bash
set -euo pipefail

# Usage:
# PYTHON_CMD=./.venv/bin/python INPUT_ROOT=data/test-m4 OUTPUT_DIR=reports/ilp_results \
# bash scripts/generate_ilp_report_assets.sh

PYTHON_CMD="${PYTHON_CMD:-python}"
INPUT_ROOT="${INPUT_ROOT:-data/test-m4}"
OUTPUT_DIR="${OUTPUT_DIR:-reports/ilp_results}"

"$PYTHON_CMD" validation/generate_ilp_report_assets.py \
  --input_root "$INPUT_ROOT" \
  --output_dir "$OUTPUT_DIR"
