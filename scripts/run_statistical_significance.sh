#!/usr/bin/env bash
set -euo pipefail

log_msg() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

if [ -z "${PYTHON_CMD:-}" ]; then
    PYTHON_CMD="python3"
fi

if [ -z "${CONSOLIDATED_CSV:-}" ]; then
    log_msg "ERROR: CONSOLIDATED_CSV is required."
    exit 1
fi

if [ -z "${OUTPUT_CSV:-}" ]; then
    log_msg "ERROR: OUTPUT_CSV is required."
    exit 1
fi

log_msg "Running Statistical Significance Analysis (Cohen's d, Z-Test)"
"$PYTHON_CMD" validation/run_statistical_significance.py \
    --consolidated_csv "$CONSOLIDATED_CSV" \
    --output_csv "$OUTPUT_CSV"

log_msg "Statistical Significance complete: $OUTPUT_CSV"
