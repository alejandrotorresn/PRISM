#!/bin/bash
set -euo pipefail

# Usage:
# PYTHON_CMD=./.venv/bin/python \
# BEST_CSV=reports/ilp_results/ilp_best_per_model.csv \
# CONSOLIDATED_CSV=reports/ilp_results/ilp_pareto_consolidated.csv \
# OUT_DIR=reports/ilp_results/latex \
# bash scripts/export_ilp_tables_latex.sh

PYTHON_CMD="${PYTHON_CMD:-python}"
BEST_CSV="${BEST_CSV:-reports/ilp_results/ilp_best_per_model.csv}"
CONSOLIDATED_CSV="${CONSOLIDATED_CSV:-reports/ilp_results/ilp_pareto_consolidated.csv}"
OUT_DIR="${OUT_DIR:-reports/ilp_results/latex}"

"$PYTHON_CMD" validation/export_ilp_tables_latex.py \
  --best_csv "$BEST_CSV" \
  --consolidated_csv "$CONSOLIDATED_CSV" \
  --output_dir "$OUT_DIR"
