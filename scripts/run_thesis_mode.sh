#!/bin/bash
set -euo pipefail

# ===============================================================================
# THESIS MODE ORCHESTRATOR
# ===============================================================================
# End-to-end orchestration for doctoral-grade experimental campaigns:
#   1) Profiling campaign (run_experiments.sh)
#   2) ILP solve + Pareto sweep per configuration
#   3) Optional hybrid runtime execution (currently simple_mlp-oriented)
#   4) Consolidated report assets + LaTeX export
#
# This script does not replace methodological decisions (statistics protocol,
# acceptance thresholds, or final model-accuracy evaluation criteria).
# It operationalizes the full pipeline with reproducible logs and structure.
# ===============================================================================

# Optional conda activation (kept consistent with project scripts)
if [ -f ~/anaconda3/etc/profile.d/conda.sh ]; then
  source ~/anaconda3/etc/profile.d/conda.sh
  _had_nounset=0
  if [[ "$-" == *u* ]]; then
    _had_nounset=1
    set +u
  fi
  conda activate prism_env 2>/dev/null || true
  if [ "$_had_nounset" -eq 1 ]; then
    set -u
  fi
fi

source "$(dirname "$0")/sanitize_cuda_env.sh"
sanitize_cuda_runtime_env

PYTHON_CMD="${PYTHON_CMD:-python}"
HOST_TAG="${HOST_TAG:-$(hostname)}"
PROFILE="${PROFILE:-custom}"
DATASETS_DIR="${DATASETS_DIR:-datasets}"
DOWNLOAD_DATASETS="${DOWNLOAD_DATASETS:-true}"

# Preset profiles (can still be overridden by explicit env vars)
case "$PROFILE" in
  doctoral_minimal)
    : "${MODELS_CSV:=simple_mlp,resnet50,resnet152,vit_b16}"
    : "${OPTIMIZERS_CSV:=SGD,AdamW}"
    : "${PRECISIONS_CSV:=fp32}"
    : "${BATCH_SIZES_CSV:=8,32,64}"
    : "${REPEATS:=5}"
    : "${WARMUP:=3}"
    : "${MEASURE:=10}"
    : "${K_SIGMA:=1.0}"
    : "${GPU_BUDGETS_MB:=400,600,800,1000,1200}"
    : "${RUN_HYBRID:=true}"
    ;;
  doctoral_full)
    : "${MODELS_CSV:=simple_mlp,resnet50,resnet152,vit_b16,bert_base,gpt2_small,distilgpt2}"
    : "${OPTIMIZERS_CSV:=SGD,AdamW,RMSprop}"
    : "${PRECISIONS_CSV:=fp32,bf16}"
    : "${BATCH_SIZES_CSV:=8,16,32,64,128}"
    : "${REPEATS:=7}"
    : "${WARMUP:=5}"
    : "${MEASURE:=15}"
    : "${K_SIGMA:=1.0}"
    : "${GPU_BUDGETS_MB:=300,400,600,800,1000,1200,1600,2000}"
    : "${RUN_HYBRID:=true}"
    ;;
  quick_smoke)
    : "${MODELS_CSV:=simple_mlp}"
    : "${OPTIMIZERS_CSV:=SGD}"
    : "${PRECISIONS_CSV:=fp32}"
    : "${BATCH_SIZES_CSV:=8}"
    : "${REPEATS:=1}"
    : "${WARMUP:=1}"
    : "${MEASURE:=1}"
    : "${GPU_BUDGETS_MB:=200,400}"
    : "${RUN_HYBRID:=false}"
    : "${ALLOW_LOW_QUALITY_STATS:=true}"
    ;;
  custom)
    ;;
  *)
    echo "[ERROR] Unsupported PROFILE: $PROFILE (use custom|quick_smoke|doctoral_minimal|doctoral_full)" >&2
    exit 2
    ;;
esac

# Campaign axes (overridable from environment)
MODELS_CSV="${MODELS_CSV:-simple_mlp,resnet50,resnet152,vit_b16,bert_base,gpt2_small,distilgpt2}"
OPTIMIZERS_CSV="${OPTIMIZERS_CSV:-SGD,AdamW}"
PRECISIONS_CSV="${PRECISIONS_CSV:-fp32}"
BATCH_SIZES_CSV="${BATCH_SIZES_CSV:-8,32}"

# Profiling controls
REPEATS="${REPEATS:-5}"
SEED_BASE="${SEED_BASE:-42}"
WARMUP="${WARMUP:-3}"
MEASURE="${MEASURE:-10}"
FORCE_THREADS="${FORCE_THREADS:-0}"
USE_SKIP_CPU="${USE_SKIP_CPU:-false}"
ENABLE_RAPL="${ENABLE_RAPL:-true}"
AUTO_AGGREGATE_STATS="${AUTO_AGGREGATE_STATS:-true}"
FAIL_FAST="${FAIL_FAST:-true}"
DRY_RUN="${DRY_RUN:-false}"

# ILP controls
K_SIGMA="${K_SIGMA:-1.0}"
W_TIME="${W_TIME:-1.0}"
W_ENERGY="${W_ENERGY:-0.0}"
W_TRANSFER="${W_TRANSFER:-1.0}"
BACKEND="${BACKEND:-auto}"
HW_AGGREGATE="${HW_AGGREGATE:-max}"
HW_DISPERSION_K="${HW_DISPERSION_K:-0.0}"
GPU_BUDGETS_MB="${GPU_BUDGETS_MB:-400,600,800,1000,1200}"
GPU_MEM_BUDGET_MB="${GPU_MEM_BUDGET_MB:-1e18}"
CPU_MEM_BUDGET_MB="${CPU_MEM_BUDGET_MB:-1e18}"
MEMORY_MODEL="${MEMORY_MODEL:-peak_approx}"
PEAK_ACTIVATION_OVERLAP="${PEAK_ACTIVATION_OVERLAP:-0.35}"
STRICT_GRAPH_MAPPING="${STRICT_GRAPH_MAPPING:-true}"
STRICT_TRANSFER_MAPPING="${STRICT_TRANSFER_MAPPING:-true}"
ALLOW_LOW_QUALITY_STATS="${ALLOW_LOW_QUALITY_STATS:-false}"
ALLOW_TRANSFER_CALIBRATION_FALLBACK="${ALLOW_TRANSFER_CALIBRATION_FALLBACK:-false}"
ALLOW_FALLBACK_GRAPH_TRACE="${ALLOW_FALLBACK_GRAPH_TRACE:-false}"

# Single-replicate profiling naturally produces low_sample quality flags.
# Keep ILP stages operational unless the caller explicitly requested strict mode.
if [[ "$REPEATS" =~ ^[0-9]+$ ]] && [ "$REPEATS" -lt 2 ] && [ "${ALLOW_LOW_QUALITY_STATS:-false}" != "true" ]; then
  ALLOW_LOW_QUALITY_STATS=true
fi

# Orchestration toggles
RUN_PROFILING="${RUN_PROFILING:-true}"
RUN_ILP="${RUN_ILP:-true}"
RUN_HYBRID="${RUN_HYBRID:-false}"
RUN_REPORTS="${RUN_REPORTS:-true}"

# Hybrid runtime controls (optional)
HYBRID_STEPS="${HYBRID_STEPS:-10}"
HYBRID_BATCH_SIZE="${HYBRID_BATCH_SIZE:-8}"
HYBRID_ENABLE_ASYNC_TRANSFER="${HYBRID_ENABLE_ASYNC_TRANSFER:-true}"
HYBRID_ENABLE_PREFETCH="${HYBRID_ENABLE_PREFETCH:-true}"
HYBRID_ALLOW_CPU_FALLBACK="${HYBRID_ALLOW_CPU_FALLBACK:-false}"
HYBRID_COMPARE_BASELINES="${HYBRID_COMPARE_BASELINES:-true}"
HYBRID_EXECUTION_MODE="${HYBRID_EXECUTION_MODE:-auto}"
HYBRID_PLAN_SELECTION="${HYBRID_PLAN_SELECTION:-pareto_best}"

BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-data/${HOST_TAG}/results_thesis_mode}"
REPORTS_DIR="${REPORTS_DIR:-reports/ilp_results/${HOST_TAG}_thesis_mode}"
LATEX_DIR="${LATEX_DIR:-${REPORTS_DIR}/latex}"
LOG_DIR="${LOG_DIR:-logs}"
LOG_FILE="${LOG_DIR}/thesis_mode_$(date +%Y%m%d_%H%M%S).txt"

mkdir -p "$BASE_OUTPUT_DIR" "$REPORTS_DIR" "$LATEX_DIR" "$LOG_DIR"

log_msg() {
  local msg="$1"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $msg" | tee -a "$LOG_FILE"
}

section() {
  local title="$1"
  log_msg "===================================================================="
  log_msg "$title"
  log_msg "===================================================================="
}

is_true() {
  [ "$1" = true ]
}

normalize_output_dir_for_host_sh() {
  local output_dir="$1"
  local host_name="$2"

  if [ -z "$output_dir" ]; then
    printf '%s' "$output_dir"
    return
  fi

  local normalized="${output_dir//\\//}"
  local leading_slash=""
  if [[ "$normalized" == /* ]]; then
    leading_slash="/"
  fi

  local IFS='/'
  local raw_parts=()
  local parts=()
  read -r -a raw_parts <<< "$normalized"

  local p
  for p in "${raw_parts[@]}"; do
    if [ -n "$p" ]; then
      parts+=("$p")
    fi
  done

  if [ ${#parts[@]} -eq 0 ]; then
    printf '%s' "$output_dir"
    return
  fi

  local data_idx=-1
  local i
  for i in "${!parts[@]}"; do
    if [ "${parts[$i]}" = "data" ]; then
      data_idx=$i
      break
    fi
  done

  if [ "$data_idx" -lt 0 ]; then
    printf '%s' "$output_dir"
    return
  fi

  if [ $((data_idx + 1)) -lt ${#parts[@]} ] && [ "${parts[$((data_idx + 1))]}" = "$host_name" ]; then
    printf '%s' "$output_dir"
    return
  fi

  local before=()
  local after=()
  if [ "$data_idx" -gt 0 ]; then
    before=("${parts[@]:0:$data_idx}")
  fi
  if [ $((data_idx + 1)) -lt ${#parts[@]} ]; then
    after=("${parts[@]:$((data_idx + 1))}")
  fi

  local rebuilt=()
  rebuilt+=("${before[@]}")
  rebuilt+=("data" "$host_name")
  rebuilt+=("${after[@]}")

  local joined=""
  for p in "${rebuilt[@]}"; do
    if [ -z "$joined" ]; then
      joined="$p"
    else
      joined="$joined/$p"
    fi
  done

  printf '%s%s' "$leading_slash" "$joined"
}

discover_config_dirs() {
  local base_dir="$1"
  local host_name="$2"
  local normalized_dir
  normalized_dir="$(normalize_output_dir_for_host_sh "$base_dir" "$host_name")"

  mapfile -t CONFIG_DIRS < <(
    {
      [ -d "$base_dir" ] && find "$base_dir" -type d -name 'batch_*'
      if [ "$normalized_dir" != "$base_dir" ] && [ -d "$normalized_dir" ]; then
        find "$normalized_dir" -type d -name 'batch_*'
      fi
    } | sort -u
  )

  if [ ${#CONFIG_DIRS[@]} -gt 0 ]; then
    local unique_roots
    unique_roots="$(printf '%s\n' "${CONFIG_DIRS[@]}" | sed -E 's|(.*?/batch_[^/]+).*|\1|' | sort -u | head -n 3 | paste -sd ', ' -)"
    log_msg "Discovered ${#CONFIG_DIRS[@]} config dirs (sample roots: ${unique_roots})"
  fi
}

has_pareto_files() {
  local root_dir="$1"
  [ -d "$root_dir" ] || return 1
  find "$root_dir" -type f -name '*_pareto_sweep.csv' -print -quit | grep -q .
}

resolve_report_input_root() {
  local base_dir="$1"
  local host_name="$2"
  local normalized_dir
  normalized_dir="$(normalize_output_dir_for_host_sh "$base_dir" "$host_name")"

  if has_pareto_files "$base_dir"; then
    printf '%s' "$base_dir"
    return
  fi

  if [ "$normalized_dir" != "$base_dir" ] && has_pareto_files "$normalized_dir"; then
    printf '%s' "$normalized_dir"
    return
  fi

  # Fallback keeps previous behavior, preserving clear error reporting downstream.
  printf '%s' "$base_dir"
}

section "THESIS MODE - PREFLIGHT"
log_msg "PROFILE=$PROFILE"
log_msg "PYTHON_CMD=$PYTHON_CMD"
log_msg "HOST_TAG=$HOST_TAG"
log_msg "BASE_OUTPUT_DIR=$BASE_OUTPUT_DIR"
log_msg "REPORTS_DIR=$REPORTS_DIR"
log_msg "LATEX_DIR=$LATEX_DIR"
log_msg "DATASETS_DIR=$DATASETS_DIR"
log_msg "MODELS_CSV=$MODELS_CSV"
log_msg "OPTIMIZERS_CSV=$OPTIMIZERS_CSV"
log_msg "PRECISIONS_CSV=$PRECISIONS_CSV"
log_msg "BATCH_SIZES_CSV=$BATCH_SIZES_CSV"
log_msg "REPEATS=$REPEATS"
log_msg "WARMUP=$WARMUP MEASURE=$MEASURE"
if [ "$ALLOW_LOW_QUALITY_STATS" = true ]; then
  log_msg "ALLOW_LOW_QUALITY_STATS=true"
fi
log_msg "DRY_RUN=$DRY_RUN"

if is_true "$DOWNLOAD_DATASETS"; then
  section "STEP 1/5 - DATASET PREPARATION"
  "$PYTHON_CMD" scripts/download_datasets.py \
    --models "$MODELS_CSV" \
    --datasets_root "$DATASETS_DIR" \
    >> "$LOG_FILE" 2>&1
  log_msg "Dataset preparation finished"
fi

if is_true "$RUN_PROFILING"; then
  section "STEP 2/5 - PROFILING CAMPAIGN"
  MODELS_CSV="$MODELS_CSV" \
  OPTIMIZERS_CSV="$OPTIMIZERS_CSV" \
  PRECISIONS_CSV="$PRECISIONS_CSV" \
  BATCH_SIZES_CSV="$BATCH_SIZES_CSV" \
  BASE_OUTPUT_DIR="$BASE_OUTPUT_DIR" \
  DATASETS_DIR="$DATASETS_DIR" \
  DOWNLOAD_DATASETS=false \
  PYTHON_CMD="$PYTHON_CMD" \
  WARMUP="$WARMUP" \
  MEASURE="$MEASURE" \
  REPEATS="$REPEATS" \
  SEED_BASE="$SEED_BASE" \
  FORCE_THREADS="$FORCE_THREADS" \
  USE_SKIP_CPU="$USE_SKIP_CPU" \
  ENABLE_RAPL="$ENABLE_RAPL" \
  AUTO_AGGREGATE_STATS="$AUTO_AGGREGATE_STATS" \
  FAIL_FAST="$FAIL_FAST" \
  DRY_RUN="$DRY_RUN" \
  bash scripts/run_experiments.sh >> "$LOG_FILE" 2>&1
  log_msg "Profiling campaign finished"
fi

if is_true "$RUN_ILP"; then
  section "STEP 3/5 - ILP PARTITION + PARETO PER CONFIG"

  if is_true "$DRY_RUN"; then
    log_msg "DRY_RUN=true -> skipping ILP and Pareto execution (no real artifacts expected)"
  else

    discover_config_dirs "$BASE_OUTPUT_DIR" "$HOST_TAG"

    if [ ${#CONFIG_DIRS[@]} -eq 0 ]; then
      log_msg "ERROR: No configuration directories found under $BASE_OUTPUT_DIR"
      exit 1
    fi

    for cfg in "${CONFIG_DIRS[@]}"; do
      model="$(basename "$(dirname "$(dirname "$(dirname "$cfg")")")")"

      log_msg "ILP partition -> model=$model cfg=$cfg"
      MODEL="$model" \
      CONFIG_DIR="$cfg" \
      PYTHON_CMD="$PYTHON_CMD" \
      K_SIGMA="$K_SIGMA" \
      W_TIME="$W_TIME" \
      W_ENERGY="$W_ENERGY" \
      W_TRANSFER="$W_TRANSFER" \
      GPU_MEM_BUDGET_MB="$GPU_MEM_BUDGET_MB" \
      CPU_MEM_BUDGET_MB="$CPU_MEM_BUDGET_MB" \
      MEMORY_MODEL="$MEMORY_MODEL" \
      PEAK_ACTIVATION_OVERLAP="$PEAK_ACTIVATION_OVERLAP" \
      BACKEND="$BACKEND" \
      HW_AGGREGATE="$HW_AGGREGATE" \
      HW_DISPERSION_K="$HW_DISPERSION_K" \
      STRICT_GRAPH_MAPPING="$STRICT_GRAPH_MAPPING" \
      STRICT_TRANSFER_MAPPING="$STRICT_TRANSFER_MAPPING" \
      ALLOW_LOW_QUALITY_STATS="$ALLOW_LOW_QUALITY_STATS" \
      ALLOW_TRANSFER_CALIBRATION_FALLBACK="$ALLOW_TRANSFER_CALIBRATION_FALLBACK" \
      ALLOW_FALLBACK_GRAPH_TRACE="$ALLOW_FALLBACK_GRAPH_TRACE" \
      bash scripts/run_ilp_partition.sh >> "$LOG_FILE" 2>&1

      log_msg "Pareto sweep -> model=$model cfg=$cfg"
      MODEL="$model" \
      CONFIG_DIR="$cfg" \
      PYTHON_CMD="$PYTHON_CMD" \
      GPU_BUDGETS_MB="$GPU_BUDGETS_MB" \
      CPU_MEM_BUDGET_MB="$CPU_MEM_BUDGET_MB" \
      MEMORY_MODEL="$MEMORY_MODEL" \
      PEAK_ACTIVATION_OVERLAP="$PEAK_ACTIVATION_OVERLAP" \
      K_SIGMA="$K_SIGMA" \
      W_TIME="$W_TIME" \
      W_ENERGY="$W_ENERGY" \
      W_TRANSFER="$W_TRANSFER" \
      BACKEND="$BACKEND" \
      HW_AGGREGATE="$HW_AGGREGATE" \
      HW_DISPERSION_K="$HW_DISPERSION_K" \
      STRICT_GRAPH_MAPPING="$STRICT_GRAPH_MAPPING" \
      STRICT_TRANSFER_MAPPING="$STRICT_TRANSFER_MAPPING" \
      ALLOW_LOW_QUALITY_STATS="$ALLOW_LOW_QUALITY_STATS" \
      ALLOW_TRANSFER_CALIBRATION_FALLBACK="$ALLOW_TRANSFER_CALIBRATION_FALLBACK" \
      ALLOW_FALLBACK_GRAPH_TRACE="$ALLOW_FALLBACK_GRAPH_TRACE" \
      bash scripts/run_ilp_pareto_sweep.sh >> "$LOG_FILE" 2>&1
    done
  fi

  log_msg "ILP and Pareto stage finished"
fi

if is_true "$RUN_HYBRID"; then
  section "STEP 4/5 - OPTIONAL HYBRID RUNTIME EXECUTION"
  if is_true "$DRY_RUN"; then
    log_msg "DRY_RUN=true -> skipping hybrid runtime execution"
    log_msg "Hybrid runtime stage finished"
  else
  discover_config_dirs "$BASE_OUTPUT_DIR" "$HOST_TAG"

  for cfg in "${CONFIG_DIRS[@]}"; do
    model="$(basename "$(dirname "$(dirname "$(dirname "$cfg")")")")"

    hybrid_assignment_csv="$cfg/ilp_solution/ilp_assignment.csv"
    hybrid_cut_edges_csv="$cfg/ilp_solution/ilp_cut_edges.csv"
    hybrid_output_dir="$cfg/ilp_solution/hybrid_execution"
    hybrid_plan_mode="default_ilp"
    hybrid_plan_source="$cfg/ilp_solution"
    hybrid_plan_source_csv=""
    hybrid_plan_gpu_budget_mb=""
    hybrid_plan_objective=""

    if [ "$HYBRID_PLAN_SELECTION" = "pareto_best" ]; then
      pareto_info="$($PYTHON_CMD - "$cfg" "$model" <<'PY'
import json
import sys
from pathlib import Path

cfg = Path(sys.argv[1])
model = sys.argv[2]
summary_path = cfg / f"{model}_pareto_summary.json"
if not summary_path.exists():
    raise SystemExit(2)

payload = json.loads(summary_path.read_text())
row = payload.get("best_feasible_row")
if not row:
    raise SystemExit(3)

source_csv = payload.get("output_csv") or row.get("source_csv") or ""
print(
    "|".join(
        [
            str(row.get("gpu_budget_mb", "")),
            str(row.get("ilp_objective", "")),
            str(row.get("ilp_status", "")),
            source_csv,
        ]
    )
)
PY
      )" || true

      if [ -n "$pareto_info" ]; then
        IFS='|' read -r hybrid_plan_gpu_budget_mb hybrid_plan_objective hybrid_plan_status hybrid_plan_source_csv <<< "$pareto_info"
        budget_slug="$(printf '%s' "$hybrid_plan_gpu_budget_mb" | tr '.' 'p')"
        hybrid_plan_source="$cfg/ilp_solution_pareto_best_budget_${budget_slug}"
        hybrid_assignment_csv="$hybrid_plan_source/ilp_assignment.csv"
        hybrid_cut_edges_csv="$hybrid_plan_source/ilp_cut_edges.csv"
        hybrid_output_dir="$hybrid_plan_source/hybrid_execution"
        hybrid_plan_mode="pareto_best"

        if [ ! -f "$hybrid_assignment_csv" ] || [ ! -f "$hybrid_cut_edges_csv" ]; then
          log_msg "Materializing Pareto-best hybrid plan -> cfg=$cfg budget_mb=$hybrid_plan_gpu_budget_mb objective=$hybrid_plan_objective"
          MODEL="$model" \
          CONFIG_DIR="$cfg" \
          PYTHON_CMD="$PYTHON_CMD" \
          K_SIGMA="$K_SIGMA" \
          W_TIME="$W_TIME" \
          W_ENERGY="$W_ENERGY" \
          W_TRANSFER="$W_TRANSFER" \
          GPU_MEM_BUDGET_MB="$hybrid_plan_gpu_budget_mb" \
          CPU_MEM_BUDGET_MB="$CPU_MEM_BUDGET_MB" \
          MEMORY_MODEL="$MEMORY_MODEL" \
          PEAK_ACTIVATION_OVERLAP="$PEAK_ACTIVATION_OVERLAP" \
          BACKEND="$BACKEND" \
          HW_AGGREGATE="$HW_AGGREGATE" \
          HW_DISPERSION_K="$HW_DISPERSION_K" \
          OUT_DIR="$hybrid_plan_source" \
          STRICT_GRAPH_MAPPING="$STRICT_GRAPH_MAPPING" \
          STRICT_TRANSFER_MAPPING="$STRICT_TRANSFER_MAPPING" \
          ALLOW_LOW_QUALITY_STATS="$ALLOW_LOW_QUALITY_STATS" \
          ALLOW_TRANSFER_CALIBRATION_FALLBACK="$ALLOW_TRANSFER_CALIBRATION_FALLBACK" \
          ALLOW_FALLBACK_GRAPH_TRACE="$ALLOW_FALLBACK_GRAPH_TRACE" \
          bash scripts/run_ilp_partition.sh >> "$LOG_FILE" 2>&1
        fi
      else
        log_msg "Pareto-best selection unavailable for cfg=$cfg; falling back to default ilp_solution"
      fi
    elif [ "$HYBRID_PLAN_SELECTION" != "default_ilp" ]; then
      log_msg "Unsupported HYBRID_PLAN_SELECTION=$HYBRID_PLAN_SELECTION for cfg=$cfg; falling back to default ilp_solution"
    fi

    if [ ! -f "$hybrid_assignment_csv" ] || [ ! -f "$hybrid_cut_edges_csv" ]; then
      log_msg "Skipping hybrid runtime for cfg=$cfg (missing plan artifacts under $hybrid_plan_source)"
      continue
    fi

    HYBRID_FLAGS=()
    if is_true "$HYBRID_ENABLE_ASYNC_TRANSFER"; then
      HYBRID_FLAGS+=(--enable_async_transfer)
    fi
    if is_true "$HYBRID_ENABLE_PREFETCH"; then
      HYBRID_FLAGS+=(--enable_prefetch)
    fi
    if is_true "$HYBRID_ALLOW_CPU_FALLBACK"; then
      HYBRID_FLAGS+=(--allow_cpu_fallback)
    fi
    if is_true "$HYBRID_COMPARE_BASELINES"; then
      HYBRID_FLAGS+=(--compare_baselines)
    fi
    HYBRID_PLAN_FLAGS=(
      --plan_selection_mode "$hybrid_plan_mode"
      --plan_source "$hybrid_plan_source"
    )
    if [ -n "$hybrid_plan_source_csv" ]; then
      HYBRID_PLAN_FLAGS+=(--plan_source_csv "$hybrid_plan_source_csv")
    fi
    if [ -n "$hybrid_plan_gpu_budget_mb" ]; then
      HYBRID_PLAN_FLAGS+=(--plan_gpu_budget_mb "$hybrid_plan_gpu_budget_mb")
    fi
    if [ -n "$hybrid_plan_objective" ]; then
      HYBRID_PLAN_FLAGS+=(--plan_objective "$hybrid_plan_objective")
    fi

    log_msg "Hybrid runtime -> cfg=$cfg plan_mode=$hybrid_plan_mode plan_source=$hybrid_plan_source"
    "$PYTHON_CMD" validation/run_hybrid_execution.py \
      --config_dir "$cfg" \
      --assignment_csv "$hybrid_assignment_csv" \
      --cut_edges_csv "$hybrid_cut_edges_csv" \
      --model "$model" \
      --batch_size "$HYBRID_BATCH_SIZE" \
      --datasets_root "$DATASETS_DIR" \
      --require_datasets \
      --steps "$HYBRID_STEPS" \
      --execution_mode "$HYBRID_EXECUTION_MODE" \
      --output_dir "$hybrid_output_dir" \
        "${HYBRID_PLAN_FLAGS[@]}" \
      "${HYBRID_FLAGS[@]}" \
      >> "$LOG_FILE" 2>&1
  done

  log_msg "Hybrid runtime stage finished"
  fi
fi

if is_true "$RUN_REPORTS"; then
  section "STEP 5/5 - CONSOLIDATED REPORTS + LATEX"

  if is_true "$DRY_RUN"; then
    log_msg "DRY_RUN=true -> skipping consolidated report and LaTeX generation"
    log_msg "Report stage finished"
  else

  REPORT_INPUT_ROOT="$(resolve_report_input_root "$BASE_OUTPUT_DIR" "$HOST_TAG")"
  log_msg "Report input root: $REPORT_INPUT_ROOT"

  PYTHON_CMD="$PYTHON_CMD" \
  INPUT_ROOT="$REPORT_INPUT_ROOT" \
  OUTPUT_DIR="$REPORTS_DIR" \
  bash scripts/generate_ilp_report_assets.sh >> "$LOG_FILE" 2>&1

  PYTHON_CMD="$PYTHON_CMD" \
  BEST_CSV="$REPORTS_DIR/ilp_best_per_model.csv" \
  CONSOLIDATED_CSV="$REPORTS_DIR/ilp_pareto_consolidated.csv" \
  HYBRID_CSV="$REPORTS_DIR/hybrid_execution_best_per_model.csv" \
  OUT_DIR="$LATEX_DIR" \
  bash scripts/export_ilp_tables_latex.sh >> "$LOG_FILE" 2>&1

  cat > "$REPORTS_DIR/THESIS_MODE_PROTOCOL_CHECKLIST.md" <<EOF
# Thesis Mode Protocol Checklist

This run executed the project-wide thesis mode orchestration.

## Generated roots
- Base output: $BASE_OUTPUT_DIR
- Reports: $REPORTS_DIR
- LaTeX: $LATEX_DIR
- Log: $LOG_FILE

## Automatically covered by script
- Profiling campaign grid
- Replicate aggregation
- ILP partition and Pareto sweep per config
- Consolidated report assets and LaTeX tables
- Optional hybrid runtime traces aligned to $HYBRID_PLAN_SELECTION plans (FX and export-backed DAG paths when supported)

## Must still be validated explicitly (doctoral criteria)
- Final model quality metrics (accuracy/loss/AUC) vs baselines in target tasks
- Statistical significance and effect size for claimed gains
- Multi-hardware robustness matrix and threats-to-validity analysis

EOF

  log_msg "Report stage finished"
  fi
fi

section "THESIS MODE COMPLETE"
log_msg "Base output: $BASE_OUTPUT_DIR"
log_msg "Reports: $REPORTS_DIR"
log_msg "LaTeX: $LATEX_DIR"
log_msg "Log: $LOG_FILE"
