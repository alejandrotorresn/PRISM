#!/bin/bash
set -euo pipefail

# =======================================================================================
# THESIS SMOKE WORKFLOW (REAL MACHINE)
# =======================================================================================
# Purpose:
#   Execute a reduced but end-to-end doctoral workflow on real hardware:
#   1) Profile a small configuration grid (FP32 only)
#   2) Aggregate replicate stats per configuration
#   3) Build/solve ILP partition + Pareto sweep
#   4) Generate consolidated report assets (CSV/plots/markdown)
#   5) Export LaTeX tables
#
# Notes:
# - This is a verification workflow (small campaign), not the full thesis grid.
# - Output is host-scoped under data/<hostname>/ to preserve machine-specific profiles.
# - Every step is logged and documented for reproducibility.
# =======================================================================================

# Optional conda activation (kept consistent with existing scripts)
if [ -f ~/anaconda3/etc/profile.d/conda.sh ]; then
  source ~/anaconda3/etc/profile.d/conda.sh
  conda activate thesis_env 2>/dev/null || true
fi

source "$(dirname "$0")/sanitize_cuda_env.sh"
sanitize_cuda_runtime_env

# -------------------------------------------------------------------------------
# USER-TUNABLE PARAMETERS (SMOKE SCALE)
# -------------------------------------------------------------------------------
PYTHON_CMD="${PYTHON_CMD:-python}"
PROFILER_SCRIPT="${PROFILER_SCRIPT:-src/profiler.py}"
DATASET_SCRIPT="${DATASET_SCRIPT:-scripts/download_datasets.py}"
HOST_TAG="${HOST_TAG:-$(hostname)}"
DATASETS_DIR="${DATASETS_DIR:-datasets}"
DOWNLOAD_DATASETS="${DOWNLOAD_DATASETS:-true}"

# Reduced campaign axes (deliberately small, preconfigured for real-machine verification)
MODELS=(${MODELS:-simple_mlp resnet50})
OPTIMIZERS=(${OPTIMIZERS:-SGD AdamW})
BATCH_SIZES=(${BATCH_SIZES:-8 32})
PRECISION="fp32"  # Required by request: FP32-only smoke workflow

# Runtime controls
REPEATS="${REPEATS:-3}"         # 3 replicates by default for more stable robust stats
SEED_BASE="${SEED_BASE:-42}"
WARMUP="${WARMUP:-2}"
MEASURE="${MEASURE:-5}"
FORCE_THREADS="${FORCE_THREADS:-0}"
USE_SKIP_CPU="${USE_SKIP_CPU:-false}"
ENABLE_RAPL="${ENABLE_RAPL:-true}"

# ILP controls
K_SIGMA="${K_SIGMA:-1.0}"
W_TIME="${W_TIME:-1.0}"
W_ENERGY="${W_ENERGY:-0.0}"
W_TRANSFER="${W_TRANSFER:-1.0}"
BACKEND="${BACKEND:-auto}"
HW_AGGREGATE="${HW_AGGREGATE:-max}"
HW_DISPERSION_K="${HW_DISPERSION_K:-0.0}"
STRICT_GRAPH_MAPPING="${STRICT_GRAPH_MAPPING:-true}"
STRICT_TRANSFER_MAPPING="${STRICT_TRANSFER_MAPPING:-true}"
ALLOW_LOW_QUALITY_STATS="${ALLOW_LOW_QUALITY_STATS:-true}"
ALLOW_TRANSFER_CALIBRATION_FALLBACK="${ALLOW_TRANSFER_CALIBRATION_FALLBACK:-false}"
ALLOW_FALLBACK_GRAPH_TRACE="${ALLOW_FALLBACK_GRAPH_TRACE:-false}"
GPU_BUDGETS_MB="${GPU_BUDGETS_MB:-500,1000,2000}"
CPU_MEM_BUDGET_MB="${CPU_MEM_BUDGET_MB:-1e18}"

# Paths
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-data/${HOST_TAG}/results_smoke}"
REPORTS_DIR="${REPORTS_DIR:-reports/ilp_results/${HOST_TAG}_smoke}"
LATEX_DIR="${LATEX_DIR:-${REPORTS_DIR}/latex}"
LOG_DIR="${LOG_DIR:-logs}"
LOG_FILE="${LOG_DIR}/thesis_smoke_workflow_$(date +%Y%m%d_%H%M%S).txt"

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

check_gpu() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    local gpu_name
    gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
    log_msg "GPU detected: ${gpu_name}"
    return 0
  fi
  log_msg "No GPU detected (nvidia-smi not found)."
  return 1
}

# -------------------------------------------------------------------------------
# STEP 0: PREFLIGHT SUMMARY
# -------------------------------------------------------------------------------
section "STEP 0/6 - WORKFLOW PREFLIGHT"
log_msg "Python command: $PYTHON_CMD"
log_msg "Profiler script: $PROFILER_SCRIPT"
log_msg "Datasets dir: $DATASETS_DIR"
log_msg "Host tag: $HOST_TAG"
log_msg "Output root: $BASE_OUTPUT_DIR"
log_msg "Reports root: $REPORTS_DIR"
log_msg "Models: ${MODELS[*]}"
log_msg "Optimizers: ${OPTIMIZERS[*]}"
log_msg "Batch sizes: ${BATCH_SIZES[*]}"
log_msg "Precision: $PRECISION"
log_msg "Repeats: $REPEATS"
log_msg "Warmup/Measure: $WARMUP/$MEASURE"
log_msg "Threads override: $FORCE_THREADS"
log_msg "Skip CPU: $USE_SKIP_CPU"
log_msg "RAPL enabled: $ENABLE_RAPL"

HAS_GPU=false
if check_gpu; then
  HAS_GPU=true
fi

if [ "$USE_SKIP_CPU" = true ] && [ "$HAS_GPU" = false ]; then
  log_msg "WARNING: USE_SKIP_CPU=true but no GPU available; forcing USE_SKIP_CPU=false."
  USE_SKIP_CPU=false
fi

# -------------------------------------------------------------------------------
# STEP 1: DATASET PREPARATION
# -------------------------------------------------------------------------------
if [ "$DOWNLOAD_DATASETS" = true ]; then
  section "STEP 1/7 - DATASET PREPARATION"
  DATASET_MODELS_CSV="$(IFS=,; echo "${MODELS[*]}")"
  log_msg "Preparing datasets for models: $DATASET_MODELS_CSV"
  "$PYTHON_CMD" "$DATASET_SCRIPT" --models "$DATASET_MODELS_CSV" --datasets_root "$DATASETS_DIR" >> "$LOG_FILE" 2>&1
  log_msg "OK: dataset preparation finished"
fi

# -------------------------------------------------------------------------------
# STEP 2: PROFILING CAMPAIGN (REDUCED GRID, FP32 ONLY)
# -------------------------------------------------------------------------------
section "STEP 2/7 - PROFILING REDUCED GRID (FP32 ONLY)"

TOTAL=$(( ${#MODELS[@]} * ${#OPTIMIZERS[@]} * ${#BATCH_SIZES[@]} * REPEATS ))
DONE=0

for model in "${MODELS[@]}"; do
  for optimizer in "${OPTIMIZERS[@]}"; do
    for batch in "${BATCH_SIZES[@]}"; do
      CFG_DIR="$BASE_OUTPUT_DIR/$model/$optimizer/$PRECISION/batch_${batch}"
      mkdir -p "$CFG_DIR"

      log_msg "Config: model=$model optimizer=$optimizer precision=$PRECISION batch=$batch"

      for ((repeat_idx=1; repeat_idx<=REPEATS; repeat_idx++)); do
        DONE=$((DONE + 1))
        RUN_ID=$(printf "run_%03d" "$repeat_idx")
        RUN_SEED=$((SEED_BASE + repeat_idx - 1))
        RUN_DIR="$CFG_DIR/$RUN_ID"
        mkdir -p "$RUN_DIR"

        CMD=(
          "$PYTHON_CMD" "$PROFILER_SCRIPT"
          --model "$model"
          --optimizer "$optimizer"
          --precision "$PRECISION"
          --batch_size "$batch"
          --warmup "$WARMUP"
          --measure "$MEASURE"
          --output_dir "$RUN_DIR"
          --datasets_root "$DATASETS_DIR"
          --require_datasets
          --seed "$RUN_SEED"
          --run_id "$RUN_ID"
        )

        if [ "$USE_SKIP_CPU" = true ]; then
          CMD+=(--skip_cpu)
        elif [ "$ENABLE_RAPL" = true ]; then
          CMD+=(--rapl)
        fi

        if [ "$FORCE_THREADS" -gt 0 ]; then
          CMD+=(--num_threads "$FORCE_THREADS")
        fi

        log_msg "[$DONE/$TOTAL] Running $RUN_ID (seed=$RUN_SEED)"
        if "${CMD[@]}" >> "$LOG_FILE" 2>&1; then
          log_msg "OK: $RUN_ID completed"
        else
          rc=$?
          log_msg "ERROR: $RUN_ID failed with exit code=$rc"
          exit "$rc"
        fi
      done
    done
  done
done

# -------------------------------------------------------------------------------
# STEP 3: AGGREGATE METRICS STATS PER CONFIGURATION
# -------------------------------------------------------------------------------
section "STEP 3/7 - AGGREGATE REPLICATE METRICS"

for model in "${MODELS[@]}"; do
  for optimizer in "${OPTIMIZERS[@]}"; do
    for batch in "${BATCH_SIZES[@]}"; do
      CFG_DIR="$BASE_OUTPUT_DIR/$model/$optimizer/$PRECISION/batch_${batch}"
      AGG_CSV="$CFG_DIR/${model}_metrics_stats.csv"
      log_msg "Aggregating stats: $CFG_DIR"
      "$PYTHON_CMD" validation/aggregate_metrics_stats.py \
        --input_dir "$CFG_DIR" \
        --output_csv "$AGG_CSV" \
        >> "$LOG_FILE" 2>&1
      log_msg "OK: $AGG_CSV"
    done
  done
done

# -------------------------------------------------------------------------------
# STEP 4: BUILD/SOLVE ILP PARTITION PER CONFIGURATION
# -------------------------------------------------------------------------------
section "STEP 4/7 - ILP PARTITION SOLVES"

for model in "${MODELS[@]}"; do
  for optimizer in "${OPTIMIZERS[@]}"; do
    for batch in "${BATCH_SIZES[@]}"; do
      CFG_DIR="$BASE_OUTPUT_DIR/$model/$optimizer/$PRECISION/batch_${batch}"
      ILP_OUT="$CFG_DIR/ilp_solution"
      log_msg "ILP partition: $CFG_DIR"

      MODEL="$model" \
      CONFIG_DIR="$CFG_DIR" \
      OUT_DIR="$ILP_OUT" \
      PYTHON_CMD="$PYTHON_CMD" \
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
      bash scripts/run_ilp_partition.sh \
      >> "$LOG_FILE" 2>&1

      log_msg "OK: ILP partition saved in $ILP_OUT"
    done
  done
done

# -------------------------------------------------------------------------------
# STEP 5: ILP PARETO SWEEP PER CONFIGURATION
# -------------------------------------------------------------------------------
section "STEP 5/7 - ILP PARETO SWEEPS"

for model in "${MODELS[@]}"; do
  for optimizer in "${OPTIMIZERS[@]}"; do
    for batch in "${BATCH_SIZES[@]}"; do
      CFG_DIR="$BASE_OUTPUT_DIR/$model/$optimizer/$PRECISION/batch_${batch}"
      log_msg "Pareto sweep: $CFG_DIR"

      MODEL="$model" \
      CONFIG_DIR="$CFG_DIR" \
      PYTHON_CMD="$PYTHON_CMD" \
      GPU_BUDGETS_MB="$GPU_BUDGETS_MB" \
      CPU_MEM_BUDGET_MB="$CPU_MEM_BUDGET_MB" \
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
      bash scripts/run_ilp_pareto_sweep.sh \
      >> "$LOG_FILE" 2>&1

      log_msg "OK: Pareto sweep generated in $CFG_DIR"
    done
  done
done

# -------------------------------------------------------------------------------
# STEP 6: GENERATE CONSOLIDATED REPORT ASSETS (CSV + PLOTS + MD)
# -------------------------------------------------------------------------------
section "STEP 6/7 - GENERATE REPORT ASSETS"

PYTHON_CMD="$PYTHON_CMD" \
INPUT_ROOT="$BASE_OUTPUT_DIR" \
OUTPUT_DIR="$REPORTS_DIR" \
bash scripts/generate_ilp_report_assets.sh \
>> "$LOG_FILE" 2>&1

log_msg "OK: consolidated report assets in $REPORTS_DIR"

# -------------------------------------------------------------------------------
# STEP 7: EXPORT LATEX TABLES
# -------------------------------------------------------------------------------
section "STEP 7/7 - EXPORT LATEX TABLES"

PYTHON_CMD="$PYTHON_CMD" \
BEST_CSV="$REPORTS_DIR/ilp_best_per_model.csv" \
CONSOLIDATED_CSV="$REPORTS_DIR/ilp_pareto_consolidated.csv" \
OUT_DIR="$LATEX_DIR" \
bash scripts/export_ilp_tables_latex.sh \
>> "$LOG_FILE" 2>&1

log_msg "OK: LaTeX tables in $LATEX_DIR"

# -------------------------------------------------------------------------------
# FINAL SUMMARY
# -------------------------------------------------------------------------------
section "WORKFLOW COMPLETE"
log_msg "Host-scoped outputs: $BASE_OUTPUT_DIR"
log_msg "Report assets:       $REPORTS_DIR"
log_msg "LaTeX tables:        $LATEX_DIR"
log_msg "Execution log:       $LOG_FILE"
log_msg "Done. This smoke workflow completed the full thesis pipeline on reduced FP32 settings."
