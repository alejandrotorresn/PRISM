#!/bin/bash

# =======================================================================================
# MASTER EXPERIMENT EXECUTION SCRIPT - Advanced Hybrid Profiler
# =======================================================================================
# Purpose: Execute an exhaustive Grid Search over Models, Batches, Precisions,
# and Optimizers to generate the cost database for the ILP model (PhD Thesis).
#
# IMPORTANT CHANGES (Zombie Thread Fix):
# - Added '--skip_cpu' flag to extract GPU data independently of CPU FP16 issues
# - Added '--num_threads N' to override SLURM single-core CPU affinity
# - Use '--skip_cpu --num_threads 16' for quick GPU-only profiling
#
# Author: Luis Alejandro Torres
# Dependencies: profiler.py, python3, nvidia-smi (optional)
# =======================================================================================

set -euo pipefail

# Activate Conda Environment
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

# Global Configuration
PYTHON_CMD="${PYTHON_CMD:-python}"
PROFILER_SCRIPT="${PROFILER_SCRIPT:-src/profiler.py}"
DATASET_SCRIPT="${DATASET_SCRIPT:-scripts/download_datasets.py}"
HOST_TAG="${HOST_TAG:-$(hostname)}"
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-data/${HOST_TAG}/results}"
DATASETS_DIR="${DATASETS_DIR:-datasets}"
HOST_TAG="${HOST_TAG:-$(hostname)}"
LOG_DIR="${LOG_DIR:-logs/${HOST_TAG}/experiments}"
LOG_FILE="${LOG_DIR}/experiments_$(date +%Y%m%d_%H%M%S).txt"

# Create log directory early; output directory is normalized later before execution.
mkdir -p "$LOG_DIR"

# --- GRID SEARCH SPACE DEFINITION ---
# How to use this script:
# 1) Define campaign axes below (models, batches, precisions, optimizers).
# 2) Optionally override behavior via environment variables (e.g., SMOKE_MODE=true).
# 3) Run script; it iterates the cartesian product and writes one artifact set per batch.
# 4) Inspect the generated timestamped log in logs/experiments/ and results tree in data/results/.

# 1. Models: Vision (ResNet, ViT), NLP (BERT, GPT2), Baseline (MLP)
MODELS=("resnet50" "resnet152" "vit_b16" "bert_base" "gpt2_small" "distilgpt2" "simple_mlp")
# MODELS=("vit_b16")  # Fast test: single model
MODELS_CSV="${MODELS_CSV:-}"

# 2. Batch Sizes (Memory Scalability Analysis)
BATCH_SIZES=(8 16 32 64 128 256)
# BATCH_SIZES=(32)  # Fast test: single batch
BATCH_SIZES_CSV="${BATCH_SIZES_CSV:-}"

# 3. Precisions: FP32 (baseline), FP16 (mixed/tensor cores), BF16 (modern)
PRECISIONS=("fp32" "fp16" "bf16")
# PRECISIONS=("fp32")  # Fast test: single precision
PRECISIONS_CSV="${PRECISIONS_CSV:-}"

# 4. Optimizers (Memory State Overhead)
OPTIMIZERS=("SGD" "SGD_momentum" "Adam" "AdamW" "RMSprop" "Adagrad" "Adadelta")
# OPTIMIZERS=("SGD" "Adam")  # Fast test: two optimizers
OPTIMIZERS_CSV="${OPTIMIZERS_CSV:-}"

# Profiler Parameters
WARMUP="${WARMUP:-3}"   # Number of warmup iterations
MEASURE="${MEASURE:-10}" # Number of measurement iterations

# --- ZOMBIE THREAD FIX FLAGS ---
# When profiling ViT-B16 or other models where CPU FP16 emulation is slow:
# --skip_cpu:      Skip CPU profiling entirely
# --num_threads N: Force N CPU threads even on SLURM single-core allocation
#                  (Set to physical core count if available)
USE_SKIP_CPU="${USE_SKIP_CPU:-false}"      # Set to 'true' to skip CPU (GPU-only mode)
FORCE_THREADS="${FORCE_THREADS:-0}"         # 0 = auto-detect, >0 = force threads (e.g. 16)
SMOKE_MODE="${SMOKE_MODE:-false}"           # true = quick sanity run (1 model, 1 optimizer, 1 precision, 1 batch)
REPEATS="${REPEATS:-1}"                      # Number of replicated runs per configuration (for robust stats)
SEED_BASE="${SEED_BASE:-42}"                 # Seed base: seed_i = SEED_BASE + (i-1)
AUTO_AGGREGATE_STATS="${AUTO_AGGREGATE_STATS:-true}"  # true = create metrics_stats.csv per config folder
AGGREGATOR_SCRIPT="${AGGREGATOR_SCRIPT:-validation/aggregate_metrics_stats.py}"
ENABLE_RAPL="${ENABLE_RAPL:-true}"          # true = pass --rapl when CPU profiling is enabled
FAIL_FAST="${FAIL_FAST:-false}"             # true = abort the campaign on first profiler/aggregation failure
ALLOW_PARTIAL_FAILURES="${ALLOW_PARTIAL_FAILURES:-false}"  # true = keep exit code 0 even when some runs fail
DRY_RUN="${DRY_RUN:-false}"                 # true = print commands and validate preflight without executing runs
DOWNLOAD_DATASETS="${DOWNLOAD_DATASETS:-true}"
OOM_RETRY_ENABLED="${OOM_RETRY_ENABLED:-true}"
OOM_RETRY_MIN_BATCH="${OOM_RETRY_MIN_BATCH:-1}"
OOM_RETRY_BACKOFF="${OOM_RETRY_BACKOFF:-2}"
OOM_SKIP_CONFIGS="${OOM_SKIP_CONFIGS:-true}"

# --- CPU (DRAM) OOM PROTECTION ---
# Isolate each profiler run inside a cgroup with a memory ceiling so that the
# Linux OOM killer only terminates the child process, not the campaign shell.
# CPU_MEM_LIMIT_PCT: percentage of MemAvailable (from /proc/meminfo) to allow.
#   Recalculated dynamically before every profiler invocation.
USE_CGROUP_MEM_LIMIT="${USE_CGROUP_MEM_LIMIT:-true}"    # true = wrap with systemd-run MemoryMax
CPU_MEM_LIMIT_PCT="${CPU_MEM_LIMIT_PCT:-95}"             # % of MemAvailable at invocation time
PROFILER_TIMEOUT_SECS="${PROFILER_TIMEOUT_SECS:-0}"      # 0 = disabled; >0 = kill after N seconds

if [ "$SMOKE_MODE" = true ]; then
    # Smoke mode intentionally shrinks the grid for a fast end-to-end health check.
    MODELS=("simple_mlp")
    BATCH_SIZES=(8)
    PRECISIONS=("fp32")
    OPTIMIZERS=("SGD")
    WARMUP=1
    MEASURE=1
fi

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

log_msg() {
    local msg="$1"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $msg" | tee -a "$LOG_FILE"
}

is_true() {
    [ "$1" = true ]
}

trim_spaces() {
    local value="$1"
    value="${value#${value%%[![:space:]]*}}"
    value="${value%${value##*[![:space:]]}}"
    printf '%s' "$value"
}

join_by_comma() {
    local IFS=,
    printf '%s' "$*"
}

apply_csv_override() {
    local csv_value="$1"
    local array_name="$2"
    local label="$3"
    local numeric_only="${4:-false}"
    local IFS=','
    local raw_items=()
    local parsed_items=()
    local raw_item
    local item
    local parsed_item

    [ -n "$csv_value" ] || return 0

    read -r -a raw_items <<< "$csv_value"
    for raw_item in "${raw_items[@]}"; do
        item="$(trim_spaces "$raw_item")"
        [ -n "$item" ] || continue
        if [ "$numeric_only" = true ] && ! [[ "$item" =~ ^[0-9]+$ ]]; then
            log_msg "[ERROR] $label override contains a non-integer value: $item"
            exit 1
        fi
        parsed_items+=("$item")
    done

    if [ ${#parsed_items[@]} -eq 0 ]; then
        log_msg "[ERROR] $label override was provided but no valid values were parsed."
        exit 1
    fi

    eval "$array_name=()"
    for parsed_item in "${parsed_items[@]}"; do
        eval "$array_name+=(\"\$parsed_item\")"
    done
}

apply_grid_overrides() {
    apply_csv_override "$MODELS_CSV" MODELS "MODELS_CSV"
    apply_csv_override "$BATCH_SIZES_CSV" BATCH_SIZES "BATCH_SIZES_CSV" true
    apply_csv_override "$PRECISIONS_CSV" PRECISIONS "PRECISIONS_CSV"
    apply_csv_override "$OPTIMIZERS_CSV" OPTIMIZERS "OPTIMIZERS_CSV"
}

require_command_or_executable() {
    local cmd="$1"
    local label="$2"

    if [[ "$cmd" == *" "* ]]; then
        # Allow executable paths containing spaces, but reject command+args patterns.
        if [ -x "$cmd" ]; then
            return 0
        fi
        log_msg "[ERROR] $label appears to include spaces but is not an executable path: $cmd"
        exit 1
    fi

    if [[ "$cmd" == */* ]]; then
        if [ ! -x "$cmd" ]; then
            log_msg "[ERROR] $label is not executable: $cmd"
            exit 1
        fi
        return 0
    fi

    if ! command -v "$cmd" >/dev/null 2>&1; then
        log_msg "[ERROR] $label not found in PATH: $cmd"
        exit 1
    fi
}

run_python_help_check() {
    local script_path="$1"
    local label="$2"

    if [ ! -f "$script_path" ]; then
        log_msg "[ERROR] $label not found: $script_path"
        exit 1
    fi

    if ! "$PYTHON_CMD" "$script_path" --help >/dev/null 2>&1; then
        log_msg "[ERROR] Preflight failed for $label: $script_path"
        log_msg "[ERROR] Check PYTHON_CMD, dependencies, and local imports before launching the campaign."
        exit 1
    fi
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

    local new_parts=("${parts[@]:0:$((data_idx + 1))}" "$host_name" "${parts[@]:$((data_idx + 1))}")
    local joined
    joined=$(IFS=/; echo "${new_parts[*]}")
    printf '%s%s' "$leading_slash" "$joined"
}

handle_failure() {
    local exit_code="$1"
    local context="$2"

    if is_true "$FAIL_FAST"; then
        log_msg "  FAIL_FAST=true -> aborting campaign at: $context"
        exit "$exit_code"
    fi
}

is_oom_failure_from_log() {
    local log_file="$1"
    if [ ! -f "$log_file" ]; then
        return 1
    fi

    tail -n 200 "$log_file" | tr '[:upper:]' '[:lower:]' | grep -E -q \
        "profiling failed after oom retries were exhausted|oom retries were exhausted|cuda out of memory|out of memory|cuda error: out of memory|hip out of memory|cublas.*alloc"
}

# --- CPU (DRAM) OOM protection helpers ---

get_mem_available_mb() {
    # Returns MemAvailable from /proc/meminfo in MiB.
    # MemAvailable is the kernel's estimate of how much memory is available
    # for starting new applications, without swapping.
    local mem_avail_kb
    mem_avail_kb=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo 2>/dev/null)
    if [ -z "$mem_avail_kb" ] || [ "$mem_avail_kb" -le 0 ] 2>/dev/null; then
        echo "0"
        return 1
    fi
    echo $(( mem_avail_kb / 1024 ))
}

is_cpu_oom_kill() {
    # Detect if the process was killed by SIGKILL (exit code 137),
    # which is the signal sent by cgroup MemoryMax enforcement or
    # the Linux OOM killer.  Also detect exit code 9 (raw SIGKILL
    # when bash reports $? = 128+signal for some shells).
    local exit_code="$1"
    [ "$exit_code" -eq 137 ] || [ "$exit_code" -eq 9 ]
}

is_timeout_kill() {
    # Detect if the process was killed by the timeout command (exit code 124).
    local exit_code="$1"
    [ "$exit_code" -eq 124 ]
}

run_profiler_sandboxed() {
    # Execute the profiler command inside a memory-limited cgroup scope.
    # Usage: run_profiler_sandboxed <cmd_array...>
    # Returns the exit code of the profiler process.
    #
    # Protection layers (applied in order):
    #   1. systemd-run --user --scope -p MemoryMax  (cgroup isolation)
    #   2. timeout (optional, if PROFILER_TIMEOUT_SECS > 0)
    #   3. Plain execution (fallback if systemd-run is unavailable)
    #
    # The cgroup MemoryMax is computed as CPU_MEM_LIMIT_PCT% of the
    # *current* MemAvailable, so it adapts to the actual system state
    # at the moment each run starts.

    local cmd=("$@")
    local wrapper=()
    local mem_limit_mb=0
    local mem_avail_mb=0

    # --- Layer 1: cgroup memory ceiling ---
    if is_true "$USE_CGROUP_MEM_LIMIT" && command -v systemd-run >/dev/null 2>&1; then
        mem_avail_mb="$(get_mem_available_mb)"
        if [ "$mem_avail_mb" -gt 0 ] 2>/dev/null; then
            mem_limit_mb=$(( mem_avail_mb * CPU_MEM_LIMIT_PCT / 100 ))
            if [ "$mem_limit_mb" -gt 0 ]; then
                wrapper=(systemd-run --user --scope -q -p "MemoryMax=${mem_limit_mb}M" -p "MemorySwapMax=0")
                log_msg "  🛡️ CGROUP: MemAvailable=${mem_avail_mb}MB → limit=${mem_limit_mb}MB (${CPU_MEM_LIMIT_PCT}%) swap=0"
            fi
        else
            log_msg "  ⚠ CGROUP: Could not read MemAvailable; running without memory limit"
        fi
    fi

    # --- Layer 2: optional timeout ---
    if [ "$PROFILER_TIMEOUT_SECS" -gt 0 ] 2>/dev/null; then
        wrapper+=(timeout --signal=KILL "$PROFILER_TIMEOUT_SECS")
    fi

    # --- Execute ---
    "${wrapper[@]}" "${cmd[@]}"
    return $?
}

mark_oom_skipped_config() {
    local out_dir="$1"
    local model="$2"
    local optimizer="$3"
    local precision="$4"
    local batch="$5"
    local run_id="$6"
    local exit_code="$7"

    cat > "$out_dir/oom_skipped_config.json" <<EOF
{
  "status": "oom_skipped",
  "model": "$model",
  "optimizer": "$optimizer",
  "precision": "$precision",
  "batch_size": $batch,
  "failed_run_id": "$run_id",
  "exit_code": $exit_code,
  "timestamp_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF
}

mark_cpu_oom_skipped_config() {
    local out_dir="$1"
    local model="$2"
    local optimizer="$3"
    local precision="$4"
    local batch="$5"
    local run_id="$6"
    local exit_code="$7"
    local mem_limit_mb="${8:-unknown}"

    cat > "$out_dir/oom_cpu_skipped_config.json" <<EOF
{
  "status": "oom_cpu_killed",
  "model": "$model",
  "optimizer": "$optimizer",
  "precision": "$precision",
  "batch_size": $batch,
  "failed_run_id": "$run_id",
  "exit_code": $exit_code,
  "mem_limit_mb": "$mem_limit_mb",
  "timestamp_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF
}

preflight_checks() {
    require_command_or_executable "$PYTHON_CMD" "PYTHON_CMD"
    run_python_help_check "$PROFILER_SCRIPT" "Profiler script"
    run_python_help_check "$DATASET_SCRIPT" "Dataset download script"

    if is_true "$AUTO_AGGREGATE_STATS"; then
        run_python_help_check "$AGGREGATOR_SCRIPT" "Aggregator script"
    fi
}

check_gpu() {
    local gpu_info
    local gpu_count
    local gpu_name

    if gpu_info=$("$PYTHON_CMD" -c 'import torch; count = torch.cuda.device_count(); name = torch.cuda.get_device_name(0) if count > 0 else ""; print(f"{count}|{name}")' 2>/dev/null); then
        gpu_count="${gpu_info%%|*}"
        gpu_name="${gpu_info#*|}"
        if [ "$gpu_count" -gt 0 ]; then
            if command -v nvidia-smi >/dev/null 2>&1; then
                gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
            fi
            log_msg "✓ Detected $gpu_count GPU(s): $gpu_name"
            return 0
        fi
    fi

    log_msg "⚠ WARNING: No CUDA GPU detected by PyTorch. Running in CPU-only mode."
    return 1
}

print_summary() {
    log_msg "========================================================"
    log_msg "EXPERIMENT CONFIGURATION SUMMARY"
    log_msg "========================================================"
    log_msg "Script: $PROFILER_SCRIPT"
    log_msg "Output Directory: $BASE_OUTPUT_DIR"
    log_msg "Datasets Directory: $DATASETS_DIR"
    log_msg "Log File: $LOG_FILE"
    log_msg "Models: ${MODELS[*]}"
    log_msg "Batch Sizes: ${BATCH_SIZES[*]}"
    log_msg "Precisions: ${PRECISIONS[*]}"
    log_msg "Optimizers: ${OPTIMIZERS[*]}"
    log_msg "Warmup Steps: $WARMUP | Measurement Steps: $MEASURE"
    log_msg "Skip CPU Profiling: $USE_SKIP_CPU"
    log_msg "Enable RAPL: $ENABLE_RAPL"
    log_msg "Force CPU Threads: $FORCE_THREADS (0=auto)"
    log_msg "Smoke Mode: $SMOKE_MODE"
    log_msg "Replicates per configuration: $REPEATS"
    log_msg "Seed Base: $SEED_BASE"
    log_msg "Auto Aggregate Stats: $AUTO_AGGREGATE_STATS"
    log_msg "Fail Fast: $FAIL_FAST"
    log_msg "OOM Retry Enabled: $OOM_RETRY_ENABLED"
    log_msg "OOM Retry Min Batch: $OOM_RETRY_MIN_BATCH"
    log_msg "OOM Retry Backoff: $OOM_RETRY_BACKOFF"
    log_msg "OOM Skip Configs: $OOM_SKIP_CONFIGS"
    log_msg "CPU OOM Protection (cgroup): $USE_CGROUP_MEM_LIMIT"
    log_msg "CPU Mem Limit: ${CPU_MEM_LIMIT_PCT}% of MemAvailable (dynamic)"
    log_msg "Profiler Timeout: ${PROFILER_TIMEOUT_SECS}s (0=disabled)"
    log_msg "Dry Run: $DRY_RUN"
    log_msg "MODELS_CSV Override: ${MODELS_CSV:-<none>}"
    log_msg "BATCH_SIZES_CSV Override: ${BATCH_SIZES_CSV:-<none>}"
    log_msg "PRECISIONS_CSV Override: ${PRECISIONS_CSV:-<none>}"
    log_msg "OPTIMIZERS_CSV Override: ${OPTIMIZERS_CSV:-<none>}"
    log_msg "========================================================"
}

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
# Execution flow (high level):
# A) Print campaign config and detect GPU availability.
# B) Normalize incompatible options (e.g., disable USE_SKIP_CPU if there is no GPU).
# C) Iterate model/optimizer/precision/batch combinations.
# D) Build profiler command safely as argument array (space-safe, no eval).
# E) Execute, log success/failure, and continue the campaign.

apply_grid_overrides
BASE_OUTPUT_DIR="$(normalize_output_dir_for_host_sh "$BASE_OUTPUT_DIR" "$HOST_TAG")"
mkdir -p "$BASE_OUTPUT_DIR"

log_msg "Starting Advanced Profiler Experiment Campaign"
preflight_checks
print_summary
if check_gpu; then
    HAS_GPU=true
else
    HAS_GPU=false
fi

if [ "$USE_SKIP_CPU" = true ] && [ "$HAS_GPU" = false ]; then
    log_msg "⚠ WARNING: USE_SKIP_CPU=true requires GPU execution target. Disabling USE_SKIP_CPU for this run."
    USE_SKIP_CPU=false
fi

if is_true "$DOWNLOAD_DATASETS"; then
    DATASET_MODELS_CSV="$(join_by_comma "${MODELS[@]}")"
    log_msg "Preparing datasets for models: $DATASET_MODELS_CSV"
    "$PYTHON_CMD" "$DATASET_SCRIPT" --models "$DATASET_MODELS_CSV" --datasets_root "$DATASETS_DIR" >> "$LOG_FILE" 2>&1
    log_msg "Dataset preparation finished"
fi

TOTAL_EXPERIMENTS=$((${#MODELS[@]} * ${#OPTIMIZERS[@]} * ${#PRECISIONS[@]} * ${#BATCH_SIZES[@]} * REPEATS))
ATTEMPTED=0
SUCCEEDED_RUNS=0
FAILED_RUNS=0
FAILED_AGGREGATIONS=0
OOM_SKIPPED_CONFIGS=0
OOM_SKIPPED_RUNS=0
CPU_OOM_SKIPPED_CONFIGS=0
CPU_OOM_SKIPPED_RUNS=0
TIMEOUT_SKIPPED_RUNS=0

# Main Loop: Model -> Optimizer -> Precision -> Batch Size
# Each loop iteration corresponds to one profiler execution and one output folder.
for model in "${MODELS[@]}"; do
    for optimizer in "${OPTIMIZERS[@]}"; do
        for precision in "${PRECISIONS[@]}"; do
            
            # Output directory structure: data/results/{model}/{optimizer}/{precision}/batch_{N}/
            
            log_msg "────────────────────────────────────────────────────────"
            log_msg "Profiling Block: Model=$model | Optimizer=$optimizer | Precision=$precision"
            log_msg "────────────────────────────────────────────────────────"

            for batch in "${BATCH_SIZES[@]}"; do
                # Keep one isolated output directory per batch to avoid file overwrite.
                OUT_DIR="$BASE_OUTPUT_DIR/$model/$optimizer/$precision/batch_${batch}"
                mkdir -p "$OUT_DIR"
                BATCH_SKIPPED_OOM=false
                
                if [ -f "$OUT_DIR/${model}_metrics_stats.csv" ]; then
                    log_msg "  → SKIPPING BATCH: Batch $batch already profiled and aggregated (resuming)."
                    ATTEMPTED=$((ATTEMPTED + REPEATS))
                    SUCCEEDED_RUNS=$((SUCCEEDED_RUNS + REPEATS))
                    continue
                elif [ -f "$OUT_DIR/oom_skipped_config.json" ] || [ -f "$OUT_DIR/oom_cpu_skipped_config.json" ]; then
                    log_msg "  → SKIPPING BATCH: Batch $batch previously marked as OOM-skipped (resuming)."
                    ATTEMPTED=$((ATTEMPTED + REPEATS))
                    if [ -f "$OUT_DIR/oom_cpu_skipped_config.json" ]; then
                        CPU_OOM_SKIPPED_CONFIGS=$((CPU_OOM_SKIPPED_CONFIGS + 1))
                        CPU_OOM_SKIPPED_RUNS=$((CPU_OOM_SKIPPED_RUNS + REPEATS))
                    else
                        OOM_SKIPPED_CONFIGS=$((OOM_SKIPPED_CONFIGS + 1))
                        OOM_SKIPPED_RUNS=$((OOM_SKIPPED_RUNS + REPEATS))
                    fi
                    continue
                fi

                for ((repeat_idx=1; repeat_idx<=REPEATS; repeat_idx++)); do
                    ATTEMPTED=$((ATTEMPTED + 1))
                    PROGRESS="[$ATTEMPTED/$TOTAL_EXPERIMENTS]"
                    RUN_ID=$(printf "run_%03d" "$repeat_idx")
                    RUN_SEED=$((SEED_BASE + repeat_idx - 1))
                    RUN_OUT_DIR="$OUT_DIR/$RUN_ID"
                    mkdir -p "$RUN_OUT_DIR"

                    if [ -f "$RUN_OUT_DIR/${model}_meta.json" ]; then
                        log_msg "  → SKIPPING RUN: $RUN_ID already exists (resuming)."
                        SUCCEEDED_RUNS=$((SUCCEEDED_RUNS + 1))
                        continue
                    fi

                    log_msg "$PROGRESS Processing: batch_size=$batch | replicate=$RUN_ID | seed=$RUN_SEED"

                    # Build profiler command with all flags (array-safe, no eval)
                    CMD=(
                        "$PYTHON_CMD" "$PROFILER_SCRIPT"
                        --model "$model"
                        --batch_size "$batch"
                        --precision "$precision"
                        --optimizer "$optimizer"
                        --warmup "$WARMUP"
                        --measure "$MEASURE"
                        --output_dir "$RUN_OUT_DIR"
                        --datasets_root "$DATASETS_DIR"
                        --require_datasets
                        --seed "$RUN_SEED"
                        --run_id "$RUN_ID"
                        --oom_retry_min_batch "$OOM_RETRY_MIN_BATCH"
                        --oom_retry_backoff "$OOM_RETRY_BACKOFF"
                    )
                    if [ "$OOM_RETRY_ENABLED" = true ]; then
                        CMD+=(--oom_retry_enabled)
                    else
                        CMD+=(--no-oom_retry_enabled)
                    fi
                    
                    if [ "${FORCE_NO_GPU:-false}" = true ]; then
                        CMD+=(--no_gpu)
                    fi
                    if [ "$USE_SKIP_CPU" = false ] && [ "$ENABLE_RAPL" = true ]; then
                        # RAPL is only meaningful when CPU profiling is enabled.
                        CMD+=(--rapl)  # Enable CPU RAPL energy measurement
                    fi

                    # Add zombie thread fix flags if configured
                    if [ "$USE_SKIP_CPU" = true ]; then
                        CMD+=(--skip_cpu)
                        log_msg "  → Skipping CPU profiling (--skip_cpu enabled)"
                    fi
                    if [ "$FORCE_THREADS" -gt 0 ]; then
                        CMD+=(--num_threads "$FORCE_THREADS")
                        log_msg "  → Forcing $FORCE_THREADS CPU threads (--num_threads)"
                    fi

                    if [ "$DRY_RUN" = true ]; then
                        log_msg "  [DRY_RUN] ${CMD[*]}"
                        SUCCEEDED_RUNS=$((SUCCEEDED_RUNS + 1))
                    # Execute profiler inside sandboxed cgroup with memory ceiling
                    elif run_profiler_sandboxed "${CMD[@]}" >> "$LOG_FILE" 2>&1; then
                        SUCCEEDED_RUNS=$((SUCCEEDED_RUNS + 1))
                        log_msg "  ✓ SUCCESS: Batch $batch replicate $RUN_ID complete"
                    else
                        EXIT_CODE=$?

                        # --- Priority 1: CPU/DRAM OOM (SIGKILL from cgroup or kernel) ---
                        if is_cpu_oom_kill "$EXIT_CODE"; then
                            CPU_OOM_SKIPPED_CONFIGS=$((CPU_OOM_SKIPPED_CONFIGS + 1))
                            CPU_OOM_SKIPPED_RUNS=$((CPU_OOM_SKIPPED_RUNS + 1))
                            mark_cpu_oom_skipped_config "$OUT_DIR" "$model" "$optimizer" "$precision" "$batch" "$RUN_ID" "$EXIT_CODE" "${mem_limit_mb:-unknown}"
                            log_msg "  🧠 CPU OOM KILL: Batch $batch replicate $RUN_ID (exit code: $EXIT_CODE, signal SIGKILL)"
                            log_msg "  Process exceeded CPU/DRAM memory limit. Marked -> $OUT_DIR/oom_cpu_skipped_config.json"
                            BATCH_SKIPPED_OOM=true
                            break

                        # --- Priority 2: Timeout kill ---
                        elif is_timeout_kill "$EXIT_CODE"; then
                            TIMEOUT_SKIPPED_RUNS=$((TIMEOUT_SKIPPED_RUNS + 1))
                            FAILED_RUNS=$((FAILED_RUNS + 1))
                            log_msg "  ⏱️ TIMEOUT KILL: Batch $batch replicate $RUN_ID exceeded ${PROFILER_TIMEOUT_SECS}s"
                            BATCH_SKIPPED_OOM=true
                            break

                        # --- Priority 3: GPU VRAM OOM (detected from Python log output) ---
                        elif is_true "$OOM_SKIP_CONFIGS" && is_oom_failure_from_log "$LOG_FILE"; then
                            OOM_SKIPPED_CONFIGS=$((OOM_SKIPPED_CONFIGS + 1))
                            OOM_SKIPPED_RUNS=$((OOM_SKIPPED_RUNS + 1))
                            mark_oom_skipped_config "$OUT_DIR" "$model" "$optimizer" "$precision" "$batch" "$RUN_ID" "$EXIT_CODE"
                            log_msg "  ⚠ OOM SKIP: Batch $batch replicate $RUN_ID (exit code: $EXIT_CODE)"
                            log_msg "  Marked as OOM-skipped config -> $OUT_DIR/oom_skipped_config.json"
                            
                            log_msg "  [!] Iniciando Analytical Rescue Regime (--no_gpu) para trazado ILP..."
                            RESCUE_CMD=("${CMD[@]}")
                            RESCUE_CMD+=("--no_gpu")
                            
                            if run_profiler_sandboxed "${RESCUE_CMD[@]}" >> "$LOG_FILE" 2>&1; then
                                log_msg "  ✓ RESCUE SUCCESS: El trazado CPU fue exitoso. ILP podrá usar fallback analítico."
                                SUCCEEDED_RUNS=$((SUCCEEDED_RUNS + 1))
                                BATCH_SKIPPED_OOM=false
                                FORCE_NO_GPU=true
                            else
                                RESCUE_EXIT=$?
                                if is_cpu_oom_kill "$RESCUE_EXIT"; then
                                    CPU_OOM_SKIPPED_CONFIGS=$((CPU_OOM_SKIPPED_CONFIGS + 1))
                                    log_msg "  🧠 RESCUE CPU OOM KILL: El trazado CPU también excedió la memoria DRAM."
                                else
                                    log_msg "  ✗ RESCUE FAILURE: El trazado CPU también falló (exit: $RESCUE_EXIT). No habrá datos ILP."
                                fi
                                BATCH_SKIPPED_OOM=true
                                break
                            fi
                            
                            log_msg "  Continuando con la campaña (réplicas restantes en modo --no_gpu)."

                        # --- Priority 4: Generic failure ---
                        else
                            FAILED_RUNS=$((FAILED_RUNS + 1))
                            log_msg "  ✗ FAILURE: Batch $batch replicate $RUN_ID (exit code: $EXIT_CODE)"
                            log_msg "  Probable causes: Out of Memory (OOM), unsupported precision, or hardware constraint"
                            handle_failure "$EXIT_CODE" "$model/$optimizer/$precision/batch_${batch}/$RUN_ID"
                        fi
                    fi

                    # Allow GPU memory to cool down
                    sleep 1
                done

                if [ "$BATCH_SKIPPED_OOM" = true ]; then
                    log_msg "  ↷ SKIP AGGREGATION: batch_$batch marked as OOM-skipped."
                    continue
                fi

                if [ "$AUTO_AGGREGATE_STATS" = true ]; then
                    AGG_OUT="$OUT_DIR/${model}_metrics_stats.csv"
                    if [ "$DRY_RUN" = true ]; then
                        log_msg "  [DRY_RUN] $PYTHON_CMD $AGGREGATOR_SCRIPT --input_dir $OUT_DIR --output_csv $AGG_OUT"
                    elif "$PYTHON_CMD" "$AGGREGATOR_SCRIPT" --input_dir "$OUT_DIR" --output_csv "$AGG_OUT" >> "$LOG_FILE" 2>&1; then
                        log_msg "  ✓ STATS: Aggregated replicate metrics -> $AGG_OUT"
                    else
                        FAILED_AGGREGATIONS=$((FAILED_AGGREGATIONS + 1))
                        log_msg "  ✗ STATS FAILURE: Aggregation failed for $OUT_DIR"
                        handle_failure 1 "$model/$optimizer/$precision/batch_${batch}/aggregation"
                    fi
                fi
            done
        done
    done
done

# ==============================================================================
# COMPLETION SUMMARY
# ==============================================================================

log_msg "========================================================"
log_msg "EXPERIMENT CAMPAIGN COMPLETED"
log_msg "========================================================"
log_msg "Total Experiments: $TOTAL_EXPERIMENTS"
log_msg "Attempted Runs: $ATTEMPTED"
log_msg "Successful Runs: $SUCCEEDED_RUNS"
log_msg "Failed Runs: $FAILED_RUNS"
log_msg "Failed Aggregations: $FAILED_AGGREGATIONS"
log_msg "OOM-skipped Configurations (GPU): $OOM_SKIPPED_CONFIGS"
log_msg "OOM-skipped Runs (GPU): $OOM_SKIPPED_RUNS"
log_msg "CPU OOM-killed Configurations: $CPU_OOM_SKIPPED_CONFIGS"
log_msg "CPU OOM-killed Runs: $CPU_OOM_SKIPPED_RUNS"
log_msg "Timeout-killed Runs: $TIMEOUT_SKIPPED_RUNS"
log_msg "Output Directory: $BASE_OUTPUT_DIR"
log_msg "Log File: $LOG_FILE"
log_msg "Allow Partial Failures: $ALLOW_PARTIAL_FAILURES"
log_msg ""
log_msg "Next Steps:"
log_msg "  1. Check log file for errors: cat $LOG_FILE"
log_msg "  2. Verify output structure: ls -R $BASE_OUTPUT_DIR"
log_msg "  3. Analyze results with ILP model (see thesis Chapter 3)"
log_msg "========================================================"

if [ "$FAILED_RUNS" -gt 0 ] || [ "$FAILED_AGGREGATIONS" -gt 0 ]; then
    log_msg "[ERROR] Campaign finished with failures. Review $LOG_FILE before using the collected data."
    if [ "$ALLOW_PARTIAL_FAILURES" = true ]; then
        log_msg "[WARN] ALLOW_PARTIAL_FAILURES=true -> returning success so downstream stages can continue."
    else
        exit 1
    fi
fi
