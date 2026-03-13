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

set -e  # Exit on error

# Activate Conda Environment
if [ -f ~/anaconda3/etc/profile.d/conda.sh ]; then
    source ~/anaconda3/etc/profile.d/conda.sh
    conda activate thesis_env 2>/dev/null || true
fi

# Global Configuration
PYTHON_CMD="${PYTHON_CMD:-python}"
PROFILER_SCRIPT="${PROFILER_SCRIPT:-src/profiler.py}"
HOST_TAG="${HOST_TAG:-$(hostname)}"
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-data/${HOST_TAG}/results}"
LOG_DIR="${LOG_DIR:-logs}"
LOG_FILE="${LOG_DIR}/experiments_$(date +%Y%m%d_%H%M%S).txt"

# Create directories
mkdir -p "$BASE_OUTPUT_DIR" "$LOG_DIR"

# --- GRID SEARCH SPACE DEFINITION ---
# How to use this script:
# 1) Define campaign axes below (models, batches, precisions, optimizers).
# 2) Optionally override behavior via environment variables (e.g., SMOKE_MODE=true).
# 3) Run script; it iterates the cartesian product and writes one artifact set per batch.
# 4) Inspect the generated timestamped log in logs/ and results tree in data/results/.

# 1. Models: Vision (ResNet, ViT), NLP (BERT, GPT2), Baseline (MLP)
MODELS=("resnet50" "resnet152" "vit_b16" "bert_base" "gpt2_small" "simple_mlp")
# MODELS=("vit_b16")  # Fast test: single model

# 2. Batch Sizes (Memory Scalability Analysis)
BATCH_SIZES=(8 16 32 64 128 256)
# BATCH_SIZES=(32)  # Fast test: single batch

# 3. Precisions: FP32 (baseline), FP16 (mixed/tensor cores), BF16 (modern)
PRECISIONS=("fp32" "fp16" "bf16")
# PRECISIONS=("fp32")  # Fast test: single precision

# 4. Optimizers (Memory State Overhead)
OPTIMIZERS=("SGD" "SGD_momentum" "Adam" "AdamW" "RMSprop" "Adagrad" "Adadelta")
# OPTIMIZERS=("SGD" "Adam")  # Fast test: two optimizers

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

check_gpu() {
    if command -v nvidia-smi &> /dev/null; then
        GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
        log_msg "✓ Detected $GPU_COUNT GPU(s): $GPU_NAME"
        return 0
    else
        log_msg "⚠ WARNING: nvidia-smi not found. Running in CPU-only mode."
        return 1
    fi
}

print_summary() {
    log_msg "========================================================"
    log_msg "EXPERIMENT CONFIGURATION SUMMARY"
    log_msg "========================================================"
    log_msg "Script: $PROFILER_SCRIPT"
    log_msg "Output Directory: $BASE_OUTPUT_DIR"
    log_msg "Log File: $LOG_FILE"
    log_msg "Models: ${MODELS[*]}"
    log_msg "Batch Sizes: ${BATCH_SIZES[*]}"
    log_msg "Precisions: ${PRECISIONS[*]}"
    log_msg "Optimizers: ${OPTIMIZERS[*]}"
    log_msg "Warmup Steps: $WARMUP | Measurement Steps: $MEASURE"
    log_msg "Skip CPU Profiling: $USE_SKIP_CPU"
    log_msg "Force CPU Threads: $FORCE_THREADS (0=auto)"
    log_msg "Smoke Mode: $SMOKE_MODE"
    log_msg "Replicates per configuration: $REPEATS"
    log_msg "Seed Base: $SEED_BASE"
    log_msg "Auto Aggregate Stats: $AUTO_AGGREGATE_STATS"
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

log_msg "Starting Advanced Profiler Experiment Campaign"
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

TOTAL_EXPERIMENTS=$((${#MODELS[@]} * ${#OPTIMIZERS[@]} * ${#PRECISIONS[@]} * ${#BATCH_SIZES[@]} * REPEATS))
COMPLETED=0

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

                for ((repeat_idx=1; repeat_idx<=REPEATS; repeat_idx++)); do
                    COMPLETED=$((COMPLETED + 1))
                    PROGRESS="[$COMPLETED/$TOTAL_EXPERIMENTS]"
                    RUN_ID=$(printf "run_%03d" "$repeat_idx")
                    RUN_SEED=$((SEED_BASE + repeat_idx - 1))
                    RUN_OUT_DIR="$OUT_DIR/$RUN_ID"
                    mkdir -p "$RUN_OUT_DIR"

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
                        --seed "$RUN_SEED"
                        --run_id "$RUN_ID"
                    )
                    if [ "$USE_SKIP_CPU" = false ]; then
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

                    # Execute profiler with error handling
                    if "${CMD[@]}" >> "$LOG_FILE" 2>&1; then
                        log_msg "  ✓ SUCCESS: Batch $batch replicate $RUN_ID complete"
                    else
                        EXIT_CODE=$?
                        log_msg "  ✗ FAILURE: Batch $batch replicate $RUN_ID (exit code: $EXIT_CODE)"
                        log_msg "  Probable causes: Out of Memory (OOM), unsupported precision, or hardware constraint"

                        # Optional: uncomment to skip remaining batches on first failure
                        # log_msg "  Skipping remaining batches for this block..."
                        # break
                    fi

                    # Allow GPU memory to cool down
                    sleep 1
                done

                if [ "$AUTO_AGGREGATE_STATS" = true ]; then
                    if [ -f "$AGGREGATOR_SCRIPT" ]; then
                        AGG_OUT="$OUT_DIR/${model}_metrics_stats.csv"
                        if "$PYTHON_CMD" "$AGGREGATOR_SCRIPT" --input_dir "$OUT_DIR" --output_csv "$AGG_OUT" >> "$LOG_FILE" 2>&1; then
                            log_msg "  ✓ STATS: Aggregated replicate metrics -> $AGG_OUT"
                        else
                            log_msg "  ⚠ STATS WARNING: Aggregation failed for $OUT_DIR (continuing)"
                        fi
                    else
                        log_msg "  ⚠ STATS WARNING: Aggregator script not found at $AGGREGATOR_SCRIPT"
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
log_msg "Completed: $COMPLETED"
log_msg "Output Directory: $BASE_OUTPUT_DIR"
log_msg "Log File: $LOG_FILE"
log_msg ""
log_msg "Next Steps:"
log_msg "  1. Check log file for errors: cat $LOG_FILE"
log_msg "  2. Verify output structure: ls -R $BASE_OUTPUT_DIR"
log_msg "  3. Analyze results with ILP model (see thesis Chapter 3)"
log_msg "========================================================"
