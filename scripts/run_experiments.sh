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
PYTHON_CMD="python"
PROFILER_SCRIPT="src/profiler.py"
BASE_OUTPUT_DIR="data/results"
LOG_DIR="logs"
LOG_FILE="${LOG_DIR}/experiments_$(date +%Y%m%d_%H%M%S).txt"

# Create directories
mkdir -p "$BASE_OUTPUT_DIR" "$LOG_DIR"

# --- GRID SEARCH SPACE DEFINITION ---

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
WARMUP=3   # Number of warmup iterations
MEASURE=10 # Number of measurement iterations

# --- ZOMBIE THREAD FIX FLAGS ---
# When profiling ViT-B16 or other models where CPU FP16 emulation is slow:
# --skip_cpu:      Skip CPU profiling entirely
# --num_threads N: Force N CPU threads even on SLURM single-core allocation
#                  (Set to physical core count if available)
USE_SKIP_CPU=false      # Set to 'true' to skip CPU (GPU-only mode)
FORCE_THREADS=0         # 0 = auto-detect, >0 = force threads (e.g. 16)

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
    log_msg "========================================================"
}

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

log_msg "Starting Advanced Profiler Experiment Campaign"
print_summary
check_gpu

TOTAL_EXPERIMENTS=$((${#MODELS[@]} * ${#OPTIMIZERS[@]} * ${#PRECISIONS[@]} * ${#BATCH_SIZES[@]}))
COMPLETED=0

# Main Loop: Model -> Optimizer -> Precision -> Batch Size
for model in "${MODELS[@]}"; do
    for optimizer in "${OPTIMIZERS[@]}"; do
        for precision in "${PRECISIONS[@]}"; do
            
            # Output directory structure: data/results/{model}/{optimizer}/{precision}/
            OUT_DIR="$BASE_OUTPUT_DIR/$model/$optimizer/$precision"
            mkdir -p "$OUT_DIR"
            
            log_msg "────────────────────────────────────────────────────────"
            log_msg "Profiling Block: Model=$model | Optimizer=$optimizer | Precision=$precision"
            log_msg "────────────────────────────────────────────────────────"

            for batch in "${BATCH_SIZES[@]}"; do
                COMPLETED=$((COMPLETED + 1))
                PROGRESS="[$COMPLETED/$TOTAL_EXPERIMENTS]"
                
                log_msg "$PROGRESS Processing: batch_size=$batch"
                
                # Build profiler command with all flags
                CMD="$PYTHON_CMD $PROFILER_SCRIPT"
                CMD="$CMD --model $model"
                CMD="$CMD --batch_size $batch"
                CMD="$CMD --precision $precision"
                CMD="$CMD --optimizer $optimizer"
                CMD="$CMD --warmup $WARMUP"
                CMD="$CMD --measure $MEASURE"
                CMD="$CMD --output_dir $OUT_DIR"
                CMD="$CMD --rapl"  # Enable CPU RAPL energy measurement
                
                # Add zombie thread fix flags if configured
                if [ "$USE_SKIP_CPU" = true ]; then
                    CMD="$CMD --skip_cpu"
                    log_msg "  → Skipping CPU profiling (--skip_cpu enabled)"
                fi
                if [ "$FORCE_THREADS" -gt 0 ]; then
                    CMD="$CMD --num_threads $FORCE_THREADS"
                    log_msg "  → Forcing $FORCE_THREADS CPU threads (--num_threads)"
                fi
                
                # Execute profiler with error handling
                if eval "$CMD" >> "$LOG_FILE" 2>&1; then
                    log_msg "  ✓ SUCCESS: Batch $batch complete"
                else
                    EXIT_CODE=$?
                    log_msg "  ✗ FAILURE: Batch $batch (exit code: $EXIT_CODE)"
                    log_msg "  Probable causes: Out of Memory (OOM), unsupported precision, or hardware constraint"
                    
                    # Optional: uncomment to skip remaining batches on first failure
                    # log_msg "  Skipping remaining batches for this block..."
                    # break
                fi
                
                # Allow GPU memory to cool down
                sleep 1
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
