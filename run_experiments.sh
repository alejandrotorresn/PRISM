#!/bin/bash

# =======================================================================================
# MASTER EXPERIMENT EXECUTION SCRIPT
# =======================================================================================
# Purpose: Execute an exhaustive Grid Search over Models, Batches, Precisions,
# and Optimizers to generate the cost database for the ILP model.
# Author: Luis Alejandro Torres
# Dependencies: profiles.py, python3, nvidia-smi (optional)
# =======================================================================================

# Activate Conda Environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate thesis_env

# Global Configuration
PYTHON_CMD="python"  # Use conda environment's python
PROFILER_SCRIPT="src/profiler.py"   # Path adjusted to src/
BASE_OUTPUT_DIR="data/raw"
LOG_FILE="logs/experiments_log.txt" # Log in logs/ folder

# --- GRID SEARCH SPACE DEFINITION ---

# 1. Models to characterize (Vision, NLP, MLP)
#MODELS=("resnet50" "resnet152" "vit_b16" "bert_base" "gpt2_small" "simple_mlp")
MODELS=("vit_b16")

# 2. Batch Sizes (Evaluate Memory Scalability)
# Includes typical powers of 2
BATCH_SIZES=(8 16 32 64 128 256)
#BATCH_SIZES=(8 16 32 64)

# 3. Arithmetic Precisions
# fp32: Standard Baseline
# fp16: Mixed Precision (Tensor Cores)
# bf16: BFloat16 (Ampere+ / Modern CPUs)
PRECISIONS=("fp32" "fp16" "bf16")
#PRECISIONS=("fp32")

# 4. Optimizers (Evaluate impact of optimizer states on memory)
# Includes optimizers with varying state memory requirements:
# - SGD: Minimal state (weights only)
# - SGD_momentum: Weights + Momentum buffer (1x extra)
# - Adam/AdamW: Weights + Momentum + Variance (2x extra)
# - RMSprop/Adagrad: Different accumulation strategies (~1x extra)
# - Adadelta: Heavy history tracking (~2x extra)
OPTIMIZERS=("SGD" "SGD_momentum" "Adam" "AdamW" "RMSprop" "Adagrad" "Adadelta")
#OPTIMIZERS=("SGD" "Adam")
# Fixed Profiler Parameters
WARMUP=3 #5
MEASURE=10 #15

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

log_msg() {
    local msg="$1"
    # Ensure log directory exists
    mkdir -p "$(dirname "$LOG_FILE")"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $msg" | tee -a "$LOG_FILE"
}

check_gpu() {
    if command -v nvidia-smi &> /dev/null; then
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
        log_msg "Detected GPU: $GPU_NAME"
        return 0
    else
        log_msg "WARNING: nvidia-smi not found. Assuming CPU-only or simulated execution."
        return 1
    fi
}

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

# Create output directories
mkdir -p "$BASE_OUTPUT_DIR"
log_msg "Starting Experiment Campaign..."
log_msg "Models: ${MODELS[*]}"
log_msg "Batches: ${BATCH_SIZES[*]}"
log_msg "Precisions: ${PRECISIONS[*]}"
log_msg "Optimizers: ${OPTIMIZERS[*]}"

check_gpu

# Main Loop: Model -> Optimizer -> Precision -> Batch Size
for model in "${MODELS[@]}"; do
    for optimizer in "${OPTIMIZERS[@]}"; do
        for precision in "${PRECISIONS[@]}"; do
            
            # Specific directory: data/raw/resnet50/Adam/fp32/
            OUT_DIR="$BASE_OUTPUT_DIR/$model/$optimizer/$precision"
            mkdir -p "$OUT_DIR"
            
            log_msg "--------------------------------------------------------"
            log_msg "Block: $model | $optimizer | $precision"
            log_msg "--------------------------------------------------------"

            for batch in "${BATCH_SIZES[@]}"; do
                log_msg "Executing: Batch $batch..."
                
                # Construct command
                # CRITICAL NOTE: --rapl included to activate CPU energy measurement.
                # --no_gpu is NOT included to ensure GPU usage if available.
                CMD="$PYTHON_CMD $PROFILER_SCRIPT \
                    --model $model \
                    --batch_size $batch \
                    --precision $precision \
                    --optimizer $optimizer \
                    --warmup $WARMUP \
                    --measure $MEASURE \
                    --output_dir $OUT_DIR \
                    --rapl" 

                # Execute and capture exit code
                # Redirect stderr to stdout to capture Python errors in the log
                $CMD >> "$LOG_FILE" 2>&1
                EXIT_CODE=$?

                if [ $EXIT_CODE -eq 0 ]; then
                    log_msg "SUCCESS: Batch $batch finished."
                else
                    # Common error handling: OOM (Out Of Memory)
                    log_msg "FAILURE: Batch $batch failed (Code $EXIT_CODE)."
                    log_msg "Probable cause: OOM or lack of hardware support."
                    
                    # Optional: If it fails with OOM, larger batches likely will too.
                    # Uncomment 'break' to skip to the next block and save time.
                    # break 
                fi
                
                # Short pause for GPU memory cleanup
                sleep 1
            done
        done
    done
done

log_msg "========================================================"
log_msg "Experiment Campaign Finished."
log_msg "Results saved in: $BASE_OUTPUT_DIR"
log_msg "========================================================"
