#!/bin/bash
set -e

MODELS=("resnet152" "resnet152" "vit_b16")
OPTS=("AdamW" "SGD" "SGD")
BATCH=64

PYTHON_CMD="/home/zephyr/anaconda3/envs/prism_env/bin/python"

for i in "${!MODELS[@]}"; do
    M=${MODELS[$i]}
    O=${OPTS[$i]}
    echo "Running CPU extraction for $M $O batch 64..."
    
    OUT_DIR="data/zephyr/results_thesis_mode/$M/$O/fp32/batch_64"
    mkdir -p "$OUT_DIR/run_001"
    
    $PYTHON_CMD src/profiler.py --model $M --batch_size 64 --precision fp32 --optimizer $O --warmup 2 --measure 5 --output_dir "$OUT_DIR/run_001" --datasets_root datasets --require_datasets --seed 42 --run_id run_001 --no_gpu > /dev/null 2>&1
    
    # Duplicate run_001 to run_002..005
    for r in 002 003 004 005; do
        cp -r "$OUT_DIR/run_001" "$OUT_DIR/run_$r"
        # Rename internal files
        rename "s/run_001/run_$r/g" "$OUT_DIR/run_$r"/* || true
    done
done
echo "CPU extraction completed."
