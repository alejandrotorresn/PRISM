#!/bin/bash
set -e

MODELS=("resnet152" "resnet152" "vit_b16")
OPTS=("AdamW" "SGD" "SGD")

PYTHON_CMD="/home/zephyr/anaconda3/envs/prism_env/bin/python"

for i in "${!MODELS[@]}"; do
    M=${MODELS[$i]}
    O=${OPTS[$i]}
    echo "Aggregating for $M $O batch 64..."
    OUT_DIR="data/zephyr/results_thesis_mode/$M/$O/fp32/batch_64"
    AGG_OUT="$OUT_DIR/${M}_metrics_stats.csv"
    
    $PYTHON_CMD validation/aggregate_metrics_stats.py --input_dir "$OUT_DIR" --output_csv "$AGG_OUT"
done
echo "Aggregation completed."
