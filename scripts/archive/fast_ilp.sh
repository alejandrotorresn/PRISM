#!/bin/bash
set -e

MODELS=("resnet152" "resnet152" "vit_b16")
OPTS=("AdamW" "SGD" "SGD")

export PYTHON_CMD="/home/zephyr/anaconda3/envs/prism_env/bin/python"
export PATH="/home/zephyr/anaconda3/envs/prism_env/bin:$PATH"

for i in "${!MODELS[@]}"; do
    M=${MODELS[$i]}
    O=${OPTS[$i]}
    echo "Running ILP for $M $O batch 64..."
    OUT_DIR="data/zephyr/results_thesis_mode/$M/$O/fp32/batch_64"
    
    MODEL="$M" CONFIG_DIR="$OUT_DIR" bash scripts/run_ilp_partition.sh
    MODEL="$M" CONFIG_DIR="$OUT_DIR" bash scripts/run_ilp_pareto_sweep.sh
done
echo "ILP completed."
