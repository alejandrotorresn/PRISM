#!/bin/bash
set -euo pipefail

# Properly initialize conda for a non-interactive shell (plain
# `conda activate` fails with "Run 'conda init' before 'conda activate'"
# when this script is run via `bash script.sh` instead of sourced in an
# already-initialized interactive shell -- conda's shell *functions* don't
# propagate to subshells/scripts, only PATH does).
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prism_env

# ===== resnet50 ===== (already run manually, 2026-10-13 -- regime A + regime B
# complete, see reports/zephyr/convergence_study/resnet50/)
# PROFILE=custom \
# MODELS_CSV=resnet50 \
# OPTIMIZERS_CSV=SGD \
# PRECISIONS_CSV=fp32 \
# BATCH_SIZES_CSV=64,192 \
# REPEATS=2 WARMUP=1 MEASURE=3 \
# RUN_HYBRID=true \
# PHASE4_ACTIVATION=true \
# PHASE4_ENABLE_RECOMPUTE=true \
# PHASE4_ENABLE_CHECKPOINT=true \
# PHASE4_BACKEND=joint \
# PHASE4_CALIBRATE_RECOMPUTE_CURVE=true \
# GPU_EXPANSION=true \
# bash scripts/run_thesis_mode.sh
#
# BATCH_SIZE_resnet50_A=64 \
# BATCH_SIZE_resnet50_B=192 \
# SEEDS_CSV=42 \
# MAX_STEPS=200 \
# MODELS_CSV=resnet50 \
# PRECISIONS_CSV=fp32 \
# bash convergence_validation/run_convergence_study.sh


# ===== resnet152 ===== (already run manually, 2026-10-13 -- regime A + regime B
# complete, see reports/zephyr/convergence_study/resnet152/)
# PROFILE=custom \
# MODELS_CSV=resnet152 \
# OPTIMIZERS_CSV=SGD \
# PRECISIONS_CSV=fp32 \
# BATCH_SIZES_CSV=32,96 \
# REPEATS=2 WARMUP=1 MEASURE=3 \
# RUN_HYBRID=true \
# PHASE4_ACTIVATION=true \
# PHASE4_ENABLE_RECOMPUTE=true \
# PHASE4_ENABLE_CHECKPOINT=true \
# PHASE4_BACKEND=joint \
# PHASE4_CALIBRATE_RECOMPUTE_CURVE=true \
# GPU_EXPANSION=true \
# bash scripts/run_thesis_mode.sh
#
# BATCH_SIZE_resnet152_A=32 \
# BATCH_SIZE_resnet152_B=96 \
# SEEDS_CSV=42 \
# MAX_STEPS=200 \
# MODELS_CSV=resnet152 \
# PRECISIONS_CSV=fp32 \
# bash convergence_validation/run_convergence_study.sh


# ===== distilgpt2 =====
# NOTE: no MEMORY_MODEL here -- Phase4 (joint backend + calibrated recompute
# curve) requires the default "topological" memory model; combining it with
# "topological_tight" raises an explicit ValueError (see fix/09 README).
PROFILE=custom \
MODELS_CSV=distilgpt2 \
OPTIMIZERS_CSV=AdamW \
PRECISIONS_CSV=fp32 \
BATCH_SIZES_CSV=16,64 \
REPEATS=2 WARMUP=1 MEASURE=3 \
RUN_HYBRID=true \
PHASE4_ACTIVATION=true \
PHASE4_ENABLE_RECOMPUTE=true \
PHASE4_ENABLE_CHECKPOINT=true \
PHASE4_BACKEND=joint \
PHASE4_CALIBRATE_RECOMPUTE_CURVE=true \
GPU_EXPANSION=true \
bash scripts/run_thesis_mode.sh

BATCH_SIZE_distilgpt2_A=16 \
BATCH_SIZE_distilgpt2_B=64 \
SEEDS_CSV=42 \
MAX_STEPS=200 \
MODELS_CSV=distilgpt2 \
PRECISIONS_CSV=fp32 \
bash convergence_validation/run_convergence_study.sh


# ===== gpt2_small =====
PROFILE=custom \
MODELS_CSV=gpt2_small \
OPTIMIZERS_CSV=AdamW \
PRECISIONS_CSV=fp32 \
BATCH_SIZES_CSV=8,32 \
REPEATS=2 WARMUP=1 MEASURE=3 \
RUN_HYBRID=true \
PHASE4_ACTIVATION=true \
PHASE4_ENABLE_RECOMPUTE=true \
PHASE4_ENABLE_CHECKPOINT=true \
PHASE4_BACKEND=joint \
PHASE4_CALIBRATE_RECOMPUTE_CURVE=true \
GPU_EXPANSION=true \
bash scripts/run_thesis_mode.sh

BATCH_SIZE_gpt2_small_A=8 \
BATCH_SIZE_gpt2_small_B=32 \
SEEDS_CSV=42 \
MAX_STEPS=200 \
MODELS_CSV=gpt2_small \
PRECISIONS_CSV=fp32 \
bash convergence_validation/run_convergence_study.sh


# ===== vit_b16 =====
# NOTE on batch sizes: unlike resnet50/resnet152/distilgpt2/gpt2_small, there
# is NO local-GPU evidence-based default for this model (fix/01's survey only
# covered those 4; vit_b16's only prior data is from the 32GB-VRAM CHUC-4
# cluster at batch 512, not representative of this 8GB local GPU). 16/64 is a
# conservative starting guess (same pattern as distilgpt2/gpt2_small's
# successful small-batch pairing) -- watch STEP 2/5 (profiling) closely on
# the first real run; the OOM-retry fallback in src/profiler.py will adapt if
# 64 is still too large for regime B, but it hasn't been exercised for this
# model+optimizer combination before.
# NOTE on optimizer: convergence_validation/run_convergence_study.sh's
# optimizer_for_model()/lr_for_model() only special-case resnet50/resnet152
# for OPTIMIZER_VISION/LR_VISION -- everything else (including vit_b16) falls
# through to OPTIMIZER_LM=adamw/LR_LM=0.00005. Using AdamW here too keeps the
# profiled config dir (OPTIMIZERS_CSV below) consistent with what
# run_convergence_study.sh will actually look for.
PROFILE=custom \
MODELS_CSV=vit_b16 \
OPTIMIZERS_CSV=AdamW \
PRECISIONS_CSV=fp32 \
BATCH_SIZES_CSV=16,64 \
REPEATS=2 WARMUP=1 MEASURE=3 \
RUN_HYBRID=true \
PHASE4_ACTIVATION=true \
PHASE4_ENABLE_RECOMPUTE=true \
PHASE4_ENABLE_CHECKPOINT=true \
PHASE4_BACKEND=joint \
PHASE4_CALIBRATE_RECOMPUTE_CURVE=true \
GPU_EXPANSION=true \
bash scripts/run_thesis_mode.sh

BATCH_SIZE_vit_b16_A=16 \
BATCH_SIZE_vit_b16_B=64 \
SEEDS_CSV=42 \
MAX_STEPS=200 \
MODELS_CSV=vit_b16 \
PRECISIONS_CSV=fp32 \
bash convergence_validation/run_convergence_study.sh


# ===== bert_base =====
# Same caveats as vit_b16 above: no local-GPU evidence-based batch default,
# falls through to OPTIMIZER_LM=adamw/LR_LM=0.00005 in run_convergence_study.sh.
PROFILE=custom \
MODELS_CSV=bert_base \
OPTIMIZERS_CSV=AdamW \
PRECISIONS_CSV=fp32 \
BATCH_SIZES_CSV=16,64 \
REPEATS=2 WARMUP=1 MEASURE=3 \
RUN_HYBRID=true \
PHASE4_ACTIVATION=true \
PHASE4_ENABLE_RECOMPUTE=true \
PHASE4_ENABLE_CHECKPOINT=true \
PHASE4_BACKEND=joint \
PHASE4_CALIBRATE_RECOMPUTE_CURVE=true \
GPU_EXPANSION=true \
bash scripts/run_thesis_mode.sh

BATCH_SIZE_bert_base_A=16 \
BATCH_SIZE_bert_base_B=64 \
SEEDS_CSV=42 \
MAX_STEPS=200 \
MODELS_CSV=bert_base \
PRECISIONS_CSV=fp32 \
bash convergence_validation/run_convergence_study.sh
