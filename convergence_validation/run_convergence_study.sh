#!/usr/bin/env bash
# Orchestrates the full convergence-study matrix: model x precision x regime
# x seed. Each cell is one call to run_convergence_study.py.
#
# Follows the same "env-var CSV list" convention already used by
# scripts/run_thesis_mode.sh (MODELS_CSV, PRECISIONS_CSV, etc.) so this
# script feels familiar to anyone who has run that one. It does NOT call
# scripts/run_thesis_mode.sh or modify it — it only reads the ILP solutions
# that previous runs of it already left on disk under data/.
#
# Regime A (sufficiency): reads <config_dir>/ilp_solution/ — the full-budget
#   plan — and runs both `ilp_plan` (PRISM) and `all_gpu` for a direct
#   comparison at the same batch size, chosen (per model) to be one where
#   All-GPU genuinely fits.
# Regime B (OOM rescue / constrained budget): reads
#   <config_dir>/ilp_solution_pareto_best_budget_*/ — the reduced-budget
#   plan — and runs `ilp_plan` (PRISM), `all_gpu_gradaccum` (a smaller
#   micro-batch, accumulated up to the same effective batch size) as a
#   secondary, documented-as-imperfect baseline, AND `greedy_plan` (a
#   naive-but-real device-partitioning heuristic, built on the fly by
#   build_greedy_plan.py at the SAME budget as the ILP plan) for a three-way
#   PRISM vs All-GPU vs Greedy comparison. The batch chosen for regime B
#   (per model) is one where a direct all-at-once All-GPU run genuinely
#   OOMs on the vision models (resnet50/resnet152); the two language models
#   never OOM up to batch 512 on the big-VRAM cluster GPUs this was
#   evidenced against (chuc-4/kinovis-2/paccaA100.unicartagena.edu.co — see
#   fix/01_accuracy_convergence/README.md), so their regime B stays a
#   budget-constrained-but-not-literally-OOMing scenario, same batch as
#   regime A.
#
# Batch size defaults (DEFAULT_BATCH_A/B below) come from a direct
# All-GPU-feasibility survey of those three clusters' own profiling data,
# not a guess — override per model/regime with e.g. BATCH_SIZE_resnet50_B=512
# if a different cluster's thresholds differ.
#
# Usage:
#   bash convergence_validation/run_convergence_study.sh
#   MODELS_CSV=resnet50 EPOCHS=1 SEEDS_CSV=42 bash convergence_validation/run_convergence_study.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_CMD="${PYTHON_CMD:-python3}"

# --- Experiment grid -------------------------------------------------------
MODELS_CSV="${MODELS_CSV:-resnet50,resnet152,distilgpt2,gpt2_small}"
PRECISIONS_CSV="${PRECISIONS_CSV:-fp32,bf16}"
REGIMES_CSV="${REGIMES_CSV:-A,B}"
SEEDS_CSV="${SEEDS_CSV:-42,43,44}"

# Generic fallback batch size, only used for a model with no evidence-based
# default below and no explicit override. Prefer BATCH_SIZE_<model>_<regime>
# (most specific), then BATCH_SIZE_<model> (same batch for both regimes),
# then the evidence-based DEFAULT_BATCH_A/B below, then this generic value.
BATCH_SIZE="${BATCH_SIZE:-64}"
GRADACCUM_MICRO_BATCH="${GRADACCUM_MICRO_BATCH:-8}"

# Evidence-based per-model batch sizes for regime A (All-GPU fits -- direct
# PRISM-vs-All-GPU comparison) and regime B (the harder case each regime is
# actually meant to exercise). Measured by surveying `all_gpu_status` across
# the pareto-sweep CSVs already on disk for chuc-4, kinovis-2 and
# paccaA100.unicartagena.edu.co (same hardware config per model, confirmed
# identical OOM threshold across all three):
#   - resnet50 (SGD):  All-GPU feasible up to batch 256, OOMs at batch 512.
#   - resnet152 (SGD): All-GPU feasible up to batch 128, OOMs at batch 256.
#   - distilgpt2/gpt2_small (AdamW): All-GPU never OOMs up to batch 512 on
#     these GPUs -- regime B for these two stays a budget-constrained (not
#     literally OOMing) scenario at the SAME batch as regime A.
# See fix/01_accuracy_convergence/README.md for the full survey.
declare -A DEFAULT_BATCH_A=( [resnet50]=256 [resnet152]=128 [distilgpt2]=128 [gpt2_small]=128 )
declare -A DEFAULT_BATCH_B=( [resnet50]=512 [resnet152]=256 [distilgpt2]=128 [gpt2_small]=128 )

# Whether to also build and run a greedy-heuristic plan (build_greedy_plan.py)
# for regime B, alongside ilp_plan and all_gpu_gradaccum. Regime A is skipped
# for greedy by design: at a full/unconstrained budget, greedy and the ILP
# both trivially put everything on GPU, so there is nothing informative to
# compare there -- the interesting divergence only shows up once the budget
# is actually constrained.
RUN_GREEDY="${RUN_GREEDY:-true}"

# Optimizer/learning-rate defaults differ by model family, not just one
# global value: smoke-testing this pipeline on distilgpt2 showed its loss
# exploding from 6.2 to 10.9 (near-random) within 2 SGD steps at lr=0.01 —
# fine for the lightly fine-tuned vision CNNs, catastrophic for a pretrained
# transformer. AdamW at a much smaller lr keeps the language models stable.
# See convergence_validation/mixed_precision.py::optimizer_cls_from_name.
LR_VISION="${LR_VISION:-0.01}"
OPTIMIZER_VISION="${OPTIMIZER_VISION:-sgd}"
LR_LM="${LR_LM:-0.00005}"
OPTIMIZER_LM="${OPTIMIZER_LM:-adamw}"

EPOCHS="${EPOCHS:-7}"
# Caps optimizer steps regardless of EPOCHS (passed straight to
# run_convergence_study.py --max_steps). Unset/empty = no cap, i.e. run full
# EPOCHS. Useful on a single workstation GPU where a full epoch over a real
# dataset (hundreds to low-thousands of steps per model) would take far
# longer than needed to see a convergence trend.
MAX_STEPS="${MAX_STEPS:-}"
EVAL_EVERY_STEPS="${EVAL_EVERY_STEPS:-50}"
EVAL_BATCHES="${EVAL_BATCHES:-10}"
SEQ_LENGTH="${SEQ_LENGTH:-128}"
DATASETS_ROOT="${DATASETS_ROOT:-$REPO_ROOT/datasets}"
GPU_ID="${GPU_ID:-0}"

HOST_TAG="${HOST_TAG:-$(hostname)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/reports/$HOST_TAG/convergence_study}"

# Real OS-level cgroup memory ceiling for each training subprocess, mirroring
# scripts/run_thesis_mode.sh's run_sandboxed/detect_available_ram_mb. This
# script had no such backstop: a single misbehaving cell (e.g. resnet50
# regime B at a batch size that pushes CPU memory hard) could run unbounded
# in this shell's own cgroup, build up sustained memory pressure, and get
# the WHOLE desktop session caught by systemd-oomd / the kernel OOM killer
# (observed killing an unrelated terminal and VS Code itself) instead of
# just this one process. MemorySwapMax=0 makes the kernel SIGKILL only the
# offending subprocess on breach, before pressure has a chance to build.
USE_CGROUP_MEM_LIMIT="${USE_CGROUP_MEM_LIMIT:-true}"
CPU_MEM_LIMIT_PCT="${CPU_MEM_LIMIT_PCT:-95}"

detect_available_ram_mb() {
  if [ -r /proc/meminfo ]; then
    awk '/^MemAvailable:/ { printf "%d", $2 / 1024 }' /proc/meminfo
  else
    echo ""
  fi
}

run_sandboxed() {
  local cmd=("$@")
  local wrapper=()
  if [ "$USE_CGROUP_MEM_LIMIT" = "true" ] && command -v systemd-run >/dev/null 2>&1; then
    local mem_avail_mb mem_limit_mb
    mem_avail_mb="$(detect_available_ram_mb)"
    if [ -n "$mem_avail_mb" ] && [ "$mem_avail_mb" -gt 0 ] 2>/dev/null; then
      mem_limit_mb=$(( mem_avail_mb * CPU_MEM_LIMIT_PCT / 100 ))
      if [ "$mem_limit_mb" -gt 0 ]; then
        wrapper=(systemd-run --user --scope -q -p "MemoryMax=${mem_limit_mb}M" -p "MemorySwapMax=0")
        echo "  CGROUP: MemAvailable=${mem_avail_mb}MB -> limit=${mem_limit_mb}MB (${CPU_MEM_LIMIT_PCT}%) swap=0"
      fi
    else
      echo "  [WARN] CGROUP: no se pudo leer MemAvailable; corriendo sin limite de memoria"
    fi
  fi
  "${wrapper[@]}" "${cmd[@]}"
}

# BF16 only exists on the KINOVIS host in this thesis's campaign (AMX/AVX-512
# BF16 support); skip bf16 configs elsewhere instead of failing the run.
BF16_HOST_HINT="${BF16_HOST_HINT:-kinovis}"

# data/ accumulates ILP-plan results from every host that ever profiled a
# model (this repo's data/ already mixes zephyr, chuc-4, kinovis-2 and
# paccaA100.unicartagena.edu.co). find_config_dir() filters its search by
# this value so a run on one cluster always picks up ITS OWN profiled plan,
# never another cluster's by accident. Defaults to HOST_TAG (this run's own
# host); override explicitly to deliberately read another host's plan
# (e.g. for local post-processing of a cluster's results).
CONFIG_HOST_FILTER="${CONFIG_HOST_FILTER:-}"

IFS=',' read -ra MODELS <<< "$MODELS_CSV"
IFS=',' read -ra PRECISIONS <<< "$PRECISIONS_CSV"
IFS=',' read -ra REGIMES <<< "$REGIMES_CSV"
IFS=',' read -ra SEEDS <<< "$SEEDS_CSV"

echo "=== convergence_validation: run_convergence_study.sh ==="
echo "Models:     ${MODELS[*]}"
echo "Precisions: ${PRECISIONS[*]}"
echo "Regimes:    ${REGIMES[*]}"
echo "Seeds:      ${SEEDS[*]}"
echo "Epochs:     $EPOCHS"
echo "Host tag:   $HOST_TAG (config lookup filter: ${CONFIG_HOST_FILTER:-$HOST_TAG})"
echo "Output:     $OUTPUT_ROOT"
echo

# One-time tiny_shakespeare split (no-op if already prepared).
if [ ! -f "$REPO_ROOT/convergence_validation/data/tiny_shakespeare/val.txt" ]; then
  echo "Preparing tiny_shakespeare train/val split (one-time)..."
  "$PYTHON_CMD" "$REPO_ROOT/convergence_validation/prepare_tiny_shakespeare_split.py"
fi

# Locate the data/<host>/.../<model>/<optimizer_class_name>/<precision>/batch_<N>/
# directory that already contains a solved ILP plan for this model/batch/
# precision/optimizer. Two filters matter here, both to avoid silently
# picking up a plan that does not match what this script is about to run:
#   - host_filter (always set by the caller below -- see CONFIG_HOST_FILTER)
#     so a run on one cluster never grabs an ILP plan profiled on a
#     DIFFERENT cluster's hardware just because data/ happens to hold
#     results from several hosts at once (this repo's data/ already mixes
#     zephyr, chuc-4, kinovis-2 and paccaA100.unicartagena.edu.co).
#   - optimizer_class_name (the profiling run's optimizer, e.g. "SGD" or
#     "AdamW" -- matches torch.optim.<Name>.__class__.__name__) so the plan
#     we read matches the optimizer this script will ACTUALLY train with
#     (see optimizer_for_model below). The ILP's memory estimate is
#     optimizer-dependent (AdamW keeps 2x extra per-parameter state vs
#     SGD); mixing them wouldn't crash (a plan solved for AdamW's heavier
#     footprint is still safe to use with plain SGD) but would compare
#     PRISM against a budget that was never the one actually exercised.
# Returns the first match; echoes nothing if none exists.
find_config_dir() {
  local model="$1" precision="$2" batch_size="$3" host_filter="$4" optimizer_class_name="$5"
  # host_filter is matched ONLY against the immediate data/<host>/ child
  # directory name, via a shell glob restricting the search roots -- NOT
  # via `find -path "*<filter>*"` against the full absolute path. This
  # repo can live under a path that itself contains the hostname as a
  # substring (e.g. this machine's own home directory is /home/zephyr/...,
  # same as HOST_TAG=zephyr), which would make a naive `-path "*zephyr*"`
  # match EVERY path in the repo and silently defeat the whole point of
  # this filter. Confirmed this the hard way: an earlier version of this
  # function returned a chuc-4-profiled plan while HOST_TAG=zephyr,
  # because /home/zephyr/... satisfied "*zephyr*" regardless of which
  # data/<host>/ subtree the match actually came from.
  local search_roots=()
  if [ -n "$host_filter" ]; then
    for d in "$REPO_ROOT"/data/*"$host_filter"*/; do
      [ -d "$d" ] && search_roots+=("$d")
    done
    if [ "${#search_roots[@]}" -eq 0 ]; then
      return 0
    fi
  else
    search_roots=("$REPO_ROOT/data")
  fi
  # Excludes any path with a legacy/ segment. All prior runs were archived
  # into a single data/legacy/<host>/... tree (not nested inside each
  # data/<host>/) specifically so new campaign output written straight to
  # data/<host>/... never collides with them. Since host_filter's glob
  # above (data/*<host>*/) already never matches "data/legacy/" (it does
  # not contain the host name as a substring), this exclusion is mostly
  # defense in depth for the host_filter="" case, which searches all of
  # data/ directly and WOULD otherwise descend into data/legacy/ too.
  find "${search_roots[@]}" -type d -regextype posix-extended \
    -regex ".*/${model}/${optimizer_class_name}/${precision}/batch_${batch_size}" \
    -not -path "*/legacy/*" \
    2>/dev/null | sort | head -n 1
}

# Vision (resnet50, resnet152) vs. language (distilgpt2, gpt2_small) pick
# different (optimizer, lr) defaults — see the LR_VISION/LR_LM comment above.
# Classifies a model as vision/non-vision by querying the project's single
# source of truth (src/data/dataset_registry.py::VISION_MODELS), not a
# hardcoded case list duplicated here -- a hardcoded list here would silently
# misclassify any model added to VISION_MODELS later (confirmed: vit_b16 was
# added to VISION_MODELS but this script's old resnet50|resnet152 case list
# never got updated, so it fell through to the LM defaults unnoticed).
# Loaded by explicit file path, not `from data.dataset_registry import ...`,
# to avoid a known import collision with the top-level data/ directory (see
# convergence_validation/datasets.py's own docstring for the same issue).
is_vision_model() {
  "$PYTHON_CMD" -c "
import importlib.util, sys
from pathlib import Path
registry_path = Path('$REPO_ROOT') / 'src' / 'data' / 'dataset_registry.py'
spec = importlib.util.spec_from_file_location('_prism_dataset_registry', registry_path)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)
sys.exit(0 if sys.argv[1] in mod.VISION_MODELS else 1)
" "$1"
}

lr_for_model() {
  if is_vision_model "$1"; then echo "$LR_VISION"; else echo "$LR_LM"; fi
}

optimizer_for_model() {
  if is_vision_model "$1"; then echo "$OPTIMIZER_VISION"; else echo "$OPTIMIZER_LM"; fi
}

# Maps the --optimizer value used by this script (sgd/adamw) to the
# torch.optim.<Name> class name baked into the profiling directory layout
# (.../<model>/<OptimizerClassName>/<precision>/batch_<N>/), so
# find_config_dir reads the plan profiled with the SAME optimizer this
# script is about to train with.
optimizer_class_name() {
  case "$1" in
    sgd) echo "SGD" ;;
    adamw) echo "AdamW" ;;
    *) echo "$1" ;;
  esac
}

# Resolves the batch size for (model, regime) with this precedence:
#   1. BATCH_SIZE_<model>_<regime>  (most specific, e.g. BATCH_SIZE_resnet50_B=512)
#   2. BATCH_SIZE_<model>           (same batch for both regimes -- legacy/simple override)
#   3. DEFAULT_BATCH_A/B[<model>]   (evidence-based default, see above)
#   4. BATCH_SIZE                   (generic fallback)
model_batch_size_for_regime() {
  local model="$1" regime="$2"
  local specific_var="BATCH_SIZE_${model}_${regime}"
  local legacy_var="BATCH_SIZE_${model}"
  local evidence_default=""
  if [ "$regime" = "A" ]; then
    evidence_default="${DEFAULT_BATCH_A[$model]:-}"
  else
    evidence_default="${DEFAULT_BATCH_B[$model]:-}"
  fi
  local fallback="${!legacy_var:-${evidence_default:-$BATCH_SIZE}}"
  echo "${!specific_var:-$fallback}"
}

run_one() {
  local model="$1" precision="$2" regime="$3" seed="$4" mode="$5"
  local batch_size="$6" accumulation_steps="$7" config_dir="$8"
  local lr optimizer
  lr="$(lr_for_model "$model")"
  optimizer="$(optimizer_for_model "$model")"

  local out_dir="$OUTPUT_ROOT/$model/$precision/regime_${regime}/seed_${seed}/${mode}"
  echo "--- $model | $precision | regime $regime | seed $seed | $mode (batch=$batch_size, accum=$accumulation_steps, optimizer=$optimizer, lr=$lr) ---"

  local extra_args=()
  if [ "$mode" = "ilp_plan" ] || [ "$mode" = "greedy_plan" ]; then
    extra_args+=(--config_dir "$config_dir")
  fi
  if [ -n "$MAX_STEPS" ]; then
    extra_args+=(--max_steps "$MAX_STEPS")
  fi

  # Do not let `set -e` abort the whole campaign on one cell's non-zero exit:
  # run_convergence_study.py returns 2 for a gracefully-handled OOM/mismatch
  # result (see its main()), which is expected, loggable data for this
  # campaign -- not a reason to skip every remaining model/seed/regime.
  run_sandboxed "$PYTHON_CMD" "$REPO_ROOT/convergence_validation/run_convergence_study.py" \
    --model "$model" \
    --precision "$precision" \
    --mode "$mode" \
    --regime "$regime" \
    --batch_size "$batch_size" \
    --accumulation_steps "$accumulation_steps" \
    --datasets_root "$DATASETS_ROOT" \
    --epochs "$EPOCHS" \
    --lr "$lr" \
    --optimizer "$optimizer" \
    --seed "$seed" \
    --eval_every_steps "$EVAL_EVERY_STEPS" \
    --eval_batches "$EVAL_BATCHES" \
    --seq_length "$SEQ_LENGTH" \
    --gpu_id "$GPU_ID" \
    --output_dir "$out_dir" \
    "${extra_args[@]}" || echo "  (non-zero exit, logged above -- continuing campaign)"
}

for model in "${MODELS[@]}"; do
  for precision in "${PRECISIONS[@]}"; do
    # CONFIG_HOST_FILTER defaults to this run's own HOST_TAG, so
    # find_config_dir only ever picks up an ILP plan profiled on THIS
    # host/cluster -- data/ already mixes results from zephyr, chuc-4,
    # kinovis-2 and paccaA100.unicartagena.edu.co, and without this filter
    # a plain alphabetical `sort | head -n 1` could silently return a plan
    # profiled on a different cluster's hardware. bf16 additionally falls
    # back to BF16_HOST_HINT when CONFIG_HOST_FILTER was left at its
    # HOST_TAG default and that default does not already look like the
    # bf16-capable host (covers the common case of post-processing bf16
    # results from a machine other than the one that profiled them).
    host_filter="${CONFIG_HOST_FILTER:-$HOST_TAG}"
    if [ "$precision" = "bf16" ] && [ "$host_filter" = "$HOST_TAG" ] && [[ "$HOST_TAG" != *"$BF16_HOST_HINT"* ]]; then
      host_filter="$BF16_HOST_HINT"
    fi

    for regime in "${REGIMES[@]}"; do
      model_batch_size="$(model_batch_size_for_regime "$model" "$regime")"

      # find_config_dir already resolves straight to the batch_N directory
      # that contains ilp_solution/ (and, for regime B, a sibling
      # ilp_solution_pareto_best_budget_*/) on the host selected by
      # host_filter above, profiled with the same optimizer this script
      # will actually train with -- same lookup for both regimes, only the
      # follow-up check differs.
      config_dir="$(find_config_dir "$model" "$precision" "$model_batch_size" "$host_filter" "$(optimizer_class_name "$(optimizer_for_model "$model")")")"
      if [ -z "$config_dir" ]; then
        echo "SKIP: no config dir found for $model/$precision/batch_$model_batch_size (regime $regime)"
        continue
      fi

      if [ "$regime" = "A" ] && [ ! -d "$config_dir/ilp_solution" ]; then
        echo "SKIP: $config_dir has no ilp_solution/ (regime A, full budget)"
        continue
      fi
      if [ "$regime" = "B" ] && ! compgen -G "$config_dir/ilp_solution_pareto_best_budget_*" > /dev/null; then
        echo "SKIP: no regime-B (reduced budget) ILP solution found under $config_dir"
        continue
      fi

      # Regime B also builds (once per config_dir, not once per seed -- the
      # plan does not depend on the training seed) a greedy-heuristic plan
      # at the SAME budget as the ILP's, for a three-way PRISM vs All-GPU
      # vs Greedy comparison. Built on demand rather than shipped as static
      # data because it is cheap (seconds, pure CSV read + heuristic, no
      # training) and keeps this script self-sufficient against any
      # config_dir that already has an ilp_solution_pareto_best_budget_*/.
      run_greedy=false
      if [ "$regime" = "B" ] && [ "$RUN_GREEDY" = "true" ]; then
        if compgen -G "$config_dir/greedy_solution_budget_*" > /dev/null; then
          run_greedy=true
        elif "$PYTHON_CMD" "$REPO_ROOT/convergence_validation/build_greedy_plan.py" \
              --config_dir "$config_dir" --model "$model"; then
          run_greedy=true
        else
          echo "  (greedy plan build failed for $config_dir -- skipping greedy_plan for this cell, continuing campaign)"
        fi
      fi

      for seed in "${SEEDS[@]}"; do
        run_one "$model" "$precision" "$regime" "$seed" "ilp_plan" "$model_batch_size" 1 "$config_dir"

        if [ "$regime" = "A" ]; then
          run_one "$model" "$precision" "$regime" "$seed" "all_gpu" "$model_batch_size" 1 "$config_dir"
        else
          accumulation_steps=$(( model_batch_size / GRADACCUM_MICRO_BATCH ))
          if [ "$accumulation_steps" -lt 1 ]; then accumulation_steps=1; fi
          run_one "$model" "$precision" "$regime" "$seed" "all_gpu_gradaccum" "$GRADACCUM_MICRO_BATCH" "$accumulation_steps" "$config_dir"
          if [ "$run_greedy" = "true" ]; then
            run_one "$model" "$precision" "$regime" "$seed" "greedy_plan" "$model_batch_size" 1 "$config_dir"
          fi
        fi
      done
    done
  done
done

echo
echo "Done. Outputs under: $OUTPUT_ROOT"
