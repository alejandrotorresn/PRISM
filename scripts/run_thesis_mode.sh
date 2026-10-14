#!/bin/bash
set -euo pipefail

# ===============================================================================
# THESIS MODE ORCHESTRATOR
# ===============================================================================
# End-to-end orchestration for doctoral-grade experimental campaigns:
#   1) Profiling campaign (run_experiments.sh)
#   2) ILP solve + Pareto sweep per configuration
#   3) Optional hybrid runtime execution (currently simple_mlp-oriented)
#   4) Consolidated report assets + LaTeX export
#
# This script does not replace methodological decisions (statistics protocol,
# acceptance thresholds, or final model-accuracy evaluation criteria).
# It operationalizes the full pipeline with reproducible logs and structure.
# ===============================================================================

# Optional conda activation (kept consistent with project scripts)
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

PYTHON_CMD="${PYTHON_CMD:-python}"
HOST_TAG="${HOST_TAG:-$(hostname)}"
PROFILE="${PROFILE:-custom}"
DATASETS_DIR="${DATASETS_DIR:-datasets}"
DOWNLOAD_DATASETS="${DOWNLOAD_DATASETS:-true}"

# Preset profiles (can still be overridden by explicit env vars)
case "$PROFILE" in
  doctoral_minimal)
    : "${MODELS_CSV:=simple_mlp,resnet50,resnet152,vit_b16}"
    : "${OPTIMIZERS_CSV:=SGD,AdamW}"
    : "${PRECISIONS_CSV:=fp32}"
    : "${BATCH_SIZES_CSV:=8,32,64}"
    : "${REPEATS:=5}"
    : "${WARMUP:=3}"
    : "${MEASURE:=10}"
    : "${K_SIGMA:=1.0}"
    : "${GPU_BUDGETS_MB:=auto}"
    : "${RUN_HYBRID:=true}"
    : "${FAIL_FAST:=false}"
    : "${ALLOW_PARTIAL_PROFILING_FAILURES:=true}"
    : "${STRICT_GRAPH_MAPPING:=true}"
    : "${STRICT_TRANSFER_MAPPING:=true}"
    : "${STRICT_METRIC_VALIDITY:=false}"
    : "${FAIL_FAST_PROFILE_ERRORS:=false}"
    : "${ALLOW_LOW_QUALITY_STATS:=false}"
    : "${ALLOW_TRANSFER_CALIBRATION_FALLBACK:=false}"
    : "${ALLOW_FALLBACK_GRAPH_TRACE:=false}"
    ;;
  doctoral_full)
    : "${MODELS_CSV:=simple_mlp,resnet50,resnet152,vit_b16,bert_base,gpt2_small,distilgpt2}"
    : "${OPTIMIZERS_CSV:=SGD,AdamW,RMSprop}"
    : "${PRECISIONS_CSV:=fp32,bf16}"
    : "${BATCH_SIZES_CSV:=8,16,32,64,128,256,512}"
    : "${REPEATS:=7}"
    : "${WARMUP:=5}"
    : "${MEASURE:=15}"
    : "${K_SIGMA:=1.0}"
    : "${GPU_BUDGETS_MB:=auto}"
    : "${RUN_HYBRID:=true}"
    : "${FAIL_FAST:=false}"
    : "${ALLOW_PARTIAL_PROFILING_FAILURES:=true}"
    : "${STRICT_GRAPH_MAPPING:=true}"
    : "${STRICT_TRANSFER_MAPPING:=true}"
    : "${STRICT_METRIC_VALIDITY:=false}"
    : "${FAIL_FAST_PROFILE_ERRORS:=false}"
    : "${ALLOW_LOW_QUALITY_STATS:=false}"
    : "${ALLOW_TRANSFER_CALIBRATION_FALLBACK:=false}"
    : "${ALLOW_FALLBACK_GRAPH_TRACE:=false}"
    ;;
  doctoral_diagnostic)
    : "${MODELS_CSV:=simple_mlp,resnet50,resnet152,vit_b16}"
    : "${OPTIMIZERS_CSV:=SGD,AdamW}"
    : "${PRECISIONS_CSV:=fp32}"
    : "${BATCH_SIZES_CSV:=8,32,64}"
    : "${REPEATS:=5}"
    : "${WARMUP:=3}"
    : "${MEASURE:=10}"
    : "${K_SIGMA:=1.0}"
    : "${GPU_BUDGETS_MB:=auto}"
    : "${RUN_HYBRID:=true}"
    : "${FAIL_FAST:=false}"
    : "${ALLOW_PARTIAL_PROFILING_FAILURES:=true}"
    : "${STRICT_GRAPH_MAPPING:=false}"
    : "${STRICT_TRANSFER_MAPPING:=false}"
    : "${STRICT_METRIC_VALIDITY:=false}"
    : "${FAIL_FAST_PROFILE_ERRORS:=false}"
    : "${ALLOW_LOW_QUALITY_STATS:=true}"
    : "${ALLOW_TRANSFER_CALIBRATION_FALLBACK:=true}"
    : "${ALLOW_FALLBACK_GRAPH_TRACE:=true}"
    ;;
  quick_smoke)
    : "${MODELS_CSV:=simple_mlp}"
    : "${OPTIMIZERS_CSV:=SGD}"
    : "${PRECISIONS_CSV:=fp32}"
    : "${BATCH_SIZES_CSV:=8}"
    : "${REPEATS:=1}"
    : "${WARMUP:=1}"
    : "${MEASURE:=1}"
    : "${GPU_BUDGETS_MB:=auto}"
    : "${RUN_HYBRID:=false}"
    : "${STRICT_GRAPH_MAPPING:=false}"
    : "${STRICT_TRANSFER_MAPPING:=false}"
    : "${STRICT_METRIC_VALIDITY:=false}"
    : "${FAIL_FAST_PROFILE_ERRORS:=false}"
    : "${ALLOW_LOW_QUALITY_STATS:=true}"
    : "${ALLOW_TRANSFER_CALIBRATION_FALLBACK:=true}"
    : "${ALLOW_FALLBACK_GRAPH_TRACE:=true}"
    ;;
  custom)
    ;;
  *)
    echo "[ERROR] Unsupported PROFILE: $PROFILE (use custom|quick_smoke|doctoral_minimal|doctoral_full|doctoral_diagnostic)" >&2
    exit 2
    ;;
esac

# Campaign axes (overridable from environment)
MODELS_CSV="${MODELS_CSV:-simple_mlp,resnet50,resnet152,vit_b16,bert_base,gpt2_small,distilgpt2}"
OPTIMIZERS_CSV="${OPTIMIZERS_CSV:-SGD,AdamW}"
PRECISIONS_CSV="${PRECISIONS_CSV:-fp32}"
BATCH_SIZES_CSV="${BATCH_SIZES_CSV:-8,32}"

# validation/run_hybrid_execution.py's own --lr default (0.01) is a single
# global value, fine for the lightly fine-tuned vision CNNs (SGD) but
# catastrophic for a pretrained transformer under AdamW. Confirmed
# directly: distilgpt2's hybrid-execution smoke test (STEP 4/5 below) had
# its loss explode from 6.14 to 28.67 within 10 steps at lr=0.01 --
# reproduced identically with ZERO activation strategies selected (not a
# Phase 4/checkpoint bug), and settled into a sane, improving 6.04-6.19
# range at lr=5e-5. This does NOT affect profiling or the ILP's own
# decisions (lr changes neither a layer's measured memory nor its
# optimizer.step() time -- see src/runner/training_profiler.py's
# _build_optimizer, which only closes over lr's VALUE, never its
# magnitude affecting cost); it only matters for whether this smoke
# test's reported loss trajectory means anything. Same split already
# applied in convergence_validation/run_convergence_study.sh
# (LR_VISION/LR_LM) -- mirrored here with the same values rather than
# duplicated with different ones.
LR_VISION="${LR_VISION:-0.01}"
LR_LM="${LR_LM:-0.00005}"

# Profiling controls
REPEATS="${REPEATS:-5}"
SEED_BASE="${SEED_BASE:-42}"
WARMUP="${WARMUP:-3}"
MEASURE="${MEASURE:-10}"
FORCE_THREADS="${FORCE_THREADS:-0}"
USE_SKIP_CPU="${USE_SKIP_CPU:-false}"
ENABLE_RAPL="${ENABLE_RAPL:-true}"
AUTO_AGGREGATE_STATS="${AUTO_AGGREGATE_STATS:-true}"
FAIL_FAST="${FAIL_FAST:-false}"
ALLOW_PARTIAL_PROFILING_FAILURES="${ALLOW_PARTIAL_PROFILING_FAILURES:-true}"
DRY_RUN="${DRY_RUN:-false}"

# ILP controls
K_SIGMA="${K_SIGMA:-1.0}"
W_ENERGY="${W_ENERGY:-0.5}"
W_TIME="${W_TIME:-0.5}"
W_TRANSFER="${W_TRANSFER:-1.0}"
BACKEND="${BACKEND:-auto}"
HW_AGGREGATE="${HW_AGGREGATE:-max}"
HW_DISPERSION_K="${HW_DISPERSION_K:-0.0}"
GPU_BUDGETS_MB="${GPU_BUDGETS_MB:-auto}"
GPU_MEM_BUDGET_MB="${GPU_MEM_BUDGET_MB:-1e18}"
# Default changed from an unconstrained 1e18 to "auto" (resolved below from
# real detected system RAM) after a real local-machine freeze: once the ILP's
# GPU memory estimate was corrected (see training_profiler.py's
# _current_batch_scale fix) to no longer underestimate GPU usage, the ILP
# started correctly offloading much more to CPU than before -- but with no
# real CPU budget ceiling, nothing stopped it from planning more CPU memory
# than the machine actually has, which froze the whole desktop (swap
# thrashing), not a catchable CUDA OOM. GPU has always had this kind of real
# ceiling (detect_gpu_vram_mb + GPU_VRAM_SAFETY_RATIO); CPU never did. An
# explicit numeric override (or 1e18 to deliberately restore the old
# unconstrained behavior) still works -- "auto" is just the new default.
CPU_MEM_BUDGET_MB="${CPU_MEM_BUDGET_MB:-auto}"
# 0.90, matching GPU_VRAM_SAFETY_RATIO's convention. An earlier version of
# this fix used 0.6, reasoning that a wrong CPU budget is more dangerous
# than a wrong GPU one (swap-thrashing the whole desktop vs a catchable
# torch.OutOfMemoryError). That reasoning was wrong: cpu_mem[n] already gets
# the same nodal_sum/peak_approx/topological treatment as gpu_mem[n] in
# src/ilp/model_builder.py::_build_effective_memory -- it is not a cruder
# model that needs extra compensation. The real backstop against a wrong
# estimate is the cgroup kill-switch below (run_sandboxed), which costs no
# usable capacity and only triggers on a genuine runtime breach. Stacking a
# punitive 0.6 planning-time ratio on top of that was starving PRISM of the
# CPU headroom that is its whole reason to exist as a complementary
# resource to a small GPU, and was pushing Phase 4 into far more
# recompute/checkpoint than the real budget ever required.
CPU_MEM_SAFETY_RATIO="${CPU_MEM_SAFETY_RATIO:-0.9}"
# Real OS-level backstop for the hybrid training subprocess itself (STEP
# 4/5), mirroring the cgroup protection scripts/run_experiments.sh already
# applies to profiling (USE_CGROUP_MEM_LIMIT/CPU_MEM_LIMIT_PCT there). Without
# this, CPU_MEM_SAFETY_RATIO above is only a planning-time budget fed to the
# ILP -- nothing stops the real training process from exceeding it at
# runtime (framework overhead, async-transfer buffers, a misestimate) and
# swap-thrashing the whole desktop. With MemorySwapMax=0, the kernel kills
# just that one process cleanly once it hits the ceiling, instead.
USE_CGROUP_MEM_LIMIT="${USE_CGROUP_MEM_LIMIT:-true}"
CPU_MEM_LIMIT_PCT="${CPU_MEM_LIMIT_PCT:-95}"
MEMORY_MODEL="${MEMORY_MODEL:-topological}"
PEAK_ACTIVATION_OVERLAP="${PEAK_ACTIVATION_OVERLAP:-0.35}"
STRICT_GRAPH_MAPPING="${STRICT_GRAPH_MAPPING:-true}"
STRICT_TRANSFER_MAPPING="${STRICT_TRANSFER_MAPPING:-true}"
STRICT_METRIC_VALIDITY="${STRICT_METRIC_VALIDITY:-false}"
FAIL_FAST_PROFILE_ERRORS="${FAIL_FAST_PROFILE_ERRORS:-false}"
ALLOW_LOW_QUALITY_STATS="${ALLOW_LOW_QUALITY_STATS:-false}"
ALLOW_TRANSFER_CALIBRATION_FALLBACK="${ALLOW_TRANSFER_CALIBRATION_FALLBACK:-false}"
ALLOW_FALLBACK_GRAPH_TRACE="${ALLOW_FALLBACK_GRAPH_TRACE:-false}"
REGIME="${REGIME:-diagnostic}"
ENFORCE_CONVEX_WEIGHTS="${ENFORCE_CONVEX_WEIGHTS:-false}"

# Single-replicate profiling naturally produces low_sample quality flags.
# Keep ILP stages operational only for non-official diagnostic/custom paths.
if [[ "$REPEATS" =~ ^[0-9]+$ ]] && [ "$REPEATS" -lt 2 ]; then
  if [ "$PROFILE" = "doctoral_minimal" ] || [ "$PROFILE" = "doctoral_full" ]; then
    echo "[ERROR] PROFILE=$PROFILE requires REPEATS>=2 to preserve strict statistical quality." >&2
    echo "[ERROR] Use PROFILE=doctoral_diagnostic (or custom) for exploratory low-replicate runs." >&2
    exit 2
  fi
  if [ "${ALLOW_LOW_QUALITY_STATS:-false}" != "true" ]; then
    ALLOW_LOW_QUALITY_STATS=true
  fi
fi

if [ "$PROFILE" = "doctoral_minimal" ] || [ "$PROFILE" = "doctoral_full" ]; then
  REGIME="deterministic"
  ENFORCE_CONVEX_WEIGHTS=true
  if [ "${ALLOW_LOW_QUALITY_STATS:-false}" = "true" ]; then
    echo "[ERROR] PROFILE=$PROFILE does not allow ALLOW_LOW_QUALITY_STATS=true under strict official mode." >&2
    exit 2
  fi
fi

if [ "$PROFILE" = "doctoral_diagnostic" ]; then
  : "${REGIME:=diagnostic}"
fi

# Orchestration toggles
RUN_PROFILING="${RUN_PROFILING:-true}"
RUN_ILP="${RUN_ILP:-true}"
RUN_HYBRID="${RUN_HYBRID:-false}"
RUN_REPORTS="${RUN_REPORTS:-true}"

# Phase 4 (activation-strategy optimization: recompute/checkpoint). Off by
# default -- no change to any existing campaign's behavior unless opted in.
# Forwarded to every scripts/run_ilp_partition.sh call below (the main ILP
# solve AND the Pareto-best materialization under STEP 4). See
# scripts/run_ilp_partition.sh for what each flag does.
PHASE4_ACTIVATION="${PHASE4_ACTIVATION:-false}"
PHASE4_BACKEND="${PHASE4_BACKEND:-auto}"
PHASE4_ENABLE_RECOMPUTE="${PHASE4_ENABLE_RECOMPUTE:-true}"
PHASE4_ENABLE_CHECKPOINT="${PHASE4_ENABLE_CHECKPOINT:-true}"
PHASE4_W_IO="${PHASE4_W_IO:-0.0}"
PHASE4_RECOMPUTE_PENALTY="${PHASE4_RECOMPUTE_PENALTY:-0.5}"
PHASE4_CHECKPOINT_PENALTY="${PHASE4_CHECKPOINT_PENALTY:-0.3}"

# Calibrated recompute curve (fix/07_gpu_recompute_joint_decision/README.md).
# Off by default, same posture as PHASE4_ACTIVATION above -- no change to
# any existing campaign's behavior unless opted in. Requires
# PHASE4_BACKEND=joint when enabled (run_ilp_partition.py enforces this and
# fails loudly, logged as a [WARN] by the safety-ratio search loop below,
# rather than silently ignoring it). Measures the REAL recompute memory
# curve by loading the real model and running real forward+backward passes
# on GPU, at EACH attempt of the safety-ratio/Phase4-target search loop
# below -- meaningfully slower per attempt than the non-calibrated path.
# Validated end to end (real training, not just predicted memory) for
# resnet50, distilgpt2, and gpt2_small: in every tested case the calibrated
# plan matched or beat the non-calibrated plan on BOTH real memory and real
# wall-clock step time for the same budget.
PHASE4_CALIBRATE_RECOMPUTE_CURVE="${PHASE4_CALIBRATE_RECOMPUTE_CURVE:-false}"
CALIBRATION_PRECISION="${CALIBRATION_PRECISION:-fp32}"
CALIBRATION_SEQ_LENGTH="${CALIBRATION_SEQ_LENGTH:-128}"
CALIBRATION_DATASETS_ROOT="${CALIBRATION_DATASETS_ROOT:-datasets}"
CALIBRATION_FRACTIONS="${CALIBRATION_FRACTIONS:-0.0,0.25,0.5,0.75,1.0}"

# Post-solve GPU expansion (fix/08_gpu_memory_expansion_and_underestimation/README.md).
# Off by default, same posture as PHASE4_CALIBRATE_RECOMPUTE_CURVE above --
# see scripts/run_ilp_partition.sh for what it does and why it also needs
# CALIBRATION_BATCH_SIZE (shares that requirement/those flags with the
# recompute curve). Forwarded only at the Pareto-best materialization call
# site below (forced false at the earlier Pareto-sweep-feeding call site,
# same reasoning as PHASE4_CALIBRATE_RECOMPUTE_CURVE there).
GPU_EXPANSION="${GPU_EXPANSION:-false}"
GPU_EXPANSION_BUDGET="${GPU_EXPANSION_BUDGET:-200}"

# Upfront safety margin applied when GPU_BUDGETS_MB=auto builds its
# candidate ladder (see scripts/run_ilp_pareto_sweep.sh). 1.0 = no change
# (prior behavior). Superseded as the primary safety mechanism by
# GPU_VRAM_SAFETY_RATIO_CSV below -- kept available as a manual override
# for the Pareto sweep's own candidate ladder, but left at 1.0 by default
# since the adaptive search already finds the real working ratio per
# (model, batch, hardware) instead of guessing one value upfront for
# every config.
GPU_VRAM_SAFETY_RATIO="${GPU_VRAM_SAFETY_RATIO:-1.0}"

# Adaptive safety-ratio search for the Pareto-best plan materialized under
# STEP 4 (see the hybrid-runtime block below). Tries each ratio in order
# (loosest/fastest first) against the budget the Pareto sweep already
# picked as "best feasible", re-materializing the plan (cheap: one ILP
# re-solve, no re-profiling, no re-sweep) and probing it with the existing
# HYBRID_STEPS-step validation run at the CONFIG'S OWN batch size (not
# HYBRID_BATCH_SIZE's default of 8). Stops at the first ratio that survives
# the probe without OOM.
#
# Shortened from a 6-value ladder (1.0 down to 0.85) to just 0.90 plus one
# fallback step: that longer ladder, combined with PHASE4_TARGET_GPU_UTILIZATION_CSV's
# old 8-value sweep down to 0.60, existed to brute-force compensate for the
# ILP's per-layer memory estimate silently underestimating real usage for
# layers past an OOM-retry point during profiling (training_profiler.py's
# output_bytes/mem_mb). That measurement is now rescaled to the real batch
# size at the source, so the baseline ILP's own gpu_mem_budget_mb constraint
# is trustworthy on its own -- 0.90 real margin, with one conservative
# fallback, should cover residual runtime overhead (cuDNN workspace,
# allocator fragmentation, async-transfer buffers) without needing to
# re-discover a working ratio from scratch via a 48-combination search.
GPU_VRAM_SAFETY_RATIO_CSV="${GPU_VRAM_SAFETY_RATIO_CSV:-0.90,0.85}"

# Inner search level, tried BEFORE falling back to a smaller safety_ratio
# above: at each trial budget, sweep src/ilp/solve.py::_solve_phase4_greedy's
# own target_gpu_utilization (loosest/fastest first) so the Phase4 heuristic
# only applies extra recompute/checkpoint near the real edge of the
# (already-correct, already-budget-respecting) baseline solution, instead
# of as an independent ~40% haircut on top of it. Only used when
# PHASE4_ACTIVATION=true. Shortened from 8 values (0.95 down to 0.60) to 3
# for the same reason as GPU_VRAM_SAFETY_RATIO_CSV above -- see
# src/ilp/model_builder.py::ILPConfig4.target_gpu_utilization.
PHASE4_TARGET_GPU_UTILIZATION_CSV="${PHASE4_TARGET_GPU_UTILIZATION_CSV:-0.95,0.90,0.85}"

# Hybrid runtime controls (optional)
HYBRID_STEPS="${HYBRID_STEPS:-10}"
HYBRID_BATCH_SIZE="${HYBRID_BATCH_SIZE:-8}"
HYBRID_ENABLE_ASYNC_TRANSFER="${HYBRID_ENABLE_ASYNC_TRANSFER:-true}"
HYBRID_ENABLE_PREFETCH="${HYBRID_ENABLE_PREFETCH:-true}"
HYBRID_ALLOW_CPU_FALLBACK="${HYBRID_ALLOW_CPU_FALLBACK:-false}"
HYBRID_COMPARE_BASELINES="${HYBRID_COMPARE_BASELINES:-true}"
HYBRID_EXECUTION_MODE="${HYBRID_EXECUTION_MODE:-auto}"
HYBRID_PLAN_SELECTION="${HYBRID_PLAN_SELECTION:-pareto_best}"

BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-data/${HOST_TAG}/results_thesis_mode/${PROFILE}}"
REPORTS_DIR="${REPORTS_DIR:-reports/${HOST_TAG}/${PROFILE}}"
LATEX_DIR="${LATEX_DIR:-${REPORTS_DIR}/latex}"
LOG_DIR="${LOG_DIR:-logs/${HOST_TAG}/thesis_mode}"
LOG_FILE="${LOG_DIR}/thesis_mode_$(date +%Y%m%d_%H%M%S).txt"

mkdir -p "$BASE_OUTPUT_DIR" "$REPORTS_DIR" "$LATEX_DIR" "$LOG_DIR"

cat > "$BASE_OUTPUT_DIR/README_RUN_CONTENTS.txt" <<EOF
PRISM thesis-mode run artifacts

Host tag: $HOST_TAG
Profile: $PROFILE
Seed base: $SEED_BASE
Repeats: $REPEATS

Expected subtree per configuration:
  <model>/<optimizer>/<precision>/batch_<N>/

Inside each batch_<N> directory, the pipeline may generate:
  - run_*/<model>_metrics.csv
  - run_*/<model>_meta.json
  - run_*/<model>_graph_edges.csv
  - run_*/<model>_transfer_edges.csv
  - <model>_metrics_stats.csv
  - <model>_pareto_sweep.csv
  - <model>_pareto_summary.json
  - ilp_solution*/

Companion outputs for this run are stored in:
  Reports: $REPORTS_DIR
  LaTeX:   $LATEX_DIR
  Log:     $LOG_FILE
EOF

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

is_true() {
  [ "$1" = true ]
}

# Vision vs. language models pick different learning rates for the
# hybrid-execution smoke test -- see the LR_VISION/LR_LM comment above.
# simple_mlp (synthetic weights, no real pretraining) is not sensitive to
# this either way; it falls through to LR_LM along with the other names
# not in VISION_MODELS.
# Classification queried from the project's single source of truth
# (src/data/dataset_registry.py::VISION_MODELS), not a hardcoded case list
# duplicated here -- a duplicated list silently drifts out of sync when a
# model is added there (confirmed: this exact bug existed in
# convergence_validation/run_convergence_study.sh's own copy of this
# function, which still only matched resnet50|resnet152 after vit_b16 was
# added to VISION_MODELS).
is_vision_model() {
  "$PYTHON_CMD" -c "
import importlib.util, sys
spec = importlib.util.spec_from_file_location('_prism_dataset_registry', 'src/data/dataset_registry.py')
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)
sys.exit(0 if sys.argv[1] in mod.VISION_MODELS else 1)
" "$1"
}

lr_for_model() {
  if is_vision_model "$1"; then echo "$LR_VISION"; else echo "$LR_LM"; fi
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

  local before=()
  local after=()
  if [ "$data_idx" -gt 0 ]; then
    before=("${parts[@]:0:$data_idx}")
  fi
  if [ $((data_idx + 1)) -lt ${#parts[@]} ]; then
    after=("${parts[@]:$((data_idx + 1))}")
  fi

  local rebuilt=()
  rebuilt+=("${before[@]}")
  rebuilt+=("data" "$host_name")
  rebuilt+=("${after[@]}")

  local joined=""
  for p in "${rebuilt[@]}"; do
    if [ -z "$joined" ]; then
      joined="$p"
    else
      joined="$joined/$p"
    fi
  done

  printf '%s%s' "$leading_slash" "$joined"
}

discover_config_dirs() {
  local base_dir="$1"
  local host_name="$2"
  local normalized_dir
  normalized_dir="$(normalize_output_dir_for_host_sh "$base_dir" "$host_name")"

  mapfile -t CONFIG_DIRS < <(
    {
      [ -d "$base_dir" ] && find "$base_dir" -type d -name 'batch_*'
      if [ "$normalized_dir" != "$base_dir" ] && [ -d "$normalized_dir" ]; then
        find "$normalized_dir" -type d -name 'batch_*'
      fi
    } | sort -u
  )

  if [ ${#CONFIG_DIRS[@]} -gt 0 ]; then
    local unique_roots
    unique_roots="$(printf '%s\n' "${CONFIG_DIRS[@]}" | sed -E 's|(.*?/batch_[^/]+).*|\1|' | sort -u | head -n 3 | paste -sd ', ' - || true)"
    log_msg "Discovered ${#CONFIG_DIRS[@]} config dirs (sample roots: ${unique_roots})"
  fi
}

has_pareto_files() {
  local root_dir="$1"
  [ -d "$root_dir" ] || return 1
  find "$root_dir" -type f -name '*_pareto_sweep.csv' -print -quit | grep -q .
}

resolve_report_input_root() {
  local base_dir="$1"
  local host_name="$2"
  local normalized_dir
  normalized_dir="$(normalize_output_dir_for_host_sh "$base_dir" "$host_name")"

  if has_pareto_files "$base_dir"; then
    printf '%s' "$base_dir"
    return
  fi

  if [ "$normalized_dir" != "$base_dir" ] && has_pareto_files "$normalized_dir"; then
    printf '%s' "$normalized_dir"
    return
  fi

  # Fallback keeps previous behavior, preserving clear error reporting downstream.
  printf '%s' "$base_dir"
}

# ---- GPU VRAM auto-detection utilities ----
detect_gpu_vram_mb() {
  # Returns total GPU VRAM in MiB from nvidia-smi, or empty string on failure.
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -i 0 --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -d '[:space:]'
  else
    echo ""
  fi
}

# ---- System RAM auto-detection (mirrors detect_gpu_vram_mb above) ----
detect_available_ram_mb() {
  # Returns currently available system RAM in MiB from /proc/meminfo's
  # MemAvailable (already accounts for reclaimable cache/buffers, unlike
  # MemFree), or empty string on failure. Deliberately a live reading, not
  # MemTotal: other processes on a shared desktop machine are already using
  # part of it at the moment this orchestrator runs.
  if [ -r /proc/meminfo ]; then
    awk '/^MemAvailable:/ { printf "%d", $2 / 1024 }' /proc/meminfo
  else
    echo ""
  fi
}

run_sandboxed() {
  # Execute a command inside a cgroup memory ceiling, mirroring
  # scripts/run_experiments.sh's run_profiler_sandboxed. Used to wrap the
  # hybrid training subprocess (STEP 4/5) with the same real OS-level
  # backstop profiling already had -- CPU_MEM_SAFETY_RATIO only bounds the
  # ILP's plan, this bounds what the process can actually do at runtime.
  # MemorySwapMax=0 means the kernel SIGKILLs just this process on breach
  # instead of letting the whole desktop swap-thrash.
  local cmd=("$@")
  local wrapper=()
  if is_true "$USE_CGROUP_MEM_LIMIT" && command -v systemd-run >/dev/null 2>&1; then
    local mem_avail_mb mem_limit_mb
    mem_avail_mb="$(detect_available_ram_mb)"
    if [ -n "$mem_avail_mb" ] && [ "$mem_avail_mb" -gt 0 ] 2>/dev/null; then
      mem_limit_mb=$(( mem_avail_mb * CPU_MEM_LIMIT_PCT / 100 ))
      if [ "$mem_limit_mb" -gt 0 ]; then
        wrapper=(systemd-run --user --scope -q -p "MemoryMax=${mem_limit_mb}M" -p "MemorySwapMax=0")
        log_msg "  🛡️ CGROUP: MemAvailable=${mem_avail_mb}MB -> limit=${mem_limit_mb}MB (${CPU_MEM_LIMIT_PCT}%) swap=0"
      fi
    else
      log_msg "  [WARN] CGROUP: Could not read MemAvailable; running hybrid execution without a memory limit"
    fi
  fi
  "${wrapper[@]}" "${cmd[@]}"
}

compute_dynamic_budgets() {
  # Computes GPU memory budget values as percentages of total VRAM.
  # Usage: compute_dynamic_budgets <vram_mb> [percentages_csv]
  # Example: compute_dynamic_budgets 8188 "90,85,80,75,70,50" -> "7369,6959,6550,6141,5731,4094"
  local vram_mb="$1"
  local percentages_csv="${2:-90,85,80,75,70,50}"
  local budgets=""
  IFS=',' read -r -a percentages <<< "$percentages_csv"
  for pct in "${percentages[@]}"; do
    local val=$(( vram_mb * pct / 100 ))
    if [ -z "$budgets" ]; then
      budgets="$val"
    else
      budgets="$budgets,$val"
    fi
  done
  echo "$budgets"
}

section "THESIS MODE - PREFLIGHT"
log_msg "PROFILE=$PROFILE"
log_msg "PYTHON_CMD=$PYTHON_CMD"
log_msg "HOST_TAG=$HOST_TAG"
log_msg "BASE_OUTPUT_DIR=$BASE_OUTPUT_DIR"
log_msg "REPORTS_DIR=$REPORTS_DIR"
log_msg "LATEX_DIR=$LATEX_DIR"
log_msg "DATASETS_DIR=$DATASETS_DIR"
log_msg "MODELS_CSV=$MODELS_CSV"
log_msg "OPTIMIZERS_CSV=$OPTIMIZERS_CSV"
log_msg "PRECISIONS_CSV=$PRECISIONS_CSV"
log_msg "BATCH_SIZES_CSV=$BATCH_SIZES_CSV"
log_msg "REPEATS=$REPEATS"
log_msg "WARMUP=$WARMUP MEASURE=$MEASURE"
log_msg "FAIL_FAST=$FAIL_FAST"
if [ "$ALLOW_PARTIAL_PROFILING_FAILURES" = true ]; then
  log_msg "ALLOW_PARTIAL_PROFILING_FAILURES=true"
fi
if [ "$ALLOW_LOW_QUALITY_STATS" = true ]; then
  log_msg "ALLOW_LOW_QUALITY_STATS=true"
fi
if [ "$ALLOW_TRANSFER_CALIBRATION_FALLBACK" = true ]; then
  log_msg "ALLOW_TRANSFER_CALIBRATION_FALLBACK=true"
fi
if [ "$ALLOW_FALLBACK_GRAPH_TRACE" = true ]; then
  log_msg "ALLOW_FALLBACK_GRAPH_TRACE=true"
fi
log_msg "DRY_RUN=$DRY_RUN"

# ---- Resolve GPU_BUDGETS_MB=auto to dynamic values ----
if [ "$GPU_BUDGETS_MB" = "auto" ]; then
  GPU_VRAM_TOTAL_MB="$(detect_gpu_vram_mb)"
  if [ -z "$GPU_VRAM_TOTAL_MB" ] || [ "$GPU_VRAM_TOTAL_MB" -le 0 ] 2>/dev/null; then
    log_msg "ERROR: GPU_BUDGETS_MB=auto but could not detect GPU VRAM via nvidia-smi."
    log_msg "       Set GPU_BUDGETS_MB explicitly (e.g. GPU_BUDGETS_MB=5000,6000,7000)."
    exit 1
  fi
  # GPU_VRAM_SAFETY_RATIO scales the detected VRAM down before the 90/85/.../50
  # percentage ladder is computed off it, so every resulting budget (including
  # the most generous, 90%) carries that much extra headroom against the
  # ILP's own memory-model estimate running under real measured usage --
  # see the GPU_VRAM_SAFETY_RATIO comment near the other orchestration
  # toggles above for the measurement that motivated this.
  GPU_VRAM_EFFECTIVE_MB="$(awk -v v="$GPU_VRAM_TOTAL_MB" -v r="$GPU_VRAM_SAFETY_RATIO" 'BEGIN{printf "%d", v*r}')"
  if [ "$GPU_VRAM_EFFECTIVE_MB" != "$GPU_VRAM_TOTAL_MB" ]; then
    log_msg "GPU_VRAM_SAFETY_RATIO=$GPU_VRAM_SAFETY_RATIO -> effective VRAM for budget ladder: ${GPU_VRAM_EFFECTIVE_MB} MiB (of ${GPU_VRAM_TOTAL_MB} MiB detected)"
  fi
  GPU_BUDGETS_MB="$(compute_dynamic_budgets "$GPU_VRAM_EFFECTIVE_MB" "90,85,80,75,70,50")"
  log_msg "GPU VRAM detected: ${GPU_VRAM_TOTAL_MB} MiB"
  log_msg "Dynamic GPU budgets (90%,85%,80%,75%,70%,50%): $GPU_BUDGETS_MB"
else
  log_msg "GPU budgets (manual): $GPU_BUDGETS_MB"
fi

# ---- Resolve CPU_MEM_BUDGET_MB=auto to a dynamic value (mirrors GPU above) ----
if [ "$CPU_MEM_BUDGET_MB" = "auto" ]; then
  CPU_RAM_AVAILABLE_MB="$(detect_available_ram_mb)"
  if [ -z "$CPU_RAM_AVAILABLE_MB" ] || [ "$CPU_RAM_AVAILABLE_MB" -le 0 ] 2>/dev/null; then
    log_msg "ERROR: CPU_MEM_BUDGET_MB=auto but could not read /proc/meminfo."
    log_msg "       Set CPU_MEM_BUDGET_MB explicitly (e.g. CPU_MEM_BUDGET_MB=12000), or 1e18 to disable this safety limit."
    exit 1
  fi
  CPU_MEM_BUDGET_MB="$(awk -v v="$CPU_RAM_AVAILABLE_MB" -v r="$CPU_MEM_SAFETY_RATIO" 'BEGIN{printf "%d", v*r}')"
  log_msg "System RAM available: ${CPU_RAM_AVAILABLE_MB} MiB"
  log_msg "CPU_MEM_SAFETY_RATIO=$CPU_MEM_SAFETY_RATIO -> CPU memory budget: ${CPU_MEM_BUDGET_MB} MiB"
else
  log_msg "CPU memory budget (manual): $CPU_MEM_BUDGET_MB"
fi

if is_true "$DOWNLOAD_DATASETS"; then
  section "STEP 1/5 - DATASET PREPARATION"
  "$PYTHON_CMD" scripts/download_datasets.py \
    --models "$MODELS_CSV" \
    --datasets_root "$DATASETS_DIR" \
    >> "$LOG_FILE" 2>&1
  log_msg "Dataset preparation finished"
fi

if is_true "$RUN_PROFILING"; then
  section "STEP 2/5 - PROFILING CAMPAIGN"
  MODELS_CSV="$MODELS_CSV" \
  OPTIMIZERS_CSV="$OPTIMIZERS_CSV" \
  PRECISIONS_CSV="$PRECISIONS_CSV" \
  BATCH_SIZES_CSV="$BATCH_SIZES_CSV" \
  BASE_OUTPUT_DIR="$BASE_OUTPUT_DIR" \
  DATASETS_DIR="$DATASETS_DIR" \
  DOWNLOAD_DATASETS=false \
  PYTHON_CMD="$PYTHON_CMD" \
  WARMUP="$WARMUP" \
  MEASURE="$MEASURE" \
  REPEATS="$REPEATS" \
  SEED_BASE="$SEED_BASE" \
  FORCE_THREADS="$FORCE_THREADS" \
  USE_SKIP_CPU="$USE_SKIP_CPU" \
  ENABLE_RAPL="$ENABLE_RAPL" \
  AUTO_AGGREGATE_STATS="$AUTO_AGGREGATE_STATS" \
  FAIL_FAST="$FAIL_FAST" \
  ALLOW_PARTIAL_FAILURES="$ALLOW_PARTIAL_PROFILING_FAILURES" \
  USE_CGROUP_MEM_LIMIT="${USE_CGROUP_MEM_LIMIT:-true}" \
  CPU_MEM_LIMIT_PCT="${CPU_MEM_LIMIT_PCT:-95}" \
  PROFILER_TIMEOUT_SECS="${PROFILER_TIMEOUT_SECS:-0}" \
  DRY_RUN="$DRY_RUN" \
  LR_VISION="$LR_VISION" \
  LR_LM="$LR_LM" \
  bash scripts/run_experiments.sh >> "$LOG_FILE" 2>&1
  log_msg "Profiling campaign finished"
fi

if is_true "$RUN_ILP"; then
  section "STEP 3/5 - ILP PARTITION + PARETO PER CONFIG"

  if [[ "$MEMORY_MODEL" != "topological" && "$MEMORY_MODEL" != "topological_tight" ]]; then
    log_msg "ERROR: MEMORY_MODEL='$MEMORY_MODEL' is invalid. Must be 'topological' or 'topological_tight'."
    exit 1
  fi

  if is_true "$DRY_RUN"; then
    log_msg "DRY_RUN=true -> skipping ILP and Pareto execution (no real artifacts expected)"
  else

    discover_config_dirs "$BASE_OUTPUT_DIR" "$HOST_TAG"

    if [ ${#CONFIG_DIRS[@]} -eq 0 ]; then
      log_msg "ERROR: No configuration directories found under $BASE_OUTPUT_DIR"
      exit 1
    fi

    for cfg in "${CONFIG_DIRS[@]}"; do
      model="$(basename "$(dirname "$(dirname "$(dirname "$cfg")")")")"

      log_msg "ILP partition -> model=$model cfg=$cfg"
      if ! MODEL="$model" \
      CONFIG_DIR="$cfg" \
      PYTHON_CMD="$PYTHON_CMD" \
      K_SIGMA="$K_SIGMA" \
      W_TIME="$W_TIME" \
      W_ENERGY="$W_ENERGY" \
      W_TRANSFER="$W_TRANSFER" \
      GPU_MEM_BUDGET_MB="$GPU_MEM_BUDGET_MB" \
      CPU_MEM_BUDGET_MB="$CPU_MEM_BUDGET_MB" \
      MEMORY_MODEL="$MEMORY_MODEL" \
      PEAK_ACTIVATION_OVERLAP="$PEAK_ACTIVATION_OVERLAP" \
      BACKEND="$BACKEND" \
      HW_AGGREGATE="$HW_AGGREGATE" \
      HW_DISPERSION_K="$HW_DISPERSION_K" \
      REGIME="$REGIME" \
      ENFORCE_CONVEX_WEIGHTS="$ENFORCE_CONVEX_WEIGHTS" \
      STRICT_GRAPH_MAPPING="$STRICT_GRAPH_MAPPING" \
      STRICT_TRANSFER_MAPPING="$STRICT_TRANSFER_MAPPING" \
      STRICT_METRIC_VALIDITY="$STRICT_METRIC_VALIDITY" \
      ALLOW_LOW_QUALITY_STATS="$ALLOW_LOW_QUALITY_STATS" \
      ALLOW_TRANSFER_CALIBRATION_FALLBACK="$ALLOW_TRANSFER_CALIBRATION_FALLBACK" \
      ALLOW_FALLBACK_GRAPH_TRACE="$ALLOW_FALLBACK_GRAPH_TRACE" \
      PHASE4_ACTIVATION="$PHASE4_ACTIVATION" \
      PHASE4_BACKEND="$PHASE4_BACKEND" \
      PHASE4_ENABLE_RECOMPUTE="$PHASE4_ENABLE_RECOMPUTE" \
      PHASE4_ENABLE_CHECKPOINT="$PHASE4_ENABLE_CHECKPOINT" \
      PHASE4_W_IO="$PHASE4_W_IO" \
      PHASE4_RECOMPUTE_PENALTY="$PHASE4_RECOMPUTE_PENALTY" \
      PHASE4_CHECKPOINT_PENALTY="$PHASE4_CHECKPOINT_PENALTY" \
      PHASE4_CALIBRATE_RECOMPUTE_CURVE=false \
      GPU_EXPANSION=false \
      bash scripts/run_ilp_partition.sh >> "$LOG_FILE" 2>&1; then
        log_msg "[WARN] ILP partition failed for $cfg. Extracting failure traceback from log:"
        tail -n 15 "$LOG_FILE" | while IFS= read -r line; do
          log_msg "    [LOG] $line"
        done
        log_msg "[WARN] Skipping Pareto sweep for $cfg due to ILP partition failure..."
        continue
      fi

      log_msg "Pareto sweep -> model=$model cfg=$cfg"
      if ! MODEL="$model" \
      CONFIG_DIR="$cfg" \
      PYTHON_CMD="$PYTHON_CMD" \
      GPU_BUDGETS_MB="$GPU_BUDGETS_MB" \
      CPU_MEM_BUDGET_MB="$CPU_MEM_BUDGET_MB" \
      MEMORY_MODEL="$MEMORY_MODEL" \
      PEAK_ACTIVATION_OVERLAP="$PEAK_ACTIVATION_OVERLAP" \
      K_SIGMA="$K_SIGMA" \
      W_TIME="$W_TIME" \
      W_ENERGY="$W_ENERGY" \
      W_TRANSFER="$W_TRANSFER" \
      BACKEND="$BACKEND" \
      HW_AGGREGATE="$HW_AGGREGATE" \
      HW_DISPERSION_K="$HW_DISPERSION_K" \
      REGIME="$REGIME" \
      ENFORCE_CONVEX_WEIGHTS="$ENFORCE_CONVEX_WEIGHTS" \
      STRICT_GRAPH_MAPPING="$STRICT_GRAPH_MAPPING" \
      STRICT_TRANSFER_MAPPING="$STRICT_TRANSFER_MAPPING" \
      STRICT_METRIC_VALIDITY="$STRICT_METRIC_VALIDITY" \
      FAIL_FAST_PROFILE_ERRORS="$FAIL_FAST_PROFILE_ERRORS" \
      ALLOW_LOW_QUALITY_STATS="$ALLOW_LOW_QUALITY_STATS" \
      ALLOW_TRANSFER_CALIBRATION_FALLBACK="$ALLOW_TRANSFER_CALIBRATION_FALLBACK" \
      ALLOW_FALLBACK_GRAPH_TRACE="$ALLOW_FALLBACK_GRAPH_TRACE" \
      bash scripts/run_ilp_pareto_sweep.sh >> "$LOG_FILE" 2>&1; then
        log_msg "[WARN] Pareto sweep failed for $cfg. Extracting failure traceback from log:"
        tail -n 15 "$LOG_FILE" | while IFS= read -r line; do
          log_msg "    [LOG] $line"
        done
        continue
      fi
    done
  fi

  log_msg "ILP and Pareto stage finished"
fi

if is_true "$RUN_HYBRID"; then
  section "STEP 4/5 - OPTIONAL HYBRID RUNTIME EXECUTION"
  if is_true "$DRY_RUN"; then
    log_msg "DRY_RUN=true -> skipping hybrid runtime execution"
    log_msg "Hybrid runtime stage finished"
  else
  discover_config_dirs "$BASE_OUTPUT_DIR" "$HOST_TAG"

  for cfg in "${CONFIG_DIRS[@]}"; do
    model="$(basename "$(dirname "$(dirname "$(dirname "$cfg")")")")"

    hybrid_assignment_csv="$cfg/ilp_solution/ilp_assignment.csv"
    hybrid_cut_edges_csv="$cfg/ilp_solution/ilp_cut_edges.csv"
    hybrid_output_dir="$cfg/ilp_solution/hybrid_execution"
    hybrid_plan_mode="default_ilp"
    hybrid_plan_source="$cfg/ilp_solution"
    hybrid_plan_source_csv=""
    hybrid_plan_gpu_budget_mb=""
    hybrid_plan_objective=""
    hybrid_plan_sim_time_ms=""
    hybrid_plan_sim_energy_j=""
    hybrid_already_executed=false

    # Built once per cfg -- reused by both the adaptive safety-ratio search
    # below (Pareto-best path) and the single run_hybrid_execution.py call
    # at the end of this loop body (default_ilp path, or any attempt the
    # search already ran).
    HYBRID_FLAGS=()
    if is_true "$HYBRID_ENABLE_ASYNC_TRANSFER"; then
      HYBRID_FLAGS+=(--enable_async_transfer)
    fi
    if is_true "$HYBRID_ENABLE_PREFETCH"; then
      HYBRID_FLAGS+=(--enable_prefetch)
    fi
    if is_true "$HYBRID_ALLOW_CPU_FALLBACK"; then
      HYBRID_FLAGS+=(--allow_cpu_fallback)
    fi
    if is_true "$HYBRID_COMPARE_BASELINES"; then
      HYBRID_FLAGS+=(--compare_baselines)
    fi

    if [ "$HYBRID_PLAN_SELECTION" = "pareto_best" ]; then
      pareto_info="$($PYTHON_CMD - "$cfg" "$model" <<'PY'
import json
import sys
from pathlib import Path

cfg = Path(sys.argv[1])
model = sys.argv[2]
summary_path = cfg / f"{model}_pareto_summary.json"
if not summary_path.exists():
    raise SystemExit(2)

payload = json.loads(summary_path.read_text())
row = payload.get("best_feasible_row")
if not row:
    raise SystemExit(3)

source_csv = payload.get("output_csv") or row.get("source_csv") or ""
print(
    "|".join(
        [
            str(row.get("gpu_budget_mb", "")),
            str(row.get("ilp_objective", "")),
            str(row.get("ilp_status", "")),
      str(row.get("sim_time_ms", "")),
      str(row.get("sim_energy_j", "")),
            source_csv,
        ]
    )
)
PY
      )" || true

      if [ -n "$pareto_info" ]; then
        IFS='|' read -r hybrid_plan_gpu_budget_mb hybrid_plan_objective hybrid_plan_status hybrid_plan_sim_time_ms hybrid_plan_sim_energy_j hybrid_plan_source_csv <<< "$pareto_info"
        hybrid_plan_mode="pareto_best"

        # Probe at the config's OWN batch size (parsed from its directory
        # name, e.g. ".../batch_128" -> 128) -- NOT HYBRID_BATCH_SIZE's
        # generic default of 8. A plan solved for batch 128 must be
        # exercised at batch 128, or an OOM that only appears at the real
        # batch size stays invisible to this probe (confirmed: with the
        # default batch-8 probe, this step reported success even on a
        # configuration that OOM'd immediately at its real batch size
        # inside convergence_validation's actual training run).
        probe_batch_size="$(basename "$cfg" | sed -nE 's/^batch_([0-9]+)$/\1/p')"
        if [ -z "$probe_batch_size" ]; then
          probe_batch_size="$HYBRID_BATCH_SIZE"
        fi

        # Adaptive safety-ratio search: try each ratio in
        # GPU_VRAM_SAFETY_RATIO_CSV (loosest/fastest first) against the
        # budget the Pareto sweep already picked as "best feasible",
        # re-materializing (cheap: one ILP re-solve, no re-profiling, no
        # re-sweep) and probing with the real HYBRID_STEPS-step hybrid
        # execution at probe_batch_size. Stops at the first ratio that
        # survives without OOM -- this becomes the plan actually used
        # (hybrid_already_executed=true skips the redundant second run
        # below).
        base_budget_mb="$hybrid_plan_gpu_budget_mb"
        IFS=',' read -r -a safety_ratios <<< "$GPU_VRAM_SAFETY_RATIO_CSV"
        if [ "$PHASE4_ACTIVATION" = true ]; then
          IFS=',' read -r -a phase4_targets <<< "$PHASE4_TARGET_GPU_UTILIZATION_CSV"
        else
          # Phase4 is off, so this value is never read by the solver --
          # one pass keeps the inner loop a no-op instead of needlessly
          # re-materializing the same non-Phase4 plan multiple times.
          phase4_targets=("$PHASE4_TARGET_GPU_UTILIZATION")
        fi
        safety_search_start_s=$SECONDS
        safety_search_attempt=0
        safety_search_success=false
        safety_search_final_target=""

        for safety_ratio in "${safety_ratios[@]}"; do
          trial_budget_mb="$(awk -v b="$base_budget_mb" -v r="$safety_ratio" 'BEGIN{printf "%.1f", b*r}')"
          budget_slug="$(printf '%s' "$trial_budget_mb" | tr '.' 'p')"
          hybrid_plan_source="$cfg/ilp_solution_pareto_best_budget_${budget_slug}"
          hybrid_assignment_csv="$hybrid_plan_source/ilp_assignment.csv"
          hybrid_cut_edges_csv="$hybrid_plan_source/ilp_cut_edges.csv"
          hybrid_output_dir="$hybrid_plan_source/hybrid_execution"
          hybrid_plan_gpu_budget_mb="$trial_budget_mb"

          for phase4_target in "${phase4_targets[@]}"; do
          safety_search_attempt=$((safety_search_attempt + 1))
          log_msg "Safety-ratio/Phase4-target search attempt $safety_search_attempt -> cfg=$cfg ratio=$safety_ratio trial_budget_mb=$trial_budget_mb phase4_target_gpu_utilization=$phase4_target probe_batch_size=$probe_batch_size"

          # Always re-materialize (no "skip if files already exist" check):
          # a directory from an earlier ratio/target combination, or from a
          # previous unrelated run of this script, could already sit at
          # this exact budget_slug with a DIFFERENT phase4_target baked in
          # -- reusing it unchecked would silently test the wrong target.
          # The re-solve itself is cheap (one ILP solve, no re-profiling,
          # no re-sweep; seconds in practice).
          log_msg "Materializing Pareto-best hybrid plan -> cfg=$cfg budget_mb=$trial_budget_mb phase4_target=$phase4_target objective=$hybrid_plan_objective"
          if ! MODEL="$model" \
          CONFIG_DIR="$cfg" \
          PYTHON_CMD="$PYTHON_CMD" \
          K_SIGMA="$K_SIGMA" \
          W_TIME="$W_TIME" \
          W_ENERGY="$W_ENERGY" \
          W_TRANSFER="$W_TRANSFER" \
          GPU_MEM_BUDGET_MB="$trial_budget_mb" \
          CPU_MEM_BUDGET_MB="$CPU_MEM_BUDGET_MB" \
          MEMORY_MODEL="$MEMORY_MODEL" \
          PEAK_ACTIVATION_OVERLAP="$PEAK_ACTIVATION_OVERLAP" \
          BACKEND="$BACKEND" \
          HW_AGGREGATE="$HW_AGGREGATE" \
          HW_DISPERSION_K="$HW_DISPERSION_K" \
          OUT_DIR="$hybrid_plan_source" \
          REGIME="$REGIME" \
          ENFORCE_CONVEX_WEIGHTS="$ENFORCE_CONVEX_WEIGHTS" \
          STRICT_GRAPH_MAPPING="$STRICT_GRAPH_MAPPING" \
          STRICT_TRANSFER_MAPPING="$STRICT_TRANSFER_MAPPING" \
          STRICT_METRIC_VALIDITY="$STRICT_METRIC_VALIDITY" \
          ALLOW_LOW_QUALITY_STATS="$ALLOW_LOW_QUALITY_STATS" \
          ALLOW_TRANSFER_CALIBRATION_FALLBACK="$ALLOW_TRANSFER_CALIBRATION_FALLBACK" \
          ALLOW_FALLBACK_GRAPH_TRACE="$ALLOW_FALLBACK_GRAPH_TRACE" \
          PHASE4_ACTIVATION="$PHASE4_ACTIVATION" \
          PHASE4_BACKEND="$PHASE4_BACKEND" \
          PHASE4_ENABLE_RECOMPUTE="$PHASE4_ENABLE_RECOMPUTE" \
          PHASE4_ENABLE_CHECKPOINT="$PHASE4_ENABLE_CHECKPOINT" \
          PHASE4_W_IO="$PHASE4_W_IO" \
          PHASE4_RECOMPUTE_PENALTY="$PHASE4_RECOMPUTE_PENALTY" \
          PHASE4_CHECKPOINT_PENALTY="$PHASE4_CHECKPOINT_PENALTY" \
          PHASE4_TARGET_GPU_UTILIZATION="$phase4_target" \
          PHASE4_CALIBRATE_RECOMPUTE_CURVE="$PHASE4_CALIBRATE_RECOMPUTE_CURVE" \
          CALIBRATION_BATCH_SIZE="$probe_batch_size" \
          CALIBRATION_PRECISION="$CALIBRATION_PRECISION" \
          CALIBRATION_SEQ_LENGTH="$CALIBRATION_SEQ_LENGTH" \
          CALIBRATION_DATASETS_ROOT="$CALIBRATION_DATASETS_ROOT" \
          CALIBRATION_FRACTIONS="$CALIBRATION_FRACTIONS" \
          GPU_EXPANSION="$GPU_EXPANSION" \
          GPU_EXPANSION_BUDGET="$GPU_EXPANSION_BUDGET" \
          bash scripts/run_ilp_partition.sh >> "$LOG_FILE" 2>&1; then
            log_msg "[WARN] Materialization failed at ratio=$safety_ratio target=$phase4_target budget_mb=$trial_budget_mb for $cfg. Trying next..."
            continue
          fi

          PROBE_PLAN_FLAGS=(--plan_selection_mode "$hybrid_plan_mode" --plan_source "$hybrid_plan_source" --plan_gpu_budget_mb "$trial_budget_mb")
          if [ -n "$hybrid_plan_objective" ]; then
            PROBE_PLAN_FLAGS+=(--plan_objective "$hybrid_plan_objective")
          fi
          if [ -n "$hybrid_plan_sim_time_ms" ]; then
            PROBE_PLAN_FLAGS+=(--plan_sim_time_ms "$hybrid_plan_sim_time_ms")
          fi
          if [ -n "$hybrid_plan_sim_energy_j" ]; then
            PROBE_PLAN_FLAGS+=(--plan_sim_energy_j "$hybrid_plan_sim_energy_j")
          fi

          # Deliberately NOT "${HYBRID_FLAGS[@]}" here: that array includes
          # --compare_baselines whenever HYBRID_COMPARE_BASELINES=true (the
          # default), which makes validation/run_hybrid_execution.py run
          # all_cpu AND all_gpu in the SAME process before ever reaching
          # ilp_plan. all_gpu at this probe_batch_size is exactly the
          # configuration we already know OOMs on a memory-constrained card
          # -- and since _run_single_safe only catches
          # HybridExecutionUnsupportedError (not OutOfMemoryError), that
          # unrelated all_gpu baseline crashes the whole process before the
          # actual plan under test ever runs. Confirmed this the hard way:
          # every ratio in GPU_VRAM_SAFETY_RATIO_CSV "failed" identically
          # until this was excluded -- the search was never testing the
          # plan it was supposed to be testing.
          PROBE_EXEC_FLAGS=()
          if is_true "$HYBRID_ENABLE_ASYNC_TRANSFER"; then
            PROBE_EXEC_FLAGS+=(--enable_async_transfer)
          fi
          if is_true "$HYBRID_ENABLE_PREFETCH"; then
            PROBE_EXEC_FLAGS+=(--enable_prefetch)
          fi
          if is_true "$HYBRID_ALLOW_CPU_FALLBACK"; then
            PROBE_EXEC_FLAGS+=(--allow_cpu_fallback)
          fi

          if run_sandboxed "$PYTHON_CMD" validation/run_hybrid_execution.py \
            --config_dir "$cfg" \
            --assignment_csv "$hybrid_assignment_csv" \
            --cut_edges_csv "$hybrid_cut_edges_csv" \
            --model "$model" \
            --batch_size "$probe_batch_size" \
            --datasets_root "$DATASETS_DIR" \
            --require_datasets \
            --steps "$HYBRID_STEPS" \
            --lr "$(lr_for_model "$model")" \
            --execution_mode "$HYBRID_EXECUTION_MODE" \
            --output_dir "$hybrid_output_dir" \
            "${PROBE_PLAN_FLAGS[@]}" \
            "${PROBE_EXEC_FLAGS[@]}" \
            >> "$LOG_FILE" 2>&1; then
            safety_search_success=true
            safety_search_final_target="$phase4_target"
            log_msg "Safety-ratio/Phase4-target search succeeded -> cfg=$cfg ratio=$safety_ratio target=$phase4_target budget_mb=$trial_budget_mb after $safety_search_attempt attempt(s)"
            break 2
          else
            log_msg "[WARN] Probe OOM'd (or failed) at ratio=$safety_ratio target=$phase4_target budget_mb=$trial_budget_mb probe_batch_size=$probe_batch_size for $cfg. Trying next..."
          fi
          done
        done

        safety_search_elapsed_s=$((SECONDS - safety_search_start_s))
        log_msg "Safety-ratio/Phase4-target search for cfg=$cfg took ${safety_search_elapsed_s}s over $safety_search_attempt attempt(s), success=$safety_search_success, final_phase4_target=$safety_search_final_target"
        "$PYTHON_CMD" - "$cfg" "$model" "$safety_search_attempt" "$safety_search_elapsed_s" "$safety_search_success" "$hybrid_plan_gpu_budget_mb" "$safety_search_final_target" <<'PY' >> "$LOG_FILE" 2>&1 || true
import json
import sys
from pathlib import Path

cfg, model, attempts, elapsed_s, success, budget_mb, phase4_target = sys.argv[1:8]
out = Path(cfg) / f"safety_ratio_search_{model}.json"
out.write_text(json.dumps({
    "attempts": int(attempts),
    "elapsed_s": int(elapsed_s),
    "success": success == "true",
    "final_budget_mb": float(budget_mb) if budget_mb else None,
    "final_phase4_target_gpu_utilization": float(phase4_target) if phase4_target else None,
}, indent=2))
PY

        if [ "$safety_search_success" != true ]; then
          log_msg "[WARN] Search exhausted GPU_VRAM_SAFETY_RATIO_CSV x PHASE4_TARGET_GPU_UTILIZATION_CSV for $cfg without a working plan. Skipping."
          continue
        fi
        hybrid_already_executed=true
      else
        log_msg "Pareto-best selection unavailable for cfg=$cfg; falling back to default ilp_solution"
      fi
    elif [ "$HYBRID_PLAN_SELECTION" != "default_ilp" ]; then
      log_msg "Unsupported HYBRID_PLAN_SELECTION=$HYBRID_PLAN_SELECTION for cfg=$cfg; falling back to default ilp_solution"
    fi

    if [ ! -f "$hybrid_assignment_csv" ] || [ ! -f "$hybrid_cut_edges_csv" ]; then
      log_msg "Skipping hybrid runtime for cfg=$cfg (missing plan artifacts under $hybrid_plan_source)"
      continue
    fi

    if [ "$hybrid_already_executed" != true ]; then
      HYBRID_PLAN_FLAGS=(
        --plan_selection_mode "$hybrid_plan_mode"
        --plan_source "$hybrid_plan_source"
      )
      if [ -n "$hybrid_plan_source_csv" ]; then
        HYBRID_PLAN_FLAGS+=(--plan_source_csv "$hybrid_plan_source_csv")
      fi
      if [ -n "$hybrid_plan_gpu_budget_mb" ]; then
        HYBRID_PLAN_FLAGS+=(--plan_gpu_budget_mb "$hybrid_plan_gpu_budget_mb")
      fi
      if [ -n "$hybrid_plan_objective" ]; then
        HYBRID_PLAN_FLAGS+=(--plan_objective "$hybrid_plan_objective")
      fi
      if [ -n "$hybrid_plan_sim_time_ms" ]; then
        HYBRID_PLAN_FLAGS+=(--plan_sim_time_ms "$hybrid_plan_sim_time_ms")
      fi
      if [ -n "$hybrid_plan_sim_energy_j" ]; then
        HYBRID_PLAN_FLAGS+=(--plan_sim_energy_j "$hybrid_plan_sim_energy_j")
      fi

      log_msg "Hybrid runtime -> cfg=$cfg plan_mode=$hybrid_plan_mode plan_source=$hybrid_plan_source"
      run_sandboxed "$PYTHON_CMD" validation/run_hybrid_execution.py \
        --config_dir "$cfg" \
        --assignment_csv "$hybrid_assignment_csv" \
        --cut_edges_csv "$hybrid_cut_edges_csv" \
        --model "$model" \
        --batch_size "$HYBRID_BATCH_SIZE" \
        --datasets_root "$DATASETS_DIR" \
        --require_datasets \
        --steps "$HYBRID_STEPS" \
        --lr "$(lr_for_model "$model")" \
        --execution_mode "$HYBRID_EXECUTION_MODE" \
        --output_dir "$hybrid_output_dir" \
          "${HYBRID_PLAN_FLAGS[@]}" \
        "${HYBRID_FLAGS[@]}" \
        >> "$LOG_FILE" 2>&1 || {
          log_msg "[WARN] Hybrid execution failed for $cfg. See log for details. Continuing..."
          continue
        }
    else
      log_msg "Hybrid runtime for cfg=$cfg already executed by the safety-ratio search above -- skipping redundant re-run."
    fi
  done

  log_msg "Hybrid runtime stage finished"
  fi
fi

if is_true "$RUN_REPORTS"; then
  section "STEP 5/5 - CONSOLIDATED REPORTS + LATEX"

  if is_true "$DRY_RUN"; then
    log_msg "DRY_RUN=true -> skipping consolidated report and LaTeX generation"
    log_msg "Report stage finished"
  else

  REPORT_INPUT_ROOT="$(resolve_report_input_root "$BASE_OUTPUT_DIR" "$HOST_TAG")"
  log_msg "Report input root: $REPORT_INPUT_ROOT"

  PYTHON_CMD="$PYTHON_CMD" \
  INPUT_ROOT="$REPORT_INPUT_ROOT" \
  OUTPUT_DIR="$REPORTS_DIR" \
  bash scripts/generate_ilp_report_assets.sh >> "$LOG_FILE" 2>&1

  log_msg "Computing Statistical Significance..."
  PYTHON_CMD="$PYTHON_CMD" \
  CONSOLIDATED_CSV="$REPORTS_DIR/csv/ilp_pareto_consolidated.csv" \
  OUTPUT_CSV="$REPORTS_DIR/csv/ilp_statistical_significance.csv" \
  bash scripts/run_statistical_significance.sh >> "$LOG_FILE" 2>&1

  PYTHON_CMD="$PYTHON_CMD" \
  BEST_CSV="$REPORTS_DIR/csv/ilp_best_per_model.csv" \
  CONSOLIDATED_CSV="$REPORTS_DIR/csv/ilp_pareto_consolidated.csv" \
  HYBRID_CSV="$REPORTS_DIR/csv/hybrid_execution_best_per_model.csv" \
  OUT_DIR="$LATEX_DIR" \
  bash scripts/export_ilp_tables_latex.sh >> "$LOG_FILE" 2>&1

  log_msg "Generating advanced analytics plots..."
  "$PYTHON_CMD" validation/generate_advanced_thesis_plots.py \
    --input_root "$REPORT_INPUT_ROOT" \
    --output_dir "$REPORTS_DIR" \
    --hybrid_csv "$REPORTS_DIR/csv/hybrid_execution_best_per_model.csv" >> "$LOG_FILE" 2>&1

  log_msg "Generating OOM vs ILP Comparison plots..."
  "$PYTHON_CMD" scripts/plot_all_oom_vs_ilp.py \
    --input_root "$REPORT_INPUT_ROOT" \
    --output_dir "$REPORTS_DIR/plots/oom_vs_ilp" >> "$LOG_FILE" 2>&1

  log_msg "Generating Energy vs Power Tradeoff plots..."
  "$PYTHON_CMD" scripts/plot_all_energy_tradeoff.py \
    --input_root "$REPORT_INPUT_ROOT" \
    --output_dir "$REPORTS_DIR/plots/energy_tradeoff" >> "$LOG_FILE" 2>&1

  log_msg "Generating methodological plots (Chapter 4: temporal decomposition, congestion knee, overlap tension)..."
  "$PYTHON_CMD" scripts/generate_methodological_plots.py \
    --input_root "$REPORT_INPUT_ROOT" \
    --output_dir "$REPORTS_DIR/plots/methodology" >> "$LOG_FILE" 2>&1

  PYTHON_CMD="$PYTHON_CMD" \
  INPUT_ROOT="$REPORT_INPUT_ROOT" \
  OUTPUT_DIR="$REPORTS_DIR/grid_audit" \
  EXPECTED_MODELS_CSV="$MODELS_CSV" \
  EXPECTED_OPTIMIZERS_CSV="$OPTIMIZERS_CSV" \
  EXPECTED_PRECISIONS_CSV="$PRECISIONS_CSV" \
  EXPECTED_BATCHES_CSV="$BATCH_SIZES_CSV" \
  EXPECTED_REPEATS="$REPEATS" \
  LOGS_ROOT="logs" \
  REQUIRE_HYBRID="$RUN_HYBRID" \
  bash scripts/audit_experimental_grid.sh >> "$LOG_FILE" 2>&1

  # Move the markdown summary to summary_docs/ where it belongs
  if [ -f "$REPORTS_DIR/grid_audit/GRID_AUDIT_SUMMARY.md" ]; then
    mkdir -p "$REPORTS_DIR/summary_docs"
    mv "$REPORTS_DIR/grid_audit/GRID_AUDIT_SUMMARY.md" "$REPORTS_DIR/summary_docs/"
  fi

  cat > "$REPORTS_DIR/THESIS_MODE_PROTOCOL_CHECKLIST.md" <<EOF
# Thesis Mode Protocol Checklist

This run executed the project-wide thesis mode orchestration.

## Generated roots
- Base output: $BASE_OUTPUT_DIR
- Reports: $REPORTS_DIR
- LaTeX: $LATEX_DIR
- Log: $LOG_FILE
- Grid audit: $REPORTS_DIR/grid_audit

## Automatically covered by script
- Profiling campaign grid
- Replicate aggregation
- ILP partition and Pareto sweep per config
- Consolidated report assets and LaTeX tables
- Optional hybrid runtime traces aligned to $HYBRID_PLAN_SELECTION plans (FX and export-backed DAG paths when supported)
- Grid audit outputs: completeness by configuration, failed configurations with probable cause, and doctoral-readiness flag

## Must still be validated explicitly (doctoral criteria)
- Final model quality metrics (accuracy/loss/AUC) vs baselines in target tasks
- Statistical significance and effect size for claimed gains
- Multi-hardware robustness matrix and threats-to-validity analysis

EOF

  log_msg "Report stage finished"
  fi
fi

section "THESIS MODE COMPLETE"
log_msg "Base output: $BASE_OUTPUT_DIR"
log_msg "Reports: $REPORTS_DIR"
log_msg "LaTeX: $LATEX_DIR"
log_msg "Log: $LOG_FILE"
