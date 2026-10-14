#!/bin/bash
set -euo pipefail

# Usage:
#   MODEL=simple_mlp CONFIG_DIR=data/test-m3-r2/simple_mlp/SGD/fp32/batch_8 bash scripts/run_ilp_partition.sh

source "$(dirname "$0")/sanitize_cuda_env.sh"
sanitize_cuda_runtime_env

PYTHON_CMD="${PYTHON_CMD:-python}"
MODEL="${MODEL:-simple_mlp}"
CONFIG_DIR="${CONFIG_DIR:-data/test-m3-r2/simple_mlp/SGD/fp32/batch_8}"
CONFIG_DIRS="${CONFIG_DIRS:-}"
K_SIGMA="${K_SIGMA:-1.0}"
W_TIME="${W_TIME:-1.0}"
W_ENERGY="${W_ENERGY:-0.0}"
W_TRANSFER="${W_TRANSFER:-1.0}"
GPU_MEM_BUDGET_MB="${GPU_MEM_BUDGET_MB:-1e18}"
# Default changed from an unconstrained 1e18 to "auto" after a real local
# freeze: once the ILP's GPU memory estimate stopped underestimating usage
# (see training_profiler.py's _current_batch_scale fix), it started
# correctly offloading much more to CPU -- with no CPU budget ceiling,
# nothing stopped it from planning more CPU memory than the machine has,
# which froze the whole desktop instead of raising a catchable OOM. An
# explicit numeric override (or 1e18 to restore the old unconstrained
# behavior) still works -- "auto" just resolves against real detected
# system RAM below, same pattern as GPU_BUDGETS_MB=auto in run_thesis_mode.sh.
CPU_MEM_BUDGET_MB="${CPU_MEM_BUDGET_MB:-auto}"
# 0.90, matching GPU_VRAM_SAFETY_RATIO's convention: the ILP's CPU memory
# constraint is itself a correct, hard budget (see
# src/ilp/model_builder.py::_build_effective_memory -- cpu_mem[n] gets the
# same nodal_sum/peak_approx/topological treatment as gpu_mem[n], not a
# cruder model), so it does not need a punitive extra discount on top. A
# steep ratio here (e.g. 0.60) starves PRISM of the CPU headroom that is
# its whole reason to exist as a complementary resource to a small GPU.
CPU_MEM_SAFETY_RATIO="${CPU_MEM_SAFETY_RATIO:-0.9}"

if [ "$CPU_MEM_BUDGET_MB" = "auto" ]; then
  if [ ! -r /proc/meminfo ]; then
    echo "ERROR: CPU_MEM_BUDGET_MB=auto but /proc/meminfo is not readable." >&2
    echo "       Set CPU_MEM_BUDGET_MB explicitly, or 1e18 to disable this safety limit." >&2
    exit 1
  fi
  CPU_RAM_AVAILABLE_MB="$(awk '/^MemAvailable:/ { printf "%d", $2 / 1024 }' /proc/meminfo)"
  CPU_MEM_BUDGET_MB="$(awk -v v="$CPU_RAM_AVAILABLE_MB" -v r="$CPU_MEM_SAFETY_RATIO" 'BEGIN{printf "%d", v*r}')"
  echo "System RAM available: ${CPU_RAM_AVAILABLE_MB} MiB; CPU_MEM_SAFETY_RATIO=$CPU_MEM_SAFETY_RATIO -> CPU memory budget: ${CPU_MEM_BUDGET_MB} MiB"
fi
MEMORY_MODEL="${MEMORY_MODEL:-topological}"
PEAK_ACTIVATION_OVERLAP="${PEAK_ACTIVATION_OVERLAP:-0.35}"
BACKEND="${BACKEND:-auto}"
HW_AGGREGATE="${HW_AGGREGATE:-max}"
HW_DISPERSION_K="${HW_DISPERSION_K:-0.0}"
OUT_DIR="${OUT_DIR:-${CONFIG_DIR}/ilp_solution}"
REGIME="${REGIME:-diagnostic}"
ENFORCE_CONVEX_WEIGHTS="${ENFORCE_CONVEX_WEIGHTS:-false}"
STRICT_GRAPH_MAPPING="${STRICT_GRAPH_MAPPING:-true}"
STRICT_TRANSFER_MAPPING="${STRICT_TRANSFER_MAPPING:-true}"
ALLOW_LOW_QUALITY_STATS="${ALLOW_LOW_QUALITY_STATS:-false}"
ALLOW_TRANSFER_CALIBRATION_FALLBACK="${ALLOW_TRANSFER_CALIBRATION_FALLBACK:-false}"
ALLOW_FALLBACK_GRAPH_TRACE="${ALLOW_FALLBACK_GRAPH_TRACE:-false}"
STRICT_METRIC_VALIDITY="${STRICT_METRIC_VALIDITY:-false}"

# Phase 4 (activation-strategy optimization: recompute/checkpoint). Off by
# default -- preserves exact prior behavior for any existing caller. When
# enabled, gives the ILP a lever beyond CPU/GPU device placement to fit a
# memory budget: marking a layer recompute (torch.utils.checkpoint, trades
# compute for not retaining its activation) or checkpoint (offload the
# saved activation to CPU) instead of only moving whole layers to CPU.
# src/runtime/hybrid_executor.py already implements both at runtime
# (call_module path); this was previously solved but never actually
# enabled by this script, so no real campaign plan ever used it.
PHASE4_ACTIVATION="${PHASE4_ACTIVATION:-false}"
PHASE4_BACKEND="${PHASE4_BACKEND:-auto}"
PHASE4_ENABLE_RECOMPUTE="${PHASE4_ENABLE_RECOMPUTE:-true}"
PHASE4_ENABLE_CHECKPOINT="${PHASE4_ENABLE_CHECKPOINT:-true}"
PHASE4_W_IO="${PHASE4_W_IO:-0.0}"
PHASE4_RECOMPUTE_PENALTY="${PHASE4_RECOMPUTE_PENALTY:-0.5}"
PHASE4_CHECKPOINT_PENALTY="${PHASE4_CHECKPOINT_PENALTY:-0.3}"
# Fraction of the GPU budget the Phase4 greedy heuristic aims to use once
# triggered (src/ilp/solve.py::_solve_phase4_greedy). Raised from the
# original hardcoded 0.60 to 0.95: see
# src/ilp/model_builder.py::ILPConfig4.target_gpu_utilization for why --
# the baseline solve already enforces gpu_mem_budget_mb as a hard
# constraint, so 0.60 was an independent, now-unjustified extra ~40%
# haircut on top of an already-correct, already-feasible solution.
PHASE4_TARGET_GPU_UTILIZATION="${PHASE4_TARGET_GPU_UTILIZATION:-0.95}"

# Calibrated recompute curve (fix/07_gpu_recompute_joint_decision/README.md):
# off by default, same safety posture as PHASE4_ACTIVATION above. When
# enabled (PHASE4_BACKEND must be "joint"), src/ilp/solve.py's recompute[n]
# additive-per-node assumption -- confirmed empirically to be wrong, real
# savings rise then REVERSE past a model-specific point -- is replaced by a
# curve measured for real, on the real model, before solving: a no-recompute
# baseline first, then the real memory curve for the blocks that baseline
# already put fully on GPU, then the joint MILP using that curve. Validated
# end to end (real training, not just predicted memory) for resnet50,
# distilgpt2, and gpt2_small -- in every tested case the calibrated plan
# matched or beat the no-recompute plan on BOTH memory and real wall-clock
# step time for the same budget, never only one of the two.
PHASE4_CALIBRATE_RECOMPUTE_CURVE="${PHASE4_CALIBRATE_RECOMPUTE_CURVE:-false}"
# Batch size for the real model+input this loads to measure the curve.
# Independent of GPU_MEM_BUDGET_MB -- defaults to BATCH_SIZE when this
# script is invoked from run_thesis_mode.sh, which already knows it.
CALIBRATION_BATCH_SIZE="${CALIBRATION_BATCH_SIZE:-${BATCH_SIZE:-}}"
CALIBRATION_PRECISION="${CALIBRATION_PRECISION:-fp32}"
CALIBRATION_SEQ_LENGTH="${CALIBRATION_SEQ_LENGTH:-128}"
CALIBRATION_DATASETS_ROOT="${CALIBRATION_DATASETS_ROOT:-datasets}"
CALIBRATION_FRACTIONS="${CALIBRATION_FRACTIONS:-0.0,0.25,0.5,0.75,1.0}"

# Post-solve GPU expansion (fix/08_gpu_memory_expansion_and_underestimation/README.md):
# off by default, same safety posture as the two flags above. Greedily moves
# CPU-resident layers to GPU when topology_memory.py's real sequential-peak
# estimate shows margin the solver's own (simultaneous-sum) constraint missed,
# then re-validates the result against the hybrid engine's REAL transient
# memory overhead (one real forward+backward pass per candidate, binary
# search over how many moves to keep) instead of trusting that estimate
# alone -- confirmed necessary: without this real-overhead validation, a real
# resnet50 training run stalled in an infinite CUDA OOM retry loop on an
# expanded plan topology_memory.py had deemed safe. Requires
# CALIBRATION_BATCH_SIZE (shares the same calibration args as the recompute
# curve above) to actually validate the expansion -- same requirement, same
# reasoning, same CALIBRATION_FLAGS block below.
GPU_EXPANSION="${GPU_EXPANSION:-false}"
GPU_EXPANSION_BUDGET="${GPU_EXPANSION_BUDGET:-200}"

# Fail fast if CBC is not available; strict campaigns should not start ILP with
# an unusable MILP backend and then fall back to exhaustive search limits.
CBC_BIN="${CBC_BIN:-}"
if [ -z "$CBC_BIN" ]; then
  if command -v cbc >/dev/null 2>&1; then
    CBC_BIN="$(command -v cbc)"
  else
    PY_DIR="$(dirname "$PYTHON_CMD")"
    if [ -x "$PY_DIR/cbc" ]; then
      CBC_BIN="$PY_DIR/cbc"
    fi
  fi
fi

if [ -z "$CBC_BIN" ] || [ ! -x "$CBC_BIN" ]; then
  echo "[ERROR] CBC solver executable not found or not executable." >&2
  echo "[ERROR] Required for scalable ILP solve (PuLP + CBC)." >&2
  echo "[ERROR] Conda fix: conda install -n prism_env -c conda-forge coin-or-cbc pulp" >&2
  echo "[ERROR] Current PYTHON_CMD: $PYTHON_CMD" >&2
  exit 2
fi

export PATH="$(dirname "$CBC_BIN"):$PATH"

STRICT_FLAGS=()
if [ "$STRICT_GRAPH_MAPPING" = true ]; then
  STRICT_FLAGS+=(--strict_graph_mapping)
fi
if [ "$STRICT_TRANSFER_MAPPING" = true ]; then
  STRICT_FLAGS+=(--strict_transfer_mapping)
fi
if [ "$ALLOW_LOW_QUALITY_STATS" = true ]; then
  STRICT_FLAGS+=(--allow_low_quality_stats)
fi
if [ "$ALLOW_TRANSFER_CALIBRATION_FALLBACK" = true ]; then
  STRICT_FLAGS+=(--allow_transfer_calibration_fallback)
fi
if [ "$ALLOW_FALLBACK_GRAPH_TRACE" = true ]; then
  STRICT_FLAGS+=(--allow_fallback_graph_trace)
fi
if [ "$STRICT_METRIC_VALIDITY" = true ]; then
  STRICT_FLAGS+=(--strict_metric_validity)
fi
if [ "$ENFORCE_CONVEX_WEIGHTS" = true ]; then
  STRICT_FLAGS+=(--enforce_convex_weights)
fi

PHASE4_FLAGS=()
if [ "$PHASE4_ACTIVATION" = true ]; then
  PHASE4_FLAGS+=(
    --phase4_activation
    --phase4_backend "$PHASE4_BACKEND"
    --phase4_w_io "$PHASE4_W_IO"
    --phase4_recompute_penalty "$PHASE4_RECOMPUTE_PENALTY"
    --phase4_checkpoint_penalty "$PHASE4_CHECKPOINT_PENALTY"
    --phase4_target_gpu_utilization "$PHASE4_TARGET_GPU_UTILIZATION"
  )
  if [ "$PHASE4_ENABLE_RECOMPUTE" = true ]; then
    PHASE4_FLAGS+=(--phase4_enable_recompute)
  fi
  if [ "$PHASE4_ENABLE_CHECKPOINT" = true ]; then
    PHASE4_FLAGS+=(--phase4_enable_checkpoint)
  fi
  if [ "$PHASE4_CALIBRATE_RECOMPUTE_CURVE" = true ]; then
    PHASE4_FLAGS+=(--phase4_calibrate_recompute_curve)
  fi
fi

# Shared by PHASE4_CALIBRATE_RECOMPUTE_CURVE and GPU_EXPANSION -- both need a
# real model+input loaded to measure something for real (the recompute
# curve, or the expanded plan's real transient overhead), so both require
# CALIBRATION_BATCH_SIZE and emit the same --calibration_* flags.
CALIBRATION_FLAGS=()
if [ "$PHASE4_CALIBRATE_RECOMPUTE_CURVE" = true ] || [ "$GPU_EXPANSION" = true ]; then
  if [ -z "$CALIBRATION_BATCH_SIZE" ]; then
    echo "[ERROR] PHASE4_CALIBRATE_RECOMPUTE_CURVE=true or GPU_EXPANSION=true requires CALIBRATION_BATCH_SIZE (or BATCH_SIZE) to be set." >&2
    exit 2
  fi
  CALIBRATION_FLAGS+=(
    --calibration_batch_size "$CALIBRATION_BATCH_SIZE"
    --calibration_precision "$CALIBRATION_PRECISION"
    --calibration_seq_length "$CALIBRATION_SEQ_LENGTH"
    --calibration_datasets_root "$CALIBRATION_DATASETS_ROOT"
    --calibration_fractions "$CALIBRATION_FRACTIONS"
  )
fi

# Independent of PHASE4_ACTIVATION in Python (lives in the always-on
# topology_memory correction block), so this array is built outside the
# PHASE4_ACTIVATION guard above.
EXPANSION_FLAGS=()
if [ "$GPU_EXPANSION" = true ]; then
  EXPANSION_FLAGS+=(--enable_gpu_expansion --gpu_expansion_budget "$GPU_EXPANSION_BUDGET")
fi

CONFIG_FLAGS=(--config_dir "$CONFIG_DIR")
if [ -n "$CONFIG_DIRS" ]; then
  CONFIG_FLAGS=(--config_dirs "$CONFIG_DIRS")
fi

"$PYTHON_CMD" validation/run_ilp_partition.py \
  "${CONFIG_FLAGS[@]}" \
  --model "$MODEL" \
  --regime "$REGIME" \
  --k_sigma "$K_SIGMA" \
  --w_time "$W_TIME" \
  --w_energy "$W_ENERGY" \
  --w_transfer "$W_TRANSFER" \
  --gpu_mem_budget_mb "$GPU_MEM_BUDGET_MB" \
  --cpu_mem_budget_mb "$CPU_MEM_BUDGET_MB" \
  --memory_model "$MEMORY_MODEL" \
  --peak_activation_overlap "$PEAK_ACTIVATION_OVERLAP" \
  --backend "$BACKEND" \
  --hw_aggregate "$HW_AGGREGATE" \
  --hw_dispersion_k "$HW_DISPERSION_K" \
  --output_dir "$OUT_DIR" \
  "${STRICT_FLAGS[@]}" \
  "${PHASE4_FLAGS[@]}" \
  "${CALIBRATION_FLAGS[@]}" \
  "${EXPANSION_FLAGS[@]}"
