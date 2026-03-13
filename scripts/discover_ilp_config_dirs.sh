#!/bin/bash
set -euo pipefail

# Discover multi-host ILP config directories and optionally execute ILP wrappers.
#
# Expected layout (produced by profiler/scripts):
# data/<host>/results/<model>/<optimizer>/<precision>/batch_<N>
#
# Examples:
#   bash scripts/discover_ilp_config_dirs.sh
#   MODEL=resnet50 OPTIMIZER=AdamW PRECISION=bf16 BATCH=32 bash scripts/discover_ilp_config_dirs.sh
#   MODE=partition MODEL=simple_mlp bash scripts/discover_ilp_config_dirs.sh
#   MODE=pareto MODEL=resnet50 GPU_BUDGETS_MB=400,800,1200 bash scripts/discover_ilp_config_dirs.sh

MODEL="${MODEL:-simple_mlp}"
OPTIMIZER="${OPTIMIZER:-SGD}"
PRECISION="${PRECISION:-fp32}"
BATCH="${BATCH:-8}"
RESULTS_ROOT="${RESULTS_ROOT:-data}"
MODE="${MODE:-print}"         # print | partition | pareto
OUTPUT_ENV_FILE="${OUTPUT_ENV_FILE:-}"

# Pass-through vars for execution modes.
PYTHON_CMD="${PYTHON_CMD:-python}"
GPU_BUDGETS_MB="${GPU_BUDGETS_MB:-400,600,800,1000,1200}"
HW_AGGREGATE="${HW_AGGREGATE:-max}"
HW_DISPERSION_K="${HW_DISPERSION_K:-0.0}"

if ! [[ "$BATCH" =~ ^[0-9]+$ ]]; then
  echo "[ERROR] BATCH must be an integer, got: $BATCH" >&2
  exit 1
fi

if [[ ! -d "$RESULTS_ROOT" ]]; then
  echo "[ERROR] RESULTS_ROOT does not exist: $RESULTS_ROOT" >&2
  exit 1
fi

mapfile -t CANDIDATE_DIRS < <(
  find "$RESULTS_ROOT" -type d \
    -path "*/results/$MODEL/$OPTIMIZER/$PRECISION/batch_${BATCH}" \
    | sort
)

if [[ ${#CANDIDATE_DIRS[@]} -eq 0 ]]; then
  echo "[ERROR] No matching config dirs found for:" >&2
  echo "        model=$MODEL optimizer=$OPTIMIZER precision=$PRECISION batch=$BATCH" >&2
  echo "        root=$RESULTS_ROOT" >&2
  exit 2
fi

MATCHED_DIRS=()
for d in "${CANDIDATE_DIRS[@]}"; do
  stats_a="$d/${MODEL}_metrics_stats.csv"
  stats_b="$d/metrics_stats.csv"
  if [[ ! -f "$stats_a" && ! -f "$stats_b" ]]; then
    continue
  fi

  if find "$d" -maxdepth 2 -type d -name 'run_*' | grep -q .; then
    if find "$d" -maxdepth 3 -type f -name "${MODEL}_graph_edges.csv" | grep -q . \
      && find "$d" -maxdepth 3 -type f -name "${MODEL}_transfer_edges.csv" | grep -q .; then
      MATCHED_DIRS+=("$d")
    fi
  else
    if [[ -f "$d/${MODEL}_graph_edges.csv" && -f "$d/${MODEL}_transfer_edges.csv" ]]; then
      MATCHED_DIRS+=("$d")
    fi
  fi
done

if [[ ${#MATCHED_DIRS[@]} -eq 0 ]]; then
  echo "[ERROR] Candidate dirs were found, but none contain complete ILP artifacts:" >&2
  echo "        required: metrics_stats + graph_edges + transfer_edges" >&2
  echo "        model=$MODEL optimizer=$OPTIMIZER precision=$PRECISION batch=$BATCH" >&2
  exit 4
fi

CONFIG_DIRS="$(printf "%s\n" "${MATCHED_DIRS[@]}" | paste -sd, -)"

echo "[INFO] Matched ${#MATCHED_DIRS[@]} config dirs:"
for d in "${MATCHED_DIRS[@]}"; do
  echo "  - $d"
done

echo ""
echo "[INFO] CONFIG_DIRS=\"$CONFIG_DIRS\""

if [[ -n "$OUTPUT_ENV_FILE" ]]; then
  {
    echo "MODEL=$MODEL"
    echo "OPTIMIZER=$OPTIMIZER"
    echo "PRECISION=$PRECISION"
    echo "BATCH=$BATCH"
    echo "RESULTS_ROOT=$RESULTS_ROOT"
    echo "CONFIG_DIRS=$CONFIG_DIRS"
    echo "HW_AGGREGATE=$HW_AGGREGATE"
    echo "HW_DISPERSION_K=$HW_DISPERSION_K"
  } > "$OUTPUT_ENV_FILE"
  echo "[INFO] Wrote environment snapshot: $OUTPUT_ENV_FILE"
fi

case "$MODE" in
  print)
    echo ""
    echo "[NEXT] Partition example:"
    echo "CONFIG_DIRS=\"$CONFIG_DIRS\" MODEL=$MODEL HW_AGGREGATE=$HW_AGGREGATE HW_DISPERSION_K=$HW_DISPERSION_K bash scripts/run_ilp_partition.sh"
    echo ""
    echo "[NEXT] Pareto example:"
    echo "CONFIG_DIRS=\"$CONFIG_DIRS\" MODEL=$MODEL GPU_BUDGETS_MB=$GPU_BUDGETS_MB HW_AGGREGATE=$HW_AGGREGATE HW_DISPERSION_K=$HW_DISPERSION_K bash scripts/run_ilp_pareto_sweep.sh"
    ;;
  partition)
    CONFIG_DIRS="$CONFIG_DIRS" MODEL="$MODEL" PYTHON_CMD="$PYTHON_CMD" \
    HW_AGGREGATE="$HW_AGGREGATE" HW_DISPERSION_K="$HW_DISPERSION_K" \
    bash scripts/run_ilp_partition.sh
    ;;
  pareto)
    CONFIG_DIRS="$CONFIG_DIRS" MODEL="$MODEL" PYTHON_CMD="$PYTHON_CMD" \
    GPU_BUDGETS_MB="$GPU_BUDGETS_MB" HW_AGGREGATE="$HW_AGGREGATE" HW_DISPERSION_K="$HW_DISPERSION_K" \
    bash scripts/run_ilp_pareto_sweep.sh
    ;;
  *)
    echo "[ERROR] Unsupported MODE: $MODE (use print|partition|pareto)" >&2
    exit 3
    ;;
esac
