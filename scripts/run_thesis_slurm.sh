#!/usr/bin/env bash

# ==============================================================================
# SLURM submission wrapper for PRISM final thesis campaigns.
# Submit with: sbatch ./scripts/run_thesis_slurm.sh --profile doctoral_minimal
# ==============================================================================

#SBATCH --job-name=PRISM_profiling
#SBATCH --partition=GPU
#SBATCH --nodelist=paccaA100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-/home/latorresn/PRISM}"
LAUNCH_SCRIPT="${LAUNCH_SCRIPT:-$PROJECT_ROOT/scripts/launch_slurm.sh}"

MODULE_NAME="${MODULE_NAME:-Analytics/anaconda3}"
CONDA_EXE="${CONDA_EXE:-/opt/ohpc/pub/Analytics/anaconda3/bin/conda}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-prism_env}"
GPU_VRAM_FALLBACK_MB="${GPU_VRAM_FALLBACK_MB:-40960}"
CAMPAIGN_PROFILE="${CAMPAIGN_PROFILE:-doctoral_full}"
RUN_HYBRID="${RUN_HYBRID:-true}"
FULL_SEEDS_CSV="${FULL_SEEDS_CSV:-42,43,44}"
SINGLE_SEED="${SINGLE_SEED:-42}"
FULL_REPEATS_PER_SEED="${FULL_REPEATS_PER_SEED:-2}"
NON_FULL_REPEATS="${NON_FULL_REPEATS:-1}"
DATA_MOUNT_SRC="${DATA_MOUNT_SRC:-$PROJECT_ROOT/data}"

log_msg() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

on_error() {
    local exit_code=$?
    log_msg "ERROR: run_thesis_slurm failed (exit_code=$exit_code, line=${BASH_LINENO[0]})"
    exit "$exit_code"
}

trap on_error ERR

parse_args() {
    while [ "$#" -gt 0 ]; do
        case "$1" in
            --profile)
                CAMPAIGN_PROFILE="$2"
                shift 2
                ;;
            --single-seed)
                SINGLE_SEED="$2"
                shift 2
                ;;
            --non-full-repeats)
                NON_FULL_REPEATS="$2"
                shift 2
                ;;
            --full-seeds)
                FULL_SEEDS_CSV="$2"
                shift 2
                ;;
            --full-repeats)
                FULL_REPEATS_PER_SEED="$2"
                shift 2
                ;;
            --run-hybrid)
                RUN_HYBRID="$2"
                shift 2
                ;;
            --help|-h)
                cat <<'EOF'
Usage: run_thesis_slurm.sh [options]

Options:
  --profile <quick_smoke|doctoral_minimal|doctoral_full|doctoral_diagnostic>
  --single-seed <int>
  --non-full-repeats <int>
  --full-seeds <csv>
  --full-repeats <int>
  --run-hybrid <true|false>

Examples:
  sbatch scripts/run_thesis_slurm.sh --profile doctoral_minimal --single-seed 42 --non-full-repeats 2
  sbatch --time=72:00:00 --cpus-per-task=16 scripts/run_thesis_slurm.sh --profile doctoral_full
EOF
                exit 0
                ;;
            *)
                log_msg "ERROR: Unknown argument: $1"
                exit 2
                ;;
        esac
    done
}

parse_args "$@"

case "$CAMPAIGN_PROFILE" in
    quick_smoke|doctoral_minimal|doctoral_full|doctoral_diagnostic)
        ;;
    *)
        log_msg "ERROR: Unsupported --profile '$CAMPAIGN_PROFILE'"
        log_msg "ERROR: Allowed values: quick_smoke, doctoral_minimal, doctoral_full, doctoral_diagnostic"
        exit 2
        ;;
esac

if [ ! -d "$PROJECT_ROOT" ]; then
    log_msg "ERROR: PROJECT_ROOT does not exist: $PROJECT_ROOT"
    exit 1
fi

if [ ! -f "$PROJECT_ROOT/src/data/__init__.py" ]; then
    log_msg "ERROR: PROJECT_ROOT does not look like PRISM root: $PROJECT_ROOT"
    log_msg "ERROR: Missing file: $PROJECT_ROOT/src/data/__init__.py"
    exit 1
fi

if [ ! -f "$LAUNCH_SCRIPT" ]; then
    log_msg "ERROR: launch script not found: $LAUNCH_SCRIPT"
    exit 1
fi

mkdir -p "$PROJECT_ROOT/logs/slurm"

log_msg "Starting Slurm thesis campaign"
log_msg "Job id: ${SLURM_JOB_ID:-unknown}"
log_msg "Node list: ${SLURM_JOB_NODELIST:-unknown}"
log_msg "Requested profile: $CAMPAIGN_PROFILE"
log_msg "Seeds config: SINGLE_SEED=$SINGLE_SEED FULL_SEEDS_CSV=$FULL_SEEDS_CSV"
log_msg "Project root: $PROJECT_ROOT"
log_msg "GPU VRAM fallback: ${GPU_VRAM_FALLBACK_MB} MiB"

cd "$PROJECT_ROOT"

if command -v module >/dev/null 2>&1; then
    module load "$MODULE_NAME"
elif [ -f /etc/profile.d/modules.sh ]; then
    # shellcheck disable=SC1091
    source /etc/profile.d/modules.sh
    module load "$MODULE_NAME"
fi

if [ -x "$CONDA_EXE" ]; then
    log_msg "Using conda executable: $CONDA_EXE"
fi

CAMPAIGN_PROFILE="$CAMPAIGN_PROFILE" \
MODULE_NAME="$MODULE_NAME" \
CONDA_EXE="$CONDA_EXE" \
CONDA_ENV_NAME="$CONDA_ENV_NAME" \
GPU_VRAM_FALLBACK_MB="$GPU_VRAM_FALLBACK_MB" \
DATA_MOUNT_SRC="$DATA_MOUNT_SRC" \
FULL_SEEDS_CSV="$FULL_SEEDS_CSV" \
SINGLE_SEED="$SINGLE_SEED" \
FULL_REPEATS_PER_SEED="$FULL_REPEATS_PER_SEED" \
NON_FULL_REPEATS="$NON_FULL_REPEATS" \
RUN_HYBRID="$RUN_HYBRID" \
bash "$LAUNCH_SCRIPT"

log_msg "Slurm thesis campaign completed."