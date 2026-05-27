#!/usr/bin/env bash
set -Eeuo pipefail

# OAR submission wrapper for Grid5000 final thesis campaigns.
# Submit with: oarsub -S ./scripts/run_thesis.sh

#OAR -n Thesis_Profiling
#OAR -q abaca
#OAR -q besteffort
#OAR -p esterel35
#OAR -t deploy
#OAR -l nodes=1,walltime=96:00:00
#OAR -O thesis_job.%jobid%.output
#OAR -E thesis_job.%jobid%.error

PROJECT_ROOT="${PROJECT_ROOT:-/root/PRISM}"
REMOTE_LAUNCH_SCRIPT="${REMOTE_LAUNCH_SCRIPT:-$PROJECT_ROOT/scripts/launch_grid5k.sh}"
LOCAL_LAUNCH_SCRIPT="${LOCAL_LAUNCH_SCRIPT:-scripts/launch_grid5k.sh}"
LOCAL_SCRIPTS_DIR="${LOCAL_SCRIPTS_DIR:-scripts}"
LOCAL_PROJECT_ROOT="${LOCAL_PROJECT_ROOT:-$(pwd)}"
SYNC_PROJECT_BEFORE_RUN="${SYNC_PROJECT_BEFORE_RUN:-true}"
# IMPORTANT: defaults are anchored to repository root to avoid excluding src/data.
SYNC_EXCLUDES="${SYNC_EXCLUDES:-/.git,/.venv,/logs,/reports,/data,/datasets,/books,/paper_thesis,/papers}"
KADEPLOY_FILE="${KADEPLOY_FILE:-rocky9_profiling.yaml}"

CAMPAIGN_PROFILE="${CAMPAIGN_PROFILE:-doctoral_full}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-prism_env}"
RUN_HYBRID="${RUN_HYBRID:-true}"
FULL_SEEDS_CSV="${FULL_SEEDS_CSV:-42,43,44}"
SINGLE_SEED="${SINGLE_SEED:-42}"
FULL_REPEATS_PER_SEED="${FULL_REPEATS_PER_SEED:-1}"
NON_FULL_REPEATS="${NON_FULL_REPEATS:-1}"

log_msg() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

on_error() {
    local exit_code=$?
    log_msg "ERROR: run_thesis failed (exit_code=$exit_code, line=${BASH_LINENO[0]})"
    exit "$exit_code"
}

trap on_error ERR

if [ -z "${OAR_JOB_ID:-}" ]; then
    log_msg "WARNING: OAR_JOB_ID is empty. This script should be launched by oarsub."
fi

if [ -z "${OAR_FILE_NODES:-}" ] || [ ! -f "${OAR_FILE_NODES:-}" ]; then
    log_msg "ERROR: OAR_FILE_NODES is missing. Cannot determine allocated node(s)."
    exit 1
fi

TARGET_NODE="$(sort -u "$OAR_FILE_NODES" | head -n 1 | xargs)"
if [ -z "$TARGET_NODE" ]; then
    log_msg "ERROR: Could not resolve target node from OAR_FILE_NODES."
    exit 1
fi

log_msg "Starting job ${OAR_JOB_ID:-unknown} on node: $TARGET_NODE"
log_msg "Deploying image with kadeploy3: $KADEPLOY_FILE"

# Deploy image on the reserved nodes and copy SSH key for root access.
kadeploy3 -f "$OAR_FILE_NODES" -a "$KADEPLOY_FILE" -k

if [ "$SYNC_PROJECT_BEFORE_RUN" = true ]; then
    log_msg "Synchronizing project sources to remote node..."
    ssh "root@$TARGET_NODE" "mkdir -p '$PROJECT_ROOT'"

    # Build rsync exclude args from comma-separated list.
    # Bare names are anchored to repo root (e.g., data -> /data) to prevent
    # accidental exclusions like src/data.
    IFS=',' read -r -a _sync_ex_items <<< "$SYNC_EXCLUDES"
    _rsync_excludes=()
    for _item in "${_sync_ex_items[@]}"; do
        _item="$(echo "$_item" | xargs)"
        [ -n "$_item" ] || continue
        case "$_item" in
            /*|*"*"*|*"?"*)
                _pattern="$_item"
                ;;
            *)
                _pattern="/$_item"
                ;;
        esac
        _rsync_excludes+=("--exclude=$_pattern")
    done

    rsync -az --delete "${_rsync_excludes[@]}" \
        "$LOCAL_PROJECT_ROOT/" "root@$TARGET_NODE:$PROJECT_ROOT/"
fi

log_msg "Copying campaign scripts to deployed node..."
ssh "root@$TARGET_NODE" "mkdir -p '$PROJECT_ROOT/scripts'"

scp "$LOCAL_LAUNCH_SCRIPT" "root@$TARGET_NODE:$REMOTE_LAUNCH_SCRIPT"

for dep in run_thesis_mode.sh run_experiments.sh run_ilp_partition.sh run_ilp_pareto_sweep.sh sanitize_cuda_env.sh; do
    if [ -f "$LOCAL_SCRIPTS_DIR/$dep" ]; then
        scp "$LOCAL_SCRIPTS_DIR/$dep" "root@$TARGET_NODE:$PROJECT_ROOT/scripts/$dep"
    fi
done

log_msg "Ensuring executable permissions and starting thesis campaign..."
ssh "root@$TARGET_NODE" "chmod +x '$PROJECT_ROOT/scripts/'*.sh"
ssh "root@$TARGET_NODE" \
    "cd '$PROJECT_ROOT' && \
     CAMPAIGN_PROFILE='$CAMPAIGN_PROFILE' \
     CONDA_ENV_NAME='$CONDA_ENV_NAME' \
    FULL_SEEDS_CSV='$FULL_SEEDS_CSV' \
    SINGLE_SEED='$SINGLE_SEED' \
    FULL_REPEATS_PER_SEED='$FULL_REPEATS_PER_SEED' \
    NON_FULL_REPEATS='$NON_FULL_REPEATS' \
     RUN_HYBRID='$RUN_HYBRID' \
     '$REMOTE_LAUNCH_SCRIPT'"

log_msg "Grid5000 thesis campaign completed."