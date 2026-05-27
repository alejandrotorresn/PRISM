#!/usr/bin/env bash
set -Eeuo pipefail

# OAR submission wrapper for Grid5000 final thesis campaigns.
# Submit with: oarsub -S ./scripts/run_thesis.sh

#OAR -n PRISM_profiling
#OAR -q abaca
#OAR -p musa
#OAR -t deploy
#OAR -l nodes=1,walltime=96:00:00
#OAR -O prism_job.%jobid%.output
#OAR -E prism_job.%jobid%.error


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PROJECT_ROOT="${PROJECT_ROOT:-/root/PRISM}"
REMOTE_LAUNCH_SCRIPT="${REMOTE_LAUNCH_SCRIPT:-$PROJECT_ROOT/scripts/launch_grid5k.sh}"
LOCAL_PROJECT_ROOT="${LOCAL_PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
LOCAL_LAUNCH_SCRIPT="${LOCAL_LAUNCH_SCRIPT:-$LOCAL_PROJECT_ROOT/scripts/launch_grid5k.sh}"
LOCAL_SCRIPTS_DIR="${LOCAL_SCRIPTS_DIR:-$LOCAL_PROJECT_ROOT/scripts}"
SYNC_PROJECT_BEFORE_RUN="${SYNC_PROJECT_BEFORE_RUN:-true}"
# IMPORTANT: defaults are anchored to repository root to avoid excluding src/data.
SYNC_EXCLUDES="${SYNC_EXCLUDES:-/.git,/.venv,/logs,/reports,/data,/datasets,/books,/paper_thesis,/papers}"
KADEPLOY_FILE="${KADEPLOY_FILE:-rocky9_profiling.yaml}"
KADEPLOY_HOME="${KADEPLOY_HOME:-/home/ltorresnino}"

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

resolve_local_path() {
    local candidate="$1"
    if [ -z "$candidate" ]; then
        return 1
    fi
    if [ -f "$candidate" ]; then
        printf '%s' "$candidate"
        return 0
    fi
    if [ -f "$LOCAL_PROJECT_ROOT/$candidate" ]; then
        printf '%s' "$LOCAL_PROJECT_ROOT/$candidate"
        return 0
    fi
    if [ -f "$SCRIPT_DIR/$candidate" ]; then
        printf '%s' "$SCRIPT_DIR/$candidate"
        return 0
    fi
    return 1
}

resolve_kadeploy_path() {
    local candidate="$1"
    if [ -z "$candidate" ]; then
        return 1
    fi

    # If absolute path is provided, use it as-is.
    if [[ "$candidate" = /* ]]; then
        [ -f "$candidate" ] || return 1
        printf '%s' "$candidate"
        return 0
    fi

    # Relative KADEPLOY paths are always resolved from KADEPLOY_HOME.
    if [ -f "$KADEPLOY_HOME/$candidate" ]; then
        printf '%s' "$KADEPLOY_HOME/$candidate"
        return 0
    fi

    return 1
}

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

if [ ! -f "$LOCAL_PROJECT_ROOT/src/data/__init__.py" ]; then
    log_msg "ERROR: LOCAL_PROJECT_ROOT does not look like PRISM root: $LOCAL_PROJECT_ROOT"
    log_msg "ERROR: Missing file: $LOCAL_PROJECT_ROOT/src/data/__init__.py"
    exit 1
fi

if ! RESOLVED_KADEPLOY_FILE="$(resolve_kadeploy_path "$KADEPLOY_FILE")"; then
    log_msg "ERROR: KADEPLOY file not found: $KADEPLOY_FILE"
    log_msg "Hint: KADEPLOY files are resolved from KADEPLOY_HOME=$KADEPLOY_HOME"
    log_msg "Example: KADEPLOY_FILE=PRISM/rocky9_profiling.yaml"
    exit 1
fi

if ! RESOLVED_LOCAL_LAUNCH_SCRIPT="$(resolve_local_path "$LOCAL_LAUNCH_SCRIPT")"; then
    log_msg "ERROR: launch script not found: $LOCAL_LAUNCH_SCRIPT"
    exit 1
fi

log_msg "Starting job ${OAR_JOB_ID:-unknown} on node: $TARGET_NODE"
log_msg "Deploying image with kadeploy3: $RESOLVED_KADEPLOY_FILE"

# Deploy image on the reserved nodes and copy SSH key for root access.
kadeploy3 -f "$OAR_FILE_NODES" -a "$RESOLVED_KADEPLOY_FILE" -k

if [ "$SYNC_PROJECT_BEFORE_RUN" = true ]; then
    log_msg "Synchronizing project sources to remote node..."
    if [ ! -f "$LOCAL_PROJECT_ROOT/src/data/__init__.py" ]; then
        log_msg "ERROR: Local project root seems incorrect: $LOCAL_PROJECT_ROOT"
        log_msg "ERROR: Missing required file: $LOCAL_PROJECT_ROOT/src/data/__init__.py"
        log_msg "Hint: set LOCAL_PROJECT_ROOT to the repository root before oarsub."
        exit 1
    fi

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

    if ! ssh "root@$TARGET_NODE" "test -f '$PROJECT_ROOT/src/data/__init__.py'"; then
        log_msg "ERROR: Remote sync incomplete: missing $PROJECT_ROOT/src/data/__init__.py"
        log_msg "ERROR: Aborting before campaign launch to avoid import failures."
        exit 1
    fi
fi

log_msg "Copying campaign scripts to deployed node..."
ssh "root@$TARGET_NODE" "mkdir -p '$PROJECT_ROOT/scripts'"

scp "$RESOLVED_LOCAL_LAUNCH_SCRIPT" "root@$TARGET_NODE:$REMOTE_LAUNCH_SCRIPT"

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