#!/usr/bin/env bash
set -Eeuo pipefail

# ==============================================================================
# SLURM CAMPAIGN LAUNCHER (node-side)
# Runs full thesis campaign through scripts/run_thesis_mode.sh with
# Slurm-aware environment activation and hardware-aware CPU tuning.
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"

MODULE_NAME="${MODULE_NAME:-Analytics/anaconda3}"
CONDA_EXE="${CONDA_EXE:-/opt/ohpc/pub/Analytics/anaconda3/bin/conda}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-prism_env}"
GPU_VRAM_FALLBACK_MB="${GPU_VRAM_FALLBACK_MB:-40960}"
CAMPAIGN_PROFILE="${CAMPAIGN_PROFILE:-doctoral_full}"
FULL_SEEDS_CSV="${FULL_SEEDS_CSV:-42,43,44}"
SINGLE_SEED="${SINGLE_SEED:-42}"
FULL_REPEATS_PER_SEED="${FULL_REPEATS_PER_SEED:-2}"
NON_FULL_REPEATS="${NON_FULL_REPEATS:-1}"
DATA_MOUNT_SRC="${DATA_MOUNT_SRC:-$PROJECT_ROOT/data}"
DATA_LINK="${DATA_LINK:-$PROJECT_ROOT/data}"
LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/logs/slurm}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/slurm_launch_${SLURM_JOB_ID:-adhoc}_$(date +%Y%m%d_%H%M%S).log}"

mkdir -p "$LOG_DIR"

log_msg() {
    local msg="$1"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $msg" | tee -a "$LOG_FILE"
}

on_error() {
    local exit_code=$?
    log_msg "ERROR: campaign aborted (exit_code=$exit_code, line=${BASH_LINENO[0]})."
    exit "$exit_code"
}

trap on_error ERR

activate_conda_env() {
    if command -v module >/dev/null 2>&1; then
        module load "$MODULE_NAME"
    elif [ -f /etc/profile.d/modules.sh ]; then
        # shellcheck disable=SC1091
        source /etc/profile.d/modules.sh
        module load "$MODULE_NAME"
    fi

    local candidates=(
        "/opt/ohpc/pub/Analytics/anaconda3/etc/profile.d/conda.sh"
        "/root/miniconda3/etc/profile.d/conda.sh"
        "/opt/conda/etc/profile.d/conda.sh"
        "$HOME/miniconda3/etc/profile.d/conda.sh"
        "$HOME/anaconda3/etc/profile.d/conda.sh"
    )

    if [ -x "$CONDA_EXE" ]; then
        eval "$("$CONDA_EXE" shell.bash hook)"
    elif command -v conda >/dev/null 2>&1; then
        eval "$(conda shell.bash hook)"
    else
        local candidate
        for candidate in "${candidates[@]}"; do
            if [ -f "$candidate" ]; then
                # shellcheck disable=SC1090
                source "$candidate"
                break
            fi
        done
    fi

    if ! command -v conda >/dev/null 2>&1; then
        log_msg "ERROR: conda was not found after loading $MODULE_NAME and probing CONDA_EXE=$CONDA_EXE."
        exit 1
    fi

    local had_nounset=0
    if [[ "$-" == *u* ]]; then
        had_nounset=1
        set +u
    fi

    if ! conda activate "$CONDA_ENV_NAME" 2>/dev/null; then
        log_msg "ERROR: unable to activate requested conda environment: $CONDA_ENV_NAME"
        exit 1
    fi

    if [ "$had_nounset" -eq 1 ]; then
        set -u
    fi

    log_msg "Module loaded: $MODULE_NAME"
    log_msg "Conda executable: $(command -v conda)"
    log_msg "Conda environment activated: $CONDA_ENV_NAME"
}

prepare_storage() {
    if [ "$DATA_LINK" != "$DATA_MOUNT_SRC" ]; then
        mkdir -p "$DATA_MOUNT_SRC"

        if [ -L "$DATA_LINK" ]; then
            local current_target
            current_target="$(readlink "$DATA_LINK")"
            if [ "$current_target" != "$DATA_MOUNT_SRC" ]; then
                rm -f "$DATA_LINK"
                ln -s "$DATA_MOUNT_SRC" "$DATA_LINK"
            fi
        elif [ -e "$DATA_LINK" ]; then
            if [ -d "$DATA_LINK" ]; then
                local first_entry
                first_entry="$(find "$DATA_LINK" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null || true)"
                if [ -n "$first_entry" ]; then
                    log_msg "Migrating existing data directory into Slurm-backed storage: $DATA_LINK -> $DATA_MOUNT_SRC"
                    cp -r "$DATA_LINK"/. "$DATA_MOUNT_SRC"/
                fi
                rm -rf "$DATA_LINK"
                ln -s "$DATA_MOUNT_SRC" "$DATA_LINK"
            else
                log_msg "WARNING: $DATA_LINK exists and is not a directory/symlink. Keeping it as-is."
            fi
        else
            ln -s "$DATA_MOUNT_SRC" "$DATA_LINK"
        fi
    else
        mkdir -p "$DATA_LINK"
    fi

    mkdir -p "$PROJECT_ROOT/logs"
    log_msg "Data path ready: $DATA_LINK"
}

configure_cpu_runtime() {
    log_msg "Starting hardware autodetection..."

    local cpu_vendor
    local model_name
    local phy_cores
    local allocated_cpus
    local target_threads

    cpu_vendor="$(lscpu | grep -m1 'Vendor ID' | awk -F ':' '{print $2}' | xargs || true)"
    model_name="$(lscpu | grep -m1 'Model name' | awk -F ':' '{print $2}' | xargs || true)"
    phy_cores="$(lscpu -b -p=Core,Socket | grep -v '^#' | sort -u | wc -l | xargs || true)"
    allocated_cpus="${SLURM_CPUS_PER_TASK:-${SLURM_JOB_CPUS_PER_NODE:-}}"
    allocated_cpus="${allocated_cpus%%(*}"
    allocated_cpus="$(echo "$allocated_cpus" | tr ',' '\n' | head -n1 | xargs || true)"

    if ! [[ "$phy_cores" =~ ^[0-9]+$ ]] || [ "$phy_cores" -le 0 ]; then
        phy_cores=1
    fi

    if [ "${FORCE_THREADS:-0}" -gt 0 ] 2>/dev/null; then
        target_threads="$FORCE_THREADS"
    elif [[ "$allocated_cpus" =~ ^[0-9]+$ ]] && [ "$allocated_cpus" -gt 0 ]; then
        if [ "$allocated_cpus" -lt "$phy_cores" ]; then
            target_threads="$allocated_cpus"
        else
            target_threads="$phy_cores"
        fi
    else
        target_threads="$phy_cores"
    fi

    export OMP_NUM_THREADS="$target_threads"
    export MKL_NUM_THREADS="$target_threads"
    export TORCH_NUM_THREADS="$target_threads"

    if [[ "$cpu_vendor" == "AuthenticAMD" || "$cpu_vendor" == "AMD" ]]; then
        export MKL_DEBUG_CPU_TYPE=5
        export OMP_BIND_PROC=true
        export OMP_PLACES=cores
        export OMP_PROC_BIND=close
        log_msg "CPU mode: AMD optimized"
    elif [[ "$cpu_vendor" == "GenuineIntel" ]]; then
        export KMP_AFFINITY=granularity=fine,compact,1,0
        export KMP_BLOCKTIME=0
        log_msg "CPU mode: Intel optimized"
    else
        log_msg "CPU vendor not recognized. Using generic affinity settings."
    fi

    log_msg "CPU: ${model_name:-unknown} (${cpu_vendor:-unknown}), physical_cores=$phy_cores, allocated_cpus=${allocated_cpus:-unknown}, threads=$target_threads"
}

run_campaign() {
    if [ ! -f "$PROJECT_ROOT/scripts/run_thesis_mode.sh" ]; then
        log_msg "ERROR: missing script: $PROJECT_ROOT/scripts/run_thesis_mode.sh"
        exit 1
    fi

    local host_tag
    local base_output_root
    local reports_root
    local run_namespace
    local repeats_for_run
    local seed

    host_tag="${HOST_TAG:-$(hostname)}"
    run_namespace="${RUN_NAMESPACE:-job_${SLURM_JOB_ID:-adhoc}_$(date +%Y%m%d_%H%M%S)}"
    base_output_root="${BASE_OUTPUT_DIR:-$DATA_LINK/$host_tag/thesis_runs/$run_namespace}"
    reports_root="${REPORTS_DIR:-$PROJECT_ROOT/reports/ilp_results/slurm_${host_tag}_thesis_mode}"

    log_msg "Run namespace: $run_namespace"
    log_msg "Base output root: $base_output_root"

    run_single_seed() {
        local campaign_seed="$1"
        local run_repeats="$2"
        local seed_output_dir
        local seed_reports_dir

        seed_output_dir="$base_output_root/$CAMPAIGN_PROFILE/seed_${campaign_seed}"
        seed_reports_dir="$reports_root/$CAMPAIGN_PROFILE/seed_${campaign_seed}"

        log_msg "Launching profile=$CAMPAIGN_PROFILE seed=$campaign_seed repeats=$run_repeats"
        log_msg "Output dir: $seed_output_dir"
        log_msg "Reports dir: $seed_reports_dir"

        (
            cd "$PROJECT_ROOT"

            if [ -f "scripts/sanitize_cuda_env.sh" ]; then
                # shellcheck disable=SC1091
                source "scripts/sanitize_cuda_env.sh"
                sanitize_cuda_runtime_env
            fi

            PROFILE="$CAMPAIGN_PROFILE" \
            HOST_TAG="$host_tag" \
            SEED_BASE="$campaign_seed" \
            REPEATS="$run_repeats" \
            BASE_OUTPUT_DIR="$seed_output_dir" \
            REPORTS_DIR="$seed_reports_dir" \
            LOG_DIR="$PROJECT_ROOT/logs/thesis_mode" \
            DATASETS_DIR="${DATASETS_DIR:-$PROJECT_ROOT/datasets}" \
            GPU_VRAM_FALLBACK_MB="$GPU_VRAM_FALLBACK_MB" \
            DOWNLOAD_DATASETS="${DOWNLOAD_DATASETS:-true}" \
            RUN_PROFILING="${RUN_PROFILING:-true}" \
            RUN_ILP="${RUN_ILP:-true}" \
            RUN_HYBRID="${RUN_HYBRID:-true}" \
            RUN_REPORTS="${RUN_REPORTS:-true}" \
            FAIL_FAST="${FAIL_FAST:-false}" \
            DRY_RUN="${DRY_RUN:-false}" \
            bash "scripts/run_thesis_mode.sh"
        ) | tee -a "$LOG_FILE"
    }

    if [ "$CAMPAIGN_PROFILE" = "doctoral_full" ]; then
        IFS=',' read -r -a full_seeds <<< "$FULL_SEEDS_CSV"
        repeats_for_run="$FULL_REPEATS_PER_SEED"
        if ! [[ "$repeats_for_run" =~ ^[0-9]+$ ]] || [ "$repeats_for_run" -le 0 ]; then
            log_msg "ERROR: FULL_REPEATS_PER_SEED must be a positive integer"
            exit 1
        fi
        if [ "$repeats_for_run" -lt 2 ]; then
            log_msg "WARNING: FULL_REPEATS_PER_SEED=$repeats_for_run is too low for strict official mode; using 2"
            repeats_for_run=2
        fi

        for seed in "${full_seeds[@]}"; do
            seed="$(echo "$seed" | xargs)"
            if ! [[ "$seed" =~ ^[0-9]+$ ]]; then
                log_msg "ERROR: invalid seed in FULL_SEEDS_CSV: '$seed'"
                exit 1
            fi
            run_single_seed "$seed" "$repeats_for_run"
        done
    else
        repeats_for_run="$NON_FULL_REPEATS"
        if ! [[ "$repeats_for_run" =~ ^[0-9]+$ ]] || [ "$repeats_for_run" -le 0 ]; then
            log_msg "ERROR: NON_FULL_REPEATS must be a positive integer"
            exit 1
        fi
        if [ "$CAMPAIGN_PROFILE" = "doctoral_minimal" ] && [ "$repeats_for_run" -lt 2 ]; then
            log_msg "WARNING: NON_FULL_REPEATS=$repeats_for_run is too low for strict official mode; using 2"
            repeats_for_run=2
        fi
        if ! [[ "$SINGLE_SEED" =~ ^[0-9]+$ ]]; then
            log_msg "ERROR: SINGLE_SEED must be a non-negative integer"
            exit 1
        fi
        run_single_seed "$SINGLE_SEED" "$repeats_for_run"
    fi

    log_msg "Campaign finished successfully."
}

main() {
    log_msg "Slurm node launcher started"
    log_msg "Project root: $PROJECT_ROOT"
    log_msg "Slurm job id: ${SLURM_JOB_ID:-unset}"
    log_msg "Node list: ${SLURM_JOB_NODELIST:-unset}"
    log_msg "GPU VRAM fallback: ${GPU_VRAM_FALLBACK_MB} MiB"
    activate_conda_env
    prepare_storage
    configure_cpu_runtime
    run_campaign
}

main "$@"