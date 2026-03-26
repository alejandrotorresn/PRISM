#!/bin/bash

sanitize_cuda_runtime_env() {
  local original_ld="${LD_LIBRARY_PATH:-}"
  local sanitized_parts=()
  local removed_parts=()
  local part

  if [ -n "$original_ld" ]; then
    IFS=':' read -r -a __cuda_ld_parts <<< "$original_ld"
    for part in "${__cuda_ld_parts[@]}"; do
      [ -n "$part" ] || continue
      if [[ "$part" == */stubs ]] || [[ "$part" == */stubs/* ]]; then
        removed_parts+=("$part")
        continue
      fi
      sanitized_parts+=("$part")
    done

    if [ ${#sanitized_parts[@]} -gt 0 ]; then
      export LD_LIBRARY_PATH="$(IFS=:; printf '%s' "${sanitized_parts[*]}")"
    else
      unset LD_LIBRARY_PATH
    fi
  fi

  if [ "${CUDA_HOME:-}" = "/usr/local/cuda" ]; then
    unset CUDA_HOME
  fi

  if [ "${CUDA_PATH:-}" = "/usr/local/cuda" ]; then
    unset CUDA_PATH
  fi

  if [ ${#removed_parts[@]} -gt 0 ]; then
    printf '[cuda-env] Removed stub CUDA library paths from LD_LIBRARY_PATH: %s\n' "$(IFS=:; printf '%s' "${removed_parts[*]}")" >&2
  fi
}