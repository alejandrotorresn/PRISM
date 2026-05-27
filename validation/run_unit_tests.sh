#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PRISM_ENV_NAME="prism_env"
CONDA_PYTHON="${HOME}/anaconda3/envs/${PRISM_ENV_NAME}/bin/python"

if [[ -x "${CONDA_PYTHON}" ]]; then
  PYTHON_BIN="${CONDA_PYTHON}"
else
  echo "ERROR: ${CONDA_PYTHON} not found."
  echo "This project requires the conda environment '${PRISM_ENV_NAME}'."
  echo "Create it with: conda env create -f config/environment.yml"
  exit 1
fi

echo "================================================================================"
echo "RUNNING UNIT TESTS (pytest)"
echo "================================================================================"
echo "Using Python: ${PYTHON_BIN}"
echo ""

cd "${ROOT_DIR}"

# Ensure both import styles are resolvable in heterogeneous environments:
# - src.* imports require project root on PYTHONPATH
# - legacy data.* / models.* imports require src/ on PYTHONPATH
export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
echo "Using PYTHONPATH: ${PYTHONPATH}"

"${PYTHON_BIN}" -c "import torch" >/dev/null 2>&1 || {
  echo "ERROR: torch is not available in ${PRISM_ENV_NAME}."
  echo "Install dependencies in ${PRISM_ENV_NAME} before running tests."
  echo "Example: ${PYTHON_BIN} -m pip install -r config/requirements.txt"
  exit 1
}

"${PYTHON_BIN}" -m pytest -q tests
