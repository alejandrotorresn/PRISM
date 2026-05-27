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

"${PYTHON_BIN}" - <<'PY'
import importlib.util
import pathlib
import sys

root = pathlib.Path.cwd()
print(f"Project root: {root}")
print(f"src package file exists: {(root / 'src' / '__init__.py').exists()}")
print(f"src/data package file exists: {(root / 'src' / 'data' / '__init__.py').exists()}")

src_spec = importlib.util.find_spec("src")
src_data_spec = importlib.util.find_spec("src.data")
data_spec = importlib.util.find_spec("data")

print(f"find_spec('src') -> {None if src_spec is None else src_spec.origin}")
print(f"find_spec('src.data') -> {None if src_data_spec is None else src_data_spec.origin}")
print(f"find_spec('data') -> {None if data_spec is None else data_spec.origin}")

if src_data_spec is None:
  print("ERROR: src.data is not importable. The remote repository is likely stale or incomplete.")
  print("Hint: ensure full project sync to /root/PRISM (not only scripts).")
  sys.exit(2)
PY

"${PYTHON_BIN}" -c "import torch" >/dev/null 2>&1 || {
  echo "ERROR: torch is not available in ${PRISM_ENV_NAME}."
  echo "Install dependencies in ${PRISM_ENV_NAME} before running tests."
  echo "Example: ${PYTHON_BIN} -m pip install -r config/requirements.txt"
  exit 1
}

"${PYTHON_BIN}" -m pytest -q tests
