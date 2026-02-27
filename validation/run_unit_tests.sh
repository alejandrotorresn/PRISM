#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -x "${ROOT_DIR}/.venv/bin/python" ]]; then
  PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
else
  PYTHON_BIN="python"
fi

echo "================================================================================"
echo "RUNNING UNIT TESTS (pytest)"
echo "================================================================================"
echo "Using Python: ${PYTHON_BIN}"
echo ""

cd "${ROOT_DIR}"
"${PYTHON_BIN}" -m pytest -q tests
