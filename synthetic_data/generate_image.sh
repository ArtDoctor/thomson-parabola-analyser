#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PY="${REPO_ROOT}/venv/bin/python"

if [ "$#" -gt 0 ]; then
  PARTICLE_ARGS=("$@")
else
  PARTICLE_ARGS=(
    -particle H0
    -particle H1
    -particle C1
    -particle C2
    -particle C3
    -particle C4
  )
fi

"${SCRIPT_DIR}/run_optimized.sh" "${PARTICLE_ARGS[@]}"
cd "${REPO_ROOT}"
"${PY}" -m synthetic_data.utils.hits_to_imgs_script
