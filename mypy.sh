#!/bin/bash
set -euo pipefail

# shellcheck disable=SC1090
source venv/bin/activate

if [ "$#" -eq 0 ]; then
  set -- main.py oblisk tests
fi
python -m mypy "$@"
