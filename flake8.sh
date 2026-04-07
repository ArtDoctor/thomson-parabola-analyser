#!/bin/bash
set -euo pipefail

# shellcheck disable=SC1090
source venv/bin/activate

if [ "$#" -eq 0 ]; then
  set -- .
fi
python -m flake8 "$@"
