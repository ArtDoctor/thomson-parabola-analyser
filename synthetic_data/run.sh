#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN_PATH="${SCRIPT_DIR}/thomson"
OUTPUT_PATH="${SCRIPT_DIR}/hits.txt"

# If sending to someone, remove -march=native.
g++ -O3 -march=native -fopenmp "${SCRIPT_DIR}/thomson.cpp" -o "${BIN_PATH}"
"${BIN_PATH}" "$@" > "${OUTPUT_PATH}"
rm -f "${BIN_PATH}"
