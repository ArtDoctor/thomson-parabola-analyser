#!/bin/bash
# Regression test: verify the default thomson.cpp run is deterministic.
# OpenMP can reorder lines, so we sort both runs before comparison.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN_PATH="${SCRIPT_DIR}/thomson_regression_bin"

echo "=== Regression Test ==="
echo "Compiling thomson.cpp..."
g++ -O3 -march=native -fopenmp "${SCRIPT_DIR}/thomson.cpp" -o "${BIN_PATH}"

echo "Running baseline #1..."
"${BIN_PATH}" > /tmp/hits_run_1.txt 2>/dev/null

echo "Running baseline #2..."
"${BIN_PATH}" > /tmp/hits_run_2.txt 2>/dev/null

# Sort both (row order varies due to OpenMP thread scheduling)
sort /tmp/hits_run_1.txt > /tmp/hits_run_1_sorted.txt
sort /tmp/hits_run_2.txt > /tmp/hits_run_2_sorted.txt

echo "Comparing sorted outputs..."
DIFF_LINES=$(diff /tmp/hits_run_1_sorted.txt /tmp/hits_run_2_sorted.txt | wc -l)

rm -f "${BIN_PATH}"

if [ "$DIFF_LINES" -eq 0 ]; then
    echo "PASS: Default runs are deterministic (0 diff lines)"
    exit 0
else
    echo "FAIL: $DIFF_LINES diff lines found"
    diff /tmp/hits_run_1_sorted.txt /tmp/hits_run_2_sorted.txt | head -20
    exit 1
fi
