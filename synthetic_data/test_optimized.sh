#!/bin/bash
# Compare thomson_optimized.cpp (optimized) against thomson.cpp (original)
# for multiple configurations. Both use deterministic RNG seeds.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ORIG_BIN="${SCRIPT_DIR}/thomson_orig_bin"
OPT_BIN="${SCRIPT_DIR}/thomson_opt_bin"

echo "=== Optimized vs Original Comparison ==="
echo "Compiling thomson.cpp (original)..."
g++ -O3 -march=native -fopenmp "${SCRIPT_DIR}/thomson.cpp" -o "${ORIG_BIN}"

echo "Compiling thomson_optimized.cpp (optimized)..."
g++ -O3 -march=native -fopenmp "${SCRIPT_DIR}/thomson_optimized.cpp" -o "${OPT_BIN}"

compute_stats() {
    awk '{
        y=$1; x=$2;
        sy+=y; sx+=x; sy2+=y*y; sx2+=x*x; n++
    } END {
        my=sy/n; mx=sx/n;
        sdy=sqrt(sy2/n - my*my);
        sdx=sqrt(sx2/n - mx*mx);
        printf "%.10f %.10f %.10f %.10f %d", my, sdy, mx, sdx, n
    }' "$1"
}

PASS=true

run_comparison() {
    local NAME="$1"
    shift
    local FLAGS="$@"

    echo ""
    echo "--- $NAME (flags: ${FLAGS:-none}) ---"

    echo "  Running original..."
    "${ORIG_BIN}" $FLAGS 2>/dev/null > /tmp/test_orig_${NAME}.txt

    echo "  Running optimized..."
    # Capture stderr for timing info
    "${OPT_BIN}" $FLAGS 2>/tmp/test_opt_stderr_${NAME}.txt > /tmp/test_opt_${NAME}.txt

    ORIG_STATS=$(compute_stats /tmp/test_orig_${NAME}.txt)
    OPT_STATS=$(compute_stats /tmp/test_opt_${NAME}.txt)

    ORIG_MY=$(echo "$ORIG_STATS" | awk '{print $1}')
    ORIG_SDY=$(echo "$ORIG_STATS" | awk '{print $2}')
    ORIG_MX=$(echo "$ORIG_STATS" | awk '{print $3}')
    ORIG_SDX=$(echo "$ORIG_STATS" | awk '{print $4}')
    ORIG_N=$(echo "$ORIG_STATS" | awk '{print $5}')

    OPT_MY=$(echo "$OPT_STATS" | awk '{print $1}')
    OPT_SDY=$(echo "$OPT_STATS" | awk '{print $2}')
    OPT_MX=$(echo "$OPT_STATS" | awk '{print $3}')
    OPT_SDX=$(echo "$OPT_STATS" | awk '{print $4}')
    OPT_N=$(echo "$OPT_STATS" | awk '{print $5}')

    echo "  Original:  meanY=$ORIG_MY  sdY=$ORIG_SDY  meanX=$ORIG_MX  sdX=$ORIG_SDX  N=$ORIG_N"
    echo "  Optimized: meanY=$OPT_MY   sdY=$OPT_SDY   meanX=$OPT_MX   sdX=$OPT_SDX   N=$OPT_N"

    # Compute relative differences
    DY=$(awk "BEGIN { d=($OPT_MY - $ORIG_MY) / $ORIG_MY * 100; printf \"%.6f\", d }")
    DX=$(awk "BEGIN { d=($OPT_MX - $ORIG_MX) / $ORIG_MX * 100; printf \"%.6f\", d }")
    DY_ABS=$(awk "BEGIN { d=($OPT_MY - $ORIG_MY) / $ORIG_MY * 100; printf \"%.6f\", (d < 0 ? -d : d) }")
    DX_ABS=$(awk "BEGIN { d=($OPT_MX - $ORIG_MX) / $ORIG_MX * 100; printf \"%.6f\", (d < 0 ? -d : d) }")

    echo "  Relative diff: dY=${DY}%  dX=${DX}%"

    # Show timing from optimized stderr
    OPT_TIME=$(grep "Elapsed:" /tmp/test_opt_stderr_${NAME}.txt | awk '{print $2}')
    if [ -n "$OPT_TIME" ]; then
        echo "  Optimized elapsed: ${OPT_TIME}s"
    fi

    # Check N matches
    if [ "$ORIG_N" != "$OPT_N" ]; then
        echo "  WARNING: Hit count differs! Original=$ORIG_N, Optimized=$OPT_N"
    fi

    # Threshold depends on configuration name
    local THRESH=0.01
    case "$NAME" in
        beam)    THRESH=1.0 ;;
        fringe)  THRESH=5.0 ;;
        *)       THRESH=0.01 ;;
    esac

    if awk "BEGIN { exit !($DY_ABS > $THRESH || $DX_ABS > $THRESH) }"; then
        echo "  FAIL: deviation exceeds ${THRESH}% threshold"
        PASS=false
    else
        echo "  PASS: within ${THRESH}% threshold"
    fi
}

# Test 1: Baseline (no flags)
run_comparison "baseline"

# Test 2: Relativistic
run_comparison "relativistic" "-relativistic"

# Test 3: Adaptive
run_comparison "adaptive" "-adaptive"

# Test 4: Beam
run_comparison "beam" "-beam"

# Test 5: Fringe (falls back to Boris, should be near-identical)
run_comparison "fringe" "-fringe"

# Test 6: Multi-particle
run_comparison "multi" "-particle C4 -particle H"

# Timing comparison: run both with timing
echo ""
echo "=== Performance Comparison (baseline, no flags) ==="

echo "  Timing original (3 runs)..."
for i in 1 2 3; do
    T_START=$(date +%s%N)
    "${ORIG_BIN}" 2>/dev/null > /dev/null
    T_END=$(date +%s%N)
    ELAPSED=$(awk "BEGIN { printf \"%.3f\", ($T_END - $T_START) / 1000000000 }")
    echo "    Run $i: ${ELAPSED}s"
done

echo "  Timing optimized (3 runs)..."
for i in 1 2 3; do
    "${OPT_BIN}" 2>/tmp/timing_opt_${i}.txt > /dev/null
    OPT_TIME=$(grep "Elapsed:" /tmp/timing_opt_${i}.txt | awk '{print $2}')
    echo "    Run $i: ${OPT_TIME}s"
done

rm -f "${ORIG_BIN}" "${OPT_BIN}"

echo ""
if $PASS; then
    echo "=== ALL OPTIMIZED TESTS PASSED ==="
    exit 0
else
    echo "=== SOME OPTIMIZED TESTS FAILED ==="
    exit 1
fi
