#!/bin/bash
# Physics validation: compare each feature (enabled individually) against baseline
# All runs use default C1 at low energy (0.089 MeV mean)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN_PATH="${SCRIPT_DIR}/thomson_test"

echo "=== Physics Validation ==="
echo "Compiling thomson.cpp..."
g++ -O3 -march=native -fopenmp "${SCRIPT_DIR}/thomson.cpp" -o "${BIN_PATH}"

compute_stats() {
    awk '{
        y=$1; x=$2;
        sy+=y; sx+=x; sy2+=y*y; sx2+=x*x; n++
    } END {
        my=sy/n; mx=sx/n;
        sdy=sqrt(sy2/n - my*my);
        sdx=sqrt(sx2/n - mx*mx);
        printf "%.8f %.8f %.8f %.8f %d", my, sdy, mx, sdx, n
    }' "$1"
}

echo ""
echo "Running baseline (no flags)..."
"${BIN_PATH}" 2>/dev/null > /tmp/phys_baseline.txt
BASE_STATS=$(compute_stats /tmp/phys_baseline.txt)
BASE_MY=$(echo "$BASE_STATS" | awk '{print $1}')
BASE_MX=$(echo "$BASE_STATS" | awk '{print $3}')
echo "  Baseline: meanY=$BASE_MY  meanX=$BASE_MX"

PASS=true

for FLAG in "-relativistic" "-fringe" "-adaptive" "-beam"; do
    NAME=$(echo "$FLAG" | tr -d '-')
    echo ""
    echo "Running $NAME..."
    "${BIN_PATH}" $FLAG 2>/dev/null > /tmp/phys_${NAME}.txt
    STATS=$(compute_stats /tmp/phys_${NAME}.txt)
    MY=$(echo "$STATS" | awk '{print $1}')
    SDY=$(echo "$STATS" | awk '{print $2}')
    MX=$(echo "$STATS" | awk '{print $3}')
    SDX=$(echo "$STATS" | awk '{print $4}')
    N=$(echo "$STATS" | awk '{print $5}')
    echo "  $NAME: meanY=$MY  meanX=$MX  N=$N"

    # Compute relative differences
    DY=$(awk "BEGIN { d=($MY - $BASE_MY) / $BASE_MY * 100; printf \"%.4f\", d }")
    DX=$(awk "BEGIN { d=($MX - $BASE_MX) / $BASE_MX * 100; printf \"%.4f\", d }")
    echo "  Relative diff: dY=${DY}%  dX=${DX}%"

    # Thresholds per feature
    case "$NAME" in
        relativistic) THRESH=0.01 ;;   # < 0.01% at low energy
        fringe)       THRESH=5.0  ;;   # < 5% (smooth edges shift results)
        adaptive)     THRESH=0.01 ;;   # < 0.01% (same DT in field)
        beam)         THRESH=1.0  ;;   # < 1% (mean unchanged, variance up)
    esac

    DY_ABS=$(awk "BEGIN { d=($MY - $BASE_MY) / $BASE_MY * 100; printf \"%.6f\", (d < 0 ? -d : d) }")
    DX_ABS=$(awk "BEGIN { d=($MX - $BASE_MX) / $BASE_MX * 100; printf \"%.6f\", (d < 0 ? -d : d) }")

    if awk "BEGIN { exit !($DY_ABS > $THRESH || $DX_ABS > $THRESH) }"; then
        echo "  FAIL: deviation exceeds ${THRESH}% threshold"
        PASS=false
    else
        echo "  PASS: within ${THRESH}% threshold"
    fi
done

rm -f "${BIN_PATH}"

echo ""
if $PASS; then
    echo "=== ALL PHYSICS TESTS PASSED ==="
    exit 0
else
    echo "=== SOME PHYSICS TESTS FAILED ==="
    exit 1
fi
