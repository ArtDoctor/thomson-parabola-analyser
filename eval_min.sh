#!/bin/bash
# Batch-run the CLI on all eval_min TIFFs and print aggregate metrics.
# Usage: ./eval_min.sh [output_dir]
#   output_dir defaults to outputs_eval_min (gitignored via outputs/).

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
source venv/bin/activate

OUT="${1:-outputs_eval_min}"
shopt -s nullglob
imgs=(eval_min/*.tif)
if [[ ${#imgs[@]} -eq 0 ]]; then
  echo "No images found under eval_min/*.tif" >&2
  exit 1
fi

python main.py "${imgs[@]}" -o "$OUT"
python evaluate_results.py "$OUT"
echo ""
echo "Batch markdown summary: ${OUT}/summary.md"
