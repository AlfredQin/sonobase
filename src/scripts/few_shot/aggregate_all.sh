#!/bin/bash
# Build per-dataset summaries → globally-corrected combined CSV → figures.
# Run after all 3 dataset sweeps complete.

set -e
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
export PYTHONPATH=$PYTHONPATH:$(pwd)

RESULTS_DIR=${RESULTS_DIR:-./experiments/few_shot/few_shot_results}

for DS in ACOUSLIC DDTI FUGC; do
  echo "=== aggregate $DS ==="
  uv run python -m nemo_cv.recipes.few_shot.aggregate \
    --dataset "$DS" \
    --output-dir "$RESULTS_DIR" \
    --predictions-root ./experiments/few_shot/predictions
done

echo ""
echo "=== apply BH-FDR globally + write combined_summary.csv ==="
uv run python -m nemo_cv.recipes.few_shot.apply_fdr_correction \
  --results-dir "$RESULTS_DIR"

echo ""
echo "=== render figures ==="
uv run python -m nemo_cv.recipes.few_shot.build_figures \
  --results-dir "$RESULTS_DIR"

echo ""
echo "Few-shot aggregation complete. Outputs in $RESULTS_DIR"
