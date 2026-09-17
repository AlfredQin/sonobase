#!/bin/bash
# Stage 2 — B2: catastrophic failure catalog (box-prompt run on HC18).
# Run after Stage-1 has produced per_sample_metrics.csv on the relevant datasets.
set -e
source "$(dirname "$(readlink -f "$0")")/_paths.sh"

OUT=${ANALYSIS_OUT_ROOT}/b2_failure_catalog/HC18_box_0corr
uv run python -m nemo_cv.recipes.analysis.b2_failure_catalog \
  --sam2 ${PRED_ROOT}/sam2_no_ft_HC18_box_0corr \
  --medsam2 ${PRED_ROOT}/medsam2_HC18_box_0corr \
  --sonobase ${PRED_ROOT}/sonobase_HC18_box_0corr \
  --output-dir ${OUT} \
  --top-k 24 \
  --samples-per-page 6
