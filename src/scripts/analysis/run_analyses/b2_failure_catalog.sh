#!/bin/bash
# Stage 2 — B2: catastrophic failure catalog (HC18 + CAMUS unified).
# Run after Stage-1 has produced per_sample_metrics.csv on the relevant datasets.
set -e
source "$(dirname "$(readlink -f "$0")")/_paths.sh"

# B2 expects 3 input dirs (one per model). It walks them for cross-dataset
# rows, so each input dir should contain predictions across whatever
# datasets you want to include in the catalog.

OUT=${ANALYSIS_OUT_ROOT}/b2_failure_catalog/HC18_point_0corr
uv run python -m nemo_cv.recipes.analysis.b2_failure_catalog \
  --sam2 ${PRED_ROOT}/sam2_no_ft_HC18_point_0corr \
  --medsam2 ${PRED_ROOT}/medsam2_HC18_point_0corr \
  --sonobase ${PRED_ROOT}/sonobase_HC18_point_0corr \
  --output-dir ${OUT} \
  --top-k 24 \
  --samples-per-page 6
