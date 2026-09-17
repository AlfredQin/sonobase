#!/bin/bash
# Stage 2 — A4: ACOUSLIC abdominal circumference (point-prompt run).
set -e
source "$(dirname "$(readlink -f "$0")")/_paths.sh"

OUT=${ANALYSIS_OUT_ROOT}/a4_acouslic_ac/ACOUSLIC_point_0corr
uv run python -m nemo_cv.recipes.analysis.a4_acouslic_ac \
  --runs sam2_no_ft=${PRED_ROOT}/sam2_no_ft_ACOUSLIC_point_0corr \
         medsam2=${PRED_ROOT}/medsam2_ACOUSLIC_point_0corr \
         sonobase=${PRED_ROOT}/sonobase_ACOUSLIC_point_0corr \
  --output-dir ${OUT} \
  --acouslic-csv ${ACOUSLIC_CSV}
