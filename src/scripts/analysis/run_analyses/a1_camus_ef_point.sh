#!/bin/bash
# Stage 2 — A1: CAMUS ejection fraction (point-prompt run).
set -e
source "$(dirname "$(readlink -f "$0")")/_paths.sh"

OUT=${ANALYSIS_OUT_ROOT}/a1_camus_ef/CAMUS_point_0corr
uv run python -m nemo_cv.recipes.analysis.a1_camus_ef \
  --runs sam2_no_ft=${PRED_ROOT}/sam2_no_ft_CAMUS_point_0corr \
         medsam2=${PRED_ROOT}/medsam2_CAMUS_point_0corr \
         sonobase=${PRED_ROOT}/sonobase_CAMUS_point_0corr \
  --output-dir ${OUT} \
  --camus-zip ${CAMUS_ZIP} \
  --include-gt-fit
