#!/bin/bash
# Stage 2 — T3.1: Temporal consistency on CAMUS (box-prompt run).
set -e
source "$(dirname "$(readlink -f "$0")")/_paths.sh"

OUT=${ANALYSIS_OUT_ROOT}/t3_1_temporal_consistency/CAMUS_box_0corr
uv run python -m nemo_cv.recipes.analysis.t3_1_temporal_consistency \
  --runs sam2_no_ft=${PRED_ROOT}/sam2_no_ft_CAMUS_box_0corr \
         medsam2=${PRED_ROOT}/medsam2_CAMUS_box_0corr \
         sonobase=${PRED_ROOT}/sonobase_CAMUS_box_0corr \
  --output-dir ${OUT}
