#!/bin/bash
# Stage 2 — S2: CAMUS by Image Quality (Good / Medium / Poor).
# Reads 6 Stage-1 CAMUS prediction dirs (3 models × 2 prompts).
set -e
source "$(dirname "$(readlink -f "$0")")/_paths.sh"

OUT=${ANALYSIS_OUT_ROOT}/s2_camus_quality/CAMUS_0corr
uv run python -m nemo_cv.recipes.analysis.s2_camus_quality \
  --runs sonobase_point=${PRED_ROOT}/sonobase_CAMUS_point_0corr \
         sonobase_box=${PRED_ROOT}/sonobase_CAMUS_box_0corr \
         medsam2_point=${PRED_ROOT}/medsam2_CAMUS_point_0corr \
         medsam2_box=${PRED_ROOT}/medsam2_CAMUS_box_0corr \
         sam2_no_ft_point=${PRED_ROOT}/sam2_no_ft_CAMUS_point_0corr \
         sam2_no_ft_box=${PRED_ROOT}/sam2_no_ft_CAMUS_box_0corr \
  --output-dir ${OUT} \
  --camus-zip ${CAMUS_ZIP}
