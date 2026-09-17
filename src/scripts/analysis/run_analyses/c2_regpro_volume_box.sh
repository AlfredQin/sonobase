#!/bin/bash
# Stage 2 — C2: RegPro prostate volume (box-prompt run).
set -e
source "$(dirname "$(readlink -f "$0")")/_paths.sh"

OUT=${ANALYSIS_OUT_ROOT}/c2_regpro_volume/RegPro_box_0corr
uv run python -m nemo_cv.recipes.analysis.c2_regpro_volume \
  --runs sam2_no_ft=${PRED_ROOT}/sam2_no_ft_RegPro_box_0corr \
         medsam2=${PRED_ROOT}/medsam2_RegPro_box_0corr \
         sonobase=${PRED_ROOT}/sonobase_RegPro_box_0corr \
  --output-dir ${OUT} \
  --regpro-zip ${REGPRO_ZIP}
