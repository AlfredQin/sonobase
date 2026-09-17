#!/bin/bash
# Stage 2 — B1: CAMUS per-structure mIoU/Dice (box-prompt run).
set -e
source "$(dirname "$(readlink -f "$0")")/_paths.sh"

OUT=${ANALYSIS_OUT_ROOT}/b1_camus_per_structure/CAMUS_box_0corr
uv run python -m nemo_cv.recipes.analysis.b1_camus_per_structure \
  --runs sam2_no_ft=${PRED_ROOT}/sam2_no_ft_CAMUS_box_0corr \
         medsam2=${PRED_ROOT}/medsam2_CAMUS_box_0corr \
         sonobase=${PRED_ROOT}/sonobase_CAMUS_box_0corr \
  --output-dir ${OUT}
