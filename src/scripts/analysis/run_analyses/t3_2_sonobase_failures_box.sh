#!/bin/bash
# Stage 2 — T3.2: SonoBase failure analysis (box-prompt run on HC18; mirrors B2 inverted).
set -e
source "$(dirname "$(readlink -f "$0")")/_paths.sh"

OUT=${ANALYSIS_OUT_ROOT}/t3_2_sonobase_failures/HC18_box_0corr
uv run python -m nemo_cv.recipes.analysis.t3_2_sonobase_failures \
  --sam2 ${PRED_ROOT}/sam2_no_ft_HC18_box_0corr \
  --medsam2 ${PRED_ROOT}/medsam2_HC18_box_0corr \
  --sonobase ${PRED_ROOT}/sonobase_HC18_box_0corr \
  --output-dir ${OUT} \
  --per-dataset-top-k 5 \
  --samples-per-page 6
