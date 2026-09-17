#!/bin/bash
# Stage 2 — T2.1: HC → Gestational Age (Hadlock) — box-prompt run.
set -e
source "$(dirname "$(readlink -f "$0")")/_paths.sh"

OUT=${ANALYSIS_OUT_ROOT}/t2_1_hc_ga/HC18_box_0corr
uv run python -m nemo_cv.recipes.analysis.t2_1_hc_ga \
  --runs sam2_no_ft=${PRED_ROOT}/sam2_no_ft_HC18_box_0corr \
         medsam2=${PRED_ROOT}/medsam2_HC18_box_0corr \
         sonobase=${PRED_ROOT}/sonobase_HC18_box_0corr \
  --output-dir ${OUT} \
  --hc18-zip ${HC18_ZIP}
