#!/bin/bash
# Stage 2 — A2 (box prompt). Mirrors a2_hc18_hc_point.sh.
set -e
source "$(dirname "$(readlink -f "$0")")/_paths.sh"

OUT=${ANALYSIS_OUT_ROOT}/a2_hc18_hc/HC18_box_0corr
uv run python -m nemo_cv.recipes.analysis.a2_hc18_hc \
  --runs sam2_no_ft=${PRED_ROOT}/sam2_no_ft_HC18_box_0corr \
         medsam2=${PRED_ROOT}/medsam2_HC18_box_0corr \
         sonobase=${PRED_ROOT}/sonobase_HC18_box_0corr \
  --output-dir ${OUT} \
  --hc18-zip ${HC18_ZIP} \
  --include-gt-fit
