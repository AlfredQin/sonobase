#!/bin/bash
# Stage 2 — A2: HC18 head circumference measurement error analysis (CPU-only).
#
# Consumes the three Stage-1 prediction directories produced by
# save_*_HC18_point.sh and writes per-sample CSV + analysis report + plots.

set -e
source "$(dirname "$(readlink -f "$0")")/_paths.sh"

: "${HC18_ZIP:?HC18_ZIP must be set, e.g. export HC18_ZIP=\$WORK/Dataset/UltraSound/Raw/HC.zip}"

OUT=${ANALYSIS_OUT_ROOT}/a2_hc18_hc/HC18_point_0corr

uv run python -m nemo_cv.recipes.analysis.a2_hc18_hc \
  --runs sam2_no_ft=${PRED_ROOT}/sam2_no_ft_HC18_point_0corr \
         medsam2=${PRED_ROOT}/medsam2_HC18_point_0corr \
         sonobase=${PRED_ROOT}/sonobase_HC18_point_0corr \
  --output-dir ${OUT} \
  --hc18-zip "${HC18_ZIP}" \
  --include-gt-fit
