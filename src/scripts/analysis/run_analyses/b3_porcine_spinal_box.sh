#!/bin/bash
# Stage 2 — B3: Porcine spinal cord per-class mIoU (box-prompt run).
set -e
source "$(dirname "$(readlink -f "$0")")/_paths.sh"

OUT=${ANALYSIS_OUT_ROOT}/b3_porcine_spinal/PorcineSpinalCord_box_0corr
uv run python -m nemo_cv.recipes.analysis.b3_porcine_spinal \
  --runs sam2_no_ft=${PRED_ROOT}/sam2_no_ft_PorcineSpinalCord_box_0corr \
         medsam2=${PRED_ROOT}/medsam2_PorcineSpinalCord_box_0corr \
         sonobase=${PRED_ROOT}/sonobase_PorcineSpinalCord_box_0corr \
  --output-dir ${OUT}
# NB: b3_porcine_spinal.py reads per-class info (category_id/category_name) directly
# from the Stage-1 prediction dirs; it does NOT take --saus-dir (removed 2026-05-31 —
# the launcher had drifted from the script's CLI).
