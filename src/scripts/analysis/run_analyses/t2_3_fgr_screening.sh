#!/bin/bash
# Stage 2 — T2.3: FGR screening (conditional on ACOUSLIC GA metadata).
# If ACOUSLIC metadata lacks per-video gestational-age, the recipe writes a
# "skipped" stub explaining why and exits 0.
set -e
source "$(dirname "$(readlink -f "$0")")/_paths.sh"

OUT=${ANALYSIS_OUT_ROOT}/t2_3_fgr_screening/ACOUSLIC_point_0corr
uv run python -m nemo_cv.recipes.analysis.t2_3_fgr_screening \
  --runs sam2_no_ft=${PRED_ROOT}/sam2_no_ft_ACOUSLIC_point_0corr \
         medsam2=${PRED_ROOT}/medsam2_ACOUSLIC_point_0corr \
         sonobase=${PRED_ROOT}/sonobase_ACOUSLIC_point_0corr \
  --output-dir ${OUT} \
  --acouslic-csv ${ACOUSLIC_CSV}
