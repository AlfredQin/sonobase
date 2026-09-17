#!/bin/bash
# Stage 2 — T1.2: composite Bland-Altman figures.
set -e
source "$(dirname "$(readlink -f "$0")")/_paths.sh"

OUT=${ANALYSIS_OUT_ROOT}/t1_2_bland_altman/point
A1_CSV=${ANALYSIS_OUT_ROOT}/a1_camus_ef/CAMUS_point_0corr/per_patient.csv
A2_CSV=${ANALYSIS_OUT_ROOT}/a2_hc18_hc/HC18_point_0corr/per_sample.csv
A4_CSV=${ANALYSIS_OUT_ROOT}/a4_acouslic_ac/ACOUSLIC_point_0corr/per_sample.csv
C2_CSV=${ANALYSIS_OUT_ROOT}/c2_regpro_volume/RegPro_point_0corr/per_sample.csv

# A1 box-prompt CSV (used for the main 1×2 panel)
A1_BOX_CSV=${ANALYSIS_OUT_ROOT}/a1_camus_ef/CAMUS_box_0corr/per_patient.csv

INPUTS=""
[ -f "$A1_CSV" ] && INPUTS="$INPUTS A1=$A1_CSV"
[ -f "$A2_CSV" ] && INPUTS="$INPUTS A2=$A2_CSV"
[ -f "$A4_CSV" ] && INPUTS="$INPUTS A4=$A4_CSV"
[ -f "$C2_CSV" ] && INPUTS="$INPUTS C2=$C2_CSV"

if [ -z "$INPUTS" ]; then
  echo "T1.2: no per-sample CSVs found. Run A1/A2/A4/C2 first."
  exit 1
fi

EXTRA_ARGS=""
[ -f "$A1_CSV" ] && EXTRA_ARGS="$EXTRA_ARGS --main-sonobase-point $A1_CSV"
[ -f "$A1_BOX_CSV" ] && EXTRA_ARGS="$EXTRA_ARGS --main-sonobase-box $A1_BOX_CSV"

uv run python -m nemo_cv.recipes.analysis.t1_2_bland_altman_figures \
  --inputs $INPUTS \
  --output-dir ${OUT} \
  $EXTRA_ARGS
