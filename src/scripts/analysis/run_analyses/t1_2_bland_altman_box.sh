#!/bin/bash
# Stage 2 — T1.2: composite Bland-Altman figures (box-prompt run).
set -e
source "$(dirname "$(readlink -f "$0")")/_paths.sh"

OUT=${ANALYSIS_OUT_ROOT}/t1_2_bland_altman/box
A1_CSV=${ANALYSIS_OUT_ROOT}/a1_camus_ef/CAMUS_box_0corr/per_patient.csv
A2_CSV=${ANALYSIS_OUT_ROOT}/a2_hc18_hc/HC18_box_0corr/per_sample.csv
A4_CSV=${ANALYSIS_OUT_ROOT}/a4_acouslic_ac/ACOUSLIC_box_0corr/per_sample.csv
C2_CSV=${ANALYSIS_OUT_ROOT}/c2_regpro_volume/RegPro_box_0corr/per_sample.csv

INPUTS=""
[ -f "$A1_CSV" ] && INPUTS="$INPUTS A1=$A1_CSV"
[ -f "$A2_CSV" ] && INPUTS="$INPUTS A2=$A2_CSV"
[ -f "$A4_CSV" ] && INPUTS="$INPUTS A4=$A4_CSV"
[ -f "$C2_CSV" ] && INPUTS="$INPUTS C2=$C2_CSV"

if [ -z "$INPUTS" ]; then
  echo "T1.2 (box): no per-sample CSVs found. Run A1/A2/A4/C2 box-prompt scripts first."
  exit 1
fi

# Note: T1.2's optional `--main-sonobase-{point,box}` arguments are intended
# for the main-text 1×2 Bland-Altman figure that *contrasts* point vs box
# for SonoBase. In a box-only run there is no point CSV to pair against, so
# the main-panel arguments are omitted; the supplement Bland-Altman panels
# still render from the per-(A1, A2, A4, C2) box CSVs above.
uv run python -m nemo_cv.recipes.analysis.t1_2_bland_altman_figures \
  --inputs $INPUTS \
  --output-dir ${OUT}
