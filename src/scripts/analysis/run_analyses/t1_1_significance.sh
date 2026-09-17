#!/bin/bash
# Stage 2 — T1.1: significance tests + BH-FDR across A1/A2/A4/C2.
# Requires the named per-sample CSVs to exist from prior Stage-2 runs.
set -e
source "$(dirname "$(readlink -f "$0")")/_paths.sh"

OUT=${ANALYSIS_OUT_ROOT}/t1_1_significance/point
A1_CSV=${ANALYSIS_OUT_ROOT}/a1_camus_ef/CAMUS_point_0corr/per_patient.csv
A2_CSV=${ANALYSIS_OUT_ROOT}/a2_hc18_hc/HC18_point_0corr/per_sample.csv
A4_CSV=${ANALYSIS_OUT_ROOT}/a4_acouslic_ac/ACOUSLIC_point_0corr/per_sample.csv
C2_CSV=${ANALYSIS_OUT_ROOT}/c2_regpro_volume/RegPro_point_0corr/per_sample.csv

INPUTS=""
[ -f "$A1_CSV" ] && INPUTS="$INPUTS A1=$A1_CSV"
[ -f "$A2_CSV" ] && INPUTS="$INPUTS A2=$A2_CSV"
[ -f "$A4_CSV" ] && INPUTS="$INPUTS A4=$A4_CSV"
[ -f "$C2_CSV" ] && INPUTS="$INPUTS C2=$C2_CSV"

if [ -z "$INPUTS" ]; then
  echo "T1.1: no per-sample CSVs found. Run A1/A2/A4/C2 first."
  exit 1
fi

uv run python -m nemo_cv.recipes.analysis.t1_1_significance \
  --inputs $INPUTS \
  --output-dir ${OUT}
