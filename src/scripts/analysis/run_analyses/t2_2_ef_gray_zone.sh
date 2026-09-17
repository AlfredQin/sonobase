#!/bin/bash
# Stage 2 — T2.2: EF gray zone (35-45%) reclassification. Reads A1's per_patient.csv.
set -e
source "$(dirname "$(readlink -f "$0")")/_paths.sh"

A1_CSV=${ANALYSIS_OUT_ROOT}/a1_camus_ef/CAMUS_point_0corr/per_patient.csv
OUT=${ANALYSIS_OUT_ROOT}/t2_2_ef_gray_zone/CAMUS_point_0corr

if [ ! -f "$A1_CSV" ]; then
  echo "T2.2: A1 per_patient.csv not found. Run a1_camus_ef_point.sh first."
  exit 1
fi

uv run python -m nemo_cv.recipes.analysis.t2_2_ef_gray_zone \
  --a1-per-patient-csv ${A1_CSV} \
  --output-dir ${OUT}
