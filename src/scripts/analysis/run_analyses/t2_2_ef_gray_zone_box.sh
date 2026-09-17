#!/bin/bash
# Stage 2 — T2.2: EF gray zone (35-45%) reclassification — box-prompt run.
# Reads A1's box-prompt per_patient.csv.
set -e
source "$(dirname "$(readlink -f "$0")")/_paths.sh"

A1_CSV=${ANALYSIS_OUT_ROOT}/a1_camus_ef/CAMUS_box_0corr/per_patient.csv
OUT=${ANALYSIS_OUT_ROOT}/t2_2_ef_gray_zone/CAMUS_box_0corr

if [ ! -f "$A1_CSV" ]; then
  echo "T2.2 (box): A1 per_patient.csv not found at $A1_CSV. Run a1_camus_ef_box.sh first."
  exit 1
fi

uv run python -m nemo_cv.recipes.analysis.t2_2_ef_gray_zone \
  --a1-per-patient-csv ${A1_CSV} \
  --output-dir ${OUT}
