#!/bin/bash
# Stage 2 — S1: KidneyUS by Scanner Manufacturer.
# Reads 6 Stage-1 KidneyUS prediction dirs (3 models × 2 prompts).
set -e
source "$(dirname "$(readlink -f "$0")")/_paths.sh"

: "${ANNOTATION_DIR:?ANNOTATION_DIR must be set, e.g. export ANNOTATION_DIR=\$WORK/Dataset/SaUS_Annotation/38_pt_8_bm_7_ext}"

OUT=${ANALYSIS_OUT_ROOT}/s1_kidneyus_manufacturer/KidneyUS_0corr
uv run python -m nemo_cv.recipes.analysis.s1_kidneyus_manufacturer \
  --runs sonobase_point=${PRED_ROOT}/sonobase_KidneyUS_point_0corr \
         sonobase_box=${PRED_ROOT}/sonobase_KidneyUS_box_0corr \
         medsam2_point=${PRED_ROOT}/medsam2_KidneyUS_point_0corr \
         medsam2_box=${PRED_ROOT}/medsam2_KidneyUS_box_0corr \
         sam2_no_ft_point=${PRED_ROOT}/sam2_no_ft_KidneyUS_point_0corr \
         sam2_no_ft_box=${PRED_ROOT}/sam2_no_ft_KidneyUS_box_0corr \
  --output-dir ${OUT} \
  --annotation-dir "${ANNOTATION_DIR}"
