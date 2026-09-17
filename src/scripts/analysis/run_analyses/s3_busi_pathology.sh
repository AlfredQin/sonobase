#!/bin/bash
# Stage 2 — S3: BUSI by Pathology Class (benign / malignant).
# Reads 6 Stage-1 BUSI prediction dirs (3 models × 2 prompts).
set -e
source "$(dirname "$(readlink -f "$0")")/_paths.sh"

: "${ANNOTATION_DIR:?ANNOTATION_DIR must be set, e.g. export ANNOTATION_DIR=\$WORK/Dataset/SaUS_Annotation/38_pt_8_bm_7_ext}"

OUT=${ANALYSIS_OUT_ROOT}/s3_busi_pathology/BUSI_0corr
uv run python -m nemo_cv.recipes.analysis.s3_busi_pathology \
  --runs sonobase_point=${PRED_ROOT}/sonobase_BUSI_point_0corr \
         sonobase_box=${PRED_ROOT}/sonobase_BUSI_box_0corr \
         medsam2_point=${PRED_ROOT}/medsam2_BUSI_point_0corr \
         medsam2_box=${PRED_ROOT}/medsam2_BUSI_box_0corr \
         sam2_no_ft_point=${PRED_ROOT}/sam2_no_ft_BUSI_point_0corr \
         sam2_no_ft_box=${PRED_ROOT}/sam2_no_ft_BUSI_box_0corr \
  --output-dir ${OUT} \
  --annotation-dir "${ANNOTATION_DIR}"
