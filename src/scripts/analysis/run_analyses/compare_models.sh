#!/bin/bash
# Stage 2 — cross-analysis aggregator. Scans every analysis_report.json
# under ANALYSIS_OUT_ROOT and emits a master CSV.
set -e
source "$(dirname "$(readlink -f "$0")")/_paths.sh"

OUT=${ANALYSIS_OUT_ROOT}/_cross_analysis
uv run python -m nemo_cv.recipes.analysis.compare_models \
  --analyses-root ${ANALYSIS_OUT_ROOT} \
  --output-dir ${OUT}
