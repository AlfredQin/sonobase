#!/bin/bash
# A1 CAMUS EF with the specialist runs beside SonoBase (box + point). SonoBase first: --include-gt-fit reads GT from run 1.
#   bash scripts/analysis/specialist/stage2_camus.sh [label ...]     (default: every <label>_CAMUS_none present)
source "$(dirname "$(readlink -f "$0")")/_common.sh"
: "${CAMUS_ZIP:?}"
LABELS=("$@"); [ ${#LABELS[@]} -gt 0 ] || LABELS=($(ls -d ${PRED_ROOT}/*_CAMUS_none 2>/dev/null | xargs -n1 basename | sed 's/_CAMUS_none$//'))
RUNS=(sonobase=${PRED_ROOT}/sonobase_CAMUS_box_0corr sonobase_point=${PRED_ROOT}/sonobase_CAMUS_point_0corr medsam2=${PRED_ROOT}/medsam2_CAMUS_box_0corr sam2_no_ft=${PRED_ROOT}/sam2_no_ft_CAMUS_box_0corr)
for L in "${LABELS[@]}"; do RUNS+=("${L}=${PRED_ROOT}/${L}_CAMUS_none"); done
OUT=${ANALYSIS_OUT_ROOT}/a1_camus_ef/CAMUS_specialists
uv run python -m nemo_cv.recipes.analysis.a1_camus_ef --runs "${RUNS[@]}" --include-gt-fit --output-dir "${OUT}" --camus-zip "${CAMUS_ZIP}"
echo "A1 -> ${OUT}"
