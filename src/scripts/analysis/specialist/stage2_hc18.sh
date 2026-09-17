#!/bin/bash
# A2 HC18 head circumference with the specialist runs beside SonoBase (box + point).
#   bash scripts/analysis/specialist/stage2_hc18.sh [label ...]
source "$(dirname "$(readlink -f "$0")")/_common.sh"
: "${HC18_ZIP:?}"
LABELS=("$@"); [ ${#LABELS[@]} -gt 0 ] || LABELS=($(ls -d ${PRED_ROOT}/*_HC18_none 2>/dev/null | xargs -n1 basename | sed 's/_HC18_none$//'))
RUNS=(sonobase=${PRED_ROOT}/sonobase_HC18_box_0corr sonobase_point=${PRED_ROOT}/sonobase_HC18_point_0corr medsam2=${PRED_ROOT}/medsam2_HC18_box_0corr sam2_no_ft=${PRED_ROOT}/sam2_no_ft_HC18_box_0corr)
for L in "${LABELS[@]}"; do RUNS+=("${L}=${PRED_ROOT}/${L}_HC18_none"); done
OUT=${ANALYSIS_OUT_ROOT}/a2_hc18_hc/HC18_specialists
uv run python -m nemo_cv.recipes.analysis.a2_hc18_hc --runs "${RUNS[@]}" --include-gt-fit --output-dir "${OUT}" --hc18-zip "${HC18_ZIP}"
echo "A2 -> ${OUT}"
