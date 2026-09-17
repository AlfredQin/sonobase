#!/bin/bash
# A4 ACOUSLIC AC with the specialist runs beside SonoBase (box + point), on the header
# pixel spacing (0.28 mm/px) with the ACOUSLIC v1.1 circumference file.
#   bash scripts/analysis/specialist/stage2_acouslic.sh [label ...]     (default: every <label>_ACOUSLIC_none present)
source "$(dirname "$(readlink -f "$0")")/_common.sh"
: "${ACOUSLIC_CSV:?}"
LABELS=("$@"); [ ${#LABELS[@]} -gt 0 ] || LABELS=($(ls -d ${PRED_ROOT}/*_ACOUSLIC_none 2>/dev/null | xargs -n1 basename | sed 's/_ACOUSLIC_none$//'))
RUNS=(sonobase=${PRED_ROOT}/sonobase_ACOUSLIC_box_0corr sonobase_point=${PRED_ROOT}/sonobase_ACOUSLIC_point_0corr medsam2=${PRED_ROOT}/medsam2_ACOUSLIC_box_0corr sam2_no_ft=${PRED_ROOT}/sam2_no_ft_ACOUSLIC_box_0corr)
for L in "${LABELS[@]}"; do RUNS+=("${L}=${PRED_ROOT}/${L}_ACOUSLIC_none"); done
OUT=${ANALYSIS_OUT_ROOT}/a4_acouslic_ac/ACOUSLIC_specialists
uv run python -m nemo_cv.recipes.analysis.a4_acouslic_ac --runs "${RUNS[@]}" --output-dir "${OUT}" --acouslic-csv "${ACOUSLIC_CSV}"
echo "A4 -> ${OUT}"
