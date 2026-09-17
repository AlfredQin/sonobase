#!/bin/bash
# Import one specialist's neutral-layout predictions as a Stage-1 run dir, scored on the archived
# SonoBase row population.   Usage (from src/):
#   bash scripts/analysis/specialist/import_external.sh <label> <DS> <neutral_dir> [extra importer args]
#   e.g. ... echonet_lv_zeroshot CAMUS $EXTERNAL/echonet_us/neutral/echonet_lv_zeroshot/CAMUS --unmapped-objects skip
source "$(dirname "$(readlink -f "$0")")/_common.sh"
LABEL=$1; DS=$2; NEUTRAL=$3; shift 3
RUN_DIR=${PRED_ROOT}/${LABEL}_${DS}_none
uv run python -m nemo_cv.recipes.analysis.import_external_predictions --dataset "${DS}" --neutral-dir "${NEUTRAL}" \
    --run-dir "${RUN_DIR}" --rows-from "${PRED_ROOT}/sonobase_${DS}_box_0corr" --model-label "${LABEL}" "$@"
echo "imported -> ${RUN_DIR} ($(tail -n +2 "${RUN_DIR}/per_sample_metrics.csv" | wc -l) rows)"
