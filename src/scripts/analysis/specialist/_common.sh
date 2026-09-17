#!/bin/bash
# Shared paths for the specialist-baseline Stage-2 scripts. Source from src/.
# Run dirs are written beside the SonoBase / MedSAM2 / SAM2 prediction runs,
# named <label>_<DS>_none (prompt protocol "none").
set -euo pipefail
MAIN_SRC=${MAIN_SRC:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}
PRED_ROOT=${PRED_ROOT:-${MAIN_SRC}/experiments/analysis/predictions}
ANALYSIS_OUT_ROOT=${ANALYSIS_OUT_ROOT:-${MAIN_SRC}/experiments/analysis}
EXTERNAL=${EXTERNAL:-${MAIN_SRC}/../external}
EXP_DIR=${EXP_DIR:-${MAIN_SRC}/experiments/specialist_baselines}
: "${DATASET_DIR:?}"; : "${ANNOTATION_DIR:?}"
export PYTHONPATH=${PYTHONPATH:-}:$(pwd)
