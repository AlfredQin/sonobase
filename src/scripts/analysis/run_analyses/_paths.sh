#!/bin/bash
# Common path / variable defaults sourced by every Stage-2 runner script.
#
# All raw-asset env vars must be set per user — e.g. in your .bashrc:
#   export HC18_ZIP=$WORK/Dataset/UltraSound/Raw/HC.zip
#   export CAMUS_ZIP=$WORK/Dataset/UltraSound/Raw/CAMUS.zip
#   export REGPRO_ZIP=$WORK/Dataset/UltraSound/Raw/RegPro.zip
#   export ACOUSLIC_CSV=$WORK/Dataset/UltraSound/ACOUSLIC/circumferences/fetal_abdominal_circumferences_per_sweep.csv
#   export PORCINE_SAUS=$WORK/Dataset/SaUS/PorcineSpinalCord
# Individual scripts source this and only assert on the vars they actually
# consume, so unrelated analyses don't force you to set everything at once.

PRED_ROOT=${PRED_ROOT:-./experiments/analysis/predictions}
ANALYSIS_OUT_ROOT=${ANALYSIS_OUT_ROOT:-./experiments/analysis}

# Pass-through (no default). A script that needs one of these should add:
#   : "${HC18_ZIP:?HC18_ZIP must be set, e.g. export HC18_ZIP=\$WORK/Dataset/UltraSound/Raw/HC.zip}"
HC18_ZIP=${HC18_ZIP:-}
CAMUS_ZIP=${CAMUS_ZIP:-}
REGPRO_ZIP=${REGPRO_ZIP:-}
ACOUSLIC_CSV=${ACOUSLIC_CSV:-}
PORCINE_SAUS=${PORCINE_SAUS:-}

export PYTHONPATH=$PYTHONPATH:$(pwd)
