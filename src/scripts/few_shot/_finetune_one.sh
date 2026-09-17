#!/bin/bash
# Generic single-(dataset × model × N × seed) fine-tune invocation.
# Usage: bash _finetune_one.sh <dataset> <model> <N> <seed> [extra hydra overrides...]
#
#   dataset  one of: ACOUSLIC, DDTI, FUGC
#   model    one of: sam2_no_ft, medsam2, sonobase
#   N        in {1, 2, 5, 10, 20, 30}
#   seed     in {42, 123, 456}
#
# Resolves the right .pt checkpoint per model and (for sonobase) runs the
# DCP→PT conversion lazily — same pattern as scripts/analysis/save_predictions/_save_one.sh.

set -e

DATASET=$1; MODEL=$2; N=$3; SEED=$4
shift 4 || true
EXTRA="$@"

if [ -z "$DATASET" ] || [ -z "$MODEL" ] || [ -z "$N" ] || [ -z "$SEED" ]; then
  echo "Usage: $0 <dataset> <model> <N> <seed> [extra hydra overrides...]"
  exit 1
fi

# `CHECKPOINT_DIR` must be set per user (e.g. `export CHECKPOINT_DIR=$WORK/Checkpoints`
# in your .bashrc) — points at the SAM2/MedSAM2 release weights. Checkpoint
# resolution (the `CKPT_PATH` / `SONOBASE_DCP` overrides and the lazy sonobase
# DCP→PT conversion) is delegated to the shared resolver.
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")

# ---- resolve checkpoint ----
CKPT="$(bash "$SCRIPT_DIR/../_resolve_ckpt.sh" "$MODEL")"

export HYDRA_FULL_ERROR=1
export PYTHONPATH=$PYTHONPATH:$(pwd)
export LD_LIBRARY_PATH=""

uv run python -m nemo_cv.recipes.few_shot.finetune \
  -c ./configs/few_shot -cn finetune \
  data=${DATASET} \
  experiment=${MODEL} \
  ckpt_path="$CKPT" \
  scratch.dataset_name=${DATASET} \
  scratch.N=${N} \
  scratch.fewshot_seed=${SEED} \
  ${EXTRA}
