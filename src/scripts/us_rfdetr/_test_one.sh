#!/bin/bash
# Eval-only US-RFDETR single-cell launcher (single-GPU by default).
#
# Usage:
#   bash _test_one.sh <dataset> <backbone> [<ckpt_path>] [<extra hydra overrides...>]
#
# If <ckpt_path> is omitted, defaults to the `best.pt` produced by the
# matching training run:
#   ./experiments/us_rfdetr/<dataset>/<backbone>/checkpoints/best.pt
#
# Run from src/.

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <dataset> <backbone> [<ckpt_path>] [<extra hydra overrides...>]" >&2
  exit 2
fi

DATASET="$1"
BACKBONE="$2"
shift 2

# Optional 3rd positional: explicit ckpt_path. If next arg starts with '-'
# or contains '=' we treat it as an override and use the default ckpt.
DEFAULT_BEST="./experiments/us_rfdetr/${DATASET}/${BACKBONE}/checkpoints/best.pt"
if [[ $# -gt 0 && ! "$1" == *"="* && ! "$1" == "-"* ]]; then
  CKPT_PATH="$1"
  shift
else
  CKPT_PATH="${DEFAULT_BEST}"
fi
EXTRA="$@"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./_common.sh
source "${SCRIPT_DIR}/_common.sh"

if [[ ! -f "${CKPT_PATH}" ]]; then
  echo "ERROR: checkpoint not found: ${CKPT_PATH}" >&2
  echo "  Train it first: bash train_us_rfdetr_on_${DATASET}.sh ${BACKBONE}" >&2
  exit 1
fi

EXPERIMENT="us_rfdetr_on_${DATASET}"
DS_LABEL="$(_dataset_label "${DATASET}")"

# Sonobase test runs still need the encoder pretraining `.pt` because
# the recipe re-instantiates the encoder + loads `pretrained_ckpt` BEFORE
# overlaying `best.pt` — so SONOBASE_DCP-derived path goes here too.
PRETRAINED="$(bash "${SCRIPT_DIR}/_resolve_ckpt.sh" "${BACKBONE}")"

echo "########################################################"
echo "# US-RFDETR test"
echo "#   dataset       = ${DS_LABEL} (overlay=${EXPERIMENT})"
echo "#   backbone      = ${BACKBONE}"
echo "#   ckpt_path     = ${CKPT_PATH}"
echo "#   pretrained_pt = ${PRETRAINED}"
echo "#   test_gpus     = ${TEST_GPUS}"
[ -n "${EXTRA}" ] && echo "#   extra         = ${EXTRA}"
echo "########################################################"

uv run torchrun --nproc_per_node="${TEST_GPUS}" -m nemo_cv.recipes.us_rfdetr.test \
  -c ./configs/us_rfdetr -cn test \
  experiment="${EXPERIMENT}" \
  backbone="${BACKBONE}" \
  model.pretrained_ckpt="${PRETRAINED}" \
  ckpt_path="${CKPT_PATH}" \
  ${EXTRA}

echo ""
echo "Done. Test outputs at: ./experiments/us_rfdetr/${DATASET}/${BACKBONE}/results/"
