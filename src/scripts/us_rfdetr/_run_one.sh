#!/bin/bash
# Generic single-(dataset × backbone) US-RFDETR train+test launcher.
#
# Usage:
#   bash _run_one.sh <dataset> <backbone> [<extra hydra overrides...>]
#
#   <dataset>    one of: cva_net | fetus | acouslic | bus_bra | ddti |
#                        fugc | kidneyus | luminous | mmotu_3d
#   <backbone>   one of: sam2_no_ft | medsam2 | sonobase
#
# What it does:
#   1. Resolve the pretrained encoder `.pt` for the chosen backbone via
#      `_resolve_ckpt.sh` (lazily converts sonobase DCP shards if needed).
#   2. Launch DDP training on `${TRAIN_GPUS}` GPUs (defaults to all in
#      `CUDA_VISIBLE_DEVICES`). The recipe loads the encoder weights,
#      trains for `trainer.max_epochs` epochs with per-epoch validation,
#      and writes `best.pt` whenever val_bbox_mAP improves.
#   3. After training, the recipe automatically loads `best.pt` and runs
#      the test loop on the same world. (To run test on its own use
#      `test_us_rfdetr_on_<dataset>.sh` instead.)
#
# Outputs land at:
#   ./experiments/us_rfdetr/<dataset>/<backbone>/
#     ├── checkpoints/best.pt
#     ├── training.jsonl
#     ├── validation.jsonl
#     ├── test.jsonl
#     └── results/test_metrics.json
#
# Run from src/.

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <dataset> <backbone> [<extra hydra overrides...>]" >&2
  exit 2
fi

DATASET="$1"
BACKBONE="$2"
shift 2
EXTRA="$@"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./_common.sh
source "${SCRIPT_DIR}/_common.sh"

CKPT_PATH="$(bash "${SCRIPT_DIR}/_resolve_ckpt.sh" "${BACKBONE}")"
EXPERIMENT="us_rfdetr_on_${DATASET}"
DS_LABEL="$(_dataset_label "${DATASET}")"

echo "########################################################"
echo "# US-RFDETR train+test"
echo "#   dataset       = ${DS_LABEL} (overlay=${EXPERIMENT})"
echo "#   backbone      = ${BACKBONE}"
echo "#   pretrained_pt = ${CKPT_PATH}"
echo "#   train_gpus    = ${TRAIN_GPUS}  (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES})"
[ -n "${EXTRA}" ] && echo "#   extra         = ${EXTRA}"
echo "########################################################"

uv run torchrun --nproc_per_node="${TRAIN_GPUS}" -m nemo_cv.recipes.us_rfdetr.train \
  -c ./configs/us_rfdetr -cn train \
  experiment="${EXPERIMENT}" \
  backbone="${BACKBONE}" \
  model.pretrained_ckpt="${CKPT_PATH}" \
  ${EXTRA}

echo ""
echo "Done. Outputs at: ./experiments/us_rfdetr/${DATASET}/${BACKBONE}/"
