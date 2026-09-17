#!/bin/bash
# US-RFDETR — instance-segmentation arm: 8 datasets × 3 backbones (24 cells).
#
# Also reports segm_mAP on the datasets that ship masks. The metric plumbing already
# existed — per-loader `MeanAveragePrecision(iou_type="segm")` in the recipe,
# RLE masks in the datasets, `SegmentationHead` + mask losses in the builder —
# but `model.segmentation_head` defaulted to false, so no run ever produced a
# segm number. This launcher turns it on.
#
# Submit:
#
#   sbatch scripts/us_rfdetr/slurm/us_rfdetr_seg_all.sbatch
#
# Runs alongside, and does NOT overwrite, the bbox-only results: cells here
# write to `experiments/us_rfdetr_seg/<dataset>/<backbone>/` via a
# `scratch.experiment_name` override. The published bbox table under
# `experiments/us_rfdetr/` is left untouched, and each cell here also emits its
# own bbox_mAP — a free detection-only-vs-multitask cross-check.
#
# Layout convention (array index encodes both axes):
#
#   array_id   dataset_idx = id // 3        backbone_idx = id % 3
#       0      0  fetus                     0  sam2_no_ft
#       1      0  fetus                     1  medsam2
#       2      0  fetus                     2  sonobase
#       …      …                            …
#      23      7  mmotu_3d                  2  sonobase
#
# `%2` throttles to 2 concurrent cells = 8 GPUs, the `regular` QOS ceiling.

#SBATCH --job-name=us_rfdetr_seg
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err
#SBATCH --partition=main
#SBATCH --array=0-23%2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=4
#SBATCH --mem=512G
#SBATCH --time=04:00:00

set -euo pipefail

# cva_net is absent by design: its COCO-video annotations are bbox-only
# (0 of 3691 annotations carry a `segmentation` field), so there is no
# instance-mask ground truth to score against. With segmentation_head=true
# the criterion appends the "masks" loss unconditionally, so including it
# would KeyError in loss_masks rather than merely report nothing.
DATASETS=(fetus acouslic bus_bra ddti fugc kidneyus luminous mmotu_3d)
BACKBONES=(sam2_no_ft medsam2 sonobase)

DATASET_IDX=$((SLURM_ARRAY_TASK_ID / ${#BACKBONES[@]}))
BACKBONE_IDX=$((SLURM_ARRAY_TASK_ID % ${#BACKBONES[@]}))
DATASET="${DATASETS[${DATASET_IDX}]}"
BACKBONE="${BACKBONES[${BACKBONE_IDX}]}"

# Per-dataset LR follows the SAME rule as the published bbox table: 3e-4
# everywhere, falling back to 1e-4 on FUGC, where 3e-4 diverges (validation
# mAP pinned at 0.000 for 19 of 24 epochs). Applying the identical rule keeps
# the segm table line-for-line comparable with the bbox table.
LR_OVERRIDE=""
if [[ "${DATASET}" == "fugc" ]]; then
  LR_OVERRIDE="optim.lr=1e-4 optim.lr_encoder=1e-5 optim.lr_min=1e-6 optim.warmup_lr_init=1e-6"
fi

# static_graph is required, not optional: with two_stage the segmentation head
# runs twice per forward (decoder outputs, then encoder outputs with
# skip_blocks=True), and DDP's find_unused_parameters reducer raises
# "Expected to mark a variable ready only once" on the second use.
export DATASET BACKBONE
export EXTRA_OVERRIDES="model.segmentation_head=true distributed.static_graph=true scratch.experiment_name=us_rfdetr_seg/${DATASET} ${LR_OVERRIDE} ${EXTRA_OVERRIDES:-}"

SBATCH_PATH="$(scontrol show job "${SLURM_JOB_ID}" -o | grep -oP 'Command=\K[^[:space:]]+' | head -n1)"
[ -f "${SBATCH_PATH}" ] || { echo "ERROR: cannot resolve this sbatch script's path from scontrol (got '${SBATCH_PATH}')" >&2; exit 1; }
SCRIPT_DIR="$(cd "$(dirname "${SBATCH_PATH}")" && pwd)"

echo "============================================================"
echo "# SEG arm — array task ${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_MAX}"
echo "#   → dataset=${DATASET} (idx ${DATASET_IDX}), backbone=${BACKBONE} (idx ${BACKBONE_IDX})"
[ -n "${LR_OVERRIDE}" ] && echo "#   → LR fallback active: ${LR_OVERRIDE}"
echo "============================================================"

source "${SCRIPT_DIR}/_us_rfdetr_one.sbatch"
