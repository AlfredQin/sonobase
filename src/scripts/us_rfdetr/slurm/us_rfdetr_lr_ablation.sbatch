#!/bin/bash
# US-RFDETR — learning-rate ablation arm: 8 datasets × 3 backbones at lr=1e-4.
#
# Why this exists
# ---------------
# The published bbox table runs at 3e-4 (FIX(E3)), with FUGC falling back to
# 1e-4 because 3e-4 diverges there. The E3 notes justified reconsidering 3e-4
# globally by comparing against the May-25 archive — but every May-25 cell was
# ALSO logged at base lr=3e-4, so those deltas measure the pretrain checkpoint
# and the v2 dedup splits, not the learning rate. The only controlled base-LR
# evidence on record is the single FUGC pair.
#
# This arm supplies the missing half of the grid: the same 8 datasets × 3
# backbones at 1e-4, on the same production checkpoint and the same v2 splits,
# so "3e-4 vs 1e-4" becomes a genuine single-variable comparison and the
# per-dataset LR rule can be shown to be uniform rather than a FUGC exception.
#
# Selection stays on VALIDATION only — test is computed once per cell with the
# config already chosen. Note the val splits are small (ACOUSLIC 30, MMOTU-3d
# 17, LUMINOUS 34 images), so this grid is a supporting ablation; the LR rule
# itself is divergence-triggered, not argmax-on-val, which is far more robust
# at these sizes.
#
# Submit:
#
#   sbatch scripts/us_rfdetr/slurm/us_rfdetr_lr_ablation.sbatch
#
# Writes to `experiments/us_rfdetr_lr1e-4/<dataset>/<backbone>/`, leaving the
# published 3e-4 results under `experiments/us_rfdetr/` untouched.
#
# Everything except the LR is held fixed at the published bbox-arm settings —
# 4 GPUs per cell included. GPU count is NOT free to change: it sets the
# per-rank batch size and the DistributedSampler's eval padding, both of which
# would shift the metric and break comparability with the 3e-4 table.

#SBATCH --job-name=us_rfdetr_lr1e-4
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

# Same 8 datasets as the seg arm, so the two new tables cover identical rows.
# cva_net is excluded here for comparability with the seg arm, not for lack of
# labels — it has bbox GT and its 3e-4 number stands in the published table.
DATASETS=(fetus acouslic bus_bra ddti fugc kidneyus luminous mmotu_3d)
BACKBONES=(sam2_no_ft medsam2 sonobase)

DATASET_IDX=$((SLURM_ARRAY_TASK_ID / ${#BACKBONES[@]}))
BACKBONE_IDX=$((SLURM_ARRAY_TASK_ID % ${#BACKBONES[@]}))
DATASET="${DATASETS[${DATASET_IDX}]}"
BACKBONE="${BACKBONES[${BACKBONE_IDX}]}"

# The full pre-FIX(E3) optimiser block — all four values move together, which
# is how the FUGC fallback row in the published table was run.
LR_1E4="optim.lr=1e-4 optim.lr_encoder=1e-5 optim.lr_min=1e-6 optim.warmup_lr_init=1e-6"

export DATASET BACKBONE
export EXTRA_OVERRIDES="${LR_1E4} scratch.experiment_name=us_rfdetr_lr1e-4/${DATASET} ${EXTRA_OVERRIDES:-}"

SBATCH_PATH="$(scontrol show job "${SLURM_JOB_ID}" -o | grep -oP 'Command=\K[^[:space:]]+' | head -n1)"
[ -f "${SBATCH_PATH}" ] || { echo "ERROR: cannot resolve this sbatch script's path from scontrol (got '${SBATCH_PATH}')" >&2; exit 1; }
SCRIPT_DIR="$(cd "$(dirname "${SBATCH_PATH}")" && pwd)"

echo "============================================================"
echo "# LR-ABLATION arm (1e-4) — array task ${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_MAX}"
echo "#   → dataset=${DATASET} (idx ${DATASET_IDX}), backbone=${BACKBONE} (idx ${BACKBONE_IDX})"
echo "============================================================"

source "${SCRIPT_DIR}/_us_rfdetr_one.sbatch"
