#!/bin/bash
# US-RFDETR — array job for all 27 (dataset × backbone) cells.
# 9 datasets × 3 backbones; the array index decomposes into both axes.
#
# Submit:
#
#   sbatch scripts/us_rfdetr/slurm/us_rfdetr_all.sbatch
#
# Restrict to a slice:
#
#   sbatch --array=0-8     ...     # all backbones × first 3 datasets
#   sbatch --array=18-26   ...     # only sonobase backbone
#   sbatch --array=0,3,6,9,12,15,18,21,24  ...  # only sam2_no_ft
#
# Layout convention (so the array index encodes both axes deterministically):
#
#   array_id   dataset_idx = id // 3        backbone_idx = id % 3
#       0      0  cva_net                   0  sam2_no_ft
#       1      0  cva_net                   1  medsam2
#       2      0  cva_net                   2  sonobase
#       3      1  fetus                     0  sam2_no_ft
#       …      …                            …
#      26      8  mmotu_3d                  2  sonobase
#
# Per-task logs land at <sbatch_submit_dir>/slurm_logs/<jobname>_<JobID>_<TaskID>.{out,err}

#SBATCH --job-name=us_rfdetr_all
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err
#SBATCH --partition=main
#SBATCH --array=0-26                      # 27 cells
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=4
#SBATCH --mem=512G
#SBATCH --time=04:00:00

set -euo pipefail

DATASETS=(cva_net fetus acouslic bus_bra ddti fugc kidneyus luminous mmotu_3d)
BACKBONES=(sam2_no_ft medsam2 sonobase)

DATASET_IDX=$((SLURM_ARRAY_TASK_ID / ${#BACKBONES[@]}))
BACKBONE_IDX=$((SLURM_ARRAY_TASK_ID % ${#BACKBONES[@]}))
DATASET="${DATASETS[${DATASET_IDX}]}"
BACKBONE="${BACKBONES[${BACKBONE_IDX}]}"

export DATASET BACKBONE
export EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"

SBATCH_PATH="$(scontrol show job "${SLURM_JOB_ID}" -o | grep -oP 'Command=\K[^[:space:]]+' | head -n1)"
[ -f "${SBATCH_PATH}" ] || { echo "ERROR: cannot resolve this sbatch script's path from scontrol (got '${SBATCH_PATH}')" >&2; exit 1; }
SCRIPT_DIR="$(cd "$(dirname "${SBATCH_PATH}")" && pwd)"

echo "============================================================"
echo "# Array task ${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_MAX}"
echo "#   → dataset=${DATASET} (idx ${DATASET_IDX}), backbone=${BACKBONE} (idx ${BACKBONE_IDX})"
echo "============================================================"

# Source the single-cell launcher inside this allocation. Slurm has
# already parsed our SBATCH directives; sourcing the body of
# `_us_rfdetr_one.sbatch` skips its own `#SBATCH` block (which is just
# parsed by sbatch at submission time anyway).
source "${SCRIPT_DIR}/_us_rfdetr_one.sbatch"
