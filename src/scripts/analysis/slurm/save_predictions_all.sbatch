#!/bin/bash
# Analysis Stage 1 — array job to save predictions for ALL datasets in scope.
# 7 datasets × (3 models × 2 prompts) = 42 sub-jobs run inside each
# array task's `save_all_<DATASET>.sh` wrapper.
#
# Submit:
#
#   sbatch scripts/analysis/slurm/save_predictions_all.sbatch
#
# Restrict to a slice:
#
#   sbatch --array=0,3 ...   # only HC18 and RegPro
#
# Layout:
#   array_id   DATASET
#       0      HC18
#       1      CAMUS
#       2      ACOUSLIC
#       3      RegPro
#       4      PorcineSpinalCord
#       5      BUSI
#       6      KidneyUS
#       7      MouseBrainTumor   (eval-only OOD; added 2026-05-31 for E4 v2 rerun)

#SBATCH --job-name=analysis_pred_all
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err
#SBATCH --partition=main
#SBATCH --array=0-7
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=512G
#SBATCH --time=04:00:00

set -euo pipefail

DATASETS=(HC18 CAMUS ACOUSLIC RegPro PorcineSpinalCord BUSI KidneyUS MouseBrainTumor)
DATASET="${DATASETS[${SLURM_ARRAY_TASK_ID}]}"
export DATASET

# NB: | head -n1 — for array jobs scontrol can echo a Command= line per
# concurrent task; take the first (matches run_all_cells / us_rfdetr_all guard).
SBATCH_PATH="$(scontrol show job "${SLURM_JOB_ID}" -o | grep -oP 'Command=\K[^[:space:]]+' | head -n1)"
[ -f "${SBATCH_PATH}" ] || { echo "ERROR: cannot resolve this sbatch script's path from scontrol (got '${SBATCH_PATH}')" >&2; exit 1; }
SCRIPT_DIR="$(cd "$(dirname "${SBATCH_PATH}")" && pwd)"

echo "============================================================"
echo "# Array task ${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_MAX} → DATASET=${DATASET}"
echo "============================================================"

source "${SCRIPT_DIR}/_save_predictions_one.sbatch"
