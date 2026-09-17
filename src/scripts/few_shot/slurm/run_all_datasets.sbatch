#!/bin/bash
# Few-shot adaptation (F1) — array job for all 3 datasets.
#
# Submit:
#
#   sbatch scripts/few_shot/slurm/run_all_datasets.sbatch
#
# Each task runs the full ~162-cell sweep for one dataset. Tasks run
# in parallel (limited by the cluster's GPU availability).
#
# When all 3 array tasks complete, run the aggregator to produce the
# combined CSV + figures (CPU-only, fast):
#
#   bash scripts/few_shot/aggregate_all.sh
#
# Or as a slurm follow-up job:
#
#   sbatch --dependency=afterok:<JobID> \
#     scripts/few_shot/slurm/aggregate_all.sbatch

#SBATCH --job-name=fs_sweep
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err
#SBATCH --partition=main
#SBATCH --array=0-2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=512G
#SBATCH --time=10:00:00

set -euo pipefail

DATASETS=(ACOUSLIC DDTI FUGC)
DATASET="${DATASETS[${SLURM_ARRAY_TASK_ID}]}"
export DATASET

SBATCH_PATH="$(scontrol show job "${SLURM_JOB_ID}" -o | grep -oP 'Command=\K[^[:space:]]+')"
[ -f "${SBATCH_PATH}" ] || { echo "ERROR: cannot resolve this sbatch script's path from scontrol (got '${SBATCH_PATH}')" >&2; exit 1; }
SCRIPT_DIR="$(cd "$(dirname "${SBATCH_PATH}")" && pwd)"

echo "============================================================"
echo "# Array task ${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_MAX} → DATASET=${DATASET}"
echo "============================================================"

source "${SCRIPT_DIR}/_run_dataset.sbatch"
