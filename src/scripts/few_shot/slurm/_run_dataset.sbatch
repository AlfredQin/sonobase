#!/bin/bash
# Few-shot adaptation (F1) — run the full per-dataset sweep on slurm.
#
# Each dataset → 3 models × 6 N values × 3 seeds = 54 fine-tunes
#              + 3 models × 6 N values × 3 seeds × 2 prompts = 108 evals.
# Each fine-tune is single-GPU (decoder-only training on 1-30 samples)
# and short (~minutes); the long pole is the count, not per-cell cost.
#
# Submit:
#
#   sbatch --export=ALL,DATASET=ACOUSLIC scripts/few_shot/slurm/_run_dataset.sbatch
#   sbatch --export=ALL,DATASET=DDTI     scripts/few_shot/slurm/_run_dataset.sbatch
#   sbatch --export=ALL,DATASET=FUGC     scripts/few_shot/slurm/_run_dataset.sbatch
#
# Wall time: ~6-8 hours per dataset on 1 GPU (single-GPU is enough since
# each fine-tune is tiny; running 162 sub-jobs serially).

#SBATCH --job-name=few_shot
#SBATCH --output=slurm_logs/%x_%A.out
#SBATCH --error=slurm_logs/%x_%A.err
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=512G
#SBATCH --time=10:00:00

set -euo pipefail

: "${DATASET:?DATASET must be set: ACOUSLIC | DDTI | FUGC}"

# Asset paths — all three must be set per user in your .bashrc, e.g.:
#   export CHECKPOINT_DIR=$WORK/Checkpoints
#   export DATASET_DIR=$WORK/Dataset/SaUS
#   export ANNOTATION_DIR=$WORK/Dataset/SaUS_Annotation/38_pt_8_bm_7_ext_v2
: "${CHECKPOINT_DIR:?CHECKPOINT_DIR must be set, e.g. export CHECKPOINT_DIR=\$WORK/Checkpoints}"
: "${DATASET_DIR:?DATASET_DIR must be set, e.g. export DATASET_DIR=\$WORK/Dataset/SaUS}"
: "${ANNOTATION_DIR:?ANNOTATION_DIR must be set, e.g. export ANNOTATION_DIR=\$WORK/Dataset/SaUS_Annotation/38_pt_8_bm_7_ext_v2}"
export CHECKPOINT_DIR DATASET_DIR ANNOTATION_DIR

SBATCH_PATH="$(scontrol show job "${SLURM_JOB_ID}" -o | grep -oP 'Command=\K[^[:space:]]+')"
[ -f "${SBATCH_PATH}" ] || { echo "ERROR: cannot resolve this sbatch script's path from scontrol (got '${SBATCH_PATH}')" >&2; exit 1; }
SCRIPT_DIR="$(cd "$(dirname "${SBATCH_PATH}")" && pwd)"
SRC_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${SRC_DIR}"

mkdir -p "${SLURM_SUBMIT_DIR:-$PWD}/slurm_logs"

export HYDRA_FULL_ERROR=1
export PYTHONPATH="${PYTHONPATH:-}:${SRC_DIR}"
export LD_LIBRARY_PATH=""
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

echo "============================================================"
echo "# Few-shot sweep on ${DATASET} (slurm)"
echo "#   job_id         = ${SLURM_JOB_ID}"
echo "#   nodes          = ${SLURM_NNODES} (${SLURM_JOB_NODELIST})"
echo "#   gpus_per_node  = ${SLURM_GPUS_PER_NODE}"
echo "============================================================"

srun --kill-on-bad-exit=1 \
  bash scripts/few_shot/run_dataset.sh "${DATASET}"
