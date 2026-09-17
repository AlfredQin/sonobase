#!/bin/bash
# Analysis Stage 1 — save predictions for one dataset across all 3
# models × 2 prompts (or whatever the per-dataset wrapper iterates).
#
# Submit:
#
#   sbatch --export=ALL,DATASET=HC18    scripts/analysis/slurm/_save_predictions_one.sbatch
#   sbatch --export=ALL,DATASET=ACOUSLIC scripts/analysis/slurm/_save_predictions_one.sbatch
#   sbatch --export=ALL,DATASET=KidneyUS scripts/analysis/slurm/_save_predictions_one.sbatch
#   ...
#
# Wraps `scripts/analysis/save_predictions/save_all_<DATASET>.sh`.
# DATASET ∈ {HC18, CAMUS, ACOUSLIC, RegPro, PorcineSpinalCord, BUSI, KidneyUS}.
#
# (HC18 has separate _point and _box wrappers due to its A3 iteration
# variants — pass DATASET=HC18 for both, the wrapper script handles
# the point/box split internally.)

#SBATCH --job-name=analysis_pred
#SBATCH --output=slurm_logs/%x_%A.out
#SBATCH --error=slurm_logs/%x_%A.err
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=512G
#SBATCH --time=04:00:00

set -euo pipefail

: "${DATASET:?DATASET must be set, e.g. DATASET=HC18}"

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

WRAPPER="scripts/analysis/save_predictions/save_all_${DATASET}.sh"
if [[ ! -f "${WRAPPER}" ]]; then
  echo "ERROR: no Stage-1 wrapper at ${WRAPPER}" >&2
  echo "  Available: $(ls scripts/analysis/save_predictions/save_all_*.sh)" >&2
  exit 1
fi

echo "============================================================"
echo "# Analysis Stage 1 — save predictions for ${DATASET} (slurm)"
echo "#   job_id  = ${SLURM_JOB_ID}"
echo "#   wrapper = ${WRAPPER}"
echo "============================================================"

srun --kill-on-bad-exit=1 bash "${WRAPPER}"
