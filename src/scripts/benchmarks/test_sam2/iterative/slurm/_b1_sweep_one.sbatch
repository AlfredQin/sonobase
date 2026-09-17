#!/bin/bash
# Benchmark B1 iterative-correction sweep — one (model, dataset, prompt)
# combo on slurm. Invokes the existing `_run_b1_sweep.sh` (which loops
# the underlying `test_sam2.py` recipe across iterations [0,1,3,5,7]
# and aggregates the per-iter JSONs into a single long-format CSV).
#
# Submit:
#
#   sbatch --export=ALL,MODEL=sam2_no_ft,DATASET=8_bm_7_ext,PROMPT=point \
#     scripts/benchmarks/test_sam2/iterative/slurm/_b1_sweep_one.sbatch
#
# Runs at 1 GPU (NOT 4) — multi-GPU DDP pads the eval set to a multiple of
# the world size, and those duplicate samples bias the pooled metric. The
# eval must be sample-exact, so the whole sweep is single-GPU.
#
# Wall time budget at 1 GPU: each (model, prompt) cell is 5 full 15-dataset
# evals (iters 0,1,3,5,7) — ~3-5 h for the slow sonobase/video cells, less
# for the baselines. The 12-hour limit gives headroom. Several cells run in
# parallel across the cluster (one GPU each), so total wall-clock stays low.

#SBATCH --job-name=bm_b1_sweep
#SBATCH --output=slurm_logs/%x_%A.out
#SBATCH --error=slurm_logs/%x_%A.err
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16               # 1 GPU × 16 CPUs/GPU
#SBATCH --gpus-per-node=1
#SBATCH --mem=256G                       # modest: 1-GPU eval; lower mem packs more cells per node
#SBATCH --time=12:00:00

set -euo pipefail

: "${MODEL:?MODEL must be set: sam2_no_ft | medsam2 | sonobase}"
: "${DATASET:?DATASET must be set: busi_camus | 8_bm_7_ext}"
: "${PROMPT:?PROMPT must be set: point | box}"

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
SRC_DIR="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
cd "${SRC_DIR}"

mkdir -p "${SLURM_SUBMIT_DIR:-$PWD}/slurm_logs"

# Runtime env (matches the iterative/_common.sh that _run_b1_sweep
# sources internally — re-exporting here is harmless).
export HYDRA_FULL_ERROR=1
export PYTHONPATH="${PYTHONPATH:-}:${SRC_DIR}"
export LD_LIBRARY_PATH=""
export NCCL_DEBUG=WARN
export NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"   # 1 GPU — see header (sample-exact eval, no DDP padding)

echo "============================================================"
echo "# B1 iterative sweep (slurm)"
echo "#   job_id            = ${SLURM_JOB_ID}"
echo "#   model             = ${MODEL}"
echo "#   dataset           = ${DATASET}"
echo "#   prompt            = ${PROMPT}"
echo "#   nodes             = ${SLURM_NNODES} (${SLURM_JOB_NODELIST})"
echo "#   gpus_per_node     = ${SLURM_GPUS_PER_NODE}"
echo "============================================================"

# Wrap with srun so slurm tracks the GPU allocation.
srun --kill-on-bad-exit=1 \
  bash scripts/benchmarks/test_sam2/iterative/_run_b1_sweep.sh \
    "${MODEL}" "${DATASET}" "${PROMPT}"
