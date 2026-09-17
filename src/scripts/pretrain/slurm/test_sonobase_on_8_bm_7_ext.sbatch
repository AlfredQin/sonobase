#!/bin/bash
# SonoBase per-dataset test on 8_bm_7_ext (Phase 3 of the 5-phase plan).
#
# Submit (must point at a SonoBase pretraining DCP dir):
#
#   sbatch --export=ALL,RESTORE_FROM=./experiments/sonobase/pretrain/38_pt_8_bm/hiera_b_conv_s_conv_t/LATEST \
#     scripts/pretrain/slurm/test_sonobase_on_8_bm_7_ext.sbatch
#
# Single-iteration test (no iterative-correction sweep — just the
# default `num_correction_pt_per_frame_val=0` evaluation). For the full
# iterative sweep, use the benchmark scripts instead:
#   scripts/benchmarks/test_sam2/iterative/slurm/b1_sweep_all_8_bm_7_ext.sbatch
#
# Wall time: ~15-30 min on 8 GPUs.

#SBATCH --job-name=sonobase_test
#SBATCH --output=slurm_logs/%x_%A.out
#SBATCH --error=slurm_logs/%x_%A.err
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16                # 1 GPU × 16 CPUs
#SBATCH --gpus-per-node=1                 # eval runs 1-GPU: avoids DDP duplicate-sample padding (matches the headline benchmark launcher)
#SBATCH --mem=256G
#SBATCH --time=02:00:00

set -euo pipefail

: "${RESTORE_FROM:?RESTORE_FROM must be set, e.g. RESTORE_FROM=./experiments/sonobase/pretrain/.../LATEST}"

# Test experiment to run — override per variant via --export=ALL,EXPERIMENT=...
# Default preserves the original baseline test for backwards compatibility.
EXPERIMENT="${EXPERIMENT:-test_hiera_b_conv_s_conv_t_on_8_bm_7_ext}"
EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"

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
export NCCL_DEBUG=WARN
export NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

GPUS_PER_NODE="${SLURM_GPUS_PER_NODE:-1}"
HEAD_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
HEAD_NODE_IP=$(srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" hostname --ip-address)
# Per-job rendezvous port so concurrent 1-GPU jobs packed on the same node
# don't collide on a fixed port (matches test_baseline_on_8_bm_7_ext.sbatch).
RDZV_PORT=$((20000 + SLURM_JOB_ID % 10000))

echo "============================================================"
echo "# SonoBase test on 8_bm_7_ext (slurm)"
echo "#   job_id            = ${SLURM_JOB_ID}"
echo "#   restore_from      = ${RESTORE_FROM}"
echo "#   nodes             = ${SLURM_NNODES} (${SLURM_JOB_NODELIST})"
echo "#   gpus_per_node     = ${GPUS_PER_NODE}"
echo "============================================================"

srun --kill-on-bad-exit=1 \
  uv run torchrun \
    --nnodes="${SLURM_NNODES}" \
    --nproc_per_node="${GPUS_PER_NODE}" \
    --rdzv-id="${SLURM_JOB_ID}" \
    --rdzv-backend=c10d \
    --rdzv-endpoint="${HEAD_NODE_IP}:${RDZV_PORT}" \
    -m nemo_cv.recipes.sonobase.test \
    -c ./configs/pretrain -cn test \
    experiment="${EXPERIMENT}" \
    restore_from="${RESTORE_FROM}" \
    ${EXTRA_OVERRIDES}
