#!/bin/bash
# US-RFDETR — generic single-(dataset, backbone) slurm launcher.
#
# Submit via `--export=ALL,DATASET=<...>,BACKBONE=<...>`:
#
#   sbatch --export=ALL,DATASET=acouslic,BACKBONE=sonobase \
#     scripts/us_rfdetr/slurm/_us_rfdetr_one.sbatch
#
# What it does:
#   1. Resolves the pretrained encoder `.pt` for the chosen backbone via
#      `_resolve_ckpt.sh` (lazily converts SonoBase DCP shards if needed).
#   2. Launches a single-node DDP training run on 4 GPUs.
#   3. After training, the recipe automatically loads `best.pt` and runs
#      the test loop on the same allocation.
#
# A single (dataset, backbone) cell finishes in ~20-30 minutes on
# 4×A100 / RTX 6000 — well under the slurm wall limit.
#
# Optional env knobs (pass via --export):
#   MAX_EPOCHS          override trainer.max_epochs (default: 24, from train.yaml)
#   EXTRA_OVERRIDES     extra Hydra overrides appended verbatim
#   CHECKPOINT_DIR      release-weights dir (REQUIRED — e.g. $WORK/Checkpoints)
#   DATASET_DIR         SaUS dataset root (REQUIRED — e.g. $WORK/Dataset/SaUS)
#   DET_ANNOTATION_DIR  detection-split root (REQUIRED — e.g. $WORK/Dataset/Ultrasound/Dense_Pred_Annotation)
#   SONOBASE_DCP        SonoBase pretraining DCP dir for the sonobase backbone

#SBATCH --job-name=us_rfdetr
#SBATCH --output=slurm_logs/%x_%A.out
#SBATCH --error=slurm_logs/%x_%A.err
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1               # one srun task; torchrun spawns the per-GPU procs
#SBATCH --cpus-per-task=64                # 4 GPUs × 16 CPUs/GPU
#SBATCH --gpus-per-node=4
#SBATCH --mem=512G
#SBATCH --time=04:00:00                   # ~30 min/cell typical; 4 h headroom for FUGC/ACOUSLIC

set -euo pipefail

: "${DATASET:?DATASET must be set, e.g. DATASET=acouslic}"
: "${BACKBONE:?BACKBONE must be set, e.g. BACKBONE=sonobase}"
EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"
[[ -n "${MAX_EPOCHS:-}" ]] && EXTRA_OVERRIDES="trainer.max_epochs=${MAX_EPOCHS} ${EXTRA_OVERRIDES}"

# Asset paths — CHECKPOINT_DIR, DATASET_DIR, DET_ANNOTATION_DIR must be set
# per user in your .bashrc, e.g.:
#   export CHECKPOINT_DIR=$WORK/Checkpoints
#   export DATASET_DIR=$WORK/Dataset/SaUS
#   export DET_ANNOTATION_DIR=$WORK/Dataset/Ultrasound/Dense_Pred_Annotation
: "${CHECKPOINT_DIR:?CHECKPOINT_DIR must be set, e.g. export CHECKPOINT_DIR=\$WORK/Checkpoints}"
: "${DATASET_DIR:?DATASET_DIR must be set, e.g. export DATASET_DIR=\$WORK/Dataset/SaUS}"
: "${DET_ANNOTATION_DIR:?DET_ANNOTATION_DIR must be set, e.g. export DET_ANNOTATION_DIR=\$WORK/Dataset/Ultrasound/Dense_Pred_Annotation}"
export CHECKPOINT_DIR DATASET_DIR DET_ANNOTATION_DIR
export SONOBASE_DCP="${SONOBASE_DCP:-./experiments/sonobase/pretrain/busi_camus/hiera_b_conv_s_conv_t/LOWEST_VAL}"

SBATCH_PATH="$(scontrol show job "${SLURM_JOB_ID}" -o | grep -oP 'Command=\K[^[:space:]]+' | head -n1)"
[ -f "${SBATCH_PATH}" ] || { echo "ERROR: cannot resolve this sbatch script's path from scontrol (got '${SBATCH_PATH}')" >&2; exit 1; }
SCRIPT_DIR="$(cd "$(dirname "${SBATCH_PATH}")" && pwd)"
SRC_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${SRC_DIR}"

mkdir -p "${SLURM_SUBMIT_DIR:-$PWD}/slurm_logs"

# Runtime env
export HYDRA_FULL_ERROR=1
export PYTHONPATH="${PYTHONPATH:-}:${SRC_DIR}"
export LD_LIBRARY_PATH=""
export NCCL_DEBUG=WARN
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1

# Resolve pretrained encoder weights (lazily DCP→PT for sonobase)
PRETRAINED_PT="$(bash scripts/us_rfdetr/_resolve_ckpt.sh "${BACKBONE}")"

# Distributed setup (single-node)
GPUS_PER_NODE="${SLURM_GPUS_PER_NODE:-4}"
HEAD_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
HEAD_NODE_IP=$(srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" hostname --ip-address)
# FIX(E3): per-job port so co-located array tasks don't collide on one c10d
# store (fixed 29501 caused RendezvousConnectionError when 2 cells share a node).
RDZV_PORT=$((20000 + SLURM_JOB_ID % 10000))

EXPERIMENT="us_rfdetr_on_${DATASET}"

echo "============================================================"
echo "# US-RFDETR train+test (slurm)"
echo "#   job_id            = ${SLURM_JOB_ID}"
echo "#   dataset           = ${DATASET} (overlay=${EXPERIMENT})"
echo "#   backbone          = ${BACKBONE}"
echo "#   pretrained_pt     = ${PRETRAINED_PT}"
echo "#   nodes             = ${SLURM_NNODES}  (${SLURM_JOB_NODELIST})"
echo "#   gpus_per_node     = ${GPUS_PER_NODE}"
[ -n "${EXTRA_OVERRIDES}" ] && echo "#   extra overrides   = ${EXTRA_OVERRIDES}"
echo "============================================================"

srun --kill-on-bad-exit=1 \
  uv run torchrun \
    --nnodes="${SLURM_NNODES}" \
    --nproc_per_node="${GPUS_PER_NODE}" \
    --rdzv-id="${SLURM_JOB_ID}" \
    --rdzv-backend=c10d \
    --rdzv-endpoint="${HEAD_NODE_IP}:${RDZV_PORT}" \
    -m nemo_cv.recipes.us_rfdetr.train \
    -c ./configs/us_rfdetr -cn train \
    experiment="${EXPERIMENT}" \
    backbone="${BACKBONE}" \
    model.pretrained_ckpt="${PRETRAINED_PT}" \
    ${EXTRA_OVERRIDES}

echo ""
echo "Done. Outputs at: ./experiments/us_rfdetr/${DATASET}/${BACKBONE}/"
