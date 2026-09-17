#!/bin/bash
# SonoBase pretraining — generic single-experiment slurm launcher.
#
# Submit with the experiment name in the EXPERIMENT env var (passed via
# `--export`); everything else uses sensible cluster defaults.
#
# Multi-node DDP example (2 nodes × 8 GPUs/node = 16 GPUs):
#
#   sbatch \
#     --export=ALL,EXPERIMENT=pretrain_hiera_b_conv_s_conv_t_on_38_pt_8_bm \
#     --nodes=2 \
#     --job-name=pt_hbconvs_38pt \
#     scripts/pretrain/slurm/_pretrain_one.sbatch
#
# Logs land at `<sbatch_submit_dir>/slurm_logs/<jobname>_<jobid>.{out,err}`.
# Project convention: invoke from inside the experiment dir so logs
# consolidate next to the experiment they describe, e.g.
#   mkdir -p experiments/sonobase/pretrain/38_pt_8_bm/hiera_b_conv_s_conv_t/slurm_logs
#   cd experiments/sonobase/pretrain/38_pt_8_bm/hiera_b_conv_s_conv_t
#   sbatch --export=ALL,EXPERIMENT=... <repo>/src/scripts/pretrain/slurm/_pretrain_one.sbatch
#
# Single-node DDP (8 GPUs on one node):
#
#   sbatch --export=ALL,EXPERIMENT=pretrain_hiera_t_on_38_pt_8_bm \
#     --nodes=1 \
#     scripts/pretrain/slurm/_pretrain_one.sbatch
#
# Resume a crashed run (BaseRecipe DCP checkpointer auto-resumes from
# the latest epoch in `experiments/<scratch.experiment_name>/`):
#
#   sbatch --export=ALL,EXPERIMENT=pretrain_hiera_b_conv_s_conv_t_on_38_pt_8_bm \
#     scripts/pretrain/slurm/_pretrain_one.sbatch
#   # (the recipe finds its own LATEST/ symlink — no extra flag needed)
#
# Optional env knobs (pass via --export=ALL,K=V,...):
#   GLOBAL_BATCH_SIZE   step_scheduler.global_batch_size (default 4)
#   EXTRA_OVERRIDES     extra Hydra overrides appended verbatim
#   CHECKPOINT_DIR      SAM2 release weights dir (REQUIRED — e.g. $WORK/Checkpoints)
#   DATASET_DIR         SaUS dataset root (REQUIRED — e.g. $WORK/Dataset/SaUS)
#   ANNOTATION_DIR      SaUS annotation root (REQUIRED — e.g. $WORK/Dataset/SaUS_Annotation/38_pt_8_bm_7_ext_v2)

#SBATCH --job-name=pretrain_us
#SBATCH --output=slurm_logs/%x_%A.out
#SBATCH --error=slurm_logs/%x_%A.err
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1               # ONE srun task per node — torchrun spawns the per-GPU procs
#SBATCH --cpus-per-task=128               # 8 GPUs × 16 CPUs/GPU
#SBATCH --gpus-per-node=8
#SBATCH --mem=0                           # all node memory (~1.5 TB); job already holds all 8 GPUs + 128 CPUs, so the node is exclusive anyway
#SBATCH --time=24:00:00

set -euo pipefail

# ---------------------------------------------------------------------------
#  Sanity-check env from --export
# ---------------------------------------------------------------------------
: "${EXPERIMENT:?EXPERIMENT must be set, e.g. EXPERIMENT=pretrain_hiera_b_conv_s_conv_t_on_38_pt_8_bm}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-4}"
EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"

# Asset paths — all three must be set per user in your .bashrc, e.g.:
#   export CHECKPOINT_DIR=$WORK/Checkpoints
#   export DATASET_DIR=$WORK/Dataset/SaUS
#   export ANNOTATION_DIR=$WORK/Dataset/SaUS_Annotation/38_pt_8_bm_7_ext_v2
: "${CHECKPOINT_DIR:?CHECKPOINT_DIR must be set, e.g. export CHECKPOINT_DIR=\$WORK/Checkpoints}"
: "${DATASET_DIR:?DATASET_DIR must be set, e.g. export DATASET_DIR=\$WORK/Dataset/SaUS}"
: "${ANNOTATION_DIR:?ANNOTATION_DIR must be set, e.g. export ANNOTATION_DIR=\$WORK/Dataset/SaUS_Annotation/38_pt_8_bm_7_ext_v2}"
export CHECKPOINT_DIR DATASET_DIR ANNOTATION_DIR

# Sonobase project root. Under `sbatch`, SLURM spools this script to a
# /var/spool/slurmd/... copy, so BASH_SOURCE points at the spool copy —
# NOT the repo. Recover the real submitted path from scontrol's Command=
# field (SLURM records it correctly for the running job).
SBATCH_PATH="$(scontrol show job "${SLURM_JOB_ID}" -o | grep -oP 'Command=\K[^[:space:]]+')"
[ -f "${SBATCH_PATH}" ] || { echo "ERROR: cannot resolve this sbatch script's path from scontrol (got '${SBATCH_PATH}')" >&2; exit 1; }
SCRIPT_DIR="$(cd "$(dirname "${SBATCH_PATH}")" && pwd)"
SRC_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${SRC_DIR}"

# Mirror SLURM's --output directive — defensive create for the
# follow-up jobs that inherit the same submit dir.
mkdir -p "${SLURM_SUBMIT_DIR:-$PWD}/slurm_logs"

# ---------------------------------------------------------------------------
#  Runtime env (matches scripts/pretrain/_pretrain_one.sh)
# ---------------------------------------------------------------------------
export HYDRA_FULL_ERROR=1
export PYTHONPATH="${PYTHONPATH:-}:${SRC_DIR}"
export LD_LIBRARY_PATH=""              # libcusparse / nvJitLink loader workaround

# Distributed-training niceties
export NCCL_DEBUG=WARN                 # Set to INFO for first multi-node run, then dial down
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# ---------------------------------------------------------------------------
#  Compute torchrun rendezvous
# ---------------------------------------------------------------------------
NNODES="${SLURM_NNODES}"
GPUS_PER_NODE="${SLURM_GPUS_PER_NODE:-8}"
HEAD_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
HEAD_NODE_IP=$(srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" hostname --ip-address)
RDZV_PORT=29500

echo "============================================================"
echo "# SonoBase pretrain (slurm)"
echo "#   job_id            = ${SLURM_JOB_ID}"
echo "#   job_name          = ${SLURM_JOB_NAME}"
echo "#   experiment        = ${EXPERIMENT}"
echo "#   nodes             = ${NNODES}  (${SLURM_JOB_NODELIST})"
echo "#   head_node_ip      = ${HEAD_NODE_IP}:${RDZV_PORT}"
echo "#   gpus_per_node     = ${GPUS_PER_NODE}"
echo "#   global_batch_size = ${GLOBAL_BATCH_SIZE}"
echo "#   CHECKPOINT_DIR    = ${CHECKPOINT_DIR}"
echo "#   DATASET_DIR       = ${DATASET_DIR}"
echo "#   ANNOTATION_DIR    = ${ANNOTATION_DIR}"
[ -n "${EXTRA_OVERRIDES}" ] && echo "#   extra overrides   = ${EXTRA_OVERRIDES}"
echo "============================================================"

# ---------------------------------------------------------------------------
#  Launch — one srun task per node, each invokes torchrun for the local procs
# ---------------------------------------------------------------------------
srun --kill-on-bad-exit=1 \
  uv run torchrun \
    --nnodes="${NNODES}" \
    --nproc_per_node="${GPUS_PER_NODE}" \
    --rdzv-id="${SLURM_JOB_ID}" \
    --rdzv-backend=c10d \
    --rdzv-endpoint="${HEAD_NODE_IP}:${RDZV_PORT}" \
    -m nemo_cv.recipes.sonobase.pretrain \
    -c ./configs/pretrain \
    experiment="${EXPERIMENT}" \
    step_scheduler.global_batch_size="${GLOBAL_BATCH_SIZE}" \
    ${EXTRA_OVERRIDES}
