#!/bin/bash
# Single-iteration (zero-shot, num_correction_pt_per_frame_val=0) test of one
# model on the 8 benchmark + 7 external split (15 datasets) via the benchmark
# `test_sam2` recipe — the canonical comparison path whose `test_metrics.json`
# is consumed by `compare_benchmarks.py`.
#
# Env-driven:
#   MODEL   = sam2_no_ft | medsam2 | sonobase        (required)
#   PROMPT  = point | box                            (default: point)
#
# For sonobase you MUST also set SONOBASE_DCP (or CKPT_PATH) to the production
# pretraining checkpoint, e.g.:
#   SONOBASE_DCP=./experiments/sonobase/pretrain/38_pt_8_bm/hiera_b_conv_s_conv_t_20ep/LOWEST_VAL
#
# Submit examples:
#   sbatch --export=ALL,MODEL=sam2_no_ft,PROMPT=box \
#     scripts/benchmarks/test_sam2/slurm/test_baseline_on_8_bm_7_ext.sbatch
#   sbatch --export=ALL,MODEL=sonobase,PROMPT=point,SONOBASE_DCP=./experiments/.../LOWEST_VAL \
#     scripts/benchmarks/test_sam2/slurm/test_baseline_on_8_bm_7_ext.sbatch
#
# Output: ./experiments/benchmarks/8_bm_7_ext/<model_dir>/<p0|p0_box>/results/
#
# 1 GPU per task by default — avoids the DDP duplicate-sample padding bias
# observed at multi-GPU (5217 unique samples vs 5240 padded). Use the
# benchmarks/test_sam2/iterative/slurm/b1_sweep_all_8_bm_7_ext.sbatch
# script if you need the full 5-iteration correction sweep instead.

#SBATCH --job-name=test_baseline_8bm7ext
#SBATCH --output=slurm_logs/%x_%A.out
#SBATCH --error=slurm_logs/%x_%A.err
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=256G
#SBATCH --time=02:00:00

set -euo pipefail

: "${MODEL:?MODEL must be set: sam2_no_ft | medsam2 | sonobase}"
PROMPT="${PROMPT:-point}"

# Map model -> (experiment overlay, on-disk dir). sonobase reuses the
# hiera_b_conv_s_conv_t encoder overlay and a matching results dir.
case "${MODEL}" in
  sam2_no_ft) MODEL_DIR="sam2_no_ft"; DEFAULT_EXP="test_sam2_no_ft_on_8_bm_7_ext" ;;
  medsam2)    MODEL_DIR="medsam2";    DEFAULT_EXP="test_medsam2_on_8_bm_7_ext" ;;
  sonobase)   MODEL_DIR="hiera_b_conv_s_conv_t"; DEFAULT_EXP="test_hiera_b_conv_s_conv_t_on_8_bm_7_ext" ;;
  *) echo "ERROR: MODEL must be one of {sam2_no_ft, medsam2, sonobase}, got '${MODEL}'" >&2; exit 2 ;;
esac

# Map prompt -> (box-probability override, output tag).
case "${PROMPT}" in
  point) PROB_BOX=0; TAG="p0" ;;
  box)   PROB_BOX=1; TAG="p0_box" ;;
  *) echo "ERROR: PROMPT must be one of {point, box}, got '${PROMPT}'" >&2; exit 2 ;;
esac

EXPERIMENT="${EXPERIMENT:-${DEFAULT_EXP}}"
EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"

: "${CHECKPOINT_DIR:?CHECKPOINT_DIR must be set, e.g. export CHECKPOINT_DIR=\$WORK/Checkpoints}"
: "${DATASET_DIR:?DATASET_DIR must be set, e.g. export DATASET_DIR=\$WORK/Dataset/SaUS}"
: "${ANNOTATION_DIR:?ANNOTATION_DIR must be set, e.g. export ANNOTATION_DIR=\$WORK/Dataset/SaUS_Annotation/38_pt_8_bm_7_ext}"
export CHECKPOINT_DIR DATASET_DIR ANNOTATION_DIR

SBATCH_PATH="$(scontrol show job "${SLURM_JOB_ID}" -o | grep -oP 'Command=\K[^[:space:]]+')"
[ -f "${SBATCH_PATH}" ] || { echo "ERROR: cannot resolve this sbatch script's path from scontrol (got '${SBATCH_PATH}')" >&2; exit 1; }
SCRIPT_DIR="$(cd "$(dirname "${SBATCH_PATH}")" && pwd)"
SRC_DIR="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${SRC_DIR}"

mkdir -p "${SLURM_SUBMIT_DIR:-$PWD}/slurm_logs"

# Resolve the model's .pt checkpoint via the canonical resolver. For sonobase
# this honors SONOBASE_DCP / CKPT_PATH and reuses the cached converted .pt.
CKPT_PATH="$(bash "${SRC_DIR}/scripts/_resolve_ckpt.sh" "${MODEL}")"

export HYDRA_FULL_ERROR=1
export PYTHONPATH="${PYTHONPATH:-}:${SRC_DIR}"
export LD_LIBRARY_PATH=""
export NCCL_DEBUG=WARN
export NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

GPUS_PER_NODE="${SLURM_GPUS_PER_NODE:-1}"
HEAD_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
HEAD_NODE_IP=$(srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" hostname --ip-address)
# Per-job unique rendezvous port. A FIXED port collides when several of these
# 1-GPU jobs pack onto the same node: when a short job tears down its c10d
# store on the shared port, a still-running job's next store sync fails with
# RendezvousConnectionError ("remote server shutdown or crash"). Deriving the
# port from the job id keeps concurrent jobs isolated.
RDZV_PORT=$((20000 + SLURM_JOB_ID % 10000))

EXP_NAME="benchmarks/8_bm_7_ext/${MODEL_DIR}/${TAG}"

echo "============================================================"
echo "# Benchmark p0 test on 8_bm_7_ext (slurm)"
echo "#   job_id            = ${SLURM_JOB_ID}"
echo "#   model             = ${MODEL}"
echo "#   prompt            = ${PROMPT} (prob_to_use_box=${PROB_BOX})"
echo "#   experiment        = ${EXPERIMENT}"
echo "#   ckpt_path         = ${CKPT_PATH}"
echo "#   nodes             = ${SLURM_NNODES} (${SLURM_JOB_NODELIST})"
echo "#   gpus_per_node     = ${GPUS_PER_NODE}"
echo "#   output            = ./experiments/${EXP_NAME}"
echo "============================================================"

srun --kill-on-bad-exit=1 \
  uv run torchrun \
    --nnodes="${SLURM_NNODES}" \
    --nproc_per_node="${GPUS_PER_NODE}" \
    --rdzv-id="${SLURM_JOB_ID}" \
    --rdzv-backend=c10d \
    --rdzv-endpoint="${HEAD_NODE_IP}:${RDZV_PORT}" \
    -m nemo_cv.recipes.benchmarks.test_sam2 \
    -c ./configs/test_sam2 -cn test \
    experiment="${EXPERIMENT}" \
    ckpt_path="${CKPT_PATH}" \
    scratch.experiment_name="${EXP_NAME}" \
    scratch.prob_to_use_box_input_for_eval="${PROB_BOX}" \
    ${EXTRA_OVERRIDES}
