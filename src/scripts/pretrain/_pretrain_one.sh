#!/bin/bash
# Generic single-encoder SonoBase pretraining launcher.
# Usage: bash _pretrain_one.sh <experiment> [<extra hydra overrides...>]
#
#   <experiment>  one of:
#                   pretrain_hiera_b_conv_s_conv_t_on_38_pt_8_bm   (primary TriBranchTrunk)
#                   pretrain_hiera_l_conv_b_conv_s_on_38_pt_8_bm   (Large TriBranchTrunk)
#                   pretrain_hiera_b+_on_38_pt_8_bm                (single-encoder ablations)
#                   pretrain_hiera_l_on_38_pt_8_bm
#                   pretrain_hiera_s_on_38_pt_8_bm
#                   pretrain_hiera_t_on_38_pt_8_bm
#                   ...or anything else under configs/pretrain/experiment/
#
# Environment knobs:
#   CUDA_VISIBLE_DEVICES  GPUs to use (default: 0,1,2,3)
#   GLOBAL_BATCH_SIZE     step_scheduler.global_batch_size (default: 4)
#   CHECKPOINT_DIR        SAM2 release weights dir (REQUIRED — e.g. $WORK/Checkpoints)
#   EXTRA_OVERRIDES       additional Hydra overrides appended after experiment
#
# Run from src/.

set -e

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <experiment> [<extra hydra overrides...>]" >&2
  exit 2
fi

EXPERIMENT="$1"
shift
EXTRA="$@"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NUM_GPUS=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | grep -c .)
export HYDRA_FULL_ERROR=1
export PYTHONPATH=$PYTHONPATH:$(pwd)
export LD_LIBRARY_PATH=""
: "${CHECKPOINT_DIR:?CHECKPOINT_DIR must be set, e.g. export CHECKPOINT_DIR=\$WORK/Checkpoints}"
export CHECKPOINT_DIR
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-4}"

echo "========================================================"
echo "# SonoBase pretrain"
echo "#   experiment       = ${EXPERIMENT}"
echo "#   num_gpus         = ${NUM_GPUS}"
echo "#   global_batch_size = ${GLOBAL_BATCH_SIZE}"
echo "#   checkpoint_dir   = ${CHECKPOINT_DIR}"
[ -n "${EXTRA}" ] && echo "#   extra overrides  = ${EXTRA}"
echo "========================================================"

uv run torchrun --nproc_per_node="${NUM_GPUS}" -m nemo_cv.recipes.sonobase.pretrain \
  -c ./configs/pretrain \
  experiment="${EXPERIMENT}" \
  step_scheduler.global_batch_size="${GLOBAL_BATCH_SIZE}" \
  ${EXTRA}
