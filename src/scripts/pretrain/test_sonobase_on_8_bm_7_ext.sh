#!/bin/bash
# Real downstream evaluation: 8 benchmark + 7 external test sets (15 datasets).
# Set RESTORE_FROM to a fully-trained checkpoint dir before running. Run from
# the `src/` directory so config paths resolve.

export CUDA_VISIBLE_DEVICES=1,2,3,4
export HYDRA_FULL_ERROR=1
export PYTHONPATH=$PYTHONPATH:$(pwd)
export LD_LIBRARY_PATH=""

# Set RESTORE_FROM externally before running, e.g.:
#   RESTORE_FROM=./experiments/sonobase/pretrain/38_pt_8_bm/hiera_b_conv_s_conv_t/LATEST \
#     bash test_sonobase_on_8_bm_7_ext.sh
RESTORE_FROM="${RESTORE_FROM:-}"
if [[ -z "${RESTORE_FROM}" ]]; then
  echo "ERROR: RESTORE_FROM is not set." >&2
  echo "  Example: RESTORE_FROM=./experiments/sonobase/pretrain/38_pt_8_bm/hiera_b_conv_s_conv_t/LATEST bash $0" >&2
  exit 2
fi

uv run torchrun --nproc_per_node=4 -m nemo_cv.recipes.sonobase.test \
  -c ./configs/pretrain -cn test \
  experiment=test_hiera_b_conv_s_conv_t_on_8_bm_7_ext \
  restore_from="${RESTORE_FROM}"
