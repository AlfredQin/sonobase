#!/bin/bash
# Smoke test: evaluate the busi_camus pretrained checkpoint on the BUSI + CAMUS
# test splits. Run from the `src/` directory so config paths resolve.

export CUDA_VISIBLE_DEVICES=1,2,3,4
export HYDRA_FULL_ERROR=1
export PYTHONPATH=$PYTHONPATH:$(pwd)/../
export LD_LIBRARY_PATH=""

uv run torchrun --nproc_per_node=4 -m nemo_cv.recipes.sonobase.test \
-c ./configs/pretrain -cn test \
experiment=test_hiera_b_conv_s_conv_t_on_busi_camus \
restore_from=./experiments/sonobase/pretrain/busi_camus/hiera_b_conv_s_conv_t/LOWEST_VAL
