#!/bin/bash

export CUDA_VISIBLE_DEVICES=1,2,3,4
export HYDRA_FULL_ERROR=1
export PYTHONPATH=$PYTHONPATH:$(pwd)
export LD_LIBRARY_PATH=""

uv run torchrun --nproc_per_node=4 -m nemo_cv.recipes.sonobase.pretrain \
-c ./configs/pretrain \
experiment=pretrain_hiera_b_conv_s_conv_t_on_busi_camus \
step_scheduler.global_batch_size=4
# experiment_name is inherited from the experiment YAML:
#   sonobase/pretrain/busi_camus/hiera_b_conv_s_conv_t
# Override on the CLI (`scratch.experiment_name=...`) if you want a fresh
# fork instead of auto-resuming an existing run.