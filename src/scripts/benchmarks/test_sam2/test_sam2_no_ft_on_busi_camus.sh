#!/bin/bash
# SAM2 (no fine-tune) zero-shot on BUSI + CAMUS test splits.
# Run from src/.

export CUDA_VISIBLE_DEVICES=1,2,3,4
export HYDRA_FULL_ERROR=1
export PYTHONPATH=$PYTHONPATH:$(pwd)
export LD_LIBRARY_PATH=""

: "${CHECKPOINT_DIR:?CHECKPOINT_DIR must be set, e.g. export CHECKPOINT_DIR=\$WORK/Checkpoints}"

uv run torchrun --nproc_per_node=4 -m nemo_cv.recipes.benchmarks.test_sam2 \
  -c ./configs/test_sam2 -cn test \
  experiment=test_sam2_no_ft_on_busi_camus \
  ckpt_path="${CHECKPOINT_DIR}/SAM2/sam2.1_hiera_base_plus.pt"
