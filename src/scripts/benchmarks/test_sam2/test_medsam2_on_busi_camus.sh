#!/bin/bash
# MedSAM2 zero-shot on BUSI + CAMUS test splits.
# (MedSAM2 is fine-tuned on medical images but not specifically on ultrasound.)
# Run from src/.

export CUDA_VISIBLE_DEVICES=1,2,3,4
export HYDRA_FULL_ERROR=1
export PYTHONPATH=$PYTHONPATH:$(pwd)
export LD_LIBRARY_PATH=""

: "${CHECKPOINT_DIR:?CHECKPOINT_DIR must be set, e.g. export CHECKPOINT_DIR=\$WORK/Checkpoints}"

uv run torchrun --nproc_per_node=4 -m nemo_cv.recipes.benchmarks.test_sam2 \
  -c ./configs/test_sam2 -cn test \
  experiment=test_medsam2_on_busi_camus \
  ckpt_path="${CHECKPOINT_DIR}/MedSAM2/MedSAM2_latest.pt"
