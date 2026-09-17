#!/bin/bash
# Early checkpoint: ACOUSLIC × {N=1, N=5} × 3 models × 3 seeds
# = 18 fine-tunes + 36 evaluations.

set -e
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

NS="1 5"
SEEDS="42 123 456"
MODELS="sonobase medsam2 sam2_no_ft"

for SEED in $SEEDS; do
  for N in $NS; do
    for MODEL in $MODELS; do
      bash "$SCRIPT_DIR/_finetune_one.sh" ACOUSLIC "$MODEL" "$N" "$SEED"
    done
  done
done

for SEED in $SEEDS; do
  for N in $NS; do
    for MODEL in $MODELS; do
      for PROMPT in point box; do
        bash "$SCRIPT_DIR/_eval_one.sh" ACOUSLIC "$MODEL" "$N" "$SEED" "$PROMPT"
      done
    done
  done
done
