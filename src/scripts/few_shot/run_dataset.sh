#!/bin/bash
# Run the full few-shot sweep for ONE dataset:
#   3 models × 6 N values × 3 seeds = 54 fine-tunes
# + 3 models × 6 N values × 3 seeds × 2 prompts = 108 evals.
#
# Usage: bash run_dataset.sh <DATASET>
#   DATASET ∈ {ACOUSLIC, DDTI, FUGC}

set -e
DATASET=$1
if [ -z "$DATASET" ]; then
  echo "Usage: $0 <DATASET>"; exit 1
fi

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

NS="1 2 5 10 20 30"
SEEDS="42 123 456"
MODELS="sonobase medsam2 sam2_no_ft"

echo "###############################################################"
echo "# ${DATASET} few-shot: 3 models × 6 N × 3 seeds = 54 fine-tunes"
echo "###############################################################"

for SEED in $SEEDS; do
  for N in $NS; do
    for MODEL in $MODELS; do
      EXP="${DATASET}_${MODEL}_N${N}_seed${SEED}"
      echo ""
      echo "--- [fine-tune] $EXP ---"
      bash "$SCRIPT_DIR/_finetune_one.sh" "$DATASET" "$MODEL" "$N" "$SEED"
    done
  done
done

echo ""
echo "###############################################################"
echo "# ${DATASET} few-shot: eval (point + box) per fine-tuned ckpt"
echo "###############################################################"

for SEED in $SEEDS; do
  for N in $NS; do
    for MODEL in $MODELS; do
      for PROMPT in point box; do
        echo ""
        echo "--- [eval ${PROMPT}] ${DATASET}_${MODEL}_N${N}_seed${SEED} ---"
        bash "$SCRIPT_DIR/_eval_one.sh" "$DATASET" "$MODEL" "$N" "$SEED" "$PROMPT"
      done
    done
  done
done

echo ""
echo "${DATASET} sweep complete."
