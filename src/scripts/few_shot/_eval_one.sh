#!/bin/bash
# Single-(dataset × model × N × seed × prompt) eval — runs Stage-1 predictions
# from the existing analysis stack against a fine-tuned checkpoint.
#
# Usage: bash _eval_one.sh <dataset> <model> <N> <seed> <prompt>
#
# Output dir: ./experiments/few_shot/predictions/<DATASET>_<MODEL>_N<N>_seed<S>_<prompt>_0corr/

set -e

DATASET=$1; MODEL=$2; N=$3; SEED=$4; PROMPT=$5
if [ -z "$PROMPT" ]; then
  echo "Usage: $0 <dataset> <model> <N> <seed> <prompt>  (prompt: point|box)"
  exit 1
fi

case "$MODEL" in
  sam2_no_ft|medsam2|sonobase) ;;
  *) echo "Unknown model: $MODEL"; exit 1;;
esac

EXP_NAME="${DATASET}_${MODEL}_N${N}_seed${SEED}"
CKPT="./experiments/few_shot/checkpoints/${EXP_NAME}/final.pth"
RUN_NAME="${EXP_NAME}_${PROMPT}_0corr"
OUT="./experiments/few_shot/predictions/${RUN_NAME}"

if [ ! -f "$CKPT" ]; then
  echo "[$EXP_NAME] checkpoint not found: $CKPT — run _finetune_one.sh first."
  exit 1
fi

# Resume short-circuit (FIX(E2)): skip if this prediction set is already
# complete. Mirrors the finetune.py final.pth guard so a re-submitted sweep
# only redoes missing cells.
if [ -f "$OUT/per_sample_metrics.csv" ]; then
  echo "[$RUN_NAME] predictions already exist — skipping eval ($OUT)."
  exit 0
fi

export HYDRA_FULL_ERROR=1
export PYTHONPATH=$PYTHONPATH:$(pwd)
export LD_LIBRARY_PATH=""

# Reuse the existing analysis Stage-1 recipe with the fine-tuned checkpoint.
# All test images for the dataset come from `data=${DATASET}` (the analysis
# data configs already point at the canonical project test split).
# experiment=<model> swaps the image_encoder via Hydra `override` (avoids
# the duplicate-defaults-list issue from CLI `+` overrides).
uv run python -m nemo_cv.recipes.analysis.save_predictions \
  -c ./configs/analysis -cn save_predictions \
  data=${DATASET} \
  experiment=${MODEL} \
  ckpt_path="$CKPT" \
  output_dir=${OUT} \
  run_name=${RUN_NAME} \
  prompt_protocol.type=${PROMPT} \
  prompt_protocol.num_correction_clicks=0
