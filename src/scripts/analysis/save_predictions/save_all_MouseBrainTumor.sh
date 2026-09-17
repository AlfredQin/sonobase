#!/bin/bash
# Stage-1 predictions for all 3 models on MouseBrainTumor, both prompts.
# Eval-only OOD (mouse) dataset; roles B2 + T3.2.
set -e
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
echo "########### MouseBrainTumor / 3 models × 2 prompts ###########"
for PROMPT in point box; do
  for MODEL in sam2_no_ft medsam2 sonobase; do
    echo ""
    echo "=== ${MODEL} / ${PROMPT} ==="
    bash "$SCRIPT_DIR/_save_one.sh" "$MODEL" MouseBrainTumor "$PROMPT"
  done
done
