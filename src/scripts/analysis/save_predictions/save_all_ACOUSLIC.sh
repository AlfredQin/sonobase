#!/bin/bash
# Stage-1 predictions for all 3 models on ACOUSLIC at both prompt protocols.
set -e
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
echo "########### ACOUSLIC / 3 models × 2 prompts ###########"
for PROMPT in point box; do
  for MODEL in sam2_no_ft medsam2 sonobase; do
    echo ""
    echo "=== ${MODEL} / ${PROMPT} ==="
    bash "$SCRIPT_DIR/_save_one.sh" "$MODEL" ACOUSLIC "$PROMPT"
  done
done
