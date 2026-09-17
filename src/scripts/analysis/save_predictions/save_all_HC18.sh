#!/bin/bash
# Stage-1 predictions for all 3 models on HC18 at both prompt protocols.
# Single entry point — used by the Slurm array launcher (_save_predictions_one.sbatch).
# For the A3 click-efficiency iteration sweep, see save_a3_iterations_HC18_{point,box}.sh.
set -e
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
echo "########### HC18 / 3 models × 2 prompts ###########"
for PROMPT in point box; do
  for MODEL in sam2_no_ft medsam2 sonobase; do
    echo ""
    echo "=== ${MODEL} / ${PROMPT} ==="
    bash "$SCRIPT_DIR/_save_one.sh" "$MODEL" HC18 "$PROMPT"
  done
done
