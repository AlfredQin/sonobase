#!/bin/bash
# B1 iterative-correction sweep on the production 8 benchmark + 7
# external datasets. Same logic as the busi_camus smoke wrapper,
# scaled up to the full evaluation suite.
#
# Sweeps:
#   models  = {sam2_no_ft, medsam2, sonobase}
#   prompts = {point, box}
#   iters   = {0, 1, 3, 5, 7}
#
# Each (model, prompt) combo runs `len(iters)` × `test_sam2.py` calls,
# then aggregates them into a single long-format CSV. Total cost:
#
#   3 models × 2 prompts × 5 iters × (one full 15-dataset eval)
#   = 30 single-iter eval runs
#
# Plan a long wall-clock window or split into nohup blocks. The sonobase
# entry requires SONOBASE_DCP (or CKPT_PATH) to be set to the production
# pretraining checkpoint — see scripts/_resolve_ckpt.sh.
#
# Run from src/.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATASET="8_bm_7_ext"
MODELS=(sam2_no_ft medsam2 sonobase)
PROMPTS=(point box)

echo ""
echo "########################################################"
echo "# B1 iterative sweep (PROD): ${#MODELS[@]} models × ${#PROMPTS[@]} prompts on ${DATASET}"
echo "########################################################"
echo ""

i=0
total=$(( ${#MODELS[@]} * ${#PROMPTS[@]} ))
for model in "${MODELS[@]}"; do
  for prompt in "${PROMPTS[@]}"; do
    i=$((i + 1))
    echo ""
    echo "=== [${i}/${total}] B1 sweep: ${model} | ${prompt} prompt ==="
    bash "${SCRIPT_DIR}/_run_b1_sweep.sh" "${model}" "${DATASET}" "${prompt}"
  done
done

echo ""
echo "Done. Per-(model, prompt) merged CSVs:"
for model in "${MODELS[@]}"; do
  model_dir="$model"
  if [[ "$model" == "sam2_no_ft" ]]; then model_dir="sam2"; fi
  if [[ "$model" == "sonobase" ]]; then model_dir="hiera_b_conv_s_conv_t"; fi
  for prompt in "${PROMPTS[@]}"; do
    echo "  ./experiments/benchmarks/${DATASET}/${model_dir}/b1_${prompt}/test_metrics_iterations.csv"
  done
done
