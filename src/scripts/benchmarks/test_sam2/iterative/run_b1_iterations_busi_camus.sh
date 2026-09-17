#!/bin/bash
# B1 iterative-correction sweep on the BUSI + CAMUS smoke datasets.
# Loops over (model in {sam2_no_ft, medsam2, sonobase}) × (prompt in
# {point, box}) × (iter in {0, 1, 3, 5, 7}). Each (model, prompt) combo
# produces one merged `test_metrics_iterations.{json,csv}` artifact.
#
# This is the smoke-scale equivalent of `run_b1_iterations_8_bm_7_ext.sh`
# — same workflow, smaller datasets — and is the fastest way to
# regression-test the iterative-correction infrastructure.
#
# Run from src/.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATASET="busi_camus"
MODELS=(sam2_no_ft medsam2 sonobase)
PROMPTS=(point box)

# Smoke wrapper: name the busi_camus pretrain so the sonobase model
# resolves without the caller exporting SONOBASE_DCP (the shared resolver
# requires it — no silent default). An explicit override still wins.
export SONOBASE_DCP="${SONOBASE_DCP:-./experiments/sonobase/pretrain/busi_camus/hiera_b_conv_s_conv_t/LOWEST_VAL}"

echo ""
echo "########################################################"
echo "# B1 iterative sweep: ${#MODELS[@]} models × ${#PROMPTS[@]} prompts on ${DATASET}"
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
echo "Done. Per-(model, prompt) merged CSVs at:"
for model in "${MODELS[@]}"; do
  model_dir="$model"
  if [[ "$model" == "sam2_no_ft" ]]; then model_dir="sam2"; fi
  if [[ "$model" == "sonobase" ]]; then model_dir="hiera_b_conv_s_conv_t"; fi
  for prompt in "${PROMPTS[@]}"; do
    echo "  ./experiments/benchmarks/${DATASET}/${model_dir}/b1_${prompt}/test_metrics_iterations.csv"
  done
done
