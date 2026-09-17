#!/bin/bash
# B2 standalone iterative-correction verification on BUSI + CAMUS.
# Same coverage as `run_b1_iterations_busi_camus.sh`, but each
# (model, prompt) combo runs only ONE process (the encoder is cached
# across iterations). Use this to validate that the standalone pipeline
# produces metrics consistent with the B1 default before considering
# B2 for production.
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
echo "# B2 standalone sweep: ${#MODELS[@]} models × ${#PROMPTS[@]} prompts on ${DATASET}"
echo "########################################################"
echo ""

i=0
total=$(( ${#MODELS[@]} * ${#PROMPTS[@]} ))
for model in "${MODELS[@]}"; do
  for prompt in "${PROMPTS[@]}"; do
    i=$((i + 1))
    echo ""
    echo "=== [${i}/${total}] B2 run: ${model} | ${prompt} prompt ==="
    bash "${SCRIPT_DIR}/_run_b2.sh" "${model}" "${DATASET}" "${prompt}"
  done
done

echo ""
echo "Done. Per-(model, prompt) B2 CSVs:"
for model in "${MODELS[@]}"; do
  model_dir="$model"
  if [[ "$model" == "sam2_no_ft" ]]; then model_dir="sam2"; fi
  if [[ "$model" == "sonobase" ]]; then model_dir="hiera_b_conv_s_conv_t"; fi
  for prompt in "${PROMPTS[@]}"; do
    echo "  ./experiments/benchmarks/${DATASET}/${model_dir}/b2_${prompt}/results/test_metrics_iterations.csv"
  done
done
