#!/bin/bash
# B2 standalone iterative-correction runner — invokes the
# `iterative_eval.py` recipe ONCE for a (model, dataset, prompt) combo,
# producing per-iteration metrics in a single pass with the image
# encoder run only once per batch (vs B1 which re-runs the encoder per
# iteration count).
#
# B2 is currently a *verification* tool; the production sweep uses
# `_run_b1_sweep.sh`. The two pipelines emit the same long-format
# `test_metrics_iterations.{json,csv}` schema so they can be diffed
# directly via `compare_b1_vs_b2.sh`.
#
# Usage:
#   bash _run_b2.sh <model> <dataset> <prompt> [<iter1> <iter2> ...]
#
# Default iteration set is `${DEFAULT_ITERATIONS[@]}` from `_common.sh`.
#
# Output directory:
#   ./experiments/benchmarks/<dataset>/<model_dir>/b2_<prompt>/
#     └── results/test_metrics_iterations.{json,csv}
#
# Run from src/.

set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <model> <dataset> <prompt> [<iter1> <iter2> ...]" >&2
  exit 2
fi

MODEL="$1"
DATASET="$2"
PROMPT="$3"
shift 3

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./_common.sh
source "${SCRIPT_DIR}/_common.sh"

if [[ $# -gt 0 ]]; then
  ITERS=("$@")
else
  ITERS=("${DEFAULT_ITERATIONS[@]}")
fi
# Hydra wants a Python list literal, e.g. iterations=[0,1,3,5,7]
ITERS_CSV="$(IFS=,; echo "${ITERS[*]}")"
ITERS_LITERAL="[${ITERS_CSV}]"

# Checkpoint resolution delegated to the shared scripts/_resolve_ckpt.sh
# (honors CKPT_PATH + the lazy sonobase DCP->PT conversion).
CKPT_PATH="$(CKPT_PATH="${CKPT_PATH:-}" \
  bash "${SCRIPT_DIR}/../../../_resolve_ckpt.sh" "${MODEL}")"

EXPERIMENT="$(_experiment_for_model_dataset "${MODEL}" "${DATASET}")"
MODEL_DIR="$(_model_dir_name "${MODEL}")"
PROB_BOX="$(_prob_box_for_prompt "${PROMPT}")"

EXP_NAME="benchmarks/${DATASET}/${MODEL_DIR}/b2_${PROMPT}"

echo "########################################################"
echo "# B2 standalone: ${MODEL} on ${DATASET} (${PROMPT})"
echo "# iterations:  ${ITERS_LITERAL}"
echo "# checkpoint:  ${CKPT_PATH}"
echo "# output:      ./experiments/${EXP_NAME}"
echo "# num_gpus:    ${NUM_GPUS}"
echo "########################################################"

uv run torchrun --nproc_per_node="${NUM_GPUS}" -m nemo_cv.recipes.benchmarks.iterative_eval \
  -c ./configs/test_sam2 -cn iterative_eval \
  experiment="${EXPERIMENT}" \
  ckpt_path="${CKPT_PATH}" \
  iterations="${ITERS_LITERAL}" \
  scratch.prob_to_use_box_input_for_eval="${PROB_BOX}" \
  scratch.experiment_name="${EXP_NAME}"

echo ""
echo "Done. B2 results at:"
echo "  ./experiments/${EXP_NAME}/results/test_metrics_iterations.csv"
echo "  ./experiments/${EXP_NAME}/results/test_metrics_iterations.json"
