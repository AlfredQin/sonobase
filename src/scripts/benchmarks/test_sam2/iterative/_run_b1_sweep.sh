#!/bin/bash
# B1 iterative-correction sweep — for one (model, dataset, prompt) combo:
#   1. Invoke `_run_b1_one.sh` once per iteration count, dropping a
#      separate `test_metrics.json` per iter.
#   2. Fold the per-iter JSONs into a single long-format
#      `test_metrics_iterations.csv` via `aggregate_iterations.py`.
#
# This is the production iterative-correction workflow. It sits next to
# the standalone B2 script (`_run_b2.sh`); both produce the same CSV
# schema so downstream consumers (a3_click_efficiency.py,
# compare_b1_vs_b2.sh) don't care which workflow generated their input.
#
# Usage:
#   bash _run_b1_sweep.sh <model> <dataset> <prompt> [<iter1> <iter2> ...]
#
# Default iteration set is `${DEFAULT_ITERATIONS[@]}` from `_common.sh`
# (currently 0 1 3 5 7).
#
# Output directory:
#   ./experiments/benchmarks/<dataset>/<model_dir>/b1_<prompt>/
#     ├── iter0/results/test_metrics.json
#     ├── iter1/results/test_metrics.json
#     ├── ...
#     └── test_metrics_iterations.{json,csv}   (the merged artifact)
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

MODEL_DIR="$(_model_dir_name "${MODEL}")"
SWEEP_DIR="./experiments/benchmarks/${DATASET}/${MODEL_DIR}/b1_${PROMPT}"
mkdir -p "${SWEEP_DIR}"

echo "########################################################"
echo "# B1 sweep: ${MODEL} on ${DATASET} (${PROMPT})"
echo "# iterations: ${ITERS[*]}"
echo "# output: ${SWEEP_DIR}"
echo "########################################################"

INPUTS_ARGS=()
for it in "${ITERS[@]}"; do
  echo ""
  echo "=== B1 iter=${it} ==="
  bash "${SCRIPT_DIR}/_run_b1_one.sh" "${MODEL}" "${DATASET}" "${PROMPT}" "${it}"
  INPUTS_ARGS+=( "${it}:${SWEEP_DIR}/iter${it}/results/test_metrics.json" )
done

echo ""
echo "=== Aggregating B1 sweep -> long-format CSV ==="
uv run python -m nemo_cv.recipes.benchmarks.aggregate_iterations \
  --inputs "${INPUTS_ARGS[@]}" \
  --output     "${SWEEP_DIR}/test_metrics_iterations.csv" \
  --json-output "${SWEEP_DIR}/test_metrics_iterations.json"

echo ""
echo "Done. B1 sweep merged at:"
echo "  ${SWEEP_DIR}/test_metrics_iterations.csv"
echo "  ${SWEEP_DIR}/test_metrics_iterations.json"
