#!/bin/bash
# B1 single-iteration runner — invokes the standard test_sam2.py recipe
# once with a fixed `scratch.num_correction_pt_per_frame_val`. Used as
# the inner loop of `_run_b1_sweep.sh`. NOT the user-facing entry point.
#
# Usage:
#   bash _run_b1_one.sh <model> <dataset> <prompt> <iter> [<ckpt_path>]
#
# Where:
#   <model>    = sam2_no_ft | medsam2 | sonobase
#   <dataset>  = busi_camus | 8_bm_7_ext  (must match an existing
#                experiment overlay file under configs/test_sam2/experiment/)
#   <prompt>   = point | box   (sets scratch.prob_to_use_box_input_for_eval)
#   <iter>     = integer >= 0  (sets scratch.num_correction_pt_per_frame_val)
#   <ckpt_path>= optional override; otherwise resolved per-model by
#                scripts/_resolve_ckpt.sh (CKPT_PATH / SONOBASE_DCP).
#
# The output goes to:
#   ./experiments/benchmarks/<dataset>/<model_dir>/b1_<prompt>/iter<N>/
# Each iteration count gets its own subdir so the per-iter
# `test_metrics.json` files don't clobber each other before they are
# folded together by `aggregate_iterations.py`.
#
# Run from src/.

set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "Usage: $0 <model> <dataset> <prompt> <iter> [<ckpt_path>]" >&2
  exit 2
fi

MODEL="$1"
DATASET="$2"
PROMPT="$3"
ITER="$4"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./_common.sh
source "${SCRIPT_DIR}/_common.sh"

# Checkpoint resolution is delegated to the shared scripts/_resolve_ckpt.sh
# (honors CKPT_PATH for any model + the lazy sonobase DCP->PT conversion).
# Order: explicit positional `$5` > CKPT_PATH env > per-model default.
CKPT_PATH="$(CKPT_PATH="${5:-${CKPT_PATH:-}}" \
  bash "${SCRIPT_DIR}/../../../_resolve_ckpt.sh" "${MODEL}")"

EXPERIMENT="$(_experiment_for_model_dataset "${MODEL}" "${DATASET}")"
MODEL_DIR="$(_model_dir_name "${MODEL}")"
PROB_BOX="$(_prob_box_for_prompt "${PROMPT}")"

EXP_NAME="benchmarks/${DATASET}/${MODEL_DIR}/b1_${PROMPT}/iter${ITER}"

echo "----- B1 single-iter run -----"
echo "  model      = ${MODEL}"
echo "  dataset    = ${DATASET}"
echo "  prompt     = ${PROMPT} (prob_to_use_box=${PROB_BOX})"
echo "  iter       = ${ITER}"
echo "  ckpt       = ${CKPT_PATH}"
echo "  output     = ./experiments/${EXP_NAME}"
echo "  num_gpus   = ${NUM_GPUS}"
echo "------------------------------"

# --standalone: single-node rendezvous on a free ephemeral port (not the
# fixed default 29500). Required so that several of these jobs can run
# concurrently on the same node — e.g. the 6 (model × prompt) cells packed
# onto one node's free GPUs — without colliding on a shared c10d store.
uv run torchrun --standalone --nproc_per_node="${NUM_GPUS}" -m nemo_cv.recipes.benchmarks.test_sam2 \
  -c ./configs/test_sam2 -cn test \
  experiment="${EXPERIMENT}" \
  ckpt_path="${CKPT_PATH}" \
  scratch.num_correction_pt_per_frame_val="${ITER}" \
  scratch.prob_to_use_box_input_for_eval="${PROB_BOX}" \
  scratch.experiment_name="${EXP_NAME}"
