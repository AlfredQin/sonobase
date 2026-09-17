#!/bin/bash
# Shared environment + helpers for the iterative-correction benchmark
# scripts (both B1 and B2). Source this file from the per-model /
# per-sweep wrappers — do NOT execute directly.

# Always run from the project src/ folder so that ./configs/* and
# ./experiments/* resolve correctly. Wrappers can override CUDA_VISIBLE_DEVICES
# before sourcing if they want to restrict GPUs.

if [[ -z "${CUDA_VISIBLE_DEVICES+x}" ]]; then
  export CUDA_VISIBLE_DEVICES=1,2,3,4
fi
NUM_GPUS=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | grep -c .)
export NUM_GPUS

export HYDRA_FULL_ERROR=1
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Workaround for the libcusparse / nvJitLink loader collision documented
# in the LESSONS section of src/scripts/pretrain/README.md.
export LD_LIBRARY_PATH=""

# Default iteration sweep — keep in sync with iterative_eval.yaml's
# top-level `iterations:` list. Override per-script if needed.
DEFAULT_ITERATIONS=(0 1 3 5 7)

# Checkpoint resolution (model -> .pt, including the lazy sonobase
# DCP->PT conversion and the CKPT_PATH / SONOBASE_DCP overrides) lives in
# the shared scripts/_resolve_ckpt.sh. The B1/B2 runners call it directly
# — see _run_b1_one.sh / _run_b2.sh.

# Map the model short name to the experiment overlay used by the test
# config tree. The experiment name embeds the dataset, so the pair
# (model, dataset) → experiment is deterministic.
_experiment_for_model_dataset() {
  local model="$1"
  local dataset="$2"
  case "$model" in
    sam2_no_ft) echo "test_sam2_no_ft_on_${dataset}" ;;
    medsam2)    echo "test_medsam2_on_${dataset}" ;;
    sonobase)   echo "test_hiera_b_conv_s_conv_t_on_${dataset}" ;;
    *)
      echo "ERROR: unknown model '$model' (expected sam2_no_ft/medsam2/sonobase)" >&2
      return 1 ;;
  esac
}

# The encoder-image-encoder short tag matches the directory layout on
# disk: experiments/benchmarks/<dataset>/<model_dir>/<...>. Sonobase uses
# the trunk variant name to disambiguate future encoder ablations.
_model_dir_name() {
  local model="$1"
  case "$model" in
    sam2_no_ft) echo "sam2" ;;
    medsam2)    echo "medsam2" ;;
    sonobase)   echo "hiera_b_conv_s_conv_t" ;;
    *)
      echo "ERROR: unknown model '$model'" >&2
      return 1 ;;
  esac
}

# Map prompt name to the scratch override (point=0, box=1).
_prob_box_for_prompt() {
  case "$1" in
    point) echo "0" ;;
    box)   echo "1" ;;
    *)
      echo "ERROR: prompt must be 'point' or 'box', got '$1'" >&2
      return 1 ;;
  esac
}
