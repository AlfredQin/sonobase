#!/bin/bash
# Sonobase (TriBranchTrunk) on BUSI + CAMUS test splits, using a pretrain
# checkpoint converted from DCP to .pt format.
#
# Two-step workflow:
#   1) Resolve the sonobase checkpoint (lazy DCP -> .pt conversion)
#   2) Run the benchmark recipe pointing at that .pt
# Run from src/.

set -e

export CUDA_VISIBLE_DEVICES=1,2,3,4
export HYDRA_FULL_ERROR=1
export PYTHONPATH=$PYTHONPATH:$(pwd)
export LD_LIBRARY_PATH=""

# === step 1: resolve the sonobase checkpoint (DCP -> .pt, cached) ===
# Delegated to the shared scripts/_resolve_ckpt.sh. Defaults to the
# busi_camus pretrain's best-validation epoch (LOWEST_VAL); override by
# exporting SONOBASE_DCP. The resolver does the lazy, cached DCP->PT
# conversion and emits the .pt path on stdout.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PT_FILE="$(SONOBASE_DCP="${SONOBASE_DCP:-./experiments/sonobase/pretrain/busi_camus/hiera_b_conv_s_conv_t/LOWEST_VAL}" \
  bash "${SCRIPT_DIR}/../../_resolve_ckpt.sh" sonobase)"

# === step 2: run benchmark test ===

uv run torchrun --nproc_per_node=4 -m nemo_cv.recipes.benchmarks.test_sam2 \
  -c ./configs/test_sam2 -cn test \
  experiment=test_hiera_b_conv_s_conv_t_on_busi_camus \
  ckpt_path="$PT_FILE"
