#!/bin/bash
# Larger TriBranchTrunk variant: Hiera-L + ConvNeXt-B + ConvNeXt-S on
# 38_pt_8_bm. Branch0 needs the SAM2-Large release weights for
# initialisation (default scratch.ckpt_path points at Hiera-B+).
#
# Run from src/.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${CHECKPOINT_DIR:?CHECKPOINT_DIR must be set, e.g. export CHECKPOINT_DIR=\$WORK/Checkpoints}"
exec bash "${SCRIPT_DIR}/_pretrain_one.sh" pretrain_hiera_l_conv_b_conv_s_on_38_pt_8_bm \
  scratch.ckpt_path="${CHECKPOINT_DIR}/SAM2/sam2.1_hiera_large.pt" \
  "$@"
