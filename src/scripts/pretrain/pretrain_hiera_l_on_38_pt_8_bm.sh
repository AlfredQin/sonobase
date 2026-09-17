#!/bin/bash
# Single-encoder ablation: Hiera-L on 38_pt_8_bm. Override
# scratch.ckpt_path to load Meta's released Hiera-L trunk weights.
#
# Run from src/.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${CHECKPOINT_DIR:?CHECKPOINT_DIR must be set, e.g. export CHECKPOINT_DIR=\$WORK/Checkpoints}"
exec bash "${SCRIPT_DIR}/_pretrain_one.sh" pretrain_hiera_l_on_38_pt_8_bm \
  scratch.ckpt_path="${CHECKPOINT_DIR}/SAM2/sam2.1_hiera_large.pt" \
  "$@"
