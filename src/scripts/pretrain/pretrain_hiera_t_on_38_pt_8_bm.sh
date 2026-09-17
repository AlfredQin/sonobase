#!/bin/bash
# Single-encoder ablation: Hiera-T on 38_pt_8_bm — same trunk
# architecture as MedSAM2, so this is the most direct
# encoder-architecture-matched comparison against MedSAM2's medical
# fine-tune. Override scratch.ckpt_path to load Meta's Hiera-Tiny
# trunk weights.
#
# Run from src/.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${CHECKPOINT_DIR:?CHECKPOINT_DIR must be set, e.g. export CHECKPOINT_DIR=\$WORK/Checkpoints}"
exec bash "${SCRIPT_DIR}/_pretrain_one.sh" pretrain_hiera_t_on_38_pt_8_bm \
  scratch.ckpt_path="${CHECKPOINT_DIR}/SAM2/sam2.1_hiera_tiny.pt" \
  "$@"
