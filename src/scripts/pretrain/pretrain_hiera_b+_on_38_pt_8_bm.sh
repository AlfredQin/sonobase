#!/bin/bash
# Single-encoder ablation: Hiera-B+ on 38_pt_8_bm. Default
# scratch.ckpt_path already points at sam2.1_hiera_base_plus.pt.
#
# Run from src/.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/_pretrain_one.sh" pretrain_hiera_b+_on_38_pt_8_bm "$@"
