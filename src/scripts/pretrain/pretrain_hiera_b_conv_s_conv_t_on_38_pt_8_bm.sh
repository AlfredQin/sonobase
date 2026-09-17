#!/bin/bash
# Production: TriBranchTrunk (Hiera-B + ConvNeXt-S + ConvNeXt-T) on
# 38_pt_8_bm. Primary SonoBase pretraining run — the headline encoder
# benchmarked against SAM2-no-ft and MedSAM2.
#
# Run from src/.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/_pretrain_one.sh" pretrain_hiera_b_conv_s_conv_t_on_38_pt_8_bm "$@"
