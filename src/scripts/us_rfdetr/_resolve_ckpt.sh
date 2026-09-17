#!/bin/bash
# Thin shim — the model -> checkpoint resolver now lives in one shared place,
# scripts/_resolve_ckpt.sh. Kept so existing us_rfdetr wrappers (_run_one.sh)
# keep calling `_resolve_ckpt.sh` from this directory unchanged.
#
# See scripts/_resolve_ckpt.sh for the full contract + env knobs
# (CHECKPOINT_DIR, CKPT_PATH, SONOBASE_DCP).
#
# Usage:
#   PT=$(bash _resolve_ckpt.sh <backbone>)
#   <backbone> ∈ {sam2_no_ft, medsam2, sonobase}

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
exec bash "$SCRIPT_DIR/../_resolve_ckpt.sh" "$@"
