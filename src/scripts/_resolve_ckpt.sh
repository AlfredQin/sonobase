#!/bin/bash
# Resolve a model's checkpoint to a single .pt path and print it on stdout.
# Progress/info messages go to stderr, so callers capture stdout cleanly:
#
#   CKPT="$(bash src/scripts/_resolve_ckpt.sh <model>)"
#
# This is the ONE place the model -> checkpoint mapping lives. Consumers:
#   * scripts/analysis/save_predictions/_save_one.sh
#   * scripts/few_shot/_finetune_one.sh
#   * scripts/us_rfdetr/_resolve_ckpt.sh   (thin shim -> this file)
#
#   <model>  one of: sam2_no_ft | medsam2 | sonobase
#
# Env knobs:
#   CHECKPOINT_DIR  Required for sam2_no_ft / medsam2 (holds their release
#                   weights); not needed for a sonobase or CKPT_PATH call.
#   CKPT_PATH       Per-call override for ANY model. May be an already-
#                   converted .pt (used as-is) or, for sonobase, a DCP
#                   directory (converted lazily). Wins over the default.
#   SONOBASE_DCP    sonobase only — REQUIRED unless CKPT_PATH is set. The
#                   pretrain run dir / DCP dir / `LATEST` symlink to convert.
#                   No default: a run can never silently fall back to a
#                   smoke / stale checkpoint.
#
# For sonobase a DCP directory is converted to a sibling .pt (`LATEST` ->
# `LATEST.pt`) via nemo_cv.recipes.benchmarks.convert_sonobase_dcp_to_pt and
# cached, so repeat calls are fast and idempotent.

set -euo pipefail

MODEL="${1:?Usage: $0 <sam2_no_ft|medsam2|sonobase>}"

# ---- 1. pick the source path (CKPT_PATH override wins for every model) ----
SRC="${CKPT_PATH:-}"
if [[ -z "$SRC" ]]; then
  # CHECKPOINT_DIR is required only by the sam2_no_ft / medsam2 branches.
  CKDIR_ERR="CHECKPOINT_DIR must be set, e.g. export CHECKPOINT_DIR=\$WORK/Checkpoints"
  case "$MODEL" in
    sam2_no_ft) SRC="${CHECKPOINT_DIR:?$CKDIR_ERR}/SAM2/sam2.1_hiera_base_plus.pt" ;;
    medsam2)    SRC="${CHECKPOINT_DIR:?$CKDIR_ERR}/MedSAM2/MedSAM2_latest.pt" ;;
    sonobase)
      # No default — require an explicit pretrain checkpoint so a run can
      # never silently fall back to a smoke / stale checkpoint. Set
      # SONOBASE_DCP to the pretrain run dir (or a DCP dir / `LATEST`
      # symlink), or CKPT_PATH to an already-converted .pt.
      SRC="${SONOBASE_DCP:?sonobase needs SONOBASE_DCP=<pretrain run dir> or CKPT_PATH=<converted .pt>}"
      ;;
    *) echo "ERROR: unknown model '$MODEL' (expected sam2_no_ft|medsam2|sonobase)" >&2; exit 2 ;;
  esac
fi

# ---- 2. a regular file is already a usable .pt — emit and stop ----
if [[ -f "$SRC" ]]; then
  echo "$SRC"
  exit 0
fi

# ---- 3. not a plain file ----
if [[ "$MODEL" != "sonobase" ]]; then
  echo "ERROR: $MODEL checkpoint not found (expected an existing .pt file): $SRC" >&2
  exit 1
fi
if [[ ! -d "$SRC" ]]; then
  echo "ERROR: sonobase checkpoint '$SRC' is neither a .pt file nor a DCP directory" >&2
  exit 1
fi

# ---- 4. sonobase DCP directory — convert (cached) then emit ----
# Resolve `LATEST` -> `epoch_N_step_M` so the converter records the real
# target in its provenance.
RESOLVED="$(readlink -f "$SRC" 2>/dev/null || echo "$SRC")"
# Key the cached .pt on the RESOLVED epoch dir, not the moving `LATEST`
# symlink: a retrain that advances `LATEST` then yields a fresh
# `epoch_N_step_M.pt` instead of silently reusing a stale `LATEST.pt`.
OUT="$(dirname "$SRC")/$(basename "$RESOLVED").pt"
if [[ -f "$OUT" ]]; then
  echo "[_resolve_ckpt] reusing cached $OUT" >&2
else
  echo "[_resolve_ckpt] converting DCP -> .pt" >&2
  echo "  source: $RESOLVED" >&2
  echo "  output: $OUT" >&2
  uv run python -m nemo_cv.recipes.benchmarks.convert_sonobase_dcp_to_pt \
    "$RESOLVED" "$OUT" >&2
fi
echo "$OUT"
