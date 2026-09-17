#!/bin/bash
# Generic single-(model × dataset × prompt) Stage-1 invocation.
# Usage: bash _save_one.sh <model> <dataset> <prompt> [extra hydra overrides...]
#
#   model    one of: sam2_no_ft, medsam2, sonobase
#   dataset  one of: HC18, CAMUS, ACOUSLIC, RegPro, PorcineSpinalCord
#   prompt   one of: point, box
#
# Resolves the right .pt checkpoint per model and (for sonobase) runs the
# DCP→PT conversion lazily. All other parameters can be overridden as
# additional Hydra overrides on the command line, e.g.:
#
#   bash _save_one.sh sonobase CAMUS point max_samples=10

set -e

MODEL=$1; DATASET=$2; PROMPT=$3
shift 3 || true
EXTRA="$@"

if [ -z "$MODEL" ] || [ -z "$DATASET" ] || [ -z "$PROMPT" ]; then
  echo "Usage: $0 <model> <dataset> <prompt> [extra hydra overrides...]"
  exit 1
fi

# `CHECKPOINT_DIR` must be set per user (e.g. `export CHECKPOINT_DIR=$WORK/Checkpoints`
# in your .bashrc) — points at the SAM2/MedSAM2 release weights. Checkpoint
# resolution (the `CKPT_PATH` / `SONOBASE_DCP` overrides and the lazy sonobase
# DCP→PT conversion) is delegated to the shared resolver.
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")

# ---- resolve checkpoint ----
CKPT="$(bash "$SCRIPT_DIR/../../_resolve_ckpt.sh" "$MODEL")"

# Correction-click count (click-efficiency curve). Defaults to 0, which
# reproduces the previous behaviour and the previous run name exactly — the
# existing archive must stay addressable. A non-zero value changes BOTH the run
# name and the Hydra override: passing the override alone as an EXTRA would
# collide, since every level would still land in the same `_0corr` directory.
NCORR="${NCORR:-0}"
RUN_NAME="${MODEL}_${DATASET}_${PROMPT}_${NCORR}corr"
# Optional box-prompt jitter (robustness sweep). Unset or 0 reproduces the
# previous behaviour exactly, including the run name — the existing archive must
# stay addressable. A non-zero value earns a `_jitNN` suffix (NN = percent) so a
# sweep can never overwrite the exact-box records it is compared against.
BOX_JITTER="${BOX_JITTER:-0}"
JIT_SUFFIX=""
if [ "$(awk -v j="$BOX_JITTER" 'BEGIN{print (j>0)?1:0}')" = "1" ]; then
  JIT_SUFFIX="_jit$(awk -v j="$BOX_JITTER" 'BEGIN{printf "%02d", j*100}')"
  EXTRA="${EXTRA} prompt_protocol.box_jitter_pct=${BOX_JITTER}"
fi

# Optional point-prompt sampler (prompt-noise study). `center` is the RITM
# distance-transform maximum used by every published run; `uniform` draws one
# pixel uniformly from the GT foreground, which is what SAM2 trains with.
# Earns a `_unif` suffix so a randomised run can never overwrite a center one.
if [ "${PT_SAMPLING:-center}" = "uniform" ]; then
  JIT_SUFFIX="${JIT_SUFFIX}_unif"
  EXTRA="${EXTRA} prompt_protocol.point_sampling=uniform"
fi

# Prompt-stream seed. Only meaningful alongside a randomised prompt (jittered
# box or uniform point); with neither, the prompt is deterministic and the seed
# would silently produce identical runs under different names.
if [ -n "${PROMPT_SEED:-}" ]; then
  JIT_SUFFIX="${JIT_SUFFIX}_s${PROMPT_SEED}"
  EXTRA="${EXTRA} prompt_protocol.seed=${PROMPT_SEED}"
fi

# Optional ablation knobs. Both default to the shipped behaviour and add
# nothing to the run name, so existing wrappers reproduce their records exactly.
#   MULTIMASK=0  -> single mask instead of best-of-3, run name gains `_mm0`
#   SCORE_SQ=1   -> also record iou_sq1024 / dice_sq1024 (extra columns only,
#                   native iou/dice unchanged), run name gains `_sq`
ABL_SUFFIX=""
if [ "${MULTIMASK:-1}" = "0" ]; then
  ABL_SUFFIX="${ABL_SUFFIX}_mm0"
  EXTRA="${EXTRA} prompt_protocol.multimask_output=false"
fi
if [ "${SCORE_SQ:-0}" = "1" ]; then
  ABL_SUFFIX="${ABL_SUFFIX}_sq"
  EXTRA="${EXTRA} score_square_grid=true"
fi
# Video datasets need a different lever for the same factor. `multimask_output`
# is an argument to SAM2ImagePredictor.predict and so reaches the image path
# only; the video predictor encodes a box as two corner points and multimask is
# then gated inside SAM2Base by `multimask_max_pt_num` (1 by default, hence
# single-mask). Raising it to 2 is what turns best-of-3 on for a box on video.
if [ -n "${MM_MAXPT:-}" ]; then
  ABL_SUFFIX="${ABL_SUFFIX}_mmpt${MM_MAXPT}"
  EXTRA="${EXTRA} model.multimask_max_pt_num=${MM_MAXPT}"
fi
# Free-form tail, so a corrected re-run can sit beside a superseded one instead
# of resuming into it -- the recipe skips any sample that already has a
# meta.json, so re-running into the same directory would inherit its rows.
ABL_SUFFIX="${ABL_SUFFIX}${RUN_SUFFIX:-}"

RUN_NAME="${RUN_NAME}${JIT_SUFFIX}${ABL_SUFFIX}"
# OUT_ROOT lets a smoke run write somewhere other than the archive. Hydra
# rejects a duplicate `output_dir=` override, so it has to be set here rather
# than appended to EXTRA by the caller.
OUT="${OUT_ROOT:-./experiments/analysis/predictions}/${RUN_NAME}"

export HYDRA_FULL_ERROR=1
export PYTHONPATH=$PYTHONPATH:$(pwd)
export LD_LIBRARY_PATH=""

# `experiment=${MODEL}` swaps the image encoder via Hydra `override` syntax.
# Earlier this script used `+model/image_encoder@model.image_encoder=...`
# which crashes with "model/image_encoder appears more than once in the
# final defaults list" (because model/sam2.yaml's defaults already declare
# the encoder). Fixed to match `_eval_one.sh`.
uv run python -m nemo_cv.recipes.analysis.save_predictions \
  -c ./configs/analysis -cn save_predictions \
  data=${DATASET} \
  experiment=${MODEL} \
  ckpt_path="$CKPT" \
  output_dir=${OUT} \
  run_name=${RUN_NAME} \
  prompt_protocol.type=${PROMPT} \
  prompt_protocol.num_correction_clicks=${NCORR} \
  ${EXTRA}
