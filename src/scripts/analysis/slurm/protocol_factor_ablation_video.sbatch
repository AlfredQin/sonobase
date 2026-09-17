#!/bin/bash
# Protocol-factor ablation, VIDEO datasets — completes the image-only run.
#
# The first ablation pass covered 10 of 15 datasets because both knobs reach the
# image path only. On the 5 video datasets `MULTIMASK=0` was silently inert
# (both arms returned byte-identical numbers) and no square-grid column was
# written. This closes that gap.
#
# The two factors need different levers on this path:
#
#   best-of-3 — `multimask_output` is an argument to
#     SAM2ImagePredictor.predict, so it never reaches the video path. The video
#     predictor encodes a box as two corner points (labels 2/3) and multimask is
#     then gated inside SAM2Base by `multimask_max_pt_num`, which the analysis
#     model config sets to 1. So video is ALREADY single-mask, matching the
#     pretraining harness, and the arm to add is the opposite one: raise
#     `multimask_max_pt_num` to 2 to switch best-of-3 on. If that changes
#     nothing, the gating claim is wrong and this run says so.
#
#   scoring grid — `propagate_in_video` upsamples the 256x256 logits to video
#     resolution internally and yields only those, so the square mask comes from
#     interpolating those logits a second time. That is one interpolation
#     further from the source than the image path, which scores straight from
#     the 256x256 tensor. Interpolating logits rather than a thresholded mask
#     keeps the extra step cheap, but it is not the identical operation and any
#     video-vs-image comparison of the grid effect should say so.
#
# 5 datasets x 2 arms = 10 runs. RUN_SUFFIX keeps them beside the superseded
# dirs from the first pass rather than resuming into them.
#
# Submit:
#   sbatch scripts/analysis/slurm/protocol_factor_ablation_video.sbatch

#SBATCH --job-name=ablate_vid
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err
#SBATCH --partition=main
#SBATCH --array=0-4%5
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=2
#SBATCH --mem=320G
#SBATCH --time=08:00:00

set -uo pipefail

DATASETS=(Brachial-Plexus CAMUS PFUS RegPro ACOUSLIC)
JITTER=0.1
DS="${DATASETS[${SLURM_ARRAY_TASK_ID:?submit as an array job}]}"

export CHECKPOINT_DIR="${CHECKPOINT_DIR:?set CHECKPOINT_DIR to the checkpoint root}"
export DATASET_DIR="${DATASET_DIR:?set DATASET_DIR to the converted-dataset root}"
export ANNOTATION_DIR="${ANNOTATION_DIR:?set ANNOTATION_DIR to the split-list root}"

SBATCH_PATH="$(scontrol show job "${SLURM_JOB_ID}" -o | grep -oP 'Command=\K[^[:space:]]+' | head -n1)"
[ -f "${SBATCH_PATH}" ] || { echo "ERROR: cannot resolve this sbatch script's path" >&2; exit 1; }
SCRIPT_DIR="$(cd "$(dirname "${SBATCH_PATH}")" && pwd)"
SRC_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${SRC_DIR}"

LOG_DIR="${SLURM_SUBMIT_DIR:-$PWD}/slurm_logs"
mkdir -p "${LOG_DIR}"
export HYDRA_FULL_ERROR=1
export PYTHONPATH="${PYTHONPATH:-}:${SRC_DIR}"
export LD_LIBRARY_PATH=""

SONOBASE_CKPT="${SONOBASE_PT:-${SRC_DIR}/experiments/sonobase/pretrain/38_pt_8_bm/hiera_b_conv_s_conv_t_20ep_dedup_lowlr/epoch_19_step_27539.pt}"
[ -f "${SONOBASE_CKPT}" ] || { echo "ERROR: sonobase checkpoint not found" >&2; exit 1; }

# arm 0: default gating -> single mask (the cell that should match the harness)
# arm 1: multimask_max_pt_num=2 -> best-of-3 switched on for a 2-point box
run_arm () {
  local gpu="$1" arm="$2"
  local logf="${LOG_DIR}/ablatevid_${SLURM_ARRAY_JOB_ID}_${DS}_arm${arm}.log"
  echo "  [gpu ${gpu}] ${DS} arm ${arm}"
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    export BOX_JITTER="${JITTER}" SCORE_SQ=1 RUN_SUFFIX="_v2"
    export CKPT_PATH="${SONOBASE_CKPT}"
    [ "${arm}" = "1" ] && export MM_MAXPT=2
    bash scripts/analysis/save_predictions/_save_one.sh sonobase "${DS}" box save_overlays=false
  ) >"${logf}" 2>&1
}

echo "=== video ablation: ${DS} on $(hostname) ==="
run_arm 0 0 & p0=$!
run_arm 1 1 & p1=$!
FAILED=()
wait "${p0}" || FAILED+=("arm0")
wait "${p1}" || FAILED+=("arm1")
if (( ${#FAILED[@]} > 0 )); then
  echo "${DS}: FAILED ${FAILED[*]}" >&2; exit 1
fi
echo "${DS}: both arms completed"
