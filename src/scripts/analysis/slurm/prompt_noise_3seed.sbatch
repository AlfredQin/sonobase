#!/bin/bash
# 3-seed randomised-prompt evaluation.
#
# The deterministic protocol prompts with the tight GT box, or the RITM
# distance-transform maximum. Neither is what a clinician produces, and
# neither is what SonoBase trained on — training samples boxes with
# `sample_box_points(noise=0.1)` and points with `method="uniform"`. This run
# evaluates at the training-time prompt distribution instead, three seeds deep,
# so the numbers carry a seed-to-seed SD rather than a single draw.
#
#   box    jitter 0.10  (min(side*0.10, 20px) per corner, SAM2's own model)
#   point  uniform      (one pixel drawn uniformly from the GT foreground)
#   seeds  42 / 2026 / 1337
#
# Items 4 and 5 of the deterministic protocol are held FIXED: best-of-3 on
# image / single-mask on video, and native geometry. Nothing here touches them.
#
# 15 datasets x 3 models x 2 prompts x 3 seeds = 270 runs. Output goes to its
# own root so the archive of deterministic records stays uncontaminated and
# individually addressable.
#
# Sizing: ~135 runs per 80 min at this shape (8 concurrent 2-GPU tasks), so
# budget ~2.5-3 h. Smoke-test one chunk before launching the full array: a
# prompt knob that fails to reach the model produces a complete, plausible
# and entirely meaningless result set.
#
# Submit:
#   sbatch scripts/analysis/slurm/prompt_noise_3seed.sbatch

#SBATCH --job-name=pnoise3
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err
#SBATCH --partition=main
#SBATCH --array=0-11%8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=2
#SBATCH --mem=320G
#SBATCH --time=12:00:00

set -uo pipefail

DATASETS=(BUSI Brachial-Plexus C-TRUS CAMUS HC18 PFUS RegPro TG3K
          ACOUSLIC BUS-BRA DDTI FUGC KidneyUS LUMINOUS MMOTU-3d)
MODELS=(sonobase medsam2 sam2_no_ft)
PROMPTS=(box point)
SEEDS=(42 2026 1337)
JITTER=0.1

ND=${#DATASETS[@]}; NM=${#MODELS[@]}; NP=${#PROMPTS[@]}; NS=${#SEEDS[@]}
PER_SEED=$(( ND * NM * NP ))
NCOMBO=$(( PER_SEED * NS ))          # 270
# Each array task strides by NCHUNK through the combination list, so the array
# width can be set to whatever per-user job limit your scheduler allows.
NCHUNK=12

I="${SLURM_ARRAY_TASK_ID:?this script must be submitted as an array job}"

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

OUT_ROOT_DIR="./experiments/analysis/predictions_prompt_noise"

run_combo () {
  local gpu="$1" idx="$2"
  local seed ds mdl prm r logf

  seed="${SEEDS[$(( idx / PER_SEED ))]}"
  r=$(( idx % PER_SEED ))
  prm="${PROMPTS[$(( r / (ND * NM) ))]}"
  r=$(( r % (ND * NM) ))
  ds="${DATASETS[$(( r / NM ))]}"
  mdl="${MODELS[$(( r % NM ))]}"

  logf="${LOG_DIR}/pnoise3_${SLURM_ARRAY_JOB_ID}_${I}_c${idx}_${mdl}_${ds}_${prm}_s${seed}.log"
  echo "  [gpu ${gpu}] combo ${idx}: ${mdl} / ${ds} / ${prm} / seed ${seed}"

  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    export PROMPT_SEED="${seed}"
    export OUT_ROOT="${OUT_ROOT_DIR}"
    if [ "${prm}" = "box" ]; then
      export BOX_JITTER="${JITTER}"
    else
      export PT_SAMPLING=uniform
    fi
    [[ "${mdl}" == "sonobase" ]] && export CKPT_PATH="${SONOBASE_CKPT}"
    # Overlays are a rendering aid; this study consumes per_sample_metrics.csv
    # only, and 270 runs' worth of PNGs is not worth the disk.
    bash scripts/analysis/save_predictions/_save_one.sh "${mdl}" "${ds}" "${prm}" save_overlays=false
  ) >"${logf}" 2>&1
}

MINE=()
for (( k = I; k < NCOMBO; k += NCHUNK )); do MINE+=("${k}"); done

echo "============================================================"
echo "# Prompt-noise 3-seed study — array task ${I}"
echo "#   node    = $(hostname)"
echo "#   combos  = ${MINE[*]}"
echo "#   out     = ${OUT_ROOT_DIR}"
echo "============================================================"

FAILED=()
n=${#MINE[@]}
for (( i = 0; i < n; i += 2 )); do
  a="${MINE[$i]}"
  run_combo 0 "${a}" & pa=$!
  pb=""
  if (( i + 1 < n )); then b="${MINE[$((i+1))]}"; run_combo 1 "${b}" & pb=$!; fi
  wait "${pa}" || FAILED+=("${a}")
  if [[ -n "${pb}" ]]; then wait "${pb}" || FAILED+=("${b}"); fi
done

if (( ${#FAILED[@]} > 0 )); then
  echo "task ${I}: FAILED combos ${FAILED[*]}" >&2
  exit 1
fi
echo "task ${I}: all ${n} combos completed"
