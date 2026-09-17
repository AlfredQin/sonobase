#!/bin/bash
# Click-efficiency curve under the Tables 1-2 prompt protocol.
#
# Why this exists
# ---------------
# An exact tight GT box (or the RITM distance-transform point) evaluated once is
# a different prompt distribution from the one behind Tables 1-2, which use a 10%
# jittered box (or a uniformly-sampled point) averaged over three seeds, and the
# two differ by 1-2 mIoU at correction level 0 — enough to move a curve across
# the 80% usability threshold. This regenerates the curve under the Tables 1-2
# protocol so every reported number rests on one prompt distribution.
#
#   box    jitter 0.10  (min(side*0.10, 20px) per corner, SAM2's own noise model)
#   point  uniform      (one pixel drawn uniformly from the GT foreground)
#   seeds  42 / 2026 / 1337
#   corrections 1 / 3 / 5 / 7
#
# Iteration 0 is NOT re-run: `predictions_prompt_noise/` already holds it under
# this exact protocol, harness and output schema, and those runs reproduce the
# Tables 1-2 macros to the digit. Re-running it would cost a quarter of the sweep
# to reproduce numbers we already have.
#
# Sizing
# ------
# 15 datasets x 3 models x 2 prompts x 3 seeds x 4 correction levels = 1080 runs.
# The 0-correction sibling (270 runs) took ~2.5-3 h at this shape; correction
# levels are dearer per run because each click is another forward pass, so budget
# appreciably longer and check progress rather than assuming a multiple.
#
# Each array task strides by NCHUNK through the combination list, so the array
# width can be set to whatever per-user job limit your scheduler allows.
#
# Submit:
#   sbatch scripts/analysis/slurm/click_curve_3seed.sbatch

#SBATCH --job-name=clickcurve3
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
CORRS=(1 3 5 7)
JITTER=0.1

ND=${#DATASETS[@]}; NM=${#MODELS[@]}; NP=${#PROMPTS[@]}
NS=${#SEEDS[@]};    NC=${#CORRS[@]}
PER_CORR=$(( ND * NM * NP * NS ))     # 270
NCOMBO=$(( PER_CORR * NC ))           # 1080
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
[ -f "${SONOBASE_CKPT}" ] || { echo "ERROR: sonobase checkpoint not found: ${SONOBASE_CKPT}" >&2; exit 1; }

# Its own root: the deterministic archive and the 0-correction prompt-noise
# archive both stay uncontaminated and individually addressable.
OUT_ROOT_DIR="./experiments/analysis/predictions_click_curve"

run_combo () {
  local gpu="$1" idx="$2"
  local corr seed ds mdl prm r logf

  corr="${CORRS[$(( idx / PER_CORR ))]}"
  r=$(( idx % PER_CORR ))
  seed="${SEEDS[$(( r / (ND * NM * NP) ))]}"
  r=$(( r % (ND * NM * NP) ))
  prm="${PROMPTS[$(( r / (ND * NM) ))]}"
  r=$(( r % (ND * NM) ))
  ds="${DATASETS[$(( r / NM ))]}"
  mdl="${MODELS[$(( r % NM ))]}"

  logf="${LOG_DIR}/clickcurve3_${SLURM_ARRAY_JOB_ID}_${I}_c${idx}_${mdl}_${ds}_${prm}_${corr}corr_s${seed}.log"
  echo "  [gpu ${gpu}] combo ${idx}: ${mdl} / ${ds} / ${prm} / ${corr}corr / seed ${seed}"

  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    export PROMPT_SEED="${seed}"
    export OUT_ROOT="${OUT_ROOT_DIR}"
    export NCORR="${corr}"
    if [ "${prm}" = "box" ]; then
      export BOX_JITTER="${JITTER}"
    else
      export PT_SAMPLING=uniform
    fi
    [[ "${mdl}" == "sonobase" ]] && export CKPT_PATH="${SONOBASE_CKPT}"
    # Overlays are a rendering aid; this study consumes per_sample_metrics.csv
    # only, and 1080 runs' worth of PNGs is not worth the disk.
    bash scripts/analysis/save_predictions/_save_one.sh "${mdl}" "${ds}" "${prm}" save_overlays=false
  ) >"${logf}" 2>&1
}

MINE=()
for (( k = I; k < NCOMBO; k += NCHUNK )); do MINE+=("${k}"); done

echo "============================================================"
echo "# Click-efficiency curve, Tables 1-2 protocol — task ${I}"
echo "#   node    = $(hostname)"
echo "#   combos  = ${#MINE[@]} of ${NCOMBO}"
echo "#   out     = ${OUT_ROOT_DIR}"
echo "============================================================"

# Concurrency per task. Measured on job 4624: a single combo leaves the GPU
# ~86 % idle (0-28 % utilisation, 3-6 GB of 143 GB) and spends its time in the
# per-sample Python/IO path instead, drawing 3.4-8.3 CPU cores. One combo per
# GPU therefore wasted most of both GPUs. PER_GPU combos share each GPU; at 3
# that is 6 in flight against this task's 32 cores (~28 used) and roughly half
# the GPU, which is the CPU limit rather than the GPU one.
PER_GPU="${PER_GPU:-3}"
NGPU_TASK=2
SLOTS=$(( PER_GPU * NGPU_TASK ))

FAILED=()
n=${#MINE[@]}
declare -A PID2COMBO=()
launch () {  # launch <combo-index> <slot>
  local idx="$1" slot="$2"
  run_combo "$(( slot % NGPU_TASK ))" "${idx}" &
  PID2COMBO[$!]="${idx}"
}

next=0
# Fill every slot, then top up as each finishes -- a fixed-stride barrier would
# idle the whole task waiting on its slowest member (CAMUS/PFUS are ~50x BUSI).
while (( next < n && next < SLOTS )); do
  launch "${MINE[$next]}" "${next}"; next=$(( next + 1 ))
done
while (( ${#PID2COMBO[@]} > 0 )); do
  if wait -n -p done_pid 2>/dev/null; then :; else
    [ -n "${done_pid:-}" ] && FAILED+=("${PID2COMBO[$done_pid]}")
  fi
  if [ -n "${done_pid:-}" ]; then
    unset 'PID2COMBO[$done_pid]'
    if (( next < n )); then
      launch "${MINE[$next]}" "${next}"; next=$(( next + 1 ))
    fi
  else
    break
  fi
done

if [ ${#FAILED[@]} -gt 0 ]; then
  echo "# [task ${I}] FAILED combos: ${FAILED[*]}" >&2
  exit 1
fi
echo "# [task ${I}] done — ${n} combos"
