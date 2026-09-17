#!/bin/bash
# Box-prompt robustness sweep.
#
# Hypothesis: SonoBase is less sensitive to an imprecise box
# than the baselines, because it saw jittered boxes on ultrasound during
# pretraining. If so, degrading the box should cost the baselines more than it
# costs SonoBase.
#
# What this controls for. The protocol rerun showed the baselines gaining up to
# +3.6 mIoU when moving from the pretraining harness to the analysis pipeline,
# but that move changed FOUR things at once — box jitter, best-of-3 mask
# selection, scoring grid, and propagation on video — so none of the gain can be
# attributed to jitter from those runs alone. Here the analysis protocol is held
# fixed and only `box_jitter_pct` varies, which is the comparison the hypothesis
# actually needs.
#
# Note on the premise: `sample_box_points` defaults (noise=0.1, bound=20px) are
# annotated "SAM default" — SAM2's own training used the same jitter, and
# MedSAM2 is finetuned from SAM2. So all three lineages saw noisy boxes; what
# differs is that SonoBase saw them on ultrasound. The sweep measures the curve
# regardless of which mechanism explains it.
#
# 15 datasets x 3 models x 3 jitter levels = 135 runs, box prompt only (jitter
# is meaningless for a point prompt). The jitter=0 arm already exists from the
# protocol rerun and is the curve's left-hand anchor.
#
# Run names get a `_jitNN` suffix, so this cannot overwrite the exact-box
# records it is compared against.
#
# Submit:
#
#   sbatch scripts/analysis/slurm/box_jitter_sweep.sbatch

#SBATCH --job-name=jitter_sweep
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
JITTERS=(0.05 0.1 0.2)

ND=${#DATASETS[@]}
NM=${#MODELS[@]}
NJ=${#JITTERS[@]}
PER_J=$(( ND * NM ))
NCOMBO=$(( PER_J * NJ ))
NCHUNK=12

I="${SLURM_ARRAY_TASK_ID:?this script must be submitted as an array job}"

export CHECKPOINT_DIR="${CHECKPOINT_DIR:?set CHECKPOINT_DIR to the checkpoint root}"
export DATASET_DIR="${DATASET_DIR:?set DATASET_DIR to the converted-dataset root}"
export ANNOTATION_DIR="${ANNOTATION_DIR:?set ANNOTATION_DIR to the split-list root}"

SBATCH_PATH="$(scontrol show job "${SLURM_JOB_ID}" -o | grep -oP 'Command=\K[^[:space:]]+' | head -n1)"
[ -f "${SBATCH_PATH}" ] || { echo "ERROR: cannot resolve this sbatch script's path from scontrol (got '${SBATCH_PATH}')" >&2; exit 1; }
SCRIPT_DIR="$(cd "$(dirname "${SBATCH_PATH}")" && pwd)"
SRC_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${SRC_DIR}"

LOG_DIR="${SLURM_SUBMIT_DIR:-$PWD}/slurm_logs"
mkdir -p "${LOG_DIR}"

export HYDRA_FULL_ERROR=1
export PYTHONPATH="${PYTHONPATH:-}:${SRC_DIR}"
export LD_LIBRARY_PATH=""

SONOBASE_CKPT="${SONOBASE_PT:-${SRC_DIR}/experiments/sonobase/pretrain/38_pt_8_bm/hiera_b_conv_s_conv_t_20ep_dedup_lowlr/epoch_19_step_27539.pt}"
[ -f "${SONOBASE_CKPT}" ] || { echo "ERROR: sonobase checkpoint not found at ${SONOBASE_CKPT}" >&2; exit 1; }

run_combo () {
  local gpu="$1" idx="$2"
  local jit ds mdl r logf
  jit="${JITTERS[$(( idx / PER_J ))]}"
  r=$(( idx % PER_J ))
  ds="${DATASETS[$(( r / NM ))]}"
  mdl="${MODELS[$(( r % NM ))]}"
  logf="${LOG_DIR}/jitter_${SLURM_ARRAY_JOB_ID}_${I}_c${idx}_${mdl}_${ds}_jit${jit}.log"

  echo "  [gpu ${gpu}] combo ${idx}: ${mdl} / ${ds} / box / jitter=${jit}"

  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    export BOX_JITTER="${jit}"
    [[ "${mdl}" == "sonobase" ]] && export CKPT_PATH="${SONOBASE_CKPT}"
    # Overlays are a rendering aid; the sweep only consumes per_sample_metrics.csv.
    bash scripts/analysis/save_predictions/_save_one.sh "${mdl}" "${ds}" box save_overlays=false
  ) >"${logf}" 2>&1
}

MINE=()
for (( k = I; k < NCOMBO; k += NCHUNK )); do MINE+=("${k}"); done

echo "============================================================"
echo "# Box-jitter sweep — array task ${I}"
echo "#   node   = $(hostname)"
echo "#   combos = ${MINE[*]}"
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
  echo "TASK ${I}: FAILED combos: ${FAILED[*]}" >&2
  exit 1
fi
echo "TASK ${I}: all ${n} combos completed"
