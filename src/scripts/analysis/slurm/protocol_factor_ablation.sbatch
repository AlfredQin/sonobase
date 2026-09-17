#!/bin/bash
# Protocol-factor ablation, SonoBase only.
#
# The protocol rerun left a residual: at matched box jitter (0.10) the analysis
# pipeline scores 2.08 mIoU BELOW the pretraining harness on external-7 box, and
# that residual is *attributed* to two differences -- best-of-3 mask selection
# and the scoring grid -- without being measured. The per-dataset residuals run
# from -6.1 (FUGC) to +1.9 (KidneyUS), so it is not one uniform effect either.
#
# This turns the attribution into a 2x2 measurement, at jitter 0.10 throughout:
#
#                       native grid          square-1024 grid
#   best-of-3 (mm=1)    A  (control)         B
#   single mask (mm=0)  C                    D
#
#   B - A  = scoring grid, with mask selection held fixed
#   C - A  = best-of-3, with the grid held fixed
#   D      = both flipped; D vs the harness's own number bounds whatever is
#            left over, i.e. whether the two-factor enumeration was complete.
#
# Only two inference configurations are needed, not four: `score_square_grid`
# records BOTH grids in the same run, from the same 256x256 logits, so each run
# yields one row of the table. 15 datasets x 2 configurations = 30 runs.
#
# Scoring both grids from the logits (rather than rescoring saved native masks)
# is what makes B-A clean: a saved mask has already been resampled once to the
# native frame, and resampling it again to a square would confound the grid
# effect with a round-trip artifact.
#
# Submit:
#   sbatch scripts/analysis/slurm/protocol_factor_ablation.sbatch

#SBATCH --job-name=proto_ablate
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err
#SBATCH --partition=main
#SBATCH --array=0-11%8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=2
#SBATCH --mem=320G
#SBATCH --time=08:00:00

set -uo pipefail

DATASETS=(BUSI Brachial-Plexus C-TRUS CAMUS HC18 PFUS RegPro TG3K
          ACOUSLIC BUS-BRA DDTI FUGC KidneyUS LUMINOUS MMOTU-3d)
MULTIMASKS=(1 0)
JITTER=0.1

ND=${#DATASETS[@]}
NMM=${#MULTIMASKS[@]}
NCOMBO=$(( ND * NMM ))
NCHUNK=12

I="${SLURM_ARRAY_TASK_ID:?this script must be submitted as an array job}"

export CHECKPOINT_DIR="${CHECKPOINT_DIR:?set CHECKPOINT_DIR to the checkpoint root}"
export DATASET_DIR="${DATASET_DIR:?set DATASET_DIR to the converted-dataset root}"
export ANNOTATION_DIR="${ANNOTATION_DIR:?set ANNOTATION_DIR to the split-list root}"

SBATCH_PATH="$(scontrol show job "${SLURM_JOB_ID}" -o | grep -oP 'Command=\K[^[:space:]]+' | head -n1)"
[ -f "${SBATCH_PATH}" ] || { echo "ERROR: cannot resolve this sbatch script's path from scontrol" >&2; exit 1; }
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
  local ds mm logf
  ds="${DATASETS[$(( idx / NMM ))]}"
  mm="${MULTIMASKS[$(( idx % NMM ))]}"
  logf="${LOG_DIR}/ablate_${SLURM_ARRAY_JOB_ID}_${I}_c${idx}_${ds}_mm${mm}.log"

  echo "  [gpu ${gpu}] combo ${idx}: sonobase / ${ds} / box / jitter=${JITTER} / multimask=${mm}"

  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    export BOX_JITTER="${JITTER}"
    export MULTIMASK="${mm}"
    export SCORE_SQ=1
    export CKPT_PATH="${SONOBASE_CKPT}"
    bash scripts/analysis/save_predictions/_save_one.sh sonobase "${ds}" box save_overlays=false
  ) >"${logf}" 2>&1
}

MINE=()
for (( k = I; k < NCOMBO; k += NCHUNK )); do MINE+=("${k}"); done

echo "============================================================"
echo "# Protocol-factor ablation — array task ${I}"
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
