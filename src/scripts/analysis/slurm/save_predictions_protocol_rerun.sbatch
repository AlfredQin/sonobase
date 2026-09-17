#!/bin/bash
# Analysis Stage 1 — the evaluation-protocol rerun.
#
# Nine datasets had no Stage-1 record, so their paper numbers came only from the
# pretraining harness, which scores on a square-resized grid against a jittered
# box and (on video) propagates a single prompt. This job produces the missing
# records under the analysis protocol — exact box, native-resolution scoring —
# so Tables 1/2/S2/S3 can be re-tabulated on one protocol.
#
# 9 datasets x 3 models x 2 prompts = 54 runs.
#
# Array shape (adjust to your scheduler's per-user limits):
#
#   MaxSubmitJobsPU=12   -> at most 12 array elements may sit in the queue,
#                           so the 54 runs are chunked 12 ways, not 1:1.
#   MaxJobsPU=8          -> at most 8 run at once.
#   MaxTRESPU=gpu=16     -> 8 concurrent tasks x 2 GPUs saturates the cap.
#
# `save_predictions` is single-GPU by design (no torchrun, no DDP), so a task
# uses its two GPUs by running two of its chunk's runs side by side, one pinned
# per GPU. One GPU per task would have left half the allocation idle.
#
# Submit:
#
#   sbatch scripts/analysis/slurm/save_predictions_protocol_rerun.sbatch
#
# Resumable: a sample with a meta.json is skipped and its records are re-read
# from disk, so re-submitting after a failure completes the gaps and still
# writes a complete per_sample_metrics.csv.

#SBATCH --job-name=proto_rerun
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

DATASETS=(Brachial-Plexus C-TRUS PFUS TG3K BUS-BRA DDTI FUGC LUMINOUS MMOTU-3d)
MODELS=(sonobase medsam2 sam2_no_ft)
PROMPTS=(point box)

NP=${#PROMPTS[@]}
NM=${#MODELS[@]}
NCOMBO=$(( ${#DATASETS[@]} * NM * NP ))
NCHUNK=12

I="${SLURM_ARRAY_TASK_ID:?this script must be submitted as an array job}"

# ---- assets ----
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

# Run one combo, pinned to one of this task's two GPUs.
run_combo () {
  local gpu="$1" idx="$2"
  local ds mdl prm logf
  ds="${DATASETS[$(( idx / (NM * NP) ))]}"
  mdl="${MODELS[$(( (idx % (NM * NP)) / NP ))]}"
  prm="${PROMPTS[$(( idx % NP ))]}"
  logf="${LOG_DIR}/proto_rerun_${SLURM_ARRAY_JOB_ID}_${I}_combo${idx}_${mdl}_${ds}_${prm}.log"

  echo "  [gpu ${gpu}] combo ${idx}: ${mdl} / ${ds} / ${prm} -> $(basename "${logf}")"

  # CKPT_PATH overrides every model in the resolver, so it must be set only on
  # the sonobase branch, and only for this subshell.
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    [[ "${mdl}" == "sonobase" ]] && export CKPT_PATH="${SONOBASE_CKPT}"
    bash scripts/analysis/save_predictions/_save_one.sh "${mdl}" "${ds}" "${prm}"
  ) >"${logf}" 2>&1
}

# Round-robin so consecutive combos (same dataset) land on different tasks and
# the big datasets spread out instead of stacking behind one another.
MINE=()
for (( k = I; k < NCOMBO; k += NCHUNK )); do MINE+=("${k}"); done

echo "============================================================"
echo "# Stage-1 protocol rerun — array task ${I}"
echo "#   node    = $(hostname)"
echo "#   gpus    = ${SLURM_GPUS_ON_NODE:-?}"
echo "#   combos  = ${MINE[*]}"
echo "#   src     = ${SRC_DIR}"
echo "============================================================"

FAILED=()
n=${#MINE[@]}
for (( i = 0; i < n; i += 2 )); do
  a="${MINE[$i]}"
  run_combo 0 "${a}" & pa=$!
  pb=""
  if (( i + 1 < n )); then
    b="${MINE[$((i+1))]}"
    run_combo 1 "${b}" & pb=$!
  fi

  wait "${pa}" || FAILED+=("${a}")
  if [[ -n "${pb}" ]]; then wait "${pb}" || FAILED+=("${b}"); fi
done

if (( ${#FAILED[@]} > 0 )); then
  echo "TASK ${I}: FAILED combos: ${FAILED[*]}" >&2
  echo "  (re-submitting this array task is safe — completed samples are skipped)" >&2
  exit 1
fi
echo "TASK ${I}: all ${n} combos completed"
