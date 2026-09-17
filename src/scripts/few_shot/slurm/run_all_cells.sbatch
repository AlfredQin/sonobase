#!/bin/bash
# Few-shot adaptation — PER-CELL array launcher (max parallelism).
#
# run_all_datasets.sbatch runs 54 fine-tunes + 108 evals SERIALLY inside each
# of 3 dataset tasks (3 GPUs total). This launcher puts
# ONE cell per array task (1 GPU each), so SLURM packs every idle GPU on the
# cluster. Each task does: fine-tune -> eval point -> eval box for its cell.
#
# Cell list = MODELS × DATASETS × N × SEEDS (row-major, built identically in
# every task and indexed by SLURM_ARRAY_TASK_ID):
#   MODELS=(sonobase medsam2 sam2_no_ft)  DATASETS=(ACOUSLIC DDTI FUGC)
#   N=(1 2 5 10 20 30)  SEEDS=(42 123 456)
#   ⇒ 3 × 3 × 6 × 3 = 162 cells  (array 0-161)
#
# Submit (all 162 cells, at most 32 running at once):
#
#   export CHECKPOINT_DIR=/path/to/checkpoints
#   export DATASET_DIR=/path/to/SaUS
#   export ANNOTATION_DIR=/path/to/38_pt_8_bm_7_ext
#   export SONOBASE_DCP=./experiments/sonobase/pretrain/38_pt_8_bm/hiera_b_conv_s_conv_t_20ep/LOWEST_VAL
#   sbatch --export=ALL --array=0-161%32 scripts/few_shot/slurm/run_all_cells.sbatch
#
# Subset examples:
#   --array=0-53%32          # sonobase only (first 54 cells)
#   MODELS=sonobase ...      # also restrict the built list to one model
#
# Completed cells are skipped on resubmit (finetune.py final.pth guard +
# _eval_one.sh per_sample_metrics.csv guard), so a partial array is safe to
# re-run. Logs: <submit_dir>/slurm_logs/<jobname>_<JobID>_<TaskID>.{out,err}.

#SBATCH --job-name=fs_cell
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16                # 8 cells/node × 16 = 128 CPUs
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G                         # 8 cells/node × 64G ≤ node RAM
#SBATCH --time=04:00:00                   # 1 fine-tune + 2 evals ≪ 4h

set -euo pipefail

# ---- required env (via --export=ALL) ----
: "${CHECKPOINT_DIR:?CHECKPOINT_DIR must be set}"
: "${DATASET_DIR:?DATASET_DIR must be set}"
: "${ANNOTATION_DIR:?ANNOTATION_DIR must be set}"
: "${SONOBASE_DCP:?SONOBASE_DCP must point at the production pretrain run dir / LATEST / LOWEST_VAL}"
export CHECKPOINT_DIR DATASET_DIR ANNOTATION_DIR SONOBASE_DCP

# ---- build the flat cell list (row-major; must match the array size) ----
MODELS_ARR=(${MODELS:-sonobase medsam2 sam2_no_ft})
DATASETS_ARR=(ACOUSLIC DDTI FUGC)
NS_ARR=(1 2 5 10 20 30)
SEEDS_ARR=(42 123 456)
CELLS=()
for m in "${MODELS_ARR[@]}"; do
  for d in "${DATASETS_ARR[@]}"; do
    for n in "${NS_ARR[@]}"; do
      for s in "${SEEDS_ARR[@]}"; do
        CELLS+=("${d} ${m} ${n} ${s}")
      done
    done
  done
done

IDX="${SLURM_ARRAY_TASK_ID:?run as an array job, e.g. --array=0-161%32}"
if [ "${IDX}" -ge "${#CELLS[@]}" ]; then
  echo "ERROR: array index ${IDX} >= cell count ${#CELLS[@]} (shrink --array)" >&2
  exit 1
fi
read -r DATASET MODEL N SEED <<< "${CELLS[${IDX}]}"

# ---- resolve repo src/ dir (sbatch spools the script; use scontrol) ----
# NB: for array jobs scontrol can echo a Command= line per concurrent task,
# so take the first (all tasks share the same script path).
SBATCH_PATH="$(scontrol show job "${SLURM_JOB_ID}" -o | grep -oP 'Command=\K[^[:space:]]+' | head -n1)"
[ -f "${SBATCH_PATH}" ] || { echo "ERROR: cannot resolve sbatch path from scontrol (got '${SBATCH_PATH}')" >&2; exit 1; }
SCRIPT_DIR="$(cd "$(dirname "${SBATCH_PATH}")" && pwd)"
SRC_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${SRC_DIR}"
mkdir -p "${SLURM_SUBMIT_DIR:-$PWD}/slurm_logs"

export HYDRA_FULL_ERROR=1
export PYTHONPATH="${PYTHONPATH:-}:${SRC_DIR}"
export LD_LIBRARY_PATH=""

echo "============================================================"
echo "# E2 few-shot cell ${IDX}/${#CELLS[@]} → ${DATASET} ${MODEL} N=${N} seed=${SEED}"
echo "#   job=${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}  node=$(hostname)  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<slurm>}"
echo "============================================================"

# One srun step → proper gres binding (each array task sees only its 1 GPU).
srun --kill-on-bad-exit=1 bash -c "
  set -e
  bash scripts/few_shot/_finetune_one.sh '${DATASET}' '${MODEL}' '${N}' '${SEED}'
  bash scripts/few_shot/_eval_one.sh     '${DATASET}' '${MODEL}' '${N}' '${SEED}' point
  bash scripts/few_shot/_eval_one.sh     '${DATASET}' '${MODEL}' '${N}' '${SEED}' box
"

echo "cell ${IDX} (${DATASET} ${MODEL} N=${N} seed=${SEED}) done."
