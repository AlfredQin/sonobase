#!/bin/bash
# Benchmark B1 iterative-correction sweep — array job for all 6
# (model × prompt) cells on the production 8_bm_7_ext suite.
#
# 3 models × 2 prompts = 6 array tasks.
#
# Submit:
#
#   sbatch scripts/benchmarks/test_sam2/iterative/slurm/b1_sweep_all_8_bm_7_ext.sbatch
#
# Restrict to a slice:
#
#   sbatch --array=0-1 ... # only sam2_no_ft (point + box)
#   sbatch --array=4-5 ... # only sonobase
#
# Layout convention:
#   array_id   model_idx = id // 2          prompt_idx = id % 2
#       0      0  sam2_no_ft                0  point
#       1      0  sam2_no_ft                1  box
#       2      1  medsam2                   0  point
#       …      …                            …
#       5      2  sonobase                  1  box
#
# Runs each cell at 1 GPU (NOT 4) — multi-GPU DDP pads the eval set and
# biases the pooled metric, so the eval must be sample-exact. The 6 array
# tasks each take 1 GPU and run in parallel across the cluster.
#
# Per-task wall time at 1 GPU: ~3-5 h for the slow sonobase/video cells
# (5 full 15-dataset evals: iters 0,1,3,5,7), less for the baselines.

#SBATCH --job-name=bm_b1_8bm7ext
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err
#SBATCH --partition=main
#SBATCH --array=0-5
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16               # 1 GPU × 16 CPUs/GPU
#SBATCH --gpus-per-node=1
#SBATCH --mem=256G                       # modest: 1-GPU eval; lower mem packs more cells per node
#SBATCH --time=12:00:00

set -euo pipefail

MODELS=(sam2_no_ft medsam2 sonobase)
PROMPTS=(point box)
DATASET="8_bm_7_ext"

MODEL_IDX=$((SLURM_ARRAY_TASK_ID / ${#PROMPTS[@]}))
PROMPT_IDX=$((SLURM_ARRAY_TASK_ID % ${#PROMPTS[@]}))
MODEL="${MODELS[${MODEL_IDX}]}"
PROMPT="${PROMPTS[${PROMPT_IDX}]}"

export MODEL DATASET PROMPT

# NB: for array jobs scontrol can echo a Command= line per concurrent task,
# so take the first (all tasks share the same script path). Matches the
# guard already in run_all_cells.sbatch / us_rfdetr_all.sbatch.
SBATCH_PATH="$(scontrol show job "${SLURM_JOB_ID}" -o | grep -oP 'Command=\K[^[:space:]]+' | head -n1)"
[ -f "${SBATCH_PATH}" ] || { echo "ERROR: cannot resolve this sbatch script's path from scontrol (got '${SBATCH_PATH}')" >&2; exit 1; }
SCRIPT_DIR="$(cd "$(dirname "${SBATCH_PATH}")" && pwd)"

echo "============================================================"
echo "# Array task ${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_MAX}"
echo "#   model=${MODEL}, dataset=${DATASET}, prompt=${PROMPT}"
echo "============================================================"

source "${SCRIPT_DIR}/_b1_sweep_one.sbatch"
