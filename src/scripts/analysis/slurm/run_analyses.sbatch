#!/bin/bash
# Analysis Stage 2 — CPU-only run of every analysis (point + box).
#
# Reads the per-sample CSVs / prediction PNGs that Stage 1 wrote to
# `./experiments/analysis/predictions/`, runs the 16 analyses (A1-A4,
# B1-B3, C2, T1.1-T3.2, S1-S3) on CPU, and emits PDF/PNG/JSON triples
# under `./experiments/analysis/<analysis>/<DATASET>_<prompt>_0corr/`.
#
# Two prompt protocols (point + box) → 2 array tasks. Each protocol
# runs ~14 analyses sequentially through the existing
# `run_all_analyses_<prompt>.sh` wrapper.
#
# Submit:
#
#   sbatch scripts/analysis/slurm/run_analyses.sbatch
#
# Or one prompt at a time:
#
#   sbatch --array=0 ... # point only
#   sbatch --array=1 ... # box only

#SBATCH --job-name=analysis_run
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err
#SBATCH --partition=main
#SBATCH --array=0-1                       # 0=point, 1=box
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G                        # CPU-only — no need for 512 G
#SBATCH --time=02:00:00

set -euo pipefail

PROMPTS=(point box)
PROMPT="${PROMPTS[${SLURM_ARRAY_TASK_ID}]}"

# Asset paths — all must be set per user in your .bashrc. Example:
#   export CHECKPOINT_DIR=$WORK/Checkpoints
#   export DATASET_DIR=$WORK/Dataset/SaUS
#   export ANNOTATION_DIR=$WORK/Dataset/SaUS_Annotation/38_pt_8_bm_7_ext_v2
#   export HC18_ZIP=$WORK/Dataset/UltraSound/Raw/HC.zip
#   export CAMUS_ZIP=$WORK/Dataset/UltraSound/Raw/CAMUS.zip
#   export REGPRO_ZIP=$WORK/Dataset/UltraSound/Raw/RegPro.zip
#   export ACOUSLIC_CSV=$WORK/Dataset/UltraSound/ACOUSLIC/circumferences/fetal_abdominal_circumferences_per_sweep.csv
#   export PORCINE_SAUS=$WORK/Dataset/SaUS/PorcineSpinalCord
: "${CHECKPOINT_DIR:?CHECKPOINT_DIR must be set, e.g. export CHECKPOINT_DIR=\$WORK/Checkpoints}"
: "${DATASET_DIR:?DATASET_DIR must be set, e.g. export DATASET_DIR=\$WORK/Dataset/SaUS}"
: "${ANNOTATION_DIR:?ANNOTATION_DIR must be set, e.g. export ANNOTATION_DIR=\$WORK/Dataset/SaUS_Annotation/38_pt_8_bm_7_ext_v2}"
: "${HC18_ZIP:?HC18_ZIP must be set, e.g. export HC18_ZIP=\$WORK/Dataset/UltraSound/Raw/HC.zip}"
: "${CAMUS_ZIP:?CAMUS_ZIP must be set, e.g. export CAMUS_ZIP=\$WORK/Dataset/UltraSound/Raw/CAMUS.zip}"
: "${REGPRO_ZIP:?REGPRO_ZIP must be set, e.g. export REGPRO_ZIP=\$WORK/Dataset/UltraSound/Raw/RegPro.zip}"
: "${ACOUSLIC_CSV:?ACOUSLIC_CSV must be set, e.g. export ACOUSLIC_CSV=\$WORK/Dataset/UltraSound/ACOUSLIC/circumferences/fetal_abdominal_circumferences_per_sweep.csv}"
: "${PORCINE_SAUS:?PORCINE_SAUS must be set, e.g. export PORCINE_SAUS=\$WORK/Dataset/SaUS/PorcineSpinalCord}"
export CHECKPOINT_DIR DATASET_DIR ANNOTATION_DIR HC18_ZIP CAMUS_ZIP REGPRO_ZIP ACOUSLIC_CSV PORCINE_SAUS

SBATCH_PATH="$(scontrol show job "${SLURM_JOB_ID}" -o | grep -oP 'Command=\K[^[:space:]]+')"
[ -f "${SBATCH_PATH}" ] || { echo "ERROR: cannot resolve this sbatch script's path from scontrol (got '${SBATCH_PATH}')" >&2; exit 1; }
SCRIPT_DIR="$(cd "$(dirname "${SBATCH_PATH}")" && pwd)"
SRC_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${SRC_DIR}"

mkdir -p "${SLURM_SUBMIT_DIR:-$PWD}/slurm_logs"

export PYTHONPATH="${PYTHONPATH:-}:${SRC_DIR}"
export LD_LIBRARY_PATH=""

echo "============================================================"
echo "# Analysis Stage 2 — ${PROMPT} prompt (slurm, CPU-only)"
echo "#   job_id  = ${SLURM_JOB_ID}"
echo "#   wrapper = scripts/analysis/run_analyses/run_all_analyses_${PROMPT}.sh"
echo "============================================================"

bash "scripts/analysis/run_analyses/run_all_analyses_${PROMPT}.sh"
