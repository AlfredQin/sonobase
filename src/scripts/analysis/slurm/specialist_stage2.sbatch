#!/bin/bash
# CPU job: run one specialist Stage-2 script (import / A1 / A2 / A4) from src/.
#   sbatch --export=ALL,SCRIPT=scripts/analysis/specialist/import_external.sh,ARGS="echonet_lv_zeroshot CAMUS /path/neutral --unmapped-objects skip" \
#          scripts/analysis/slurm/specialist_stage2.sbatch
# Env: DATASET_DIR ANNOTATION_DIR HC18_ZIP CAMUS_ZIP REGPRO_ZIP ACOUSLIC_CSV UV_PROJECT_ENVIRONMENT UV_NO_SYNC
#SBATCH --job-name=spec_stage2
#SBATCH --output=slurm_logs/%x_%A.out
#SBATCH --error=slurm_logs/%x_%A.err
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
set -euo pipefail
: "${SCRIPT:?SCRIPT=scripts/analysis/specialist/<name>.sh}"
SBATCH_PATH="$(scontrol show job "${SLURM_JOB_ID}" -o | grep -oP 'Command=\K[^[:space:]]+' | head -n1)"
SRC="$(cd "$(dirname "${SBATCH_PATH}")/../../.." && pwd)"; cd "${SRC}"
mkdir -p "${SLURM_SUBMIT_DIR:-$PWD}/slurm_logs"
echo "# stage2 ${SCRIPT} ${ARGS:-} (src=${SRC}) job=${SLURM_JOB_ID}"
eval bash "${SCRIPT}" ${ARGS:-}
