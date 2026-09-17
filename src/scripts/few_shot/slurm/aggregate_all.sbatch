#!/bin/bash
# Few-shot adaptation (F1) — CPU-only aggregator.
#
# Run after the per-dataset sweeps (`run_all_datasets.sbatch`) complete.
# Builds per-dataset summary CSVs, applies global BH-FDR correction,
# emits the combined CSV + publication figures.
#
# Submit standalone:
#
#   sbatch scripts/few_shot/slurm/aggregate_all.sbatch
#
# Or as a follow-up to the sweep:
#
#   sbatch --dependency=afterok:<sweep_jobid> \
#     scripts/few_shot/slurm/aggregate_all.sbatch

#SBATCH --job-name=fs_aggregate
#SBATCH --output=slurm_logs/%x_%A.out
#SBATCH --error=slurm_logs/%x_%A.err
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00

set -euo pipefail

SBATCH_PATH="$(scontrol show job "${SLURM_JOB_ID}" -o | grep -oP 'Command=\K[^[:space:]]+')"
[ -f "${SBATCH_PATH}" ] || { echo "ERROR: cannot resolve this sbatch script's path from scontrol (got '${SBATCH_PATH}')" >&2; exit 1; }
SCRIPT_DIR="$(cd "$(dirname "${SBATCH_PATH}")" && pwd)"
SRC_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${SRC_DIR}"

mkdir -p "${SLURM_SUBMIT_DIR:-$PWD}/slurm_logs"

export PYTHONPATH="${PYTHONPATH:-}:${SRC_DIR}"
export LD_LIBRARY_PATH=""

echo "============================================================"
echo "# Few-shot aggregator (slurm, CPU-only)"
echo "#   job_id   = ${SLURM_JOB_ID}"
echo "============================================================"

bash scripts/few_shot/aggregate_all.sh
