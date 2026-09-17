#!/bin/bash
# Stage 1 of dataset deduplication: hash every image in SaUS and write an
# overlap_report.json. CPU-only, multi-process. Stage 2 (generate_splits) is
# near-instant and run interactively afterwards. See
# docs/DATA_SPLITS.md.
#
# Usage (submit from the repo root so slurm_logs/ resolves):
#   DATASET_DIR=/path/to/SaUS \
#     sbatch src/scripts/data/dedup_overlap.sbatch
# Env knobs: DATASET_DIR (required), OUT (report path), HASH_ALGO (default md5).
#SBATCH --job-name=dedup_overlap
#SBATCH --output=slurm_logs/%x_%A.out
#SBATCH --error=slurm_logs/%x_%A.err
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=256G
#SBATCH --time=01:00:00

set -euo pipefail

SRC_DIR="${SRC_DIR:-${SLURM_SUBMIT_DIR:-$PWD}/src}"
DATASET_DIR="${DATASET_DIR:?set DATASET_DIR to the SaUS root}"
HASH_ALGO="${HASH_ALGO:-md5}"
OUT="${OUT:-${DATASET_DIR}/overlap_report_$(date +%Y%m%d).json}"
WORKERS="${SLURM_CPUS_PER_TASK:-128}"

cd "$SRC_DIR"
export PYTHONPATH="${PYTHONPATH:-}:${SRC_DIR}"

echo "SaUS root : $DATASET_DIR"
echo "hash algo : $HASH_ALGO"
echo "workers   : $WORKERS"
echo "output    : $OUT"

uv run python -m data.utils.check_overlap \
  --path "$DATASET_DIR" \
  --hash-algo "$HASH_ALGO" \
  --workers "$WORKERS" \
  --output "$OUT"

echo "=== done; summary ==="
uv run python -c "import json,sys; r=json.load(open('$OUT')); print(r['summary'])"
