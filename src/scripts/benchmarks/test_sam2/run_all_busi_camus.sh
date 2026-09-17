#!/bin/bash
# Run all three models (SAM2-no-ft, MedSAM2, Sonobase) on BUSI + CAMUS
# test splits, then aggregate the per-model results into a single
# comparison CSV.
# Run from src/.

set -e

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")

echo ""
echo "########################################################"
echo "# Benchmark: SAM2-no-ft / MedSAM2 / Sonobase on BUSI+CAMUS"
echo "########################################################"
echo ""

echo "=== [1/3] SAM2 (no fine-tune) ==="
bash "$SCRIPT_DIR/test_sam2_no_ft_on_busi_camus.sh"
echo ""

echo "=== [2/3] MedSAM2 ==="
bash "$SCRIPT_DIR/test_medsam2_on_busi_camus.sh"
echo ""

echo "=== [3/3] Sonobase ==="
bash "$SCRIPT_DIR/test_sonobase_on_busi_camus.sh"
echo ""

echo "=== Aggregating comparison ==="
uv run python -m nemo_cv.recipes.benchmarks.compare_benchmarks \
  --runs \
    sam2_no_ft=./experiments/benchmarks/busi_camus/sam2/p0/results/test_metrics.json \
    medsam2=./experiments/benchmarks/busi_camus/medsam2/p0/results/test_metrics.json \
    sonobase=./experiments/benchmarks/busi_camus/hiera_b_conv_s_conv_t/p0/results/test_metrics.json \
  --output ./experiments/benchmarks/busi_camus/comparison.csv

echo ""
echo "Done. Comparison written to ./experiments/benchmarks/busi_camus/comparison.csv"
