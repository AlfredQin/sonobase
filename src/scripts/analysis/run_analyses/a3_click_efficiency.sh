#!/bin/bash
# Stage 2 — A3: click-efficiency analysis.
#
# Consumes the long-format `test_metrics_iterations.csv` artifacts produced
# by the iterative-correction benchmark sweep (B1 default; see
# `src/scripts/benchmarks/test_sam2/iterative/README.md`). One CSV per
# (model × prompt) cell. The B2 standalone path produces an identical
# schema, so swapping `b1_<prompt>` -> `b2_<prompt>` below is sufficient
# to validate against the encoder-cached pipeline.
#
# Override `BENCH_ROOT` to consume a different benchmark dataset (e.g.
# `BENCH_ROOT=./experiments/benchmarks/busi_camus` for the smoke set
# during development).

set -e
source "$(dirname "$(readlink -f "$0")")/_paths.sh"

OUT="${ANALYSIS_OUT_ROOT}/a3_click_efficiency"
BENCH_ROOT="${BENCH_ROOT:-./experiments/benchmarks/8_bm_7_ext}"
PIPE="${PIPE:-b1}"   # "b1" (default) or "b2" (standalone verification)

# B1 emits `<sweep_dir>/test_metrics_iterations.csv` directly under
# `<dataset>/<model_dir>/b1_<prompt>/`. B2 places the same artifact
# under `b2_<prompt>/results/`. Both layouts are handled below.
case "${PIPE}" in
  b1) CSV_REL="test_metrics_iterations.csv" ;;
  b2) CSV_REL="results/test_metrics_iterations.csv" ;;
  *)  echo "ERROR: PIPE must be 'b1' or 'b2', got '${PIPE}'" >&2; exit 2 ;;
esac

uv run python -m nemo_cv.recipes.analysis.a3_click_efficiency \
  --inputs \
    "sam2_no_ft:point=${BENCH_ROOT}/sam2/${PIPE}_point/${CSV_REL}" \
    "sam2_no_ft:box=${BENCH_ROOT}/sam2/${PIPE}_box/${CSV_REL}" \
    "medsam2:point=${BENCH_ROOT}/medsam2/${PIPE}_point/${CSV_REL}" \
    "medsam2:box=${BENCH_ROOT}/medsam2/${PIPE}_box/${CSV_REL}" \
    "sonobase:point=${BENCH_ROOT}/hiera_b_conv_s_conv_t/${PIPE}_point/${CSV_REL}" \
    "sonobase:box=${BENCH_ROOT}/hiera_b_conv_s_conv_t/${PIPE}_box/${CSV_REL}" \
  --output-dir "${OUT}" \
  --per-dataset
