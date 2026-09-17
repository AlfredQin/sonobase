#!/bin/bash
# Verification helper — run the B1 default sweep and the B2 standalone
# sweep on BUSI + CAMUS, then diff the resulting per-iteration CSVs to
# confirm they agree within Monte-Carlo tolerance.
#
# This is the validation step the project will use before promoting B2
# to a production-default pipeline. It runs both wrappers sequentially
# (B1 first, then B2) and finally calls a small inline Python diff that
# reports per-(dataset, iteration, metric) absolute / relative deltas.
#
# Cost estimate (smoke):
#   B1: 3 models × 2 prompts × 5 iters × 1 BUSI+CAMUS eval ≈ 30 runs
#   B2: 3 models × 2 prompts × 1 single-pass run         ≈ 6 runs
#   Total ≈ 36 process launches.
#
# Run from src/.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Tolerance for the metric agreement check. RNG-driven prompt sampling
# means B1 / B2 are NOT byte-equivalent (see the docstring of
# iterative_eval.py for the explanation); aggregate metrics over a
# couple-hundred samples should agree well below this bound.
ABS_TOL="${ABS_TOL:-0.02}"   # absolute mIoU/Dice tolerance
REL_TOL="${REL_TOL:-0.05}"   # relative tolerance (5%)

DATASET="busi_camus"
MODELS=(sam2_no_ft medsam2 sonobase)
PROMPTS=(point box)

echo "########################################################"
echo "# B1 vs B2 validation on ${DATASET}"
echo "# Tolerances: |Δ| <= ${ABS_TOL}  OR  |Δ|/B1 <= ${REL_TOL}"
echo "########################################################"

if [[ "${SKIP_RUNS:-0}" != "1" ]]; then
  echo ""
  echo "=== Step 1/3: B1 sweep ==="
  bash "${SCRIPT_DIR}/run_b1_iterations_busi_camus.sh"

  echo ""
  echo "=== Step 2/3: B2 sweep ==="
  bash "${SCRIPT_DIR}/run_b2_iterations_busi_camus.sh"
else
  echo "SKIP_RUNS=1 -- skipping evaluation, comparing existing CSVs only."
fi

echo ""
echo "=== Step 3/3: Diffing B1 vs B2 per-iteration CSVs ==="

# Build the list of (label, b1_csv, b2_csv) triples for the diff.
TRIPLES=()
for model in "${MODELS[@]}"; do
  model_dir="$model"
  if [[ "$model" == "sam2_no_ft" ]]; then model_dir="sam2"; fi
  if [[ "$model" == "sonobase" ]]; then model_dir="hiera_b_conv_s_conv_t"; fi
  for prompt in "${PROMPTS[@]}"; do
    label="${model}/${prompt}"
    b1_csv="./experiments/benchmarks/${DATASET}/${model_dir}/b1_${prompt}/test_metrics_iterations.csv"
    b2_csv="./experiments/benchmarks/${DATASET}/${model_dir}/b2_${prompt}/results/test_metrics_iterations.csv"
    TRIPLES+=( "${label}|${b1_csv}|${b2_csv}" )
  done
done

# Inline Python diff. Reads the long-format CSVs (skipping the
# AGGREGATE_* tail rows for the per-dataset table; we re-aggregate from
# the per-dataset rows so any drift in the aggregator itself is caught
# separately). Reports any (dataset, iteration, metric) triple whose
# delta exceeds both tolerances.
ABS_TOL="${ABS_TOL}" REL_TOL="${REL_TOL}" \
TRIPLES_STR="$(IFS=$'\n'; echo "${TRIPLES[*]}")" \
uv run python - <<'PY'
import csv
import os
import sys
from pathlib import Path

abs_tol = float(os.environ.get("ABS_TOL", "0.02"))
rel_tol = float(os.environ.get("REL_TOL", "0.05"))

triples = [t for t in os.environ["TRIPLES_STR"].splitlines() if t.strip()]


def load(path):
    p = Path(path)
    if not p.is_file():
        return None
    rows = []
    with p.open() as f:
        r = csv.DictReader(f)
        for row in r:
            ds = row.get("dataset", "")
            if not ds or ds.startswith("AGGREGATE"):
                continue
            try:
                it = int(row["iteration"])
            except (KeyError, ValueError):
                continue
            metrics = {}
            for k, v in row.items():
                if k in ("dataset", "iteration", "n_samples"):
                    continue
                if v in (None, ""):
                    continue
                try:
                    metrics[k] = float(v)
                except ValueError:
                    pass
            rows.append((ds, it, metrics))
    return rows


total = 0
fail = 0
header = f"{'combo':<28} {'dataset':<24} {'iter':>5} {'metric':<10} {'B1':>10} {'B2':>10} {'|Δ|':>10} {'rel':>10} {'ok'}"
print(header)
print("-" * len(header))

for triple in triples:
    label, b1_path, b2_path = triple.split("|")
    b1 = load(b1_path)
    b2 = load(b2_path)
    if b1 is None or b2 is None:
        missing = b1_path if b1 is None else b2_path
        print(f"{label:<28} MISSING {missing}")
        fail += 1
        continue
    b1_idx = {(d, i): m for d, i, m in b1}
    b2_idx = {(d, i): m for d, i, m in b2}
    keys = sorted(set(b1_idx) & set(b2_idx))
    for d, i in keys:
        for m in sorted(set(b1_idx[d, i]) & set(b2_idx[d, i])):
            v1 = b1_idx[d, i][m]
            v2 = b2_idx[d, i][m]
            d_abs = abs(v1 - v2)
            d_rel = d_abs / max(abs(v1), 1e-12)
            ok = (d_abs <= abs_tol) or (d_rel <= rel_tol)
            total += 1
            if not ok:
                fail += 1
            mark = "OK" if ok else "FAIL"
            print(
                f"{label:<28} {d:<24} {i:>5d} {m:<10} {v1:>10.4f} {v2:>10.4f} "
                f"{d_abs:>10.4f} {d_rel:>10.2%} {mark}"
            )

print()
print(f"Total comparisons: {total}, failures: {fail} (tol abs<={abs_tol}, rel<={rel_tol})")
sys.exit(1 if fail > 0 else 0)
PY

echo ""
echo "Done. If the comparison above passed (no FAIL rows), B2 is consistent"
echo "with the production B1 pipeline within the configured tolerance."
