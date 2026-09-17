#!/bin/bash
# Final Stage-2 pass once every specialist run exists (CPU, project venv, from src/):
#   1. import the three ACOUSLIC-AI solutions (5-fold CV outputs merged by the inference scripts) on the archived SonoBase rows,
#   2. A4 (v1.1 ground truth, 0.28 mm/px) with the specialists beside SonoBase box / point, MedSAM2, SAM2,
#   3. T1.1 with every specialist pair (A1 CAMUS EF, A2 HC18 HC, A4 ACOUSLIC AC) in ONE BH-FDR family,
#   4. the aggregator (Table 1 rows, per-category cells, clinical scalars, paired segmentation tests) + the T1.1 CSV extracts.
#   bash scripts/analysis/specialist/finalize.sh [--skip-import]
#   (or: sbatch --export=ALL,SCRIPT=scripts/analysis/specialist/finalize.sh scripts/analysis/slurm/specialist_stage2.sbatch)
source "$(dirname "$(readlink -f "$0")")/_common.sh"
HERE="$(dirname "$(readlink -f "$0")")"
NOISED_CSV=${NOISED_CSV:?set NOISED_CSV to the 3-seed randomised-prompt record (prompt_noise_3seed.csv from scripts/analysis/aggregate_prompt_noise.py)}
[ -f "${NOISED_CSV}" ] || { echo "noised-prompt CSV not found: ${NOISED_CSV}" >&2; exit 1; }
ACOUSLIC_SOLS=(a3 a2 a1)
if [ "${1:-}" != "--skip-import" ]; then
  for sol in "${ACOUSLIC_SOLS[@]}"; do
    NEUTRAL="${EXTERNAL}/acouslic_us/neutral/${sol}/ACOUSLIC"
    python3 - "${NEUTRAL}/manifest.json" <<'PY'
import json, sys
m = json.load(open(sys.argv[1])); folds = m.get("extra", {}).get("folds", {})
assert sorted(folds) == ["0", "1", "2", "3", "4"], f"{sys.argv[1]}: folds present = {sorted(folds)}"
assert all(v["n_test"] == 60 for v in folds.values()), folds
PY
    bash "${HERE}/import_external.sh" "acouslic_${sol}_cv" ACOUSLIC "${NEUTRAL}"
    python3 - "${PRED_ROOT}/acouslic_${sol}_cv_ACOUSLIC_none/per_sample_metrics.csv" <<'PY'
import csv, sys
n = sum(1 for _ in csv.DictReader(open(sys.argv[1]))); assert n == 5868, f"{sys.argv[1]}: {n} rows (expected 5868)"; print(f"{sys.argv[1]}: {n} rows")
PY
  done
fi
bash "${HERE}/stage2_acouslic.sh" acouslic_a3_cv acouslic_a2_cv acouslic_a1_cv
T11="${ANALYSIS_OUT_ROOT}/t1_1_significance/specialists"
uv run python -m nemo_cv.recipes.analysis.t1_1_significance \
  --inputs "a1=${ANALYSIS_OUT_ROOT}/a1_camus_ef/CAMUS_specialists/per_patient.csv" \
           "a2=${ANALYSIS_OUT_ROOT}/a2_hc18_hc/HC18_specialists/per_sample.csv" \
           "a4=${ANALYSIS_OUT_ROOT}/a4_acouslic_ac/ACOUSLIC_specialists/per_sample.csv" \
  --baseline-models nnunet_resenc_m dlv3p_r50 echonet_lv_ft echonet_lv_zeroshot acouslic_a3_cv acouslic_a2_cv acouslic_a1_cv \
                    sonobase_point medsam2 sam2_no_ft \
  --target-model sonobase --output-dir "${T11}"
RES="${EXP_DIR}/results"; mkdir -p "${RES}"
uv run python scripts/analysis/specialist_vs_sonobase.py --pred-root "${PRED_ROOT}" --noised-csv "${NOISED_CSV}" \
  --specialists nnunet_resenc_m dlv3p_r50 echonet_lv_zeroshot echonet_lv_ft acouslic_a3_cv acouslic_a2_cv acouslic_a1_cv \
  --clinical "a1=${ANALYSIS_OUT_ROOT}/a1_camus_ef/CAMUS_specialists/analysis_report.json" \
             "a2=${ANALYSIS_OUT_ROOT}/a2_hc18_hc/HC18_specialists/analysis_report.json" \
             "a4=${ANALYSIS_OUT_ROOT}/a4_acouslic_ac/ACOUSLIC_specialists/analysis_report.json" \
  --out-dir "${RES}"
cp "${T11}/wilcoxon_results.csv" "${RES}/specialist_t1_1_wilcoxon.csv"
python3 - "${T11}/analysis_report.json" "${RES}/specialist_t1_1_bootstrap_ci.csv" <<'PY'
import csv, json, sys
rows = json.load(open(sys.argv[1]))["bootstrap_ci"]; cols = ["analysis", "metric", "unit", "model", "n", "mae", "ci_lo", "ci_hi"]
with open(sys.argv[2], "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); [w.writerow({c: r[c] for c in cols}) for r in rows]
print(f"{len(rows)} bootstrap rows -> {sys.argv[2]}")
PY
echo "finalize done -> ${RES}"
