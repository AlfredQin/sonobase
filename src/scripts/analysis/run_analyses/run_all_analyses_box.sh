#!/bin/bash
# Convenience wrapper: run every Stage-2 analysis at the box-prompt
# protocol, in dependency order (clinical-measurement → derived → cross).
# Each child script tolerates missing inputs and either skips or errors
# with a clear message.
#
# S1/S2/S3 subgroup analyses cover both prompts in one invocation, so
# they're listed only in `run_all_analyses_point.sh` to avoid running
# them twice; uncomment below if you want to re-run them for the box
# pass independently.
#
# A3 click-efficiency consumes the benchmark sweep CSVs (point + box
# in the same invocation), so it is also listed only in the point
# wrapper.

set -e
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
FAILED=()

run() {
  echo ""
  echo "##############################################################"
  echo "# $1"
  echo "##############################################################"
  if ! bash "$SCRIPT_DIR/$2"; then
    echo "  [warn] $2 returned non-zero (check inputs)"
    FAILED+=("$2")
  fi
}

# --- Clinical measurements first (referenced by post-hoc analyses) ---
run "A1 — CAMUS Ejection Fraction (box)" a1_camus_ef_box.sh
run "A2 — HC18 Head Circumference (box)" a2_hc18_hc_box.sh
run "A4 — ACOUSLIC Abdominal Circumference (box)" a4_acouslic_ac_box.sh
run "C2 — RegPro Prostate Volume (box)" c2_regpro_volume_box.sh

# --- Per-structure / per-dataset segmentation slices (box) ---
run "B1 — CAMUS per-structure (box)" b1_camus_per_structure_box.sh
run "B3 — Porcine spinal cord (box)" b3_porcine_spinal_box.sh

# --- Failure catalogs (box) ---
run "B2 — Catastrophic failure catalog (box)" b2_failure_catalog_box.sh
run "T3.2 — SonoBase failure analysis (box)" t3_2_sonobase_failures_box.sh

# --- Statistics & post-hoc (box) ---
run "T1.1 — Significance tests (BH-FDR, box)" t1_1_significance_box.sh
run "T1.2 — Bland-Altman composite figures (box)" t1_2_bland_altman_box.sh
run "T2.1 — HC -> GA (Hadlock, box)" t2_1_hc_ga_box.sh
run "T2.2 — EF gray zone (box)" t2_2_ef_gray_zone_box.sh
run "T2.3 — FGR screening (box, conditional)" t2_3_fgr_screening_box.sh
run "T3.1 — Temporal consistency (box)" t3_1_temporal_consistency_box.sh

# Re-run compare_models so it picks up the box subdirs that just appeared.
run "compare_models — master cross-analysis CSV" compare_models.sh

echo ""
if [ ${#FAILED[@]} -gt 0 ]; then
  echo "Stage-2 box-prompt analyses FAILED (${#FAILED[@]}): ${FAILED[*]}"
  exit 1
fi
echo "All Stage-2 box-prompt analyses completed successfully."
