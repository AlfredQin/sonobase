#!/bin/bash
# Convenience wrapper: run every Stage-2 analysis at the point-prompt
# protocol, in dependency order (clinical-measurement → derived → cross).
# Each child script tolerates missing inputs and either skips or errors
# with a clear message.

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

# --- Clinical measurement first (these are referenced by the post-hoc analyses) ---
run "A1 — CAMUS Ejection Fraction" a1_camus_ef_point.sh
run "A2 — HC18 Head Circumference" a2_hc18_hc_point.sh
run "A4 — ACOUSLIC Abdominal Circumference" a4_acouslic_ac_point.sh
run "C2 — RegPro Prostate Volume" c2_regpro_volume.sh

# --- Per-structure / per-dataset segmentation slices ---
run "B1 — CAMUS per-structure" b1_camus_per_structure.sh
run "B3 — Porcine spinal cord" b3_porcine_spinal.sh

# --- Click efficiency (requires a3 iteration runs) ---
run "A3 — Click efficiency" a3_click_efficiency.sh

# --- Failure catalogs ---
run "B2 — Catastrophic failure catalog" b2_failure_catalog.sh
run "T3.2 — SonoBase failure analysis" t3_2_sonobase_failures.sh

# --- Statistics & post-hoc ---
run "T1.1 — Significance tests (BH-FDR)" t1_1_significance.sh
run "T1.2 — Bland-Altman composite figures" t1_2_bland_altman.sh
run "T2.1 — HC -> GA (Hadlock)" t2_1_hc_ga_point.sh
run "T2.2 — EF gray zone" t2_2_ef_gray_zone.sh
run "T2.3 — FGR screening (conditional)" t2_3_fgr_screening.sh
run "T3.1 — Temporal consistency" t3_1_temporal_consistency.sh

# --- Cross-analysis aggregator ---
run "compare_models — master cross-analysis CSV" compare_models.sh

echo ""
if [ ${#FAILED[@]} -gt 0 ]; then
  echo "Stage-2 analyses FAILED (${#FAILED[@]}): ${FAILED[*]}"
  exit 1
fi
echo "All Stage-2 analyses completed successfully."
