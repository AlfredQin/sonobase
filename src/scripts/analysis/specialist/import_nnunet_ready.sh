#!/bin/bash
# Import every Benchmark dataset whose nnU-Net predictions exist in the neutral layout but that has no Stage-1 run dir yet
# (idempotent; one CPU job imports whatever is ready).   Usage (from src/):
#   bash scripts/analysis/specialist/import_nnunet_ready.sh [label=nnunet_resenc_m] [extra importer args]
source "$(dirname "$(readlink -f "$0")")/_common.sh"
LABEL=${1:-nnunet_resenc_m}; [ $# -gt 0 ] && shift
DONE=(); TODO=(); WAIT=()
for DS in BUSI Brachial-Plexus C-TRUS CAMUS HC18 PFUS RegPro TG3K; do
  NEUTRAL="${EXTERNAL}/nnunet_us/neutral/${LABEL}/${DS}"
  if [ -f "${PRED_ROOT}/${LABEL}_${DS}_none/per_sample_metrics.csv" ]; then DONE+=("${DS}"); continue; fi
  if [ ! -f "${NEUTRAL}/manifest.json" ]; then WAIT+=("${DS}"); continue; fi
  TODO+=("${DS}")
done
echo "# ${LABEL}: already imported: ${DONE[*]:-none} | importing now: ${TODO[*]:-none} | no predictions yet: ${WAIT[*]:-none}"
for DS in "${TODO[@]}"; do
  bash "$(dirname "$(readlink -f "$0")")/import_external.sh" "${LABEL}" "${DS}" "${EXTERNAL}/nnunet_us/neutral/${LABEL}/${DS}" "$@"
done
