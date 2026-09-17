#!/bin/bash
# Sequentially run US-RFDETR train+test on every (dataset × backbone)
# cell — 9 datasets × 3 backbones = 27 runs total. Each run drops its
# own `best.pt` and `test_metrics.json` under
# `./experiments/us_rfdetr/<dataset>/<backbone>/`.
#
# This is a long job — plan a multi-hour wall-clock window and launch
# under nohup. The order below is small-to-large by dataset size so a
# crash early in the sweep loses less work.
#
# Override the iteration via env vars to scope down:
#   ONLY_DATASETS="ddti acouslic"   bash run_all_us_rfdetr.sh
#   ONLY_BACKBONES="sonobase"       bash run_all_us_rfdetr.sh
#
# Run from src/.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./_common.sh
source "${SCRIPT_DIR}/_common.sh"

if [[ -n "${ONLY_DATASETS:-}" ]]; then
  read -r -a DATASETS <<< "${ONLY_DATASETS}"
else
  DATASETS=("${ALL_DATASETS[@]}")
fi
if [[ -n "${ONLY_BACKBONES:-}" ]]; then
  read -r -a BACKBONES <<< "${ONLY_BACKBONES}"
else
  BACKBONES=("${ALL_BACKBONES[@]}")
fi

TOTAL=$(( ${#DATASETS[@]} * ${#BACKBONES[@]} ))
echo ""
echo "########################################################"
echo "# US-RFDETR full sweep — ${#DATASETS[@]} datasets × ${#BACKBONES[@]} backbones = ${TOTAL} runs"
echo "########################################################"
echo "  datasets:  ${DATASETS[*]}"
echo "  backbones: ${BACKBONES[*]}"
echo ""

i=0
for ds in "${DATASETS[@]}"; do
  for bb in "${BACKBONES[@]}"; do
    i=$((i + 1))
    echo ""
    echo "########################################################"
    echo "# [${i}/${TOTAL}] $(_dataset_label "${ds}") | backbone=${bb}"
    echo "########################################################"
    bash "${SCRIPT_DIR}/_run_one.sh" "${ds}" "${bb}"
  done
done

echo ""
echo "########################################################"
echo "# Sweep done. Per-cell outputs:"
echo "########################################################"
for ds in "${DATASETS[@]}"; do
  for bb in "${BACKBONES[@]}"; do
    echo "  ./experiments/us_rfdetr/${ds}/${bb}/"
  done
done
