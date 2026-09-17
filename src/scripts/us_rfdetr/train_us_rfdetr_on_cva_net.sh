#!/bin/bash
# US-RFDETR per-dataset wrapper: train+test on CVA-Net for one or more
# backbones (defaults to all three in scope).
# Usage: bash train_us_rfdetr_on_cva_net.sh [backbone1 backbone2 ...]
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./_common.sh
source "${SCRIPT_DIR}/_common.sh"
BACKBONES=("${@:-${ALL_BACKBONES[@]}}")
for bb in "${BACKBONES[@]}"; do
  echo ""
  echo "=== US-RFDETR on CVA-Net | backbone=${bb} ==="
  bash "${SCRIPT_DIR}/_run_one.sh" cva_net "${bb}"
done
