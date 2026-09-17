#!/bin/bash
# US-RFDETR per-dataset wrapper: train+test on BUS-BRA for one or more backbones.
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./_common.sh
source "${SCRIPT_DIR}/_common.sh"
BACKBONES=("${@:-${ALL_BACKBONES[@]}}")
for bb in "${BACKBONES[@]}"; do
  echo ""
  echo "=== US-RFDETR on BUS-BRA | backbone=${bb} ==="
  bash "${SCRIPT_DIR}/_run_one.sh" bus_bra "${bb}"
done
