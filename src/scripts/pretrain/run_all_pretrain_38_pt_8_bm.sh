#!/bin/bash
# Sequentially run all 6 SonoBase pretraining variants on the
# 38_pt_8_bm dataset. Each run drops checkpoints under
# `./experiments/sonobase/pretrain/38_pt_8_bm/<encoder>/` and a
# matching `validation.jsonl` / `training.jsonl` log pair.
#
# Total wall time is dominated by the two TriBranchTrunk runs (≈ 2-3
# days each on 4×A6000 at the default scratch.num_epochs=40); the four
# single-encoder ablations are individually faster. Plan to launch
# under nohup and check progress via `tail -f` on the per-run logs.
#
# Run from src/.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 1) Headline TriBranchTrunk first — this is the model the paper cites.
echo "########################################################"
echo "# [1/6] TriBranchTrunk (Hiera-B + ConvNeXt-S + ConvNeXt-T)"
echo "########################################################"
bash "${SCRIPT_DIR}/pretrain_hiera_b_conv_s_conv_t_on_38_pt_8_bm.sh"

echo ""
echo "########################################################"
echo "# [2/6] TriBranchTrunk Large (Hiera-L + ConvNeXt-B + ConvNeXt-S)"
echo "########################################################"
bash "${SCRIPT_DIR}/pretrain_hiera_l_conv_b_conv_s_on_38_pt_8_bm.sh"

# 2) Four single-encoder ablations (smaller -> larger).
echo ""
echo "########################################################"
echo "# [3/6] Hiera-T ablation"
echo "########################################################"
bash "${SCRIPT_DIR}/pretrain_hiera_t_on_38_pt_8_bm.sh"

echo ""
echo "########################################################"
echo "# [4/6] Hiera-S ablation"
echo "########################################################"
bash "${SCRIPT_DIR}/pretrain_hiera_s_on_38_pt_8_bm.sh"

echo ""
echo "########################################################"
echo "# [5/6] Hiera-B+ ablation"
echo "########################################################"
bash "${SCRIPT_DIR}/pretrain_hiera_b+_on_38_pt_8_bm.sh"

echo ""
echo "########################################################"
echo "# [6/6] Hiera-L ablation"
echo "########################################################"
bash "${SCRIPT_DIR}/pretrain_hiera_l_on_38_pt_8_bm.sh"

echo ""
echo "Done. All 6 SonoBase pretraining runs complete."
echo "Checkpoints under: ./experiments/sonobase/pretrain/38_pt_8_bm/"
