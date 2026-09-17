#!/bin/bash
# SonoBase pretraining — array job for the 6 production encoder variants
# on the 38_pt_8_bm dataset mixture.
#
# Submit:
#
#   sbatch scripts/pretrain/slurm/pretrain_all_38_pt_8_bm.sbatch
#
# This launches 6 array tasks (one per encoder variant). Each task is a
# multi-node DDP run; tasks run independently so they can share the
# cluster (slurm interleaves them based on partition availability).
#
# Restrict to a subset:
#
#   sbatch --array=0,1 scripts/pretrain/slurm/pretrain_all_38_pt_8_bm.sbatch  # only TriBranchTrunk variants
#   sbatch --array=2-5 scripts/pretrain/slurm/pretrain_all_38_pt_8_bm.sbatch  # only single-encoder ablations
#
# Cancel one task:
#
#   scancel <jobid>_<task_idx>
#
# Per-task logs land at <sbatch_submit_dir>/slurm_logs/<jobname>_<JobID>_<TaskID>.{out,err}.

#SBATCH --job-name=pt_38pt
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err
#SBATCH --partition=main
#SBATCH --array=0-5                      # 6 encoder variants (override with --array=...)
#SBATCH --nodes=1                        # bump to 2 if scaling up; the inner _pretrain_one supports it
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus-per-node=8
#SBATCH --mem=0                          # all node memory (~1.5 TB) — job takes all 8 GPUs + 128 CPUs, so the node is exclusive
#SBATCH --time=24:00:00

set -euo pipefail

# Encoder variants in priority order (TriBranchTrunk variants first —
# they're the headline models; single-encoder ablations after).
EXPERIMENTS=(
  pretrain_hiera_b_conv_s_conv_t_on_38_pt_8_bm    # 0 — primary TriBranchTrunk (Hiera-B + ConvNeXt-S + ConvNeXt-T)
  pretrain_hiera_l_conv_b_conv_s_on_38_pt_8_bm    # 1 — Large TriBranchTrunk
  pretrain_hiera_b+_on_38_pt_8_bm                 # 2 — Hiera-B+ ablation
  pretrain_hiera_l_on_38_pt_8_bm                  # 3 — Hiera-L ablation
  pretrain_hiera_s_on_38_pt_8_bm                  # 4 — Hiera-S ablation
  pretrain_hiera_t_on_38_pt_8_bm                  # 5 — Hiera-T ablation (matches MedSAM2 trunk)
)
EXPERIMENT="${EXPERIMENTS[${SLURM_ARRAY_TASK_ID}]}"

# Per-encoder pretrained-weights override for the four single-encoder
# ablations + the Large TriBranchTrunk (otherwise scratch.ckpt_path
# defaults to Hiera-B+, which is wrong for those variants).
: "${CHECKPOINT_DIR:?CHECKPOINT_DIR must be set, e.g. export CHECKPOINT_DIR=\$WORK/Checkpoints}"
case "${EXPERIMENT}" in
  pretrain_hiera_l_conv_b_conv_s_on_38_pt_8_bm|pretrain_hiera_l_on_38_pt_8_bm)
    EXTRA_OVERRIDES="scratch.ckpt_path=${CHECKPOINT_DIR}/SAM2/sam2.1_hiera_large.pt"
    ;;
  pretrain_hiera_s_on_38_pt_8_bm)
    EXTRA_OVERRIDES="scratch.ckpt_path=${CHECKPOINT_DIR}/SAM2/sam2.1_hiera_small.pt"
    ;;
  pretrain_hiera_t_on_38_pt_8_bm)
    EXTRA_OVERRIDES="scratch.ckpt_path=${CHECKPOINT_DIR}/SAM2/sam2.1_hiera_tiny.pt"
    ;;
  *)
    EXTRA_OVERRIDES=""
    ;;
esac

# Forward to the generic single-experiment slurm script. Re-exec it via
# the same SBATCH allocation so the (nodes, gpus, mem, time) block is
# inherited automatically.
export EXPERIMENT
export EXTRA_OVERRIDES

SBATCH_PATH="$(scontrol show job "${SLURM_JOB_ID}" -o | grep -oP 'Command=\K[^[:space:]]+')"
[ -f "${SBATCH_PATH}" ] || { echo "ERROR: cannot resolve this sbatch script's path from scontrol (got '${SBATCH_PATH}')" >&2; exit 1; }
SCRIPT_DIR="$(cd "$(dirname "${SBATCH_PATH}")" && pwd)"

echo "============================================================"
echo "# Array task ${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_MAX} → experiment=${EXPERIMENT}"
[ -n "${EXTRA_OVERRIDES}" ] && echo "#   extra overrides = ${EXTRA_OVERRIDES}"
echo "============================================================"

# Bash-source so the SBATCH directives at the top of _pretrain_one.sbatch
# are NOT re-parsed by slurm — only the body executes in our allocation.
source "${SCRIPT_DIR}/_pretrain_one.sbatch"
