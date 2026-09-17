#!/bin/bash
# Shared environment + helpers for the US-RFDETR scripts.
# Source this file from per-dataset / per-backbone wrappers — do NOT
# execute directly.

# ---- env knobs ----
if [[ -z "${CUDA_VISIBLE_DEVICES+x}" ]]; then
  export CUDA_VISIBLE_DEVICES=1,2,3,4
fi
TRAIN_GPUS_DEFAULT=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | grep -c .)
export TRAIN_GPUS="${TRAIN_GPUS:-${TRAIN_GPUS_DEFAULT}}"
export TEST_GPUS="${TEST_GPUS:-1}"

export HYDRA_FULL_ERROR=1
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Workaround for the libcusparse / nvJitLink loader collision documented
# in `src/scripts/pretrain/README.md` § "Lessons Learned".
export LD_LIBRARY_PATH=""

# Asset paths — every helper inside the recipe / configs reads from
# these via OmegaConf env interpolation. All three must be set per user
# in your .bashrc:
#   export CHECKPOINT_DIR=$WORK/Checkpoints
#   export DATASET_DIR=$WORK/Dataset/SaUS
#   export DET_ANNOTATION_DIR=$WORK/Dataset/Ultrasound/Dense_Pred_Annotation
: "${CHECKPOINT_DIR:?CHECKPOINT_DIR must be set, e.g. export CHECKPOINT_DIR=\$WORK/Checkpoints}"
: "${DATASET_DIR:?DATASET_DIR must be set, e.g. export DATASET_DIR=\$WORK/Dataset/SaUS}"
: "${DET_ANNOTATION_DIR:?DET_ANNOTATION_DIR must be set, e.g. export DET_ANNOTATION_DIR=\$WORK/Dataset/Ultrasound/Dense_Pred_Annotation}"
export CHECKPOINT_DIR DATASET_DIR DET_ANNOTATION_DIR

# Default sonobase pretrain checkpoint — the best-validation epoch
# (LOWEST_VAL) of the busi_camus run. Override `SONOBASE_DCP` to use a
# different pretraining run / epoch / encoder variant.
export SONOBASE_DCP_DEFAULT="./experiments/sonobase/pretrain/busi_camus/hiera_b_conv_s_conv_t/LOWEST_VAL"
export SONOBASE_DCP="${SONOBASE_DCP:-${SONOBASE_DCP_DEFAULT}}"

# Three backbones in scope for US-RFDETR.
ALL_BACKBONES=(sam2_no_ft medsam2 sonobase)

# Nine datasets in scope.
ALL_DATASETS=(
  cva_net
  fetus
  acouslic
  bus_bra
  ddti
  fugc
  kidneyus
  luminous
  mmotu_3d
)

# Friendly dataset display name (matches the experiment overlay suffix).
_dataset_label() {
  case "$1" in
    cva_net)   echo "CVA-Net" ;;
    fetus)     echo "Fetus" ;;
    acouslic)  echo "ACOUSLIC" ;;
    bus_bra)   echo "BUS-BRA" ;;
    ddti)      echo "DDTI" ;;
    fugc)      echo "FUGC" ;;
    kidneyus)  echo "KidneyUS" ;;
    luminous)  echo "LUMINOUS" ;;
    mmotu_3d)  echo "MMOTU-3d" ;;
    *) echo "$1" ;;
  esac
}
