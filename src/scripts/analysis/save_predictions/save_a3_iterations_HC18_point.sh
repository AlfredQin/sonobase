#!/bin/bash
# A3 click-efficiency Stage-1: snapshot predictions at iterations [0,1,3,5,7]
# for all 3 models on HC18 with point initial prompt.
set -e
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
ITER='[0,1,3,5,7]'
echo "########### HC18 / point + iterations ${ITER} ###########"
for MODEL in sam2_no_ft medsam2 sonobase; do
  echo ""
  echo "=== $MODEL ==="
  RUN_NAME_OVERRIDE="${MODEL}_HC18_point_iters01357"
  OUT="./experiments/analysis/predictions/${RUN_NAME_OVERRIDE}"
  bash "$SCRIPT_DIR/_save_one.sh" "$MODEL" HC18 point \
    "+prompt_protocol.iterations=${ITER}" \
    "run_name=${RUN_NAME_OVERRIDE}" \
    "output_dir=${OUT}"
done
