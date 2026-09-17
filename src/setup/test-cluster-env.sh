#!/bin/bash
# test-cluster-env.sh — Verify the cluster environment is working.
#
# Run this after setup-cluster-env.sh to confirm everything works.
# It checks: container access, GPU visibility, venv activation,
# package isolation, and persistence.
#
# Usage:
#   cd <project>/src/setup
#   bash test-cluster-env.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_DIR="$(dirname "$SRC_DIR")"
PROJECT_NAME="$(basename "$PROJECT_DIR")"
VENV_DIR="$SRC_DIR/env/venv"

# Read container config
eval "$(grep -v '^\s*#' "$SCRIPT_DIR/container.env" | grep '=')"
CONTAINER_MOUNTS="${CONTAINER_MOUNTS:-$PROJECT_DIR:$PROJECT_DIR}"

# Find the base image (same logic as setup script)
BASE_SQSH=""
if [[ -n "${CLUSTER_BASE_SQSH:-}" && -f "$CLUSTER_BASE_SQSH" ]]; then
    BASE_SQSH="$CLUSTER_BASE_SQSH"
elif [[ -n "${USER_BASE_SQSH:-}" && -f "$USER_BASE_SQSH" ]]; then
    BASE_SQSH="$USER_BASE_SQSH"
else
    echo "FAIL: No base container image found. Run setup-cluster-env.sh first."
    exit 1
fi

PASS=0
FAIL=0

run_test() {
    local name="$1"
    local exit_code="$2"
    if [[ "$exit_code" -eq 0 ]]; then
        echo "  PASS: $name"
        ((PASS++))
    else
        echo "  FAIL: $name"
        ((FAIL++))
    fi
}

echo "============================================"
echo " Testing cluster environment: $PROJECT_NAME"
echo " Base image: $BASE_SQSH"
echo "============================================"
echo ""

# --- Test 1: Venv exists ---
echo "[Test 1] Venv exists"
test -f "$VENV_DIR/bin/activate"
run_test "src/env/venv/bin/activate exists" $?

echo ""

# --- Test 2: Container + GPU access ---
echo "[Test 2] Container access and GPU visibility (requesting GPU allocation...)"
GPU_OUTPUT=$(srun -N1 -n1 --gres=gpu:1 --time=00:03:00 \
    --container-image="$BASE_SQSH" \
    --container-mounts="$CONTAINER_MOUNTS" \
    python -c "
import torch
print(f'pytorch={torch.__version__}')
print(f'cuda={torch.version.cuda}')
print(f'gpu={torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'device={torch.cuda.get_device_name(0)}')
" 2>&1 | grep -v 'cpu-bind')

echo "$GPU_OUTPUT"
echo "$GPU_OUTPUT" | grep -q "gpu=True"
run_test "GPU is visible inside container" $?

echo ""

# --- Test 3: Venv activation inside container ---
echo "[Test 3] Venv activation inside container"
VENV_OUTPUT=$(srun -N1 -n1 --gres=gpu:1 --time=00:03:00 \
    --container-image="$BASE_SQSH" \
    --container-mounts="$CONTAINER_MOUNTS" \
    bash -c "
source '$VENV_DIR/bin/activate'
python -c \"
import sys
import torch
print(f'python={sys.executable}')
print(f'torch={torch.__version__}')
print(f'gpu={torch.cuda.is_available()}')
\"
" 2>&1 | grep -v 'cpu-bind')

echo "$VENV_OUTPUT"
echo "$VENV_OUTPUT" | grep -q "gpu=True"
run_test "GPU works with venv activated" $?

echo ""

# --- Test 4: pip install to venv (not base) ---
echo "[Test 4] Package install isolation (installing tqdm as test)..."
INSTALL_OUTPUT=$(srun -N1 -n1 --time=00:03:00 \
    --container-image="$BASE_SQSH" \
    --container-mounts="$CONTAINER_MOUNTS" \
    bash -c "
source '$VENV_DIR/bin/activate'
pip install --quiet tqdm 2>&1
python -c 'import tqdm; print(f\"tqdm={tqdm.__version__}\")'
# Verify it landed in the venv, not the base
pip show tqdm 2>/dev/null | grep -i location
" 2>&1 | grep -v 'cpu-bind' | grep -v 'DEPRECATION')

echo "$INSTALL_OUTPUT"
echo "$INSTALL_OUTPUT" | grep -q "tqdm="
run_test "pip install works inside venv" $?

echo "$INSTALL_OUTPUT" | grep -qi "location.*env/venv"
run_test "Package installed to venv (not base container)" $?

echo ""

# --- Test 5: Persistence across sessions ---
echo "[Test 5] Package persistence (new SLURM job, same venv)..."
PERSIST_OUTPUT=$(srun -N1 -n1 --time=00:02:00 \
    --container-image="$BASE_SQSH" \
    --container-mounts="$CONTAINER_MOUNTS" \
    bash -c "
source '$VENV_DIR/bin/activate'
python -c 'import tqdm; print(f\"tqdm_persisted={tqdm.__version__}\")'
python -c 'import torch; print(f\"torch_intact={torch.__version__}\")'
" 2>&1 | grep -v 'cpu-bind')

echo "$PERSIST_OUTPUT"
echo "$PERSIST_OUTPUT" | grep -q "tqdm_persisted="
run_test "Installed package persists across sessions" $?

echo "$PERSIST_OUTPUT" | grep -q "torch_intact="
run_test "Base PyTorch remains intact" $?

echo ""

# --- Test 6: Workspace mount ---
echo "[Test 6] Project directory accessible at workspace path"
MOUNT_OUTPUT=$(srun -N1 -n1 --time=00:02:00 \
    --container-image="$BASE_SQSH" \
    --container-mounts="$CONTAINER_MOUNTS" \
    bash -c "
source '$VENV_DIR/bin/activate'
python -c \"
import os
# Can we see the project files?
setup_script = '$SCRIPT_DIR/project-setup.sh'
print(f'setup_script_exists={os.path.exists(setup_script)}')
print(f'venv_exists={os.path.exists(\"$VENV_DIR/bin/activate\")}')
\"
" 2>&1 | grep -v 'cpu-bind')

echo "$MOUNT_OUTPUT"
echo "$MOUNT_OUTPUT" | grep -q "setup_script_exists=True"
run_test "Project files visible inside container" $?

echo ""
echo "============================================"
echo " Results: $PASS passed, $FAIL failed"
echo "============================================"

if [[ "$FAIL" -eq 0 ]]; then
    echo ""
    echo " All tests passed. Your environment is ready."
    echo ""
    echo " To start working:"
    echo "   srun --pty --gres=gpu:1 --time=04:00:00 \\"
    echo "       --container-image=$BASE_SQSH \\"
    echo "       --container-mounts=$CONTAINER_MOUNTS \\"
    echo "       bash -c \"source $VENV_DIR/bin/activate && bash\""
    echo ""
    exit 0
else
    echo ""
    echo " Some tests failed. Check the output above."
    echo " If the container image is wrong, update src/setup/container.env"
    echo " If the venv is missing, re-run: bash src/setup/project-setup.sh cluster"
    echo ""
    exit 1
fi
