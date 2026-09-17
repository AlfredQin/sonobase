#!/bin/bash
# setup-cluster-env.sh — Set up this project's container environment on the GPU cluster.
#
# Run this ONCE after cloning the repo on the cluster.
# It creates a Python venv that layers on top of the GPU base container,
# so your pip installs are isolated per-project and persistent across sessions,
# while the base container (PyTorch, CUDA, cuDNN) stays untouched.
#
# Usage:
#   cd <project>/src/setup
#   bash setup-cluster-env.sh
#
# What it creates:
#   src/env/venv/    — Python venv with --system-site-packages
#                      (inherits PyTorch/CUDA from base container,
#                       your pip installs go here)
#
# After setup, use the venv inside the container:
#   srun --pty --gres=gpu:1 --time=04:00:00 \
#       --container-image=<sqsh> --container-mounts=<src>:<dst> \
#       bash -c "source /path/to/src/env/venv/bin/activate && python ..."

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_DIR="$(dirname "$SRC_DIR")"
PROJECT_NAME="$(basename "$PROJECT_DIR")"
VENV_DIR="$SRC_DIR/env/venv"

echo "============================================"
echo " Setting up cluster environment"
echo " Project: $PROJECT_NAME"
echo "============================================"
echo ""

# --- Step 1: Find the base container image ---
if [[ ! -f "$SCRIPT_DIR/container.env" ]]; then
    echo "ERROR: src/setup/container.env not found."
    echo "  Copy it from the repository and set CLUSTER_BASE_SQSH."
    exit 1
fi

# Source config (expand $USER at runtime)
eval "$(grep -v '^\s*#' "$SCRIPT_DIR/container.env" | grep '=')"
# Default mount: the repository itself. Add data / checkpoint paths via
# CONTAINER_MOUNTS in container.env or in your environment.
CONTAINER_MOUNTS="${CONTAINER_MOUNTS:-$PROJECT_DIR:$PROJECT_DIR}"

BASE_SQSH=""

# Option A: Cluster-provided image (preferred — no auth, no download)
if [[ -n "${CLUSTER_BASE_SQSH:-}" && -f "$CLUSTER_BASE_SQSH" ]]; then
    BASE_SQSH="$CLUSTER_BASE_SQSH"
    echo "[1/3] Using cluster base image: $BASE_SQSH"

# Option B: Per-user copy (already pulled previously)
elif [[ -n "${USER_BASE_SQSH:-}" && -f "$USER_BASE_SQSH" ]]; then
    BASE_SQSH="$USER_BASE_SQSH"
    echo "[1/3] Using existing user base image: $BASE_SQSH"

# Option C: Pull from NGC (requires API key)
elif [[ -n "${NGC_IMAGE:-}" ]]; then
    echo "[1/3] No local base image found."
    echo "      Attempting to pull from NGC: $NGC_IMAGE"
    echo "      (Requires an NGC API key. If this fails, set CLUSTER_BASE_SQSH"
    echo "       in src/setup/container.env to an image staged on your cluster.)"
    echo ""

    if [[ -z "${ENROOT_LOGIN_PASSWORD:-}" ]]; then
        echo "ERROR: NGC API key not set."
        echo "  export ENROOT_LOGIN_USERNAME='\$oauthtoken'"
        echo "  export ENROOT_LOGIN_PASSWORD='<your-key>'"
        echo "  Then re-run this script."
        echo ""
        echo "  Or set CLUSTER_BASE_SQSH in src/setup/container.env."
        exit 1
    fi

    mkdir -p "$(dirname "$USER_BASE_SQSH")"
    enroot import -o "$USER_BASE_SQSH" "docker://nvcr.io#${NGC_IMAGE#nvcr.io/}"
    BASE_SQSH="$USER_BASE_SQSH"
    echo "      Saved to $BASE_SQSH"
else
    echo "ERROR: No base image configured."
    echo "  Set CLUSTER_BASE_SQSH or NGC_IMAGE in src/setup/container.env."
    exit 1
fi

echo ""

# --- Step 2: Create project venv inside the container ---
if [[ -d "$VENV_DIR" && -f "$VENV_DIR/bin/activate" ]]; then
    echo "[2/3] Venv already exists at src/env/venv/. Skipping creation."
else
    echo "[2/3] Creating project venv (inherits base container packages)..."
    mkdir -p "$SRC_DIR/env"

    # Run inside the container so the venv links to the container's Python
    srun --pty -N1 -n1 --time=00:10:00 \
        --container-image="$BASE_SQSH" \
        --container-mounts="$CONTAINER_MOUNTS" \
        bash -c "python -m venv --system-site-packages '$VENV_DIR' && echo 'Venv created successfully.'"

    if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
        echo "ERROR: Venv creation failed."
        exit 1
    fi
fi

echo ""

# --- Step 3: Install project dependencies if lock file exists ---
if [[ -f "$SCRIPT_DIR/requirements.lock.txt" ]]; then
    echo "[3/3] Installing packages from requirements.lock.txt..."
    srun --pty -N1 -n1 --time=00:10:00 \
        --container-image="$BASE_SQSH" \
        --container-mounts="$CONTAINER_MOUNTS" \
        bash -c "source '$VENV_DIR/bin/activate' && pip install --quiet -r '$SCRIPT_DIR/requirements.lock.txt'"
    echo "      Packages installed."
else
    echo "[3/3] No requirements.lock.txt found. Skipping."
fi

echo ""
echo "============================================"
echo " Environment ready: $PROJECT_NAME"
echo "============================================"
echo ""
echo " Base container: $BASE_SQSH"
echo " Project venv:   src/env/venv/"
echo ""
echo " Quick start:"
echo "   # Get a GPU and enter the environment:"
echo "   srun --pty --gres=gpu:1 --time=04:00:00 \\"
echo "       --container-image=$BASE_SQSH \\"
echo "       --container-mounts=$CONTAINER_MOUNTS \\"
echo "       bash -c \"source $VENV_DIR/bin/activate && bash\""
echo ""
echo " Install packages (inside the container):"
echo "   pip install <package>       # persists in src/env/venv/"
echo "   pip install torch==X.Y      # ERROR: torch is in the base container"
echo ""
echo " The base container (PyTorch, CUDA) is read-only."
echo " Your pip installs go to src/env/venv/ and persist across sessions."
echo ""
