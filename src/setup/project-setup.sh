#!/bin/bash
# Project environment setup — run once after cloning.
#
# This script sets up your development environment. It supports two modes:
#   1. Local (conda)   — for your laptop/workstation
#   2. Cluster (container + venv) — for the GPU cluster
#
# Usage:
#   cd <project>/src/setup
#   bash project-setup.sh
#
# Use `local` on a workstation and `cluster` on a Slurm cluster that runs jobs
# inside a container image (see container.env). For a locked workstation
# install from pyproject.toml + uv.lock, use setup-uv-env.sh instead.

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_DIR="$(dirname "$SRC_DIR")"
PROJECT_NAME="$(basename "$PROJECT_DIR")"

# --- Detect environment ---
on_cluster=false
if command -v srun &>/dev/null && command -v enroot &>/dev/null; then
    on_cluster=true
fi

echo "========================================"
echo "  Project Setup: $PROJECT_NAME"
echo "========================================"
echo ""

# --- Choose mode ---
if [[ "${1:-}" == "local" ]]; then
    mode="local"
elif [[ "${1:-}" == "cluster" ]]; then
    mode="cluster"
elif [[ "${1:-}" == "--refresh-container-packages" ]]; then
    mode="refresh-container-packages"
else
    echo "Where are you setting up?"
    echo ""
    if $on_cluster; then
        echo "  1) Local (conda — for your laptop)"
        echo "  2) Cluster (container + venv — for GPU training)"
        echo ""
        echo "  Detected: You appear to be on the cluster."
    else
        echo "  1) Local (conda — for your laptop)"
        echo "  2) Cluster (container + venv — for GPU training)"
        echo ""
        echo "  Detected: You appear to be on a local machine."
    fi
    echo ""
    read -rp "Choose [1/2]: " choice
    case "$choice" in
        1) mode="local" ;;
        2) mode="cluster" ;;
        *)
            echo "Invalid choice. Run again with: bash project-setup.sh local  OR  bash project-setup.sh cluster"
            exit 1
            ;;
    esac
fi

echo ""

# =============================================
# LOCAL SETUP (conda)
# =============================================
if [[ "$mode" == "local" ]]; then
    echo "--- Local Setup (conda) ---"
    echo ""

    if ! command -v conda &>/dev/null; then
        echo "ERROR: conda not found. Install Miniconda or Anaconda first:"
        echo "  https://docs.conda.io/en/latest/miniconda.html"
        exit 1
    fi

    ENV_FILE="$SCRIPT_DIR/environment.yml"
    if [[ ! -f "$ENV_FILE" ]]; then
        echo "ERROR: environment.yml not found at $ENV_FILE"
        exit 1
    fi

    ENV_NAME=$(grep 'name:' "$ENV_FILE" 2>/dev/null | head -1 | awk '{print $2}')
    if [[ -z "$ENV_NAME" ]]; then
        ENV_NAME="$PROJECT_NAME"
    fi

    # Check if env already exists
    if conda env list 2>/dev/null | grep -q "^${ENV_NAME} "; then
        echo "Conda environment '$ENV_NAME' already exists."
        read -rp "Update it? [y/N]: " update
        if [[ "$update" =~ ^[Yy] ]]; then
            echo "Updating environment..."
            conda env update -f "$ENV_FILE" --prune
        fi
    else
        echo "Creating conda environment '$ENV_NAME'..."
        conda env create -f "$ENV_FILE"
    fi

    echo ""
    echo "========================================"
    echo "  Local setup complete!"
    echo "========================================"
    echo ""
    echo "  Activate:  conda activate $ENV_NAME"
    echo ""
    echo "Next: read README.md for the phase-by-phase run instructions."

# =============================================
# CLUSTER SETUP (container + venv)
# =============================================
elif [[ "$mode" == "cluster" ]]; then
    echo "--- Cluster Setup (container + venv) ---"
    echo ""

    if [[ -f "$SCRIPT_DIR/setup-cluster-env.sh" ]]; then
        bash "$SCRIPT_DIR/setup-cluster-env.sh"
    else
        echo "ERROR: setup-cluster-env.sh not found at $SCRIPT_DIR/"
        echo "This project may not have cluster support configured."
        exit 1
    fi

    # Auto-generate container-packages.txt if missing (first-time cluster setup)
    CONTAINER_PKG_FILE="$SCRIPT_DIR/container-packages.txt"
    if [[ ! -f "$CONTAINER_PKG_FILE" ]]; then
        echo ""
        echo "--- First-time setup: generating container-packages.txt ---"
        echo "Running 'pip list --format=freeze' inside the base container..."
        # shellcheck disable=SC1091
        source "$SCRIPT_DIR/container.env" 2>/dev/null || true
        BASE_SQSH="${CLUSTER_BASE_SQSH:-${USER_BASE_SQSH:-}}"
        if [[ -n "$BASE_SQSH" && -f "$BASE_SQSH" ]]; then
            srun -N1 -n1 --time=00:05:00 \
                --container-image="$BASE_SQSH" \
                --container-mounts="${CONTAINER_MOUNTS:-$PROJECT_DIR:$PROJECT_DIR}" \
                bash -c "pip list --format=freeze" 2>&1 | grep -v '^srun' | grep -vE '^(cpu-bind|WARNING)' > "$CONTAINER_PKG_FILE" || true
            if [[ -s "$CONTAINER_PKG_FILE" ]]; then
                echo "  Generated $(wc -l < "$CONTAINER_PKG_FILE") entries in $CONTAINER_PKG_FILE"
                echo "  It records the packages the base container already provides."
            else
                echo "  WARNING: could not generate container-packages.txt. Try manually:"
                echo "    bash project-setup.sh --refresh-container-packages"
            fi
        else
            echo "  Skipping — no base container image found. Configure src/setup/container.env first."
        fi
    fi

elif [[ "$mode" == "refresh-container-packages" ]]; then
    echo "--- Regenerating container-packages.txt ---"
    echo ""
    # shellcheck disable=SC1091
    source "$SCRIPT_DIR/container.env" 2>/dev/null || true
    BASE_SQSH="${CLUSTER_BASE_SQSH:-${USER_BASE_SQSH:-}}"
    if [[ -z "$BASE_SQSH" || ! -f "$BASE_SQSH" ]]; then
        echo "ERROR: no base container image found. Check src/setup/container.env."
        exit 1
    fi
    CONTAINER_PKG_FILE="$SCRIPT_DIR/container-packages.txt"
    echo "Running 'pip list --format=freeze' inside $BASE_SQSH..."
    srun -N1 -n1 --time=00:05:00 \
        --container-image="$BASE_SQSH" \
        --container-mounts="${CONTAINER_MOUNTS:-$PROJECT_DIR:$PROJECT_DIR}" \
        bash -c "pip list --format=freeze" 2>&1 | grep -v '^srun' | grep -vE '^(cpu-bind|WARNING)' > "$CONTAINER_PKG_FILE"
    echo "Wrote $(wc -l < "$CONTAINER_PKG_FILE") entries to $CONTAINER_PKG_FILE"
    echo "Review with: head $CONTAINER_PKG_FILE"
    echo "Commit the regenerated file."
fi
