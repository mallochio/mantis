#!/usr/bin/env bash
# Run the openfugu orchestrator natively on the host (MPS/CUDA/CPU).
# litellm and llm-router should already be running in Docker on their
# published ports (3001 and 5500).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$REPO_ROOT/.venv-openfugu"
SERVE="$REPO_ROOT/openfugu-patch/serve.py"

if [[ ! -f "$SERVE" ]]; then
    echo "ERROR: $SERVE not found" >&2
    exit 1
fi

# Create venv and install dependencies if missing.
if [[ ! -d "$VENV/bin" ]]; then
    echo "[native-openfugu] creating venv at $VENV ..."
    python3 -m venv "$VENV"
fi

# shellcheck source=/dev/null
source "$VENV/bin/activate"

# Install/upgrade core orchestrator dependencies.
python3 -m pip install --quiet --upgrade pip
python3 -m pip install --quiet -e "$REPO_ROOT"

# Build the TRINITY base vector from the committed small safetensors head if needed.
if [[ ! -f "$REPO_ROOT/artifacts/model_iter_60.npy" ]]; then
    echo "[native-openfugu] building artifacts/model_iter_60.npy from router_head.safetensors ..."
    python3 "$REPO_ROOT/scripts/make_vec.py"
fi

# Load .env so native process gets the same config as the Docker stack.
# shellcheck source=/dev/null
set -a
source "$REPO_ROOT/.env"
set +a

# Auto-detect device unless the user explicitly set FUGU_CONDUCTOR_DEVICE.
if [[ -z "${FUGU_CONDUCTOR_DEVICE:-}" || "${FUGU_CONDUCTOR_DEVICE}" == "auto" ]]; then
    DEVICE=$(python3 - <<'PY'
import torch
if torch.backends.mps.is_available():
    print("mps")
elif torch.cuda.is_available():
    print("cuda:0")
else:
    print("cpu")
PY
    )
    export FUGU_CONDUCTOR_DEVICE="$DEVICE"
fi

# Default dtype: bfloat16 on mps/cuda, float32 on cpu unless user overrides.
if [[ -z "${FUGU_CONDUCTOR_DTYPE:-}" ]]; then
    if [[ "$FUGU_CONDUCTOR_DEVICE" == mps || "$FUGU_CONDUCTOR_DEVICE" == cuda* ]]; then
        export FUGU_CONDUCTOR_DTYPE=bfloat16
    else
        export FUGU_CONDUCTOR_DTYPE=float32
    fi
fi

# Point the native process at the local artifacts and the Dockerized LiteLLM proxy.
# The .env file uses Docker paths (/app/...), so override them if they do not exist.
[[ -f "${FUGU_VECTOR:-}" ]] || export FUGU_VECTOR="$REPO_ROOT/artifacts/model_iter_60.npy"
[[ -f "${FUGU_HEAD:-}" ]] || export FUGU_HEAD="$REPO_ROOT/artifacts/router_head.npy"
[[ -n "${FUGU_BASE_URL:-}" ]] || export FUGU_BASE_URL="http://127.0.0.1:3001/v1"

# Native process should listen on all interfaces so Docker containers can reach
# it via host.docker.internal on Mac, and via localhost on Linux.
export FUGU_HOST="${FUGU_HOST:-0.0.0.0}"
export FUGU_PORT="${FUGU_PORT:-8088}"

# Default to sampled generation for local Conductor (greedy decoding often
# produces degenerate or unparseable workflows with these checkpoints).
export FUGU_CONDUCTOR_DO_SAMPLE="${FUGU_CONDUCTOR_DO_SAMPLE:-true}"
export FUGU_CONDUCTOR_TEMPERATURE="${FUGU_CONDUCTOR_TEMPERATURE:-0.7}"
export FUGU_CONDUCTOR_TOP_P="${FUGU_CONDUCTOR_TOP_P:-0.9}"

echo "[native-openfugu] device=$FUGU_CONDUCTOR_DEVICE dtype=$FUGU_CONDUCTOR_DTYPE"
echo "[native-openfugu] do_sample=$FUGU_CONDUCTOR_DO_SAMPLE temp=$FUGU_CONDUCTOR_TEMPERATURE top_p=$FUGU_CONDUCTOR_TOP_P"
echo "[native-openfugu] listening on $FUGU_HOST:$FUGU_PORT"
echo "[native-openfugu] press Ctrl-C to stop"

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
exec python3 "$SERVE" --host "$FUGU_HOST" --port "$FUGU_PORT"
