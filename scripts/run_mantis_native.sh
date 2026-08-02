#!/usr/bin/env bash
# Run the mantis orchestrator natively on the host (MPS/CUDA/CPU).
# litellm and llm-router should already be running in Docker on their
# published ports (3001 and 5500).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$REPO_ROOT/.venv-mantis"
SERVE="$REPO_ROOT/openfugu-patch/serve.py"

if [[ ! -f "$SERVE" ]]; then
    echo "ERROR: $SERVE not found" >&2
    exit 1
fi

# Create venv and install dependencies if missing.
if [[ ! -d "$VENV/bin" ]]; then
    echo "[native-mantis] creating venv at $VENV ..."
    python3 -m venv "$VENV"
fi

# shellcheck source=/dev/null
source "$VENV/bin/activate"

# Install/upgrade core orchestrator dependencies.
python3 -m pip install --quiet --upgrade pip
python3 -m pip install --quiet -e "$REPO_ROOT"

# Build the TRINITY base vector from the committed small safetensors head if needed.
if [[ ! -f "$REPO_ROOT/artifacts/model_iter_60.npy" ]]; then
    echo "[native-mantis] building artifacts/model_iter_60.npy from router_head.safetensors ..."
    python3 "$REPO_ROOT/scripts/make_vec.py"
fi

# Load .env so native process gets the same config as the Docker stack.
# shellcheck source=/dev/null
set -a
source "$REPO_ROOT/.env"
set +a

CONDUCTOR_DEV="${MANTIS_CONDUCTOR_DEVICE:-${FUGU_CONDUCTOR_DEVICE:-}}"
# Auto-detect device unless explicitly set.
if [[ -z "$CONDUCTOR_DEV" || "$CONDUCTOR_DEV" == "auto" ]]; then
    CONDUCTOR_DEV=$(python3 - <<'PY'
import torch
if torch.backends.mps.is_available():
    print("mps")
elif torch.cuda.is_available():
    print("cuda:0")
else:
    print("cpu")
PY
    )
    export MANTIS_CONDUCTOR_DEVICE="$CONDUCTOR_DEV"
    export FUGU_CONDUCTOR_DEVICE="$CONDUCTOR_DEV"
fi

CONDUCTOR_DT="${MANTIS_CONDUCTOR_DTYPE:-${FUGU_CONDUCTOR_DTYPE:-}}"
# Default dtype: bfloat16 on mps/cuda, float32 on cpu unless user overrides.
if [[ -z "$CONDUCTOR_DT" ]]; then
    if [[ "$CONDUCTOR_DEV" == mps || "$CONDUCTOR_DEV" == cuda* ]]; then
        export MANTIS_CONDUCTOR_DTYPE=bfloat16
        export FUGU_CONDUCTOR_DTYPE=bfloat16
    else
        export MANTIS_CONDUCTOR_DTYPE=float32
        export FUGU_CONDUCTOR_DTYPE=float32
    fi
fi

VECTOR_FILE="${MANTIS_VECTOR:-${FUGU_VECTOR:-}}"
[[ -f "$VECTOR_FILE" ]] || export MANTIS_VECTOR="$REPO_ROOT/artifacts/model_iter_60.npy"
HEAD_FILE="${MANTIS_HEAD:-${FUGU_HEAD:-}}"
[[ -f "$HEAD_FILE" ]] || export MANTIS_HEAD="$REPO_ROOT/artifacts/router_head.npy"
BASE_URL="${MANTIS_BASE_URL:-${FUGU_BASE_URL:-}}"
[[ -n "$BASE_URL" ]] || export MANTIS_BASE_URL="http://127.0.0.1:3001/v1"

HOST_VAL="${MANTIS_HOST:-${FUGU_HOST:-0.0.0.0}}"
PORT_VAL="${MANTIS_PORT:-${FUGU_PORT:-8088}}"

export MANTIS_HOST="$HOST_VAL"
export MANTIS_PORT="$PORT_VAL"

echo "[native-mantis] device=$MANTIS_CONDUCTOR_DEVICE dtype=$MANTIS_CONDUCTOR_DTYPE"
echo "[native-mantis] listening on $MANTIS_HOST:$MANTIS_PORT"
echo "[native-mantis] press Ctrl-C to stop"

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
exec python3 "$SERVE" --host "$MANTIS_HOST" --port "$MANTIS_PORT"
