#!/usr/bin/env bash
# Run Mantis on the host with uv. No container runtime is required.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVE_DIR="$REPO_ROOT/apps/api"
cd "$REPO_ROOT"

[[ -f "$SERVE_DIR/api.py" ]] || { echo "missing $SERVE_DIR/api.py" >&2; exit 1; }
command -v uv >/dev/null || { echo "uv is required: https://docs.astral.sh/uv/" >&2; exit 1; }

# uv.lock defines the reproducible runtime. It also creates the project venv.
uv sync --locked --no-dev

if [[ ! -f "$REPO_ROOT/artifacts/model_iter_60.npy" ]]; then
    echo "[mantis] building router vector"
    uv run --no-sync python scripts/make_vec.py
fi

if [[ -f "$REPO_ROOT/.env" ]]; then
    set -a
    # .env is intentionally shell-compatible and is not committed.
    source "$REPO_ROOT/.env"
    set +a
fi

# The shared routing catalog, when present, is the authoritative configuration.
export PYTHONPATH="$REPO_ROOT/scripts${PYTHONPATH:+:$PYTHONPATH}"
if catalog_render=$(uv run --no-sync python scripts/model_catalog.py render); then
    if [[ -n "$catalog_render" ]]; then
        eval "$catalog_render"
        export MANTIS_PROVIDER_KEYS="$(uv run --no-sync python - <<'PY'
import json
import model_catalog
catalog = model_catalog.load_mantis_catalog()
if catalog is not None:
    print(json.dumps(model_catalog.resolve_provider_keys(catalog), separators=(",", ":")))
PY
)"
        export MANTIS_ENDPOINT_PROFILE=catalog
    fi
else
    echo "Mantis catalog validation failed" >&2
    exit 1
fi

if [[ -z "${MANTIS_CONDUCTOR_DEVICE:-}" || "${MANTIS_CONDUCTOR_DEVICE}" == auto ]]; then
    export MANTIS_CONDUCTOR_DEVICE="$(uv run --no-sync python - <<'PY'
import torch
print("mps" if torch.backends.mps.is_available() else "cuda:0" if torch.cuda.is_available() else "cpu")
PY
)"
fi
if [[ -z "${MANTIS_CONDUCTOR_DTYPE:-}" ]]; then
    if [[ "$MANTIS_CONDUCTOR_DEVICE" == mps || "$MANTIS_CONDUCTOR_DEVICE" == cuda* ]]; then
        export MANTIS_CONDUCTOR_DTYPE=bfloat16
    else
        export MANTIS_CONDUCTOR_DTYPE=float32
    fi
fi

[[ -f "${MANTIS_VECTOR:-}" ]] || export MANTIS_VECTOR="$REPO_ROOT/artifacts/model_iter_60.npy"
[[ -f "${MANTIS_HEAD:-}" ]] || export MANTIS_HEAD="$REPO_ROOT/artifacts/router_head.safetensors"
export MANTIS_HOST="${MANTIS_HOST:-127.0.0.1}"
export MANTIS_PORT="${MANTIS_PORT:-8088}"
# Persist orchestration runs on disk so an API restart does not lose in-flight runs.
export MANTIS_RUN_STORE="${MANTIS_RUN_STORE:-file}"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

echo "[mantis] host=$MANTIS_HOST port=$MANTIS_PORT device=$MANTIS_CONDUCTOR_DEVICE"
exec uv run --no-sync python -m uvicorn api:app --app-dir "$SERVE_DIR" \
    --host "$MANTIS_HOST" --port "$MANTIS_PORT"
