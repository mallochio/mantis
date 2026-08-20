#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/app"
cd "$REPO_ROOT"

echo "=== [Mantis Render Entrypoint] Starting initialization ==="

# 1. Environment & paths
export AI_ROUTING_CONFIG="${AI_ROUTING_CONFIG:-$REPO_ROOT/config/catalog.toml}"
export BIFROST_DATA_DIR="${BIFROST_DATA_DIR:-/tmp/bifrost}"
export MANTIS_DATA_DIR="${MANTIS_DATA_DIR:-/tmp/mantis}"
export MANTIS_ROUTER_URL="http://127.0.0.1:5500/v1"
export MANTIS_ROUTER_HOST="127.0.0.1"
export MANTIS_ROUTER_PORT="5500"
export MANTIS_HOST="0.0.0.0"
export MANTIS_PORT="${PORT:-8088}"
export MANTIS_API_KEY="${MANTIS_API_KEY:-sk-mantis-local}"
export MANTIS_ROUTER_KEY="${MANTIS_ROUTER_KEY:-sk-route-local}"
export BIFROST_API_KEY="${BIFROST_API_KEY:-sk-bifrost-local}"
export BIFROST_ENCRYPTION_KEY="${BIFROST_ENCRYPTION_KEY:-760529c75e2b39a8bb728cdcd5b2dcef91b0f1a9a4e320f7ca7da41bc88bb254}"
export OPENCODE_API_KEY="${OPENCODE_API_KEY:-${OPENCODE_GO_API_KEY:-}}"

# Secret files from Render (e.g. /etc/secrets/gcp-service-account.json)
if [[ -f "/etc/secrets/gcp-service-account.json" ]]; then
  export GOOGLE_APPLICATION_CREDENTIALS="/etc/secrets/gcp-service-account.json"
fi

mkdir -p "$BIFROST_DATA_DIR" "$MANTIS_DATA_DIR/router"

# 2. Setup Bifrost config
if [[ -f "$REPO_ROOT/config/bifrost.json" ]]; then
  cp "$REPO_ROOT/config/bifrost.json" "$BIFROST_DATA_DIR/config.json"
elif [[ -f "$REPO_ROOT/config/bifrost.template.json" ]]; then
  cp "$REPO_ROOT/config/bifrost.template.json" "$BIFROST_DATA_DIR/config.json"
fi

# Process cleanup trap
bifrost_pid=""
gateway_pid=""
mantis_pid=""

cleanup() {
  echo "[Mantis Render Entrypoint] Shutting down services..."
  [[ -n "$mantis_pid" ]] && kill -TERM "$mantis_pid" 2>/dev/null || true
  [[ -n "$gateway_pid" ]] && kill -TERM "$gateway_pid" 2>/dev/null || true
  [[ -n "$bifrost_pid" ]] && kill -TERM "$bifrost_pid" 2>/dev/null || true
  wait 2>/dev/null || true
  echo "[Mantis Render Entrypoint] Shutdown complete."
}
trap cleanup SIGTERM SIGINT SIGHUP

# 3. Start Bifrost on :8080
echo "[Mantis Render Entrypoint] 1/3 Starting Bifrost on 127.0.0.1:8080..."
bifrost -app-dir "$BIFROST_DATA_DIR" -host 127.0.0.1 -port 8080 -log-style pretty &
bifrost_pid=$!

# Wait for Bifrost ready
for i in $(seq 1 60); do
  if curl -fsS --max-time 2 "http://127.0.0.1:8080/health" >/dev/null 2>&1; then
    echo "[Mantis Render Entrypoint] Bifrost is ready."
    break
  fi
  if ! kill -0 "$bifrost_pid" 2>/dev/null; then
    echo "[Mantis Render Entrypoint] ERROR: Bifrost process died." >&2
    exit 1
  fi
  sleep 0.5
done

# 4. Start Direct Gateway / Router on :5500
echo "[Mantis Render Entrypoint] 2/3 Starting Mantis Router on 127.0.0.1:5500..."
(
  cd "$REPO_ROOT/apps/gateway"
  exec uv run --no-sync python server.py
) &
gateway_pid=$!

# Wait for Gateway ready
for i in $(seq 1 60); do
  if curl -fsS --max-time 2 "http://127.0.0.1:5500/healthz" 2>/dev/null | grep -q '"ready":true'; then
    echo "[Mantis Render Entrypoint] Mantis Router is ready."
    break
  fi
  if ! kill -0 "$gateway_pid" 2>/dev/null; then
    echo "[Mantis Render Entrypoint] ERROR: Router process died." >&2
    exit 1
  fi
  sleep 0.5
done

# 5. Start Mantis API on 0.0.0.0:$MANTIS_PORT
echo "[Mantis Render Entrypoint] 3/3 Starting Mantis API on 0.0.0.0:$MANTIS_PORT..."
export PYTHONPATH="$REPO_ROOT/scripts:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
if catalog_render=$(uv run --no-sync python scripts/model_catalog.py render 2>/dev/null); then
  if [[ -n "$catalog_render" ]]; then
    eval "$catalog_render"
    export MANTIS_PROVIDER_KEYS="$(uv run --no-sync python - <<'PY'
import json, model_catalog
catalog = model_catalog.load_mantis_catalog()
if catalog is not None:
    print(json.dumps(model_catalog.resolve_provider_keys(catalog), separators=(",", ":")))
PY
)"
    export MANTIS_ENDPOINT_PROFILE=catalog
  fi
fi

exec uv run --no-sync python -m uvicorn api:app --app-dir "$REPO_ROOT/apps/api" \
  --host "$MANTIS_HOST" --port "$MANTIS_PORT"
