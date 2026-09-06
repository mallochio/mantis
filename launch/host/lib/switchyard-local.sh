#!/usr/bin/env bash
# Start NVIDIA NeMo Switchyard for mantis/base on loopback.
#
# Regenerates routes.toml from the shared catalog, then starts
# switchyard-server. Changing [base] models or providers in the catalog is
# enough to retarget Base; this script does not hard-code them.
set -euo pipefail

SELF="$0"
while [ -L "$SELF" ]; do
  LINK_DIR="$(cd -P "$(dirname "$SELF")" && pwd)"
  SELF="$(readlink "$SELF")"
  [[ "$SELF" = /* ]] || SELF="$LINK_DIR/$SELF"
done
LIB_DIR="$(cd "$(dirname "$SELF")" && pwd)"
REPO_ROOT="$(cd "$LIB_DIR/../../.." && pwd)"

export PATH="/opt/homebrew/bin:/opt/homebrew/sbin:$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
export RUST_LOG="switchyard=debug,bedrock=debug"

if [ -f "$HOME/.zshrc" ]; then
  while IFS='=' read -r name value; do
    [[ "$name" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] && export "$name=$value"
  done < <(/bin/zsh -lc 'source "$HOME/.zshrc" >/dev/null && env')
fi
# Mantis Bedrock: use static IAM keys in eu-central-1, not SSO (expired)
unset AWS_PROFILE AWS_SESSION_TOKEN 2>/dev/null || true
export AWS_REGION="eu-central-1"
if [ -z "${AWS_ACCESS_KEY_ID:-}" ]; then
  export AWS_ACCESS_KEY_ID="$(security find-generic-password -a "$USER" -s "shell-env/AWS_ACCESS_KEY_ID" -w 2>/dev/null || true)"
fi
if [ -z "${AWS_SECRET_ACCESS_KEY:-}" ]; then
  export AWS_SECRET_ACCESS_KEY="$(security find-generic-password -a "$USER" -s "shell-env/AWS_SECRET_ACCESS_KEY" -w 2>/dev/null || true)"
fi
# Bedrock OpenAI-compatible endpoint (Grok capable) needs a bearer API key,
# not IAM keys. Shared helper mints a short-term token (12h) and persists it
# so Switchyard and Mantis (separate envs) use the same key.
# shellcheck source=bedrock-key.sh
source "$LIB_DIR/bedrock-key.sh"
ensure_bedrock_api_key
export PATH="/opt/homebrew/bin:/opt/homebrew/sbin:$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
export RUST_LOG="switchyard=debug,bedrock=debug"

DATA_DIR="${MANTIS_DATA_DIR:-$HOME/.local/share/mantis}"
LOG_DIR="$DATA_DIR/switchyard"
CONFIG_OUT="${SWITCHYARD_CONFIG:-$LOG_DIR/routes.toml}"
CATALOG="${AI_ROUTING_CONFIG:-$HOME/.config/ai-routing/catalog.toml}"
HOST="${SWITCHYARD_HOST:-127.0.0.1}"
PORT="${SWITCHYARD_PORT:-5500}"
mkdir -p "$LOG_DIR"
chmod 700 "$DATA_DIR" "$LOG_DIR"

if [ ! -f "$CATALOG" ]; then
  echo "ERROR: routing catalog not found: $CATALOG" >&2
  exit 1
fi

if [ -n "${SWITCHYARD_BIN:-}" ]; then
  :
elif command -v switchyard-server >/dev/null 2>&1; then
  SWITCHYARD_BIN="$(command -v switchyard-server)"
elif [ -x "$HOME/.cargo/bin/switchyard-server" ]; then
  SWITCHYARD_BIN="$HOME/.cargo/bin/switchyard-server"
else
  echo "ERROR: switchyard-server not found." >&2
  echo "Install with: cargo install --locked switchyard-server" >&2
  exit 1
fi

if [ "$HOST" != "127.0.0.1" ] && [ "$HOST" != "::1" ] && [ "$HOST" != "localhost" ]; then
  echo "ERROR: Switchyard must bind loopback (got host $HOST)." >&2
  exit 1
fi

export PYTHONPATH="$REPO_ROOT/scripts${PYTHONPATH:+:$PYTHONPATH}"
# uv must run inside the repo so it uses the project environment; callers often
# invoke this script via ~/Startup with cwd outside the project.
if ! (
  cd "$REPO_ROOT" &&
    uv run --no-sync python scripts/switchyard_config.py render \
      --catalog "$CATALOG" --output "$CONFIG_OUT"
); then
  echo "ERROR: failed to render Switchyard config from $CATALOG" >&2
  exit 1
fi
if ! "$SWITCHYARD_BIN" --config "$CONFIG_OUT" --dry-run; then
  echo "ERROR: switchyard-server rejected $CONFIG_OUT" >&2
  exit 1
fi

echo "restarting Switchyard on :$PORT"
if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  for pid in $(lsof -nP -iTCP:"$PORT" -sTCP:LISTEN -t 2>/dev/null); do
    kill "$pid" 2>/dev/null || true
  done
  for _ in $(seq 1 50); do
    lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1 || break
    sleep 0.1
  done
  for pid in $(lsof -nP -iTCP:"$PORT" -sTCP:LISTEN -t 2>/dev/null); do
    kill -9 "$pid" 2>/dev/null || true
  done
fi
rm -f "$LOG_DIR/server.pid"

# shellcheck source=detach.sh
source "$LIB_DIR/detach.sh"
detach_cmd "$LOG_DIR/server.pid" "$LOG_DIR/server.out" "$LOG_DIR/server.err" \
  "$SWITCHYARD_BIN" --config "$CONFIG_OUT" --host "$HOST" --port "$PORT" \
  >/dev/null

for _ in $(seq 1 150); do
  if curl -fsS --max-time 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    ROUTER_PID=$(lsof -nP -iTCP:"$PORT" -sTCP:LISTEN -t 2>/dev/null | head -n1 || true)
    [ -n "$ROUTER_PID" ] && echo "$ROUTER_PID" > "$LOG_DIR/server.pid"
    echo "Switchyard running pid ${ROUTER_PID:-} on :$PORT"
    exit 0
  fi
  sleep 0.1
done

echo "ERROR: Switchyard not running on :$PORT — see $LOG_DIR/server.err" >&2
tail -n 40 "$LOG_DIR/server.err" >&2 2>/dev/null || true
exit 1
