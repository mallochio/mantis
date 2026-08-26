#!/usr/bin/env bash
# Canonical launch script for the Bifrost AI gateway (maximhq/bifrost).
# Lives in ~/Startup/ so StartupFolder (https://github.com/FuzzyIdeas/StartupFolder)
# runs it at login. It detaches Bifrost, prints status, then exits.
#
# Loads exported env from ~/.zshrc (single source for BIFROST_API_KEY,
# BIFROST_ENCRYPTION_KEY, OPENCODE_API_KEY, AWS_*, GOOGLE_APPLICATION_CREDENTIALS,
# AZURE_OPENAI_*, ...), then starts the gateway with the npx wrapper. The wrapper
# execs the Go binary in place, so the recorded PID is the gateway process and
# inherits the zsh env natively (AWS/GCP credential chains resolve as in a shell).
#
# The gateway is stopped-and-restarted if already running so config/env changes
# take effect; if it is down it is just started.
#
# Usage: ~/Startup/bifrost-local.sh          (gateway backgrounded, prints status)
set -euo pipefail

# StartupFolder may invoke us from / and ~/Startup/bifrost-local.sh may be a
# symlink; resolve to the real script location for stable relative paths.
SELF="$0"
while [ -L "$SELF" ]; do
  LINK_DIR="$(cd -P "$(dirname "$SELF")" && pwd)"
  SELF="$(readlink "$SELF")"
  [[ "$SELF" = /* ]] || SELF="$LINK_DIR/$SELF"
done
SCRIPT_DIR="$(cd "$(dirname "$SELF")" && pwd)"

# Ensure system CLIs (lsof, curl, npx) are found under launchd's minimal PATH
# as well as an interactive shell.
export PATH="/opt/homebrew/bin:/opt/homebrew/sbin:$HOME/.local/bin:$PATH"

# StartupFolder/launchd does not read shell startup files. Import exported env
# from zsh so ~/.zshrc remains the single place for Bifrost/backend config.
if [ -f "$HOME/.zshrc" ]; then
  while IFS='=' read -r name value; do
    [[ "$name" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] && export "$name=$value"
  done < <(/bin/zsh -lc 'source "$HOME/.zshrc" >/dev/null && env')
fi

# config.json references env.OPENCODE_API_KEY; fall back to the Go key name.
export OPENCODE_API_KEY="${OPENCODE_API_KEY:-${OPENCODE_GO_API_KEY:-}}"

DATA_DIR="${BIFROST_DATA_DIR:-$HOME/.local/share/bifrost}"
LOG_DIR="$DATA_DIR/logs"
CONFIG_SRC="${BIFROST_CONFIG:-$HOME/.config/ai-routing/bifrost.json}"
mkdir -p "$LOG_DIR"
chmod 700 "$DATA_DIR" "$LOG_DIR"

# The app-dir config.json is a derived copy. The canonical file to edit is
# ~/.config/ai-routing/bifrost.json (symlinked into the mantis repo). Sync it
# on every launch so restarts always apply edits and the two can never drift.
if [ -f "$CONFIG_SRC" ] && ! cmp -s "$CONFIG_SRC" "$DATA_DIR/config.json" 2>/dev/null; then
  cp "$CONFIG_SRC" "$DATA_DIR/config.json"
  chmod 600 "$DATA_DIR/config.json"
  echo "bifrost config synced from $CONFIG_SRC"
fi

HOST="${BIFROST_HOST:-127.0.0.1}"
PORT="${BIFROST_PORT:-8080}"

# If the gateway is already listening, stop it so we start a clean instance.
# Kill every listener on this port and any other obvious bifrost process,
# regardless of recorded PID. This makes restarts robust after manual launches.
if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "bifrost already running on :$PORT — stopping for restart"
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
  pgrep -f 'bifrost-http|@maximhq/bifrost' 2>/dev/null | while IFS= read -r pid; do
    kill -9 "$pid" 2>/dev/null || true
  done
  for _ in $(seq 1 50); do
    lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1 || break
    sleep 0.1
  done
  rm -f "$DATA_DIR/server.pid"
  echo "bifrost stopped"
fi

# Prefer the npx wrapper (keeps the cached binary current); if the registry is
# unreachable, fall back to the newest already-cached binary.
# If BIFROST_VERSION is set, use the exact cached binary (or npx tag) to support
# pre-release/bespoke binaries (e.g. 2.0.0) not yet on the public npm registry.
BIFROST_VERSION="${BIFROST_VERSION:-}"
if [ -n "$BIFROST_VERSION" ]; then
  # Cached binaries live in a `v` prefixed directory (v1.6.11, v2.0.0).
  CACHED_DIR="$HOME/Library/Caches/bifrost/$BIFROST_VERSION"
  if [ ! -d "$CACHED_DIR" ] && [ -d "$HOME/Library/Caches/bifrost/v$BIFROST_VERSION" ]; then
    CACHED_DIR="$HOME/Library/Caches/bifrost/v$BIFROST_VERSION"
  fi
  CACHED_BIN="$CACHED_DIR/bin/bifrost-http-0"
  if [ -x "$CACHED_BIN" ]; then
    LAUNCH=("$CACHED_BIN")
  elif command -v npx >/dev/null 2>&1; then
    NPM_VERSION="${BIFROST_VERSION#v}"
    LAUNCH=(npx -y "@maximhq/bifrost@$NPM_VERSION")
  else
    echo "ERROR: BIFROST_VERSION=$BIFROST_VERSION but no cached binary or npx is available" >&2
    exit 1
  fi
elif command -v npx >/dev/null 2>&1; then
  LAUNCH=(npx -y @maximhq/bifrost)
else
  CACHED_BIN=$(ls -1t "$HOME/Library/Caches/bifrost"/*/bin/bifrost-http-* 2>/dev/null | head -n1 || true)
  if [ -n "${CACHED_BIN:-}" ] && [ -x "$CACHED_BIN" ]; then
    LAUNCH=("$CACHED_BIN")
  else
    echo "ERROR: neither npx nor a cached bifrost binary is available" >&2
    exit 1
  fi
fi

# shellcheck source=detach.sh
source "$SCRIPT_DIR/detach.sh"
detach_cmd "$DATA_DIR/server.pid" "$LOG_DIR/server.out" "$LOG_DIR/server.err" \
  "${LAUNCH[@]}" -app-dir "$DATA_DIR" -host "$HOST" -port "$PORT" -log-style pretty \
  >/dev/null

for _ in $(seq 1 300); do  # 60s — first boot downloads the binary
  if curl -fsS --max-time 2 "http://$HOST:$PORT/health" >/dev/null 2>&1; then
    # Record the actual TCP listener PID (the npx wrapper is a parent process
    # that execs/spawns the Go binary; the listener PID is the gateway itself).
    BIFROST_PID=$(lsof -nP -iTCP:"$PORT" -sTCP:LISTEN -t 2>/dev/null | head -n1 || true)
    [ -n "$BIFROST_PID" ] && echo "$BIFROST_PID" > "$DATA_DIR/server.pid"
    echo "bifrost running pid $BIFROST_PID on $HOST:$PORT (data: $DATA_DIR)"
    exit 0
  fi
  sleep 0.2
done

echo "ERROR: bifrost not healthy on $HOST:$PORT — see $LOG_DIR/server.err" >&2
tail -n 40 "$LOG_DIR/server.err" >&2 2>/dev/null || true
exit 1
