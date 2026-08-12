#!/usr/bin/env bash
# Canonical launch script for the RouteLLM coding-router.
# Lives in ~/Startup/ so StartupFolder (https://github.com/FuzzyIdeas/StartupFolder)
# runs it at login. It detaches the router, prints status, then exits.
#
# Loads exported env from ~/.zshrc, starts the local LiteLLM proxy if needed,
# then starts the router. The router is stopped-and-restarted if already running
# so config/env changes take effect; if it is down it is just started.
#
# Usage: ~/Startup/llm-router.sh          (router backgrounded, prints status)
set -euo pipefail

# Resources (server.py, .venv, logs/) live in the repo dir, not next to this
# script, so cd there explicitly — StartupFolder may invoke us from / and
# ~/Startup/llm-router.sh is a symlink to this file.
SELF="$0"
while [ -L "$SELF" ]; do SELF="$(readlink "$SELF")"; done
REPO_DIR="${LLM_ROUTER_DIR:-$(cd "$(dirname "$SELF")" && pwd)}"
if [ ! -d "$REPO_DIR" ]; then
  echo "ERROR: llm-router repo dir not found: $REPO_DIR" >&2
  exit 1
fi
cd "$REPO_DIR"
# shellcheck source=endpoint-profile.sh
. "$REPO_DIR/endpoint-profile.sh"

# Ensure system CLIs (lsof, security) are found under launchd's minimal PATH
# as well as an interactive shell.
export PATH="/opt/homebrew/bin:/opt/homebrew/sbin:$PATH"

# StartupFolder/launchd does not read shell startup files. Import exported env
# from zsh so ~/.zshrc remains the single place for RouteLLM/backend config.
if [ -f "$HOME/.zshrc" ]; then
  while IFS='=' read -r name value; do
    [[ "$name" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] && export "$name=$value"
  done < <(/bin/zsh -lc 'source "$HOME/.zshrc" >/dev/null && env')
fi

DATA_DIR="${MANTIS_DATA_DIR:-$HOME/.local/share/mantis}"
ROUTER_LOG_DIR="$DATA_DIR/router"
mkdir -p "$ROUTER_LOG_DIR"
chmod 700 "$DATA_DIR" "$ROUTER_LOG_DIR"
export MANTIS_DATA_DIR="$DATA_DIR"

# An explicit target source (ROUTELLM_TARGETS_JSON or the AI_ROUTING_CONFIG
# catalog) carries its own provider/credential bindings. Legacy gateway/direct
# key requirements below must not block such a deployment.
if [ -n "${ROUTELLM_TARGETS_JSON:-}" ] || [ -n "${AI_ROUTING_CONFIG:-}" ] || \
   [ -f "$HOME/.config/ai-routing/catalog.toml" ]; then
  ROUTER_EXPLICIT_SOURCE=1
else
  ROUTER_EXPLICIT_SOURCE=0
fi

# --- router ---
. ./.venv/bin/activate
# Direct endpoints only (Cloudflare gateway retired); the shared catalog or
# launcher-supplied legacy env provides provider bases and keys.
export EXPENSIVE_BASE="${EXPENSIVE_BASE:-https://openrouter.ai/api/v1}"
export EXPENSIVE_KEY="${EXPENSIVE_KEY:-${OPENROUTER_API_KEY:-}}"
export CHEAP_BASE="${CHEAP_BASE:-https://opencode.ai/zen/go/v1}"
export CHEAP_KEY="${CHEAP_KEY:-${OPENCODE_API_KEY:-}}"
export MIDDLE_BASE="${MIDDLE_BASE:-}"
export MIDDLE_KEY="${MIDDLE_KEY:-}"
# Preserve direct-provider credentials unless an explicit source is active.
if [ "$ROUTER_EXPLICIT_SOURCE" -eq 0 ]; then
  router_select_endpoint_keys
  if [ -z "$EXPENSIVE_KEY" ]; then
    echo 'ERROR: EXPENSIVE_KEY or OPENROUTER_API_KEY required for the expensive backend.' >&2
    exit 1
  fi
  if [ -z "$CHEAP_KEY" ]; then
    echo 'ERROR: CHEAP_KEY or OPENCODE_API_KEY required for the cheap backend.' >&2
    exit 1
  fi
  if [ -n "${MIDDLE_BASE:-}" ] && [ -z "${MIDDLE_KEY:-}" ]; then
    echo 'ERROR: MIDDLE_KEY is required when MIDDLE_BASE is configured.' >&2
    exit 1
  fi
fi
export ROUTELLM_CONTEXT_WINDOW="${ROUTELLM_CONTEXT_WINDOW:-auto}"
export ROUTELLM_MAX_TOKENS="${ROUTELLM_MAX_TOKENS:-131072}"
export ROUTELLM_RESP_CACHE_TTL_S="${ROUTELLM_RESP_CACHE_TTL_S:-120}"
export ROUTELLM_KEY="${ROUTELLM_KEY:-sk-route-local}"
export ROUTELLM_HOST="${ROUTELLM_HOST:-127.0.0.1}"
export ROUTELLM_PORT="${ROUTELLM_PORT:-5500}"
if [ "$ROUTELLM_HOST" != "127.0.0.1" ] && [ "$ROUTELLM_HOST" != "::1" ] && [ "$ROUTELLM_HOST" != "localhost" ]; then
  if [ -z "${ROUTELLM_KEY:-}" ] || [ "$ROUTELLM_KEY" = "sk-route-local" ]; then
    echo 'ERROR: non-loopback binding requires an externally supplied, non-default ROUTELLM_KEY.' >&2
    exit 1
  fi
fi

# If the router is already listening, stop it so we start a clean instance.
# The router is owned here and restarted to pick up config/env changes.
if lsof -nP -iTCP:"$ROUTELLM_PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "router already running on :$ROUTELLM_PORT — stopping for restart"
  # Refuse to kill a listener unless the recorded PID owns this port and its
  # command is this checkout's server. A stale PID file must fail closed.
  ROUTER_PID=""
  CANDIDATE=$(cat "$ROUTER_LOG_DIR/server.pid" 2>/dev/null || true)
  if [ -n "${CANDIDATE:-}" ]      && lsof -nP -iTCP:"$ROUTELLM_PORT" -sTCP:LISTEN -t 2>/dev/null | grep -qx "$CANDIDATE"      && [ "$(lsof -a -p "$CANDIDATE" -d cwd -Fn 2>/dev/null | sed -n 's/^n//p')" = "$REPO_DIR" ] \
     && ps -p "$CANDIDATE" -o command= 2>/dev/null | grep -Fq -- "server.py"; then
    ROUTER_PID="$CANDIDATE"
  else
    echo "ERROR: port $ROUTELLM_PORT is owned by an unverified process; refusing to kill it" >&2
    exit 1
  fi
  if [ -n "${ROUTER_PID:-}" ]; then
    kill "$ROUTER_PID" 2>/dev/null || true
    for _ in $(seq 1 50); do
      kill -0 "$ROUTER_PID" 2>/dev/null || break
      sleep 0.1
    done
    # Force-kill if it's still alive after the graceful window.
    if kill -0 "$ROUTER_PID" 2>/dev/null; then
      kill -9 "$ROUTER_PID" 2>/dev/null || true
    fi
  fi
  # Wait for the socket to actually free up so the restart doesn't hit EADDRINUSE.
  for _ in $(seq 1 50); do
    lsof -nP -iTCP:"$ROUTELLM_PORT" -sTCP:LISTEN >/dev/null 2>&1 || break
    sleep 0.1
  done
  rm -f "$ROUTER_LOG_DIR/server.pid"
  echo "router stopped"
fi

nohup python server.py </dev/null > "$ROUTER_LOG_DIR/server.out" 2> "$ROUTER_LOG_DIR/server.err" &
echo $! > "$ROUTER_LOG_DIR/server.pid"
for _ in $(seq 1 600); do  # 60s — first boot downloads the Supra checkpoint
  if lsof -nP -iTCP:"$ROUTELLM_PORT" -sTCP:LISTEN >/dev/null 2>&1; then
    if curl -fsS --max-time 2 "http://127.0.0.1:$ROUTELLM_PORT/healthz" 2>/dev/null | grep -q '"ready":true'; then
      ROUTER_PID=$(lsof -nP -iTCP:"$ROUTELLM_PORT" -sTCP:LISTEN -t 2>/dev/null | head -n1 || true)
      echo "router running pid ${ROUTER_PID:-$(cat "$ROUTER_LOG_DIR/server.pid")} on :$ROUTELLM_PORT"
      exit 0
    fi
  fi
  sleep 0.1
done

echo "ERROR: router not running on :$ROUTELLM_PORT — see $ROUTER_LOG_DIR/server.err" >&2
cat "$ROUTER_LOG_DIR/server.err" >&2 2>/dev/null || true
exit 1
