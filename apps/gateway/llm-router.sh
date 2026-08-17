#!/usr/bin/env bash
# Canonical launch script for the Mantis gateway.
# Lives in ~/Startup/ so StartupFolder (https://github.com/FuzzyIdeas/StartupFolder)
# runs it at login. It detaches the router, prints status, then exits.
#
# Loads exported env from ~/.zshrc, then starts the router. The router is stopped-and-restarted if already running
# so config/env changes take effect; if it is down it is just started.
#
# Usage: ~/Startup/llm-router.sh          (router backgrounded, prints status)
set -euo pipefail

# Resources (server.py, .venv, logs/) live in the repo dir, not next to this
# script, so cd there explicitly — StartupFolder may invoke us from / and
# ~/Startup/llm-router.sh is a symlink to this file.
SELF="$0"
while [ -L "$SELF" ]; do
  LINK_DIR="$(cd -P "$(dirname "$SELF")" && pwd)"
  SELF="$(readlink "$SELF")"
  [[ "$SELF" = /* ]] || SELF="$LINK_DIR/$SELF"
done
REPO_DIR="${LLM_ROUTER_DIR:-$(cd "$(dirname "$SELF")" && pwd)}"
if [ ! -d "$REPO_DIR" ]; then
  echo "ERROR: Mantis gateway directory not found: $REPO_DIR" >&2
  exit 1
fi
cd "$REPO_DIR"

# Ensure system CLIs (lsof, security) are found under launchd's minimal PATH
# as well as an interactive shell.
export PATH="/opt/homebrew/bin:/opt/homebrew/sbin:$PATH"

# StartupFolder/launchd does not read shell startup files. Import exported env
# from zsh so ~/.zshrc remains the single place for router/backend config.
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

# The router requires an explicit target source: MANTIS_ROUTER_TARGETS_JSON
# or a routing catalog (AI_ROUTING_CONFIG). Provider bindings and credentials
# come from that source.
if [ -z "${MANTIS_ROUTER_TARGETS_JSON:-}" ] && [ -z "${AI_ROUTING_CONFIG:-}" ] && \
   [ ! -f "$HOME/.config/ai-routing/catalog.toml" ]; then
  echo "ERROR: no router target source: set MANTIS_ROUTER_TARGETS_JSON or provide a routing catalog" >&2
  exit 1
fi

# --- router ---
. ./.venv/bin/activate
export MANTIS_CONTEXT_LENGTH="${MANTIS_CONTEXT_LENGTH:-262144}"
export MANTIS_ROUTER_MAX_TOKENS="${MANTIS_ROUTER_MAX_TOKENS:-384000}"
export MANTIS_ROUTER_RESP_CACHE_TTL_S="${MANTIS_ROUTER_RESP_CACHE_TTL_S:-120}"
export MANTIS_ROUTER_KEY="${MANTIS_ROUTER_KEY:-sk-route-local}"
export MANTIS_ROUTER_HOST="${MANTIS_ROUTER_HOST:-127.0.0.1}"
export MANTIS_ROUTER_PORT="${MANTIS_ROUTER_PORT:-5500}"
# Re-evaluate the session tier every N completed turns so a session does not
# stay on an expensive tier forever. 0 disables rescoring; 4 is a reasonable
# starting value for coding-agent sessions.
export MANTIS_ROUTER_RESCORE_EVERY_N="${MANTIS_ROUTER_RESCORE_EVERY_N:-4}"
if [ "$MANTIS_ROUTER_HOST" != "127.0.0.1" ] && [ "$MANTIS_ROUTER_HOST" != "::1" ] && [ "$MANTIS_ROUTER_HOST" != "localhost" ]; then
  if [ -z "${MANTIS_ROUTER_KEY:-}" ] || [ "$MANTIS_ROUTER_KEY" = "sk-route-local" ]; then
    echo 'ERROR: non-loopback binding requires an externally supplied, non-default MANTIS_ROUTER_KEY.' >&2
    exit 1
  fi
fi

# If the router is already listening, stop it so we start a clean instance.
# The router is owned here and restarted to pick up config/env changes.
if lsof -nP -iTCP:"$MANTIS_ROUTER_PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "gateway already running on :$MANTIS_ROUTER_PORT — stopping for restart"
  # Refuse to kill a listener unless the recorded PID owns this port and its
  # command is this checkout's server. A stale PID file must fail closed.
  ROUTER_PID=""
  CANDIDATE=$(cat "$ROUTER_LOG_DIR/server.pid" 2>/dev/null || true)
  if [ -n "${CANDIDATE:-}" ]      && lsof -nP -iTCP:"$MANTIS_ROUTER_PORT" -sTCP:LISTEN -t 2>/dev/null | grep -qx "$CANDIDATE"      && [ "$(lsof -a -p "$CANDIDATE" -d cwd -Fn 2>/dev/null | sed -n 's/^n//p')" = "$REPO_DIR" ] \
     && ps -p "$CANDIDATE" -o command= 2>/dev/null | grep -Fq -- "server.py"; then
    ROUTER_PID="$CANDIDATE"
  else
    echo "ERROR: port $MANTIS_ROUTER_PORT is owned by an unverified process; refusing to kill it" >&2
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
    lsof -nP -iTCP:"$MANTIS_ROUTER_PORT" -sTCP:LISTEN >/dev/null 2>&1 || break
    sleep 0.1
  done
  rm -f "$ROUTER_LOG_DIR/server.pid"
  echo "gateway stopped"
fi

nohup "$REPO_DIR/.venv/bin/python" server.py </dev/null > "$ROUTER_LOG_DIR/server.out" 2> "$ROUTER_LOG_DIR/server.err" &
echo $! > "$ROUTER_LOG_DIR/server.pid"
for _ in $(seq 1 600); do  # 60s — first boot downloads the Supra checkpoint
  if lsof -nP -iTCP:"$MANTIS_ROUTER_PORT" -sTCP:LISTEN >/dev/null 2>&1; then
    if curl -fsS --max-time 2 "http://127.0.0.1:$MANTIS_ROUTER_PORT/healthz" 2>/dev/null | grep -q '"ready":true'; then
      ROUTER_PID=$(lsof -nP -iTCP:"$MANTIS_ROUTER_PORT" -sTCP:LISTEN -t 2>/dev/null | head -n1 || true)
      echo "gateway running pid ${ROUTER_PID:-$(cat "$ROUTER_LOG_DIR/server.pid")} on :$MANTIS_ROUTER_PORT"
      exit 0
    fi
  fi
  sleep 0.1
done

echo "ERROR: gateway not running on :$MANTIS_ROUTER_PORT — see $ROUTER_LOG_DIR/server.err" >&2
cat "$ROUTER_LOG_DIR/server.err" >&2 2>/dev/null || true
exit 1
