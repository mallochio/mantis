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

# --- router ---
. ./.venv/bin/activate
# MF scoring uses OpenAI text-embedding-3-small; needed only in mf/bert mode
# or when SCORE_WITH_MF is enabled in supra mode. Retrieve the key when possible.
if [ -z "${OPENAI_API_KEY:-}" ]; then
  export OPENAI_API_KEY=$(security find-generic-password -s 'AI.Playground.openai.apiKey' -w 2>/dev/null || true)
fi
if [ -z "${OPENAI_API_KEY:-}" ] && [ "${ROUTELLM_ROUTER:-supra}" != "supra" ]; then
  echo 'ERROR: OPENAI_API_KEY not set and not found in Keychain; MF scoring needs it for embeddings.' >&2
  exit 1
fi
# Select the endpoint profile before assigning defaults. This avoids turning
# gateway mode into a mixed direct/gateway configuration.
ROUTER_GATEWAY=0
case "${ROUTELLM_ENDPOINT_PROFILE:-}" in
  cloudflare) ROUTER_GATEWAY=1 ;;
  direct) ROUTER_GATEWAY=0 ;;
  *)
    case "${ROUTELLM_GATEWAY_MODE:-}" in
      1|true|TRUE|yes|YES|on|ON|cloudflare) ROUTER_GATEWAY=1 ;;
      ""|0|false|FALSE|no|NO|off|OFF) ;;
      *) echo 'ERROR: invalid ROUTELLM_GATEWAY_MODE' >&2; exit 2 ;;
    esac
    [[ "${EXPENSIVE_BASE:-}${CHEAP_BASE:-}${MIDDLE_BASE:-}" == *unified-ai-gateway.siddsantham.workers.dev* ]] && ROUTER_GATEWAY=1
    if [ -z "${EXPENSIVE_BASE:-}" ] && [ -z "${CHEAP_BASE:-}" ] && [ -z "${MIDDLE_BASE:-}" ]; then
      ROUTER_GATEWAY=1
    fi
    ;;
esac
if [ "$ROUTER_GATEWAY" -eq 1 ]; then
  export EXPENSIVE_BASE="https://unified-ai-gateway.siddsantham.workers.dev/v1"
  export CHEAP_BASE="https://unified-ai-gateway.siddsantham.workers.dev/v1"
  export MIDDLE_BASE="https://unified-ai-gateway.siddsantham.workers.dev/v1"
  export AI_GATEWAY_API_KEY="${AI_GATEWAY_API_KEY:-${MANTIS_GATEWAY_API_KEY:-}}"
  export EXPENSIVE_KEY="${AI_GATEWAY_API_KEY}"
  export CHEAP_KEY="${AI_GATEWAY_API_KEY}"
  export MIDDLE_KEY="${AI_GATEWAY_API_KEY}"
  if [ -z "$AI_GATEWAY_API_KEY" ]; then
    echo 'ERROR: Cloudflare gateway endpoint requires AI_GATEWAY_API_KEY or MANTIS_GATEWAY_API_KEY' >&2
    exit 2
  fi
else
  export EXPENSIVE_BASE="${EXPENSIVE_BASE:-https://openrouter.ai/api/v1}"
  export EXPENSIVE_KEY="${EXPENSIVE_KEY:-${OPENROUTER_API_KEY:-}}"
  export CHEAP_BASE="${CHEAP_BASE:-https://opencode.ai/zen/go/v1}"
  export CHEAP_KEY="${CHEAP_KEY:-${OPENCODE_API_KEY:-}}"
  export MIDDLE_BASE="${MIDDLE_BASE:-}"
  export MIDDLE_KEY="${MIDDLE_KEY:-}"
fi
export EXPENSIVE_MODEL="${EXPENSIVE_MODEL:-openai/gpt-5.6-sol}"
export EXPENSIVE_REASONING_EFFORT="${EXPENSIVE_REASONING_EFFORT:-medium}"
export CHEAP_MODEL="${CHEAP_MODEL:-deepseek-v4-flash}"
export CHEAP_REASONING_EFFORT="${CHEAP_REASONING_EFFORT:-}"
export CHEAP_MAX_TOKENS="${CHEAP_MAX_TOKENS:-131072}"
export MIDDLE_MODEL="${MIDDLE_MODEL:-kimi-k3}"
export MIDDLE_REASONING_EFFORT="${MIDDLE_REASONING_EFFORT:-medium}"
export MIDDLE_MAX_TOKENS="${MIDDLE_MAX_TOKENS:-131072}"
# Preserve direct-provider credentials unless a gateway endpoint/profile is selected.
# Base-host auto-detection keeps existing Cloudflare gateway configurations working.
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
export ROUTELLM_CONTEXT_WINDOW="${ROUTELLM_CONTEXT_WINDOW:-262144}"
export ROUTELLM_MAX_TOKENS="${ROUTELLM_MAX_TOKENS:-131072}"
# supra = Supra-Router-51M complexity gate is the primary signal (best
# quality/cost on the Aug 5-10 log; MF score had ~zero labeler separation).
# mf/bert keep the legacy RouteLLM scoring path.
export ROUTELLM_ROUTER="${ROUTELLM_ROUTER:-supra}"
# Optional MF score for observability (x-route-score, eval bands) in supra
# mode; adds one embedding call per unscored prompt when enabled.
export ROUTELLM_SCORE_WITH_MF="${ROUTELLM_SCORE_WITH_MF:-0}"
# 0.156 = calibrated for 30% strong-model calls via RouteLLM (mf mode only).
# Canonical defaults live here; server.py mirrors them (bare-run parity).
export ROUTELLM_THRESHOLD="${ROUTELLM_THRESHOLD:-0.2}"
export ROUTELLM_RESP_CACHE_TTL_S="${ROUTELLM_RESP_CACHE_TTL_S:-120}"
export ROUTELLM_USE_SUPRA="${ROUTELLM_USE_SUPRA:-1}"
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
