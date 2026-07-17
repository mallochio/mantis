#!/usr/bin/env bash
# Canonical launch script for the RouteLLM coding-router.
# Lives in ~/Startup/ so StartupFolder (https://github.com/FuzzyIdeas/StartupFolder)
# runs it at login. It detaches the router, prints status, then exits.
#
# Loads exported env from ~/.zshrc, starts the Azure Foundry proxy (port 41437)
# if not already running, then the router. The proxy normalizes GPT-5.6 request
# fields for Azure Foundry and is shared with OpenCode. The router is
# stopped-and-restarted if already running (so config/env changes take effect);
# if it is down it is just started.
#
# Usage: ~/Startup/llm-router.sh          (proxy/router backgrounded, prints status)
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
mkdir -p logs

# Ensure homebrew tools (node) and system CLIs (lsof, security) are found under
# launchd's minimal PATH as well as an interactive shell.
export PATH="/opt/homebrew/opt/node@24/bin:/opt/homebrew/bin:/opt/homebrew/sbin:$PATH"

# StartupFolder/launchd does not read shell startup files. Import exported env
# from zsh so ~/.zshrc remains the single place for RouteLLM/backend config.
if [ -f "$HOME/.zshrc" ]; then
  while IFS='=' read -r name value; do
    [[ "$name" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] && export "$name=$value"
  done < <(/bin/zsh -lc 'source "$HOME/.zshrc" >/dev/null && env')
fi

# --- Azure Foundry proxy (shared with OpenCode) ---
export AZURE_FOUNDRY_PROXY_PORT="${AZURE_FOUNDRY_PROXY_PORT:-41437}"
PROXY_SCRIPT="${AZURE_FOUNDRY_PROXY_SCRIPT:-$HOME/.config/opencode/azure-foundry-proxy.mjs}"

if ! lsof -nP -iTCP:"$AZURE_FOUNDRY_PROXY_PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  if [ -z "${AZURE_API_KEY:-}" ]; then
    export AZURE_API_KEY=$(security find-generic-password -s 'shell-env/IH_FOUNDRY_API_KEY' -w 2>/dev/null || true)
  fi
  if [ -z "${AZURE_API_KEY:-}" ]; then
    echo 'ERROR: AZURE_API_KEY not set and not found in Keychain (shell-env/IH_FOUNDRY_API_KEY). Proxy needs it for Azure Foundry.' >&2
    exit 1
  fi
  if [ ! -f "$PROXY_SCRIPT" ]; then
    echo "ERROR: proxy script not found: $PROXY_SCRIPT" >&2
    exit 1
  fi
  nohup node "$PROXY_SCRIPT" > logs/proxy.out 2> logs/proxy.err &
  echo $! > logs/proxy.pid
  for _ in $(seq 1 50); do
    lsof -nP -iTCP:"$AZURE_FOUNDRY_PROXY_PORT" -sTCP:LISTEN >/dev/null 2>&1 && break
    sleep 0.1
  done
  if ! lsof -nP -iTCP:"$AZURE_FOUNDRY_PROXY_PORT" -sTCP:LISTEN >/dev/null 2>&1; then
    echo 'ERROR: proxy failed to start — see logs/proxy.err' >&2
    cat logs/proxy.err >&2 2>/dev/null || true
    exit 1
  fi
  echo "proxy started pid $(cat logs/proxy.pid) on :$AZURE_FOUNDRY_PROXY_PORT"
else
  echo "proxy already running on :$AZURE_FOUNDRY_PROXY_PORT"
fi

# --- router ---
. ./.venv/bin/activate
# mf router calls OpenAI text-embedding-3-small per prompt; retrieve key from
# macOS Keychain if not already in env.
if [ -z "${OPENAI_API_KEY:-}" ]; then
  export OPENAI_API_KEY=$(security find-generic-password -s 'AI.Playground.openai.apiKey' -w 2>/dev/null || true)
fi
if [ -z "${OPENAI_API_KEY:-}" ]; then
  echo 'ERROR: OPENAI_API_KEY not set and not found in Keychain. mf router needs it for embeddings.' >&2
  exit 1
fi
export EXPENSIVE_BASE="${EXPENSIVE_BASE:-http://127.0.0.1:$AZURE_FOUNDRY_PROXY_PORT/v1}"
export EXPENSIVE_KEY="${EXPENSIVE_KEY:-dummy}"
export EXPENSIVE_MODEL="${EXPENSIVE_MODEL:-gpt-5.6-luna}"
export EXPENSIVE_REASONING_EFFORT="${EXPENSIVE_REASONING_EFFORT:-ultra}"
export CHEAP_BASE="${CHEAP_BASE:-https://opencode.ai/zen/go/v1}"
if [ -z "${CHEAP_KEY:-${OPENCODE_GO_API_KEY:-${OPENCODE_API_KEY:-}}}" ]; then
  export OPENCODE_GO_API_KEY=$(security find-generic-password -s 'shell-env/OPENCODE_GO_API_KEY' -w 2>/dev/null || true)
fi
if [ -z "${CHEAP_KEY:-${OPENCODE_GO_API_KEY:-${OPENCODE_API_KEY:-}}}" ]; then
  export OPENCODE_API_KEY=$(security find-generic-password -s 'shell-env/OPENCODE_API_KEY' -w 2>/dev/null || true)
fi
export CHEAP_KEY="${CHEAP_KEY:-${OPENCODE_GO_API_KEY:-${OPENCODE_API_KEY:-}}}"
export CHEAP_MODEL="${CHEAP_MODEL:-glm-5.2}"
export CHEAP_REASONING_EFFORT="${CHEAP_REASONING_EFFORT:-max}"
export CHEAP_MAX_TOKENS="${CHEAP_MAX_TOKENS:-64000}"
export ROUTELLM_ROUTER="${ROUTELLM_ROUTER:-mf}"
# 0.156 = calibrated for 30% strong-model calls via routellm.calibrate_threshold
export ROUTELLM_THRESHOLD="${ROUTELLM_THRESHOLD:-0.156}"
export ROUTELLM_KEY="${ROUTELLM_KEY:-sk-route-local}"
export ROUTELLM_HOST="${ROUTELLM_HOST:-127.0.0.1}"
export ROUTELLM_PORT="${ROUTELLM_PORT:-5500}"

# If the router is already listening, stop it so we start a clean instance.
# (The proxy above is only started-if-down since it's shared with OpenCode; the
# router we own and restart to pick up config/env changes.)
if lsof -nP -iTCP:"$ROUTELLM_PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "router already running on :$ROUTELLM_PORT — stopping for restart"
  # Trust the recorded pid only if it still owns the port; otherwise take the
  # current listener so a stale/reused pid file can't kill the wrong process.
  ROUTER_PID=""
  if [ -f logs/server.pid ]; then
    CANDIDATE=$(cat logs/server.pid 2>/dev/null || true)
    if [ -n "${CANDIDATE:-}" ] && lsof -nP -iTCP:"$ROUTELLM_PORT" -sTCP:LISTEN -t 2>/dev/null | grep -qx "$CANDIDATE"; then
      ROUTER_PID="$CANDIDATE"
    fi
  fi
  if [ -z "${ROUTER_PID:-}" ]; then
    ROUTER_PID=$(lsof -nP -iTCP:"$ROUTELLM_PORT" -sTCP:LISTEN -t 2>/dev/null | head -n1 || true)
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
  rm -f logs/server.pid
  echo "router stopped"
fi

nohup python server.py > logs/server.out 2> logs/server.err &
echo $! > logs/server.pid
for _ in $(seq 1 50); do
  if lsof -nP -iTCP:"$ROUTELLM_PORT" -sTCP:LISTEN >/dev/null 2>&1; then
    ROUTER_PID=$(lsof -nP -iTCP:"$ROUTELLM_PORT" -sTCP:LISTEN -t 2>/dev/null | head -n1 || true)
    echo "router running pid ${ROUTER_PID:-$(cat logs/server.pid)} on :$ROUTELLM_PORT"
    exit 0
  fi
  sleep 0.1
done

echo "ERROR: router not running on :$ROUTELLM_PORT — see logs/server.err" >&2
cat logs/server.err >&2 2>/dev/null || true
exit 1
