#!/usr/bin/env bash
# llm-stack.sh — Universal switcher and launcher for Mantis (Local vs Cloud).
#
# Supports:
#   start     Start the local LLM stack (Bifrost :8080 -> Gateway :5500 -> Mantis API :8088)
#   stop      Stop the local LLM stack to save battery and RAM
#   restart   Restart the local LLM stack
#   status    Check status of local and cloud Mantis endpoints
#   cloud     Manage cloud deployment (status, test, sync, use-cloud, use-local)
#
# Usage: ~/Startup/llm-stack.sh [start|stop|restart|status|cloud]
set -euo pipefail

SELF="$0"
while [ -L "$SELF" ]; do SELF="$(readlink "$SELF")"; done
STARTUP_DIR="$(cd "$(dirname "$SELF")" && pwd)"
LIB_DIR="$STARTUP_DIR/lib"
REPO_ROOT="$(cd "$STARTUP_DIR/../.." && pwd)"

export PATH="/opt/homebrew/bin:/opt/homebrew/sbin:$HOME/.local/bin:$PATH"

if [ -f "$HOME/.zshrc" ]; then
  while IFS='=' read -r name value; do
    [[ "$name" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] && export "$name=$value"
  done < <(/bin/zsh -lc 'source "$HOME/.zshrc" >/dev/null && env')
fi

BIFROST_URL="http://127.0.0.1:8080/health"
ROUTER_URL="http://127.0.0.1:5500/healthz"
MANTIS_LOCAL_READY="http://127.0.0.1:8088/ready"
BIFROST_PIDFILE="$HOME/.local/share/bifrost/server.pid"
GATEWAY_PIDFILE="$HOME/.local/share/mantis/router/server.pid"
CATALOG="$HOME/.config/ai-routing/catalog.toml"

step() { printf '\n\033[1;36m== %s ==\033[0m\n' "$1"; }
ok()   { printf '  \033[1;32mok\033[0m    %s\n' "$1"; }
warn() { printf '  \033[1;33mwarn\033[0m  %s\n' "$1"; }
fail() { printf '  \033[1;31mFAIL\033[0m  %s\n' "$1" >&2; }

LOCK_DIR="$HOME/.local/share/llm-stack/.lock"

acquire_lock() {
  mkdir -p "$HOME/.local/share/llm-stack"
  if [ -d "$LOCK_DIR" ]; then
    local holder
    holder=$(cat "$LOCK_DIR/pid" 2>/dev/null || echo "unknown")
    if [ "$holder" != "unknown" ] && ! kill -0 "$holder" 2>/dev/null; then
      warn "removing stale llm-stack lock (pid $holder)"
      rm -rf "$LOCK_DIR"
    fi
  fi
  if ! mkdir "$LOCK_DIR" 2>/dev/null; then
    local holder
    holder=$(cat "$LOCK_DIR/pid" 2>/dev/null || echo "unknown")
    fail "another llm-stack process is already running (pid $holder). Use status or wait."
    exit 1
  fi
  echo $$ > "$LOCK_DIR/pid"
  trap 'rm -rf "$LOCK_DIR"' EXIT
}

health_ok()   { curl -fsS --max-time 3 "$1" >/dev/null 2>&1; }
gateway_ready() { curl -fsS --max-time 3 "$ROUTER_URL" 2>/dev/null | grep -q '"ready":true'; }
pid_on_port() { lsof -nP -iTCP:"$1" -sTCP:LISTEN -t 2>/dev/null | head -n1 || true; }

kill_all_listeners() {
  local port="$1" pid
  [ -n "$port" ] || return 0
  for pid in $(lsof -nP -iTCP:"$port" -sTCP:LISTEN -t 2>/dev/null); do
    kill "$pid" 2>/dev/null || true
  done
  for _ in $(seq 1 50); do
    lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1 || break
    sleep 0.1
  done
  for pid in $(lsof -nP -iTCP:"$port" -sTCP:LISTEN -t 2>/dev/null); do
    kill -9 "$pid" 2>/dev/null || true
  done
  for _ in $(seq 1 50); do
    lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1 || break
    sleep 0.1
  done
}

port_from_url() {
  echo "$1" | awk -F: '{print $3}' | cut -d/ -f1
}

wait_ready() {
  local name="$1" url="$2" timeout_s="${3:-90}" pat="${4:-}" i=0
  while [ "$i" -lt "$timeout_s" ]; do
    if [ -n "$pat" ]; then
      if curl -fsS --max-time 3 "$url" 2>/dev/null | grep -q "$pat"; then ok "$name ready"; return 0; fi
    elif health_ok "$url"; then
      ok "$name ready"; return 0
    fi
    sleep 1; i=$((i+1))
  done
  fail "$name not ready after ${timeout_s}s ($url)"
  return 1
}

start_one() {
  local no="$1" name="$2" script="$3" url="$4" timeout_s="$5" pat="${6:-}"
  step "$no starting $name"
  if [ ! -x "$script" ]; then fail "component script missing: $script"; return 1; fi
  "$script" >/dev/null 2>&1 || { fail "$script failed"; return 1; }
  wait_ready "$name" "$url" "$timeout_s" "$pat" || return 1
}

cmd_start() {
  acquire_lock
  printf '\n\033[1mStarting the local LLM stack (Bifrost -> direct gateway -> Mantis API)\033[0m\n'
  [ -f "$CATALOG" ] || warn "catalog missing: $CATALOG (required by the gateway and API)"
  start_one "1/3" "bifrost" "$LIB_DIR/bifrost-local.sh" "$BIFROST_URL" 90 || return 1
  start_one "2/3" "direct gateway" "$LIB_DIR/llm-router.sh" "$ROUTER_URL" 120 '"ready":true' || return 1
  start_one "3/3" "Mantis API" "$LIB_DIR/mantis-local.sh" "$MANTIS_LOCAL_READY" 180 || return 1
  cmd_status
}

cmd_restart() {
  acquire_lock
  printf '\n\033[1mRestarting the whole local stack\033[0m\n'
  start_one "1/3" "bifrost" "$LIB_DIR/bifrost-local.sh" "$BIFROST_URL" 90 || return 1
  start_one "2/3" "direct gateway" "$LIB_DIR/llm-router.sh" "$ROUTER_URL" 120 '"ready":true' || return 1
  start_one "3/3" "Mantis API" "$LIB_DIR/mantis-local.sh" "$MANTIS_LOCAL_READY" 180 || return 1
  cmd_status
}

cmd_stop() {
  acquire_lock
  printf '\n\033[1mStopping local stack (Mantis API -> direct gateway -> Bifrost)\033[0m\n'
  step "1/3 stopping Mantis API (:8088)"
  kill_all_listeners "$(port_from_url "$MANTIS_LOCAL_READY")"
  local mp="$HOME/.local/share/mantis/server.pid"
  if [ -f "$mp" ]; then
    local pid; pid=$(cat "$mp" 2>/dev/null || true)
    if [ -n "${pid:-}" ] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
      for _ in $(seq 1 50); do kill -0 "$pid" 2>/dev/null || break; sleep 0.1; done
      kill -9 "$pid" 2>/dev/null || true
    fi
    rm -f "$mp"
  fi
  ok "Mantis API stopped"
  step "2/3 stopping direct gateway (:5500)"
  kill_all_listeners "$(port_from_url "$ROUTER_URL")"
  rm -f "$GATEWAY_PIDFILE"
  ok "direct gateway stopped"
  step "3/3 stopping bifrost (:8080)"
  kill_all_listeners "$(port_from_url "$BIFROST_URL")"
  rm -f "$BIFROST_PIDFILE"
  ok "bifrost stopped"
}

cmd_status() {
  printf '\n\033[1mLLM Stack Status\033[0m\n'
  printf '  %-14s %-7s %-8s %s\n' SERVICE PORT PID STATE
  if health_ok "$BIFROST_URL"; then
    printf '  %-14s %-7s %-8s \033[1;32mup\033[0m\n' bifrost :8080 "$(pid_on_port 8080)"
  else
    printf '  %-14s %-7s %-8s \033[1;31mdown (saved battery)\033[0m\n' bifrost :8080 -
  fi
  if gateway_ready; then
    printf '  %-14s %-7s %-8s \033[1;32mup\033[0m\n' gateway :5500 "$(pid_on_port 5500)"
  else
    printf '  %-14s %-7s %-8s \033[1;31mdown (saved battery)\033[0m\n' gateway :5500 -
  fi
  if health_ok "$MANTIS_LOCAL_READY"; then
    printf '  %-14s %-7s %-8s \033[1;32mup\033[0m\n' mantis-api :8088 "$(pid_on_port 8088)"
  else
    printf '  %-14s %-7s %-8s \033[1;31mdown (saved battery)\033[0m\n' mantis-api :8088 -
  fi

  # Check active MANTIS_URL in environment
  local active_url="${MANTIS_URL:-http://127.0.0.1:8088/v1}"
  printf '\n  Active Environment MANTIS_URL: %s\n' "$active_url"
  if [[ "$active_url" =~ ^https?://(127\.0\.0\.1|localhost) ]]; then
    printf '  Routing target: \033[1;33mLocal Loopback\033[0m\n'
  else
    printf '  Routing target: \033[1;32mRender Cloud Instance\033[0m\n'
    if curl -fsS --max-time 5 "${active_url%/v1}/ready" >/dev/null 2>&1; then
      printf '  Cloud Status:   \033[1;32mOnline & Healthy\033[0m\n'
    else
      printf '  Cloud Status:   \033[1;31mUnreachable / Deploying\033[0m\n'
    fi
  fi
  printf '\n'
}

cmd_cloud() {
  exec "$REPO_ROOT/deploy/scripts/mantis-cloud.sh" "$@"
}

case "${1:-status}" in
  start)   cmd_start ;;
  restart) cmd_restart ;;
  status)  cmd_status ;;
  stop)    cmd_stop ;;
  cloud)   shift; cmd_cloud "$@" ;;
  *) echo "usage: $0 [start|stop|restart|status|cloud ...]" >&2; exit 2 ;;
esac
