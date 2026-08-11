#!/usr/bin/env bash
# llm-stack.sh — one script for the whole local LLM stack.
#
# Order: 1) Bifrost gateway (:8080, single upstream for both routers)
#        2) llm-router   (:5500, Supra coding router, catalog-configured)
#        3) Mantis       (:8088, conductor orchestrator, catalog mode)
#
# The shared catalog (~/.config/ai-routing/catalog.toml) is the source of
# truth for both routers; Bifrost (~/.local/share/bifrost/config.json) holds
# the upstream routes (opencode-go, Azure, Bedrock-EU, Vertex) and keys.
#
# Usage: ~/Startup/llm-stack.sh [start|restart|status|stop]
#   start    (default) start each component; running ones are restarted to load the latest catalog/config
#   restart  stop and start all three in order
#   status   health + pid summary
#   stop     stop all three in reverse order
set -euo pipefail

# StartupFolder may invoke us from / and the script may be a symlink; resolve.
SELF="$0"
while [ -L "$SELF" ]; do SELF="$(readlink "$SELF")"; done
STARTUP_DIR="$(cd "$(dirname "$SELF")" && pwd)"
LIB_DIR="$STARTUP_DIR/lib"

export PATH="/opt/homebrew/bin:/opt/homebrew/sbin:$HOME/.local/bin:$PATH"
set +e
source "$HOME/.zshrc" 2>/dev/null || true
set -e

BIFROST_URL="http://127.0.0.1:8080/health"
ROUTER_URL="http://127.0.0.1:5500/healthz"
MANTIS_URL="http://127.0.0.1:8088/ready"
BIFROST_PIDFILE="$HOME/.local/share/bifrost/server.pid"
ROUTER_PIDFILE="$HOME/.local/share/mantis/router/server.pid"
MANTIS_REPO="$HOME/Personal/other/mantis"
CATALOG="$HOME/.config/ai-routing/catalog.toml"

step() { printf '\n\033[1;36m== %s ==\033[0m\n' "$1"; }
ok()   { printf '  \033[1;32mok\033[0m    %s\n' "$1"; }
warn() { printf '  \033[1;33mwarn\033[0m  %s\n' "$1"; }
fail() { printf '  \033[1;31mFAIL\033[0m  %s\n' "$1" >&2; }

health_ok()   { curl -fsS --max-time 5 "$1" >/dev/null 2>&1; }
router_ready() { curl -fsS --max-time 5 "$ROUTER_URL" 2>/dev/null | grep -q '"ready":true'; }
pid_on_port() { lsof -nP -iTCP:"$1" -sTCP:LISTEN -t 2>/dev/null | head -n1 || true; }

wait_ready() { # name url timeout_s [grep_pattern]
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

start_one() { # step_no name script url timeout_s [grep]
  local no="$1" name="$2" script="$3" url="$4" timeout_s="$5" pat="${6:-}"
  step "$no starting $name"
  if [ ! -x "$script" ]; then fail "component script missing: $script"; return 1; fi
  "$script" >/dev/null 2>&1 || { fail "$script failed"; return 1; }
  wait_ready "$name" "$url" "$timeout_s" "$pat" || return 1
}

cmd_start() {
  printf '\n\033[1mStarting the LLM stack (Bifrost -> router -> mantis)\033[0m\n'
  printf '\033[2mComponents already running are restarted so the latest catalog/config is loaded.\033[0m\n'
  [ -f "$CATALOG" ] || warn "catalog missing: $CATALOG (routers will fall back to legacy env)"
  start_one "1/3" "bifrost" "$LIB_DIR/bifrost-local.sh" "$BIFROST_URL" 90 || return 1
  start_one "2/3" "llm-router" "$LIB_DIR/llm-router.sh" "$ROUTER_URL" 120 '"ready":true' || return 1
  start_one "3/3" "mantis" "$LIB_DIR/mantis-local.sh" "$MANTIS_URL" 180 || return 1
  cmd_status
}

cmd_restart() {
  printf '\n\033[1mRestarting the whole stack (Bifrost -> router -> mantis)\033[0m\n'
  start_one "1/3" "bifrost" "$LIB_DIR/bifrost-local.sh" "$BIFROST_URL" 90 || return 1
  start_one "2/3" "llm-router" "$LIB_DIR/llm-router.sh" "$ROUTER_URL" 120 '"ready":true' || return 1
  start_one "3/3" "mantis" "$LIB_DIR/mantis-local.sh" "$MANTIS_URL" 180 || return 1
  cmd_status
}

cmd_stop() {
  printf '\n\033[1mStopping the stack (mantis -> router -> bifrost)\033[0m\n'
  step "1/3 stopping mantis (:8088)"
  local mp="$HOME/.local/share/mantis/server.pid"
  if [ -f "$mp" ] && kill -0 "$(cat "$mp")" 2>/dev/null; then
    kill "$(cat "$mp")" 2>/dev/null || true
    rm -f "$mp"
    ok "mantis host process stopped"
  else
    ok "mantis host process not running"
  fi
  step "2/3 stopping llm-router (:5500)"
  if health_ok "$ROUTER_URL"; then
    local rp; rp=$(cat "$ROUTER_PIDFILE" 2>/dev/null || true)
    if [ -n "${rp:-}" ] && kill -0 "$rp" 2>/dev/null; then kill "$rp" 2>/dev/null || true; fi
    for _ in $(seq 1 50); do health_ok "$ROUTER_URL" || break; sleep 0.1; done
    ok "llm-router stopped"
  else
    ok "llm-router not running"
  fi
  step "3/3 stopping bifrost (:8080)"
  if health_ok "$BIFROST_URL"; then
    local bp; bp=$(cat "$BIFROST_PIDFILE" 2>/dev/null || true)
    if [ -n "${bp:-}" ] && kill -0 "$bp" 2>/dev/null; then kill "$bp" 2>/dev/null || true; fi
    for _ in $(seq 1 50); do health_ok "$BIFROST_URL" || break; sleep 0.1; done
    ok "bifrost stopped"
  else
    ok "bifrost not running"
  fi
}

cmd_status() {
  printf '\n\033[1mStack status\033[0m\n'
  printf '  %-12s %-7s %-8s %s\n' SERVICE PORT PID STATE
  if health_ok "$BIFROST_URL"; then
    printf '  %-12s %-7s %-8s \033[1;32mup\033[0m\n' bifrost :8080 "$(pid_on_port 8080)"
  else
    printf '  %-12s %-7s %-8s \033[1;31mdown\033[0m\n' bifrost :8080 -
  fi
  if router_ready; then
    printf '  %-12s %-7s %-8s \033[1;32mup\033[0m\n' llm-router :5500 "$(pid_on_port 5500)"
  else
    printf '  %-12s %-7s %-8s \033[1;31mdown\033[0m\n' llm-router :5500 -
  fi
  if health_ok "$MANTIS_URL"; then
    printf '  %-12s %-7s %-8s \033[1;32mup\033[0m\n' mantis :8088 "$(pid_on_port 8088)"
  else
    printf '  %-12s %-7s %-8s \033[1;31mdown\033[0m\n' mantis :8088 -
  fi
  if router_ready; then
    local rev; rev=$(curl -fsS --max-time 5 "$ROUTER_URL" 2>/dev/null | grep -o '"target_config_revision":"[^"]*"' | head -n1 | cut -d'"' -f4)
    printf '\n  router config: %s\n' "${rev:-unknown}"
  fi
  printf '\n'
}

case "${1:-start}" in
  start)   cmd_start ;;
  restart) cmd_restart ;;
  status)  cmd_status ;;
  stop)    cmd_stop ;;
  *) echo "usage: $0 [start|restart|status|stop]" >&2; exit 2 ;;
esac
