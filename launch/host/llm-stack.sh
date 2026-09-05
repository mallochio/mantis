#!/usr/bin/env bash
# llm-stack.sh — Local Mantis/Switchyard stack controller.
#
# Supports:
#   start [--experimental]    Start normal Base/Fusion mode, or opt into Trinity/Ultra
#   restart [--experimental]  Restart in normal or experimental mode
#   stop                      Stop the local stack
#   status                    Check local process health
#
# A bare invocation (as used by StartupFolder) starts normal Base/Fusion mode.
# Usage: ~/Startup/llm-stack.sh [start [--experimental]|restart [--experimental]|--experimental|stop|status]
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

ROUTER_URL="http://127.0.0.1:5500/health"
MANTIS_LOCAL_READY="http://127.0.0.1:8088/ready"
SWITCHYARD_PIDFILE="$HOME/.local/share/mantis/switchyard/server.pid"
CATALOG="${AI_ROUTING_CONFIG:-$HOME/.config/ai-routing/catalog.toml}"

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
switchyard_ready() { health_ok "$ROUTER_URL"; }
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
  local out
  step "$no starting $name"
  if [ ! -x "$script" ]; then fail "component script missing: $script"; return 1; fi
  if ! out="$("$script" 2>&1)"; then
    fail "$script failed"
    printf '%s\n' "$out" >&2
    return 1
  fi
  wait_ready "$name" "$url" "$timeout_s" "$pat" || return 1
}

EXPERIMENTAL_MODES=0

configure_runtime_mode() {
  if [ "$EXPERIMENTAL_MODES" = 1 ]; then
    export MANTIS_EXPERIMENTAL_MODES=1
    ok "experimental modes enabled (Trinity + Ultra)"
  else
    unset MANTIS_EXPERIMENTAL_MODES
    ok "normal mode (Base + Fusion; Trinity/Ultra gated)"
  fi
}

api_runtime_mode() {
  curl -fsS --max-time 3 "$MANTIS_LOCAL_READY" 2>/dev/null \
    | sed -n 's/.*"runtime_mode":"\([^"]*\)".*/\1/p'
}

cmd_start() {
  acquire_lock
  configure_runtime_mode
  printf '\n\033[1mStarting the local LLM stack (Switchyard -> Mantis API)\033[0m\n'
  [ -f "$CATALOG" ] || warn "catalog missing: $CATALOG (required by Switchyard and the API)"
  if switchyard_ready; then
    ok "Switchyard already ready"
  elif [ -n "$(pid_on_port 5500)" ] && wait_ready "existing Switchyard" "$ROUTER_URL" 30; then
    :
  else
    start_one "1/2" "Switchyard" "$LIB_DIR/switchyard-local.sh" "$ROUTER_URL" 60 || return 1
  fi
  local desired_mode="normal"
  [ "$EXPERIMENTAL_MODES" = 1 ] && desired_mode="experimental"
  if health_ok "$MANTIS_LOCAL_READY" && [ "$(api_runtime_mode)" = "$desired_mode" ]; then
    ok "Mantis API already ready in $desired_mode mode"
  else
    start_one "2/2" "Mantis API" "$LIB_DIR/mantis-local.sh" "$MANTIS_LOCAL_READY" 180 || return 1
  fi
  cmd_status
}

cmd_restart() {
  acquire_lock
  configure_runtime_mode
  printf '\n\033[1mRestarting the whole local stack\033[0m\n'
  start_one "1/2" "Switchyard" "$LIB_DIR/switchyard-local.sh" "$ROUTER_URL" 60 || return 1
  start_one "2/2" "Mantis API" "$LIB_DIR/mantis-local.sh" "$MANTIS_LOCAL_READY" 180 || return 1
  cmd_status
}

cmd_stop() {
  acquire_lock
  printf '\n\033[1mStopping local stack (Mantis API -> Switchyard)\033[0m\n'
  step "1/2 stopping Mantis API (:8088)"
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
  step "2/2 stopping Switchyard (:5500)"
  kill_all_listeners "$(port_from_url "$ROUTER_URL")"
  rm -f "$SWITCHYARD_PIDFILE"
  ok "Switchyard stopped"
}

print_model_config() {
  local output
  output="$(
    cd "$REPO_ROOT" && CATALOG="$CATALOG" uv run --no-sync python - 2>/dev/null <<'PY'
import os, sys, tomllib

catalog = os.environ.get("CATALOG", os.path.expanduser("~/.config/ai-routing/catalog.toml"))
if not os.path.exists(catalog):
    sys.exit(0)

with open(catalog, "rb") as f:
    data = tomllib.load(f)

base = data.get("base") or {}
targets = base.get("targets") or {}
eff = targets.get("efficient") or {}
cap = targets.get("capable") or {}
fusion = data.get("fusion") or {}
mantis = data.get("mantis") or {}
workers = mantis.get("workers") or {}

eff_model = eff.get("upstream_model", "?")
cap_model = cap.get("upstream_model", "?")
eff_effort = eff.get("reasoning_effort") or "none"
cap_effort = cap.get("reasoning_effort") or "none"
eff_tokens = eff.get("max_tokens") or "default"
cap_tokens = cap.get("max_tokens") or "default"
picker = base.get("picker") or "efficient_first"

def _slot_str(slot):
    return ", ".join(slot) if isinstance(slot, list) else str(slot)


def _slot_effort(slot):
    if isinstance(slot, list):
        slot = slot[0] if slot else None
    return (workers.get(slot) or {}).get("reasoning_effort") or "none" if slot else "none"


main_slot = fusion.get("main", "?")
side_slot = fusion.get("sidekick", "?")
main_eff = _slot_effort(main_slot)
side_eff = _slot_effort(side_slot)
providers = data.get("providers") or {}
az = providers.get("azure-foundry-router") or {}
az_url = az.get("base_url", "?")

print(f"  Base:          {picker}")
print(f"    efficient:   {eff_model} (reasoning={eff_effort}, max_tokens={eff_tokens})")
print(f"    capable:     {cap_model} (reasoning={cap_effort}, max_tokens={cap_tokens})")
print(
    f"  Fusion:        main={_slot_str(main_slot)} (reasoning={main_eff}), "
    f"sidekick={_slot_str(side_slot)} (reasoning={side_eff})"
)
print(f"  Azure-router:  mantis/azure-router -> model-router (reasoning=medium, context=256000)")
print(f"    endpoint:    {az_url}")
PY
  )" || output=""
  if [ -n "$output" ]; then
    printf '\n\033[1mMantis routing models\033[0m\n%s\n' "$output"
  fi
  return 0
}

cmd_status() {
  printf '\n\033[1mLLM Stack Status\033[0m\n'
  printf '  %-14s %-7s %-8s %s\n' SERVICE PORT PID STATE
  if switchyard_ready; then
    printf '  %-14s %-7s %-8s \033[1;32mup\033[0m\n' switchyard :5500 "$(pid_on_port 5500)"
  else
    printf '  %-14s %-7s %-8s \033[1;31mdown\033[0m\n' switchyard :5500 -
  fi
  if health_ok "$MANTIS_LOCAL_READY"; then
    printf '  %-14s %-7s %-8s \033[1;32mup\033[0m\n' mantis-api :8088 "$(pid_on_port 8088)"
  else
    printf '  %-14s %-7s %-8s \033[1;31mdown\033[0m\n' mantis-api :8088 -
  fi

  local active_url="${MANTIS_URL:-http://127.0.0.1:8088/v1}"
  printf '\n  Active Environment MANTIS_URL: %s\n' "$active_url"
  printf '  Routing target: \033[1;33mLocal Loopback\033[0m\n'
  print_model_config
  return 0
}

command="${1:-start}"
case "$command" in
  --experimental)
    EXPERIMENTAL_MODES=1
    cmd_start
    ;;
  start|restart)
    if [ "${2:-}" = "--experimental" ]; then
      EXPERIMENTAL_MODES=1
    elif [ -n "${2:-}" ]; then
      echo "usage: $0 $command [--experimental]" >&2
      exit 2
    fi
    if [ "$command" = start ]; then cmd_start; else cmd_restart; fi
    ;;
  status) cmd_status ;;
  stop) cmd_stop ;;
  *) echo "usage: $0 [start [--experimental]|restart [--experimental]|--experimental|stop|status]" >&2; exit 2 ;;
esac
