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

if [ -f "$REPO_ROOT/.env" ]; then
  set -a
  # Repo .env holds local-only secrets (gitignored); sourced for status display.
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.env"
  set +a
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

launchd_target() { printf 'gui/%s/com.mallochio.mantis.%s' "$(id -u)" "$1"; }
launchd_plist() { printf '%s/.local/share/llm-stack/agents/com.mallochio.mantis.%s.plist' "$HOME" "$1"; }
launchd_loaded() { launchctl print "$(launchd_target "$1")" >/dev/null 2>&1; }

write_launchd_jobs() {
  local args=(write --repo "$REPO_ROOT" --home "$HOME")
  [ "$EXPERIMENTAL_MODES" = 1 ] && args+=(--experimental)
  ( cd "$REPO_ROOT" && uv run --no-sync python launch/host/launchd_jobs.py "${args[@]}" >/dev/null )
}

legacy_listener() {
  local port="$1" pidfile="$2" pid
  pid=$(pid_on_port "$port")
  if [ -n "$pid" ]; then
    fail "port :$port is held by unmanaged PID $pid; stop it explicitly before migration"
    return 1
  fi
  rm -f "$pidfile"
}

bootstrap_job() {
  local name="$1" port="$2" pidfile="$3" url="$4" timeout="$5"
  if ! launchd_loaded "$name"; then
    legacy_listener "$port" "$pidfile" || return 1
    launchctl bootstrap "gui/$(id -u)" "$(launchd_plist "$name")" || return 1
  fi
  wait_ready "$name" "$url" "$timeout"
}

bootout_job() {
  local name="$1" port="$2" pidfile="$3"
  launchd_loaded "$name" || return 0
  launchctl bootout "$(launchd_target "$name")" || return 1
  for _ in $(seq 1 600); do
    if ! launchd_loaded "$name" && [ -z "$(pid_on_port "$port")" ]; then
      rm -f "$pidfile"
      return 0
    fi
    sleep 0.1
  done
  fail "$name did not release port :$port after bootout"
  return 1
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
  [ -f "$CATALOG" ] || { fail "catalog missing: $CATALOG"; return 1; }
  write_launchd_jobs || return 1
  bootstrap_job switchyard 5500 "$SWITCHYARD_PIDFILE" "$ROUTER_URL" 60 || return 1
  local desired_mode="normal"
  [ "$EXPERIMENTAL_MODES" = 1 ] && desired_mode="experimental"
  if launchd_loaded api && [ "$(api_runtime_mode)" != "$desired_mode" ]; then
    bootout_job api 8088 "$HOME/.local/share/mantis/server.pid" || return 1
  fi
  bootstrap_job api 8088 "$HOME/.local/share/mantis/server.pid" "$MANTIS_LOCAL_READY" 180 || return 1
  cmd_status
}

cmd_restart() {
  acquire_lock
  configure_runtime_mode
  printf '\n\033[1mRestarting the whole local stack\033[0m\n'
  bootout_job api 8088 "$HOME/.local/share/mantis/server.pid" || return 1
  bootout_job switchyard 5500 "$SWITCHYARD_PIDFILE" || return 1
  write_launchd_jobs || return 1
  bootstrap_job switchyard 5500 "$SWITCHYARD_PIDFILE" "$ROUTER_URL" 60 || return 1
  bootstrap_job api 8088 "$HOME/.local/share/mantis/server.pid" "$MANTIS_LOCAL_READY" 180 || return 1
  cmd_status
}

cmd_stop() {
  acquire_lock
  printf '\n\033[1mStopping local stack (Mantis API -> Switchyard)\033[0m\n'
  bootout_job api 8088 "$HOME/.local/share/mantis/server.pid" || return 1
  bootout_job switchyard 5500 "$SWITCHYARD_PIDFILE" || return 1
  legacy_listener 8088 "$HOME/.local/share/mantis/server.pid" || return 1
  legacy_listener 5500 "$SWITCHYARD_PIDFILE" || return 1
  ok "Mantis API and Switchyard stopped"
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


def _slot_effort(slot, field=None):
    if isinstance(slot, list):
        slot = slot[0] if slot else None
    if not slot:
        return "none"
    if slot in ("mantis/base", "base"):
        return "dynamic"
    role = slot.split(":", 1)[1].strip() if slot.startswith("base:") else slot
    if role in ("efficient", "capable") and base:
        target_eff = (targets.get(role) or {}).get("reasoning_effort")
        return (target_eff if target_eff and target_eff != "none" else field) or "none"
    worker = workers.get(slot)
    if worker is not None:
        return field or worker.get("reasoning_effort") or "none"
    if "|" in slot:
        return slot.rsplit("|", 1)[1].strip() or "none"
    return field or "none"


main_slot = fusion.get("main", "?")
side_slot = fusion.get("sidekick", "?")
main_eff = _slot_effort(main_slot, fusion.get("main_reasoning"))
side_eff = _slot_effort(side_slot, fusion.get("sidekick_reasoning"))
print(f"  Base:          {picker}")
print(f"    efficient:   {eff_model} (reasoning={eff_effort}, max_tokens={eff_tokens})")
print(f"    capable:     {cap_model} (reasoning={cap_effort}, max_tokens={cap_tokens})")
print(
    f"  Fusion:        main={_slot_str(main_slot)} (reasoning={main_eff}), "
    f"sidekick={_slot_str(side_slot)} (reasoning={side_eff})"
)
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
