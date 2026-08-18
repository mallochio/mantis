#!/usr/bin/env bash
# Canonical launch script for the Mantis gateway.
# Lives in ~/Startup/ so StartupFolder (https://github.com/FuzzyIdeas/StartupFolder)
# runs it at login. It detaches the router, prints status, then exits.
#
# Loads exported env from ~/.zshrc (with a hard timeout so Starship/dumb TERM
# cannot wedge the launcher), kills any previous gateway or stuck copy of this
# script, then starts server.py in a new session so it survives this script
# exiting. Running the script again is a neat restart.
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

ZSHRC_IMPORT_TIMEOUT_S="${ZSHRC_IMPORT_TIMEOUT_S:-8}"

_import_zshrc_env() {
  # Non-interactive, non-login source of ~/.zshrc. Login zsh (-lc) plus TERM=dumb
  # lets Starship block the launcher for hours. Timeout and kill the importer
  # process group if it does not finish.
  [ -f "$HOME/.zshrc" ] || return 0
  local dump item name value
  dump="$(mktemp -t llm-router-env)"
  /usr/bin/python3 - "$dump" "$ZSHRC_IMPORT_TIMEOUT_S" <<'PY'
import os, signal, subprocess, sys

dump, timeout_s = sys.argv[1], float(sys.argv[2])
env = os.environ.copy()
if env.get("TERM") in (None, "", "dumb"):
    env["TERM"] = "xterm-256color"
env.pop("STARSHIP_SHELL", None)
proc = subprocess.Popen(
    [
        "/bin/zsh",
        "-f",
        "-c",
        'source "$HOME/.zshrc" >/dev/null 2>/dev/null; exec /usr/bin/env -0',
    ],
    stdout=subprocess.PIPE,
    stderr=subprocess.DEVNULL,
    env=env,
    start_new_session=True,
)

def _kill_importer(*_args):
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except OSError:
        pass

signal.signal(signal.SIGTERM, _kill_importer)
signal.signal(signal.SIGINT, _kill_importer)
try:
    data, _ = proc.communicate(timeout=timeout_s)
except subprocess.TimeoutExpired:
    _kill_importer()
    proc.wait()
    sys.stderr.write(
        f"WARNING: ~/.zshrc env import timed out after {int(timeout_s)}s; "
        "continuing with the current environment\n"
    )
    sys.exit(0)
if proc.returncode not in (0, None):
    sys.stderr.write(
        "WARNING: ~/.zshrc env import failed; continuing with the current environment\n"
    )
    sys.exit(0)
with open(dump, "wb") as handle:
    handle.write(data)
PY
  if [ -s "$dump" ]; then
    while IFS= read -r -d '' item || [ -n "${item:-}" ]; do
      [ -n "$item" ] || continue
      name="${item%%=*}"
      value="${item#*=}"
      case "$name" in
        PWD|OLDPWD|SHLVL|_|SHELL|STARSHIP_SHELL) continue ;;
      esac
      [[ "$name" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue
      export "$name=$value"
    done < "$dump"
  fi
  rm -f "$dump"
}

_import_zshrc_env
export PATH="/opt/homebrew/bin:/opt/homebrew/sbin:$PATH"

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

_OUR_PGID="$(ps -o pgid= -p $$ 2>/dev/null | tr -d ' ')"

_is_self() {
  [ "$1" = "$$" ] || [ "$1" = "$PPID" ]
}

_kill_pid() {
  local sig="$1" pid="$2"
  [ -n "$pid" ] || return 0
  _is_self "$pid" && return 0
  kill -"$sig" "$pid" 2>/dev/null || true
}

# Kill a stuck launcher and its process group (orphaned zshrc importers),
# but never our own group. Skip editors that merely have the path in argv.
_kill_group() {
  local sig="$1" pid="$2" pgid
  [ -n "$pid" ] || return 0
  _is_self "$pid" && return 0
  pgid="$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')"
  if [ -n "$pgid" ] && [ "$pgid" != "$_OUR_PGID" ]; then
    kill -"$sig" -"$pgid" 2>/dev/null || _kill_pid "$sig" "$pid"
  else
    _kill_pid "$sig" "$pid"
  fi
}

_stuck_launcher_pids() {
  pgrep -f '(^|/)(bash|zsh|sh|/usr/bin/env) .*/llm-router\.sh' 2>/dev/null || true
}

_wait_port_clear() {
  local i
  for i in $(seq 1 50); do
    lsof -nP -iTCP:"$MANTIS_ROUTER_PORT" -sTCP:LISTEN >/dev/null 2>&1 || return 0
    sleep 0.1
  done
  return 1
}

_stop_gateway() {
  local pid
  echo "stopping gateway on :$MANTIS_ROUTER_PORT (if any)"
  if [ -f "$ROUTER_LOG_DIR/server.pid" ]; then
    _kill_pid TERM "$(cat "$ROUTER_LOG_DIR/server.pid" 2>/dev/null || true)"
  fi
  for pid in $(lsof -nP -iTCP:"$MANTIS_ROUTER_PORT" -sTCP:LISTEN -t 2>/dev/null); do
    _kill_pid TERM "$pid"
  done
  for pid in $(pgrep -f 'apps/gateway/.venv/bin/python.*server.py' 2>/dev/null || true); do
    _kill_pid TERM "$pid"
  done
  # Stuck copies of this launcher (zshrc import, leftover nohup wrappers).
  for pid in $(_stuck_launcher_pids); do
    _kill_group TERM "$pid"
  done
  _wait_port_clear || true
  for pid in $(lsof -nP -iTCP:"$MANTIS_ROUTER_PORT" -sTCP:LISTEN -t 2>/dev/null); do
    _kill_pid KILL "$pid"
  done
  for pid in $(pgrep -f 'apps/gateway/.venv/bin/python.*server.py' 2>/dev/null || true); do
    _kill_pid KILL "$pid"
  done
  for pid in $(_stuck_launcher_pids); do
    _kill_group KILL "$pid"
  done
  _wait_port_clear || true
  rm -f "$ROUTER_LOG_DIR/server.pid"
}

_daemonize_server() {
  /usr/bin/python3 - "$REPO_DIR/.venv/bin/python" "$REPO_DIR" \
    "$ROUTER_LOG_DIR/server.out" "$ROUTER_LOG_DIR/server.err" \
    "$ROUTER_LOG_DIR/server.pid" <<'PY'
import os, sys

python, cwd, log_out, log_err, pidfile = sys.argv[1:6]
r, w = os.pipe()
child = os.fork()
if child > 0:
    os.close(w)
    pid = os.read(r, 64)
    os.close(r)
    os.waitpid(child, 0)
    sys.stdout.buffer.write(pid)
    sys.exit(0)

os.close(r)
os.setsid()
if os.fork() > 0:
    os._exit(0)

os.chdir(cwd)
os.umask(0o22)
devnull = os.open(os.devnull, os.O_RDWR)
out = os.open(log_out, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
err = os.open(log_err, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
pid = os.getpid()
with open(pidfile, "w", encoding="utf-8") as handle:
    handle.write(f"{pid}\n")
os.write(w, str(pid).encode())
os.close(w)
os.dup2(devnull, 0)
os.dup2(out, 1)
os.dup2(err, 2)
for fd in (devnull, out, err):
    if fd > 2:
        os.close(fd)
os.execve(python, [python, "server.py"], os.environ)
PY
}

echo "restarting gateway on :$MANTIS_ROUTER_PORT"
_stop_gateway
started="$(_daemonize_server)"
printf '%s\n' "$started" > "$ROUTER_LOG_DIR/server.pid"

for _ in $(seq 1 600); do  # 60s — first boot downloads the Supra checkpoint
  if lsof -nP -iTCP:"$MANTIS_ROUTER_PORT" -sTCP:LISTEN >/dev/null 2>&1; then
    if curl -fsS --max-time 2 "http://127.0.0.1:$MANTIS_ROUTER_PORT/healthz" 2>/dev/null | grep -q '"ready":true'; then
      ROUTER_PID=$(lsof -nP -iTCP:"$MANTIS_ROUTER_PORT" -sTCP:LISTEN -t 2>/dev/null | head -n1 || true)
      echo "gateway running pid ${ROUTER_PID:-$started} on :$MANTIS_ROUTER_PORT"
      exit 0
    fi
  fi
  sleep 0.1
done

echo "ERROR: gateway not running on :$MANTIS_ROUTER_PORT — see $ROUTER_LOG_DIR/server.err" >&2
cat "$ROUTER_LOG_DIR/server.err" >&2 2>/dev/null || true
exit 1
