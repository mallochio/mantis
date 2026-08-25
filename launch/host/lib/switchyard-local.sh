#!/usr/bin/env bash
# Canonical launch script for NVIDIA NeMo Switchyard (mantis/base).
#
# Regenerates routes.toml from the shared catalog, then starts
# switchyard-server on loopback. Changing [base] models or providers in the
# catalog is enough to retarget Base; this script does not hard-code them.
set -euo pipefail

SELF="$0"
while [ -L "$SELF" ]; do
  LINK_DIR="$(cd -P "$(dirname "$SELF")" && pwd)"
  SELF="$(readlink "$SELF")"
  [[ "$SELF" = /* ]] || SELF="$LINK_DIR/$SELF"
done
LIB_DIR="$(cd "$(dirname "$SELF")" && pwd)"
REPO_ROOT="$(cd "$LIB_DIR/../../.." && pwd)"

export PATH="/opt/homebrew/bin:/opt/homebrew/sbin:$HOME/.cargo/bin:$HOME/.local/bin:$PATH"

ZSHRC_IMPORT_TIMEOUT_S="${ZSHRC_IMPORT_TIMEOUT_S:-8}"

_import_zshrc_env() {
  [ -f "$HOME/.zshrc" ] || return 0
  local dump item name value
  dump="$(mktemp -t switchyard-env)"
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
export PATH="/opt/homebrew/bin:/opt/homebrew/sbin:$HOME/.cargo/bin:$HOME/.local/bin:$PATH"

DATA_DIR="${MANTIS_DATA_DIR:-$HOME/.local/share/mantis}"
LOG_DIR="$DATA_DIR/switchyard"
CONFIG_OUT="${SWITCHYARD_CONFIG:-$LOG_DIR/routes.toml}"
CATALOG="${AI_ROUTING_CONFIG:-$HOME/.config/ai-routing/catalog.toml}"
HOST="${SWITCHYARD_HOST:-127.0.0.1}"
PORT="${SWITCHYARD_PORT:-5500}"
mkdir -p "$LOG_DIR"
chmod 700 "$DATA_DIR" "$LOG_DIR"

if [ ! -f "$CATALOG" ]; then
  echo "ERROR: routing catalog not found: $CATALOG" >&2
  exit 1
fi

if [ -n "${SWITCHYARD_BIN:-}" ]; then
  :
elif command -v switchyard-server >/dev/null 2>&1; then
  SWITCHYARD_BIN="$(command -v switchyard-server)"
elif [ -x "$HOME/.cargo/bin/switchyard-server" ]; then
  SWITCHYARD_BIN="$HOME/.cargo/bin/switchyard-server"
else
  echo "ERROR: switchyard-server not found." >&2
  echo "Install with: cargo install --locked switchyard-server" >&2
  exit 1
fi

if [ "$HOST" != "127.0.0.1" ] && [ "$HOST" != "::1" ] && [ "$HOST" != "localhost" ]; then
  echo "ERROR: Switchyard must bind loopback (got host $HOST)." >&2
  exit 1
fi

export PYTHONPATH="$REPO_ROOT/scripts${PYTHONPATH:+:$PYTHONPATH}"
if ! python3 "$REPO_ROOT/scripts/switchyard_config.py" render --catalog "$CATALOG" --output "$CONFIG_OUT"; then
  echo "ERROR: failed to render Switchyard config from $CATALOG" >&2
  exit 1
fi
if ! "$SWITCHYARD_BIN" --config "$CONFIG_OUT" --dry-run; then
  echo "ERROR: switchyard-server rejected $CONFIG_OUT" >&2
  exit 1
fi

_is_self() {
  [ "$1" = "$$" ] || [ "$1" = "$PPID" ]
}

_kill_pid() {
  local sig="$1" pid="$2"
  [ -n "$pid" ] || return 0
  _is_self "$pid" && return 0
  kill -"$sig" "$pid" 2>/dev/null || true
}

_wait_port_clear() {
  local i
  for i in $(seq 1 50); do
    lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1 || return 0
    sleep 0.1
  done
  return 1
}

_stop_switchyard() {
  local pid
  echo "stopping Switchyard on :$PORT (if any)"
  if [ -f "$LOG_DIR/server.pid" ]; then
    _kill_pid TERM "$(cat "$LOG_DIR/server.pid" 2>/dev/null || true)"
  fi
  for pid in $(lsof -nP -iTCP:"$PORT" -sTCP:LISTEN -t 2>/dev/null); do
    _kill_pid TERM "$pid"
  done
  _wait_port_clear || true
  for pid in $(lsof -nP -iTCP:"$PORT" -sTCP:LISTEN -t 2>/dev/null); do
    _kill_pid KILL "$pid"
  done
  _wait_port_clear || true
  rm -f "$LOG_DIR/server.pid"
}

_daemonize_server() {
  /usr/bin/python3 - "$SWITCHYARD_BIN" "$CONFIG_OUT" "$HOST" "$PORT" \
    "$LOG_DIR/server.out" "$LOG_DIR/server.err" \
    "$LOG_DIR/server.pid" <<'PY'
import os, sys

binary, config, host, port, log_out, log_err, pidfile = sys.argv[1:8]
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
os.execve(
    binary,
    [binary, "--config", config, "--host", host, "--port", port],
    os.environ,
)
PY
}

echo "restarting Switchyard on :$PORT"
_stop_switchyard
started="$(_daemonize_server)"
printf '%s\n' "$started" > "$LOG_DIR/server.pid"

for _ in $(seq 1 150); do
  if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
    if curl -fsS --max-time 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
      ROUTER_PID=$(lsof -nP -iTCP:"$PORT" -sTCP:LISTEN -t 2>/dev/null | head -n1 || true)
      echo "Switchyard running pid ${ROUTER_PID:-$started} on :$PORT"
      exit 0
    fi
  fi
  sleep 0.1
done

echo "ERROR: Switchyard not running on :$PORT — see $LOG_DIR/server.err" >&2
cat "$LOG_DIR/server.err" >&2 2>/dev/null || true
exit 1
