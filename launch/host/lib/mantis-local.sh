#!/usr/bin/env bash
# Start Mantis as a portable host process. No container runtime is required.
set -euo pipefail

export PATH="/opt/homebrew/bin:/opt/homebrew/sbin:$HOME/.local/bin:$PATH"
REPO="$HOME/Personal/other/mantis"
DATA_DIR="${MANTIS_DATA_DIR:-$HOME/.local/share/mantis}"
PIDFILE="$DATA_DIR/server.pid"
LOGFILE="$DATA_DIR/server.log"
READY_URL="http://127.0.0.1:8088/ready"
mkdir -p "$DATA_DIR"

if [[ -f "$PIDFILE" ]]; then
    pid=$(<"$PIDFILE")
    if kill -0 "$pid" 2>/dev/null; then
        kill "$pid"
        for _ in $(seq 1 50); do kill -0 "$pid" 2>/dev/null || break; sleep 0.1; done
        if kill -0 "$pid" 2>/dev/null; then
            echo "[mantis] previous process $pid did not stop; sending SIGKILL" >&2
            kill -9 "$pid" 2>/dev/null || true
            for _ in $(seq 1 50); do kill -0 "$pid" 2>/dev/null || break; sleep 0.1; done
        fi
    fi
    rm -f "$PIDFILE"
fi

# Login zsh is the canonical source for locally stored provider credentials.
# Detach into a new session so IDE/agent shell teardown cannot SIGKILL Mantis.
echo '[mantis] starting portable host process (catalog mode)...'
# shellcheck source=detach.sh
source "$(cd "$(dirname "$0")" && pwd)/detach.sh"
detach_cmd "$PIDFILE" "$LOGFILE" "$LOGFILE" \
  /bin/zsh -ic "cd '$REPO' && exec ./scripts/run_mantis_native.sh" \
  >/dev/null

echo '[mantis] checking local readiness...'
for _ in $(seq 1 180); do
    if curl -fsS --max-time 3 "$READY_URL" >/dev/null; then
        echo "[mantis] ready: $(curl -fsS --max-time 3 "$READY_URL")"
        exit 0
    fi
    sleep 1
done
echo "[mantis] startup failed; see $LOGFILE" >&2
exit 1
