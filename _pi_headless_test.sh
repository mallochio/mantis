#!/usr/bin/env bash
# Headless Pi test through the routellm provider.
# Captures pi stdout+stderr to a file and blocks until pi exits.
set -uo pipefail
OUT=/tmp/pi_route_final.out
cd "$HOME"
pi --provider routellm --model auto --no-tools --no-extensions --mode text \
   -p 'In one short sentence, what does the Python print function do?' \
   < /dev/null > "$OUT" 2>&1
rc=$?
echo "=== pi exit: $rc ==="
echo "=== pi output ==="
cat "$OUT"
echo "=== last router decision ==="
LOG="${MANTIS_DATA_DIR:-$HOME/.local/share/mantis}/router/decisions.log"
tail -1 "$LOG" 2>/dev/null