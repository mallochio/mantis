#!/usr/bin/env bash
set -euo pipefail

if [[ -f .env ]]; then
    set -a; source .env; set +a
fi

TOKEN="${MANTIS_API_KEY:-}"
[[ -n "$TOKEN" ]] || { echo "ERROR: set MANTIS_API_KEY" >&2; exit 1; }
# MANTIS_URL may be given with or without the /v1 suffix; normalize both.
BASE="${MANTIS_URL:-http://127.0.0.1:8088}"
BASE="${BASE%/}"
BASE="${BASE%/v1}"
API="$BASE/v1"

echo "1/4 health..."
out=$(curl -fsS "$BASE/health")
grep -q '"status": "ok"\|"status":"ok"' <<<"$out"
echo "  OK"

echo "2/4 readiness..."
out=$(curl -fsS "$BASE/ready")
grep -q '"status": "ready"\|"status":"ready"' <<<"$out"
echo "  OK"

echo "3/4 models..."
out=$(curl -fsS "$API/models" -H "Authorization: Bearer $TOKEN")
grep -q 'mantis-ultra' <<<"$out"
echo "  OK"

echo "4/4 chat completions..."
out=$(curl -fsS "$API/chat/completions" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"model":"mantis","messages":[{"role":"user","content":"Reply with exactly: 4"}]}')
grep -q '"object": "chat.completion"\|"object":"chat.completion"' <<<"$out"
echo "  OK"

echo "ALL CHECKS PASSED"
