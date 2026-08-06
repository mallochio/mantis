#!/usr/bin/env bash
set -euo pipefail

if [[ -f .env ]]; then
    set -a; source .env; set +a
fi

TOKEN="${MANTIS_API_KEY:-}"
[[ -n "$TOKEN" ]] || { echo "ERROR: set MANTIS_API_KEY" >&2; exit 1; }
BASE="${MANTIS_URL:-http://127.0.0.1:8088/v1}"

echo "1/4 health..."
curl -fsS "${BASE%/v1}/health" | grep -q '"status": "ok"\|"status":"ok"'
echo "  OK"

echo "2/4 readiness..."
curl -fsS "${BASE%/v1}/ready" | grep -q '"status": "ready"\|"status":"ready"'
echo "  OK"

echo "3/4 models..."
curl -fsS "$BASE/models" -H "Authorization: Bearer $TOKEN" \
  | grep -q 'mantis-ultra'
echo "  OK"

echo "4/4 chat completions..."
curl -fsS "$BASE/chat/completions" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"model":"mantis","messages":[{"role":"user","content":"Reply with exactly: 4"}]}' \
  | grep -q '"object": "chat.completion"\|"object":"chat.completion"'
echo "  OK"

echo "ALL CHECKS PASSED"
