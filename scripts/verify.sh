#!/usr/bin/env bash
set -euo pipefail

if [[ -f .env ]]; then
    # shellcheck source=/dev/null
    set -a; source .env; set +a
fi

TOKEN="${FUGU_API_KEY:-${LITELLM_KEY:-}}"
if [[ -z "$TOKEN" ]]; then
    echo "ERROR: set FUGU_API_KEY or LITELLM_KEY in .env" >&2
    exit 1
fi

echo "1/4 litellm..."
curl -sf http://127.0.0.1:3001/health/liveliness >/dev/null && echo "  OK"

echo "2/4 router responds + supra header..."
HEADERS=$(curl -sD - -o /dev/null http://127.0.0.1:5500/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"model":"auto","messages":[{"role":"user","content":"hi"}],"max_tokens":1}')
echo "$HEADERS" | grep -i "x-route-supra-complexity" && echo "  OK"

echo "3/4 openfugu trinity..."
R=$(curl -sf http://127.0.0.1:8088/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"model":"trinity","messages":[{"role":"user","content":"Reply with exactly: 4"}],"max_tokens":8}')
echo "$R" | grep -q "4" && echo "  OK"

echo "4/4 openfugu conductor routes..."
R2=$(curl -sf http://127.0.0.1:8088/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"model":"conductor","messages":[{"role":"user","content":"Reply with exactly: 4"}],"max_tokens":8}')
echo "$R2" | grep -q "4" && echo "  OK"

echo "ALL CHECKS PASSED"
