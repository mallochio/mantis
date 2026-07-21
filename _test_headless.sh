#!/usr/bin/env bash
# Headless routing self-test against the local server.
set -uo pipefail
URL=http://127.0.0.1:5500/v1/chat/completions
KEY=sk-route-local
route_one() {
  local label="$1"; local prompt="$2"; shift 2
  echo "=== $label ==="
  echo "prompt: $prompt"
  resp=$(curl -sS -m 90 -D "$H" "$URL" \
    -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
    -d "$(jq -cn --arg p "$prompt" '{model:"auto",messages:[{role:"user",content:$p}],max_tokens:24,stream:false}')")
  echo "x-route-decision: $(grep -i '^x-route-decision:' "$H" | tr -d '\r' | cut -d' ' -f2)"
  echo "x-route-score:    $(grep -i '^x-route-score:'    "$H" | tr -d '\r' | cut -d' ' -f2)"
  echo "x-route-model:    $(grep -i '^x-route-model:'    "$H" | tr -d '\r' | cut -d' ' -f2)"
  echo "downstream model:  $(echo "$resp" | jq -r '.model // .error.message' 2>/dev/null)"
  echo "reply:             $(echo "$resp" | jq -r '.choices[0].message.content // .error' 2>/dev/null | head -c 120)"
  echo
}
H=$(mktemp)
route_one "EASY (expect deepseek-v4-pro)"  "Write hello world in Python, one line"
route_one "HARD (expect luna)"    "Implement Paxos with byzantine fault tolerance and prove liveness and safety; include the formal invariant argument and the fail-stop vs fail-beautiful distinction"
route_one "MEDIUM (refactor)"     "Refactor this pymongo query to avoid N+1 by adding a covering compound index, and explain the query planner interaction with the ESR rule"
rm -f "$H"
echo "=== decision log tail ==="
tail -5 ~/.config/llm-router/logs/decisions.log 2>/dev/null | jq -c '{decision, score, model, ttfb_ms, prompt}' 2>/dev/null || tail -5 ~/.config/llm-router/logs/decisions.log