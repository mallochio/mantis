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
    -H "X-Route-Session: $SESSION" \
    -d "$(jq -cn --arg p "$prompt" '{model:"auto",messages:[{role:"user",content:$p}],max_tokens:24,stream:false}')")
  echo "x-route-decision: $(grep -i '^x-route-decision:' "$H" | tr -d '\r' | cut -d' ' -f2)"
  echo "x-route-score:    $(grep -i '^x-route-score:'    "$H" | tr -d '\r' | cut -d' ' -f2)"
  echo "x-route-model:    $(grep -i '^x-route-model:'    "$H" | tr -d '\r' | cut -d' ' -f2)"
  echo "downstream model:  $(echo "$resp" | jq -r '.model // .error.message' 2>/dev/null)"
  echo "reply:             $(echo "$resp" | jq -r '.choices[0].message.content // .error' 2>/dev/null | head -c 120)"
  echo
}
H=$(mktemp)
SESSION="headless-$(date +%s)"
route_one "EASY (cheap/middle/expensive tier)"  "Write hello world in Python, one line"
route_one "HARD (expensive tier)"    "Implement Paxos with byzantine fault tolerance and prove liveness and safety; include the formal invariant argument and the fail-stop vs fail-beautiful distinction"
route_one "MEDIUM (middle tier)"     "Refactor this pymongo query to avoid N+1 by adding a covering compound index, and explain the query planner interaction with the ESR rule"
# A short continuation should report the same session affinity decision.
route_one "CONTINUATION (sticky session)" "Proceed"
echo "=== RESPONSES (explicit endpoint) ==="
curl -sS -m 90 -D "$H" "http://127.0.0.1:5500/v1/responses" \
  -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  -H "X-Route-Session: $SESSION-responses" \
  -d '{"model":"auto","input":"Reply with exactly OK","max_output_tokens":24,"stream":false}' \
  | jq -c '{id,object,status,error}'
rm -f "$H"
LOG="${MANTIS_DATA_DIR:-$HOME/.local/share/mantis}/router/decisions.log"
tail -5 "$LOG" 2>/dev/null | jq -c '{decision, score, model, ttfb_ms, prompt}' 2>/dev/null || tail -5 "$LOG"