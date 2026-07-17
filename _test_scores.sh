#!/usr/bin/env bash
# Score coding prompts through the mf router and print decisions.
set -uo pipefail
URL=http://127.0.0.1:5500/v1/chat/completions
KEY=sk-route-local
H=$(mktemp)

score_one() {
  local prompt="$1"
  resp=$(curl -sS -m 30 -D "$H" "$URL" \
    -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
    -d "$(jq -cn --arg p "$prompt" '{model:"auto",messages:[{role:"user",content:$p}],max_tokens:1,stream:false}')")
  score=$(grep -i '^x-route-score:' "$H" | tr -d '\r' | cut -d' ' -f2)
  decision=$(grep -i '^x-route-decision:' "$H" | tr -d '\r' | cut -d' ' -f2)
  printf "%-8s %-8s  %s\n" "$score" "$decision" "${prompt:0:75}"
}

echo "score    decision  prompt"
echo "-------  --------  ------"
score_one "Write hello world in Python"
score_one "What does the print function do"
score_one "Fix the typo: prnit hello"
score_one "Write a Python function to reverse a string"
score_one "Refactor this pymongo query to avoid N+1 by adding a covering compound index, and explain the query planner interaction with the ESR rule"
score_one "Implement Paxos with byzantine fault tolerance and prove liveness and safety"
score_one "Design a distributed lock service tolerating network partitions, compare Paxos vs Raft, and prove correctness"
score_one "Write a one-line Python program that prints hello world"
rm -f "$H"