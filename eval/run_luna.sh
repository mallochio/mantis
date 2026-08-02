#!/usr/bin/env bash
set -euo pipefail
# Run the 16-fixture conductor-luna eval.
# Uses the LiteLLM planner (gpt-5.6-luna-max) instead of a local 3B model.
# Direct/trinity baselines are reused from eval/results-native-v2-scored.jsonl.

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

if [ -f .env ]; then
  set -a
  source .env
  set +a
fi

export DIRECT_MODEL="${DIRECT_MODEL:-gpt-5.6-luna-max}"
export FUGU_WORKER_MODELS="${FUGU_WORKER_MODELS:-gemini-3.6-flash-high,gpt-5.6-luna-max,gpt-5.6-sol-medium,deepseek-v4-flash-0731-xhigh,claude-opus-5-medium,claude-sonnet-5-medium,gemini-3.1-pro-preview-high}"

# Force the LiteLLM planner and clear any local Conductor checkpoint.
export FUGU_LOCAL_CONDUCTOR=""
export FUGU_CONDUCTOR_MODEL="${FUGU_CONDUCTOR_MODEL:-gpt-5.6-luna-max}"

wait_for() {
  local url="$1"
  local max_wait="${2:-120}"
  echo "[run_luna] waiting for $url..."
  for i in $(seq 1 "$max_wait"); do
    if curl -fsS "$url" >/dev/null 2>&1; then
      echo "[run_luna] $url ready"
      return 0
    fi
    sleep 2
  done
  echo "[run_luna] $url did not become healthy"
  return 1
}

echo "[run_luna] starting full Docker stack with LiteLLM conductor planner..."
docker compose up -d --force-recreate
wait_for http://localhost:3001/health/liveliness 120
wait_for http://localhost:8088/health 180

# Make sure the router is emitting Supra headers.
for i in $(seq 1 120); do
  if curl -sD - -o /dev/null http://127.0.0.1:5500/v1/chat/completions \
       -H "Content-Type: application/json" \
       -H "Authorization: Bearer ${LITELLM_KEY:-sk-mantis}" \
       -d '{"model":"auto","messages":[{"role":"user","content":"hi"}],"max_tokens":1}' 2>/dev/null | grep -qi "x-route-supra-complexity"; then
    break
  fi
  sleep 2
done

bash scripts/verify.sh

rm -f eval/results-luna.jsonl
echo "[run_luna] conductor-luna"
python3 eval/run_eval.py --config conductor-luna --fixtures eval/fixtures.jsonl --output eval/results-luna.jsonl --timeout 300

echo "[run_luna] scoring and writing report..."
python3 eval/report_luna.py

echo "[run_luna] done. Report: eval/report-luna-conductor.md"
