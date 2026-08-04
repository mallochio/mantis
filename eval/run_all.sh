#!/usr/bin/env bash
set -euo pipefail
# Run the full N=16 comparative eval.
# Restarts the openfugu container with the right Conductor env for each
# config block. Safe to re-run: it truncates eval/results.jsonl first.
#
# Conductor-luna uses the LiteLLM planner (gpt-5.6-luna-max) instead of a
# local Llama-3.2-3B checkpoint. Set MANTIS_LOCAL_CONDUCTOR to empty so the
# orchestrator falls back to the hosted planner model.

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# Load user secrets / env overrides.
if [ -f .env ]; then
  set -a
  source .env
  set +a
fi

export DIRECT_MODEL="${DIRECT_MODEL:-gpt-5.6-luna-max}"
export MANTIS_WORKER_MODELS="${MANTIS_WORKER_MODELS:-gemini-3.6-flash-high,gpt-5.6-luna-max,gpt-5.6-sol-medium,deepseek-v4-flash-0731-xhigh,claude-opus-5-medium,claude-sonnet-5-medium,gemini-3.1-pro-preview-high}"

rm -f eval/results.jsonl eval/results-scored.jsonl eval/report.md

wait_for() {
  local url="$1"
  local max_wait="${2:-120}"
  echo "[run_all] waiting for $url..."
  for i in $(seq 1 "$max_wait"); do
    if curl -fsS "$url" >/dev/null 2>&1; then
      echo "[run_all] $url ready"
      return 0
    fi
    sleep 2
  done
  echo "[run_all] $url did not become healthy"
  return 1
}

echo "[run_all] starting default stack for direct + trinity..."
docker compose up -d --build
wait_for http://localhost:3001/health/liveliness 120
wait_for http://localhost:8088/health 180

# The llm-router loads a 51M model on first request; wait until the header is present.
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

echo "[run_all] direct"
python3 eval/run_eval.py --config direct --fixtures eval/fixtures.jsonl --output eval/results.jsonl --timeout 300

echo "[run_all] trinity"
python3 eval/run_eval.py --config trinity --fixtures eval/fixtures.jsonl --output eval/results.jsonl --timeout 300

echo "[run_all] restarting openfugu for conductor-old..."
MANTIS_LOCAL_CONDUCTOR=di-zhang-fdu/openfugu-conductor-3b \
MANTIS_CONDUCTOR_MODEL=gpt-5.6-luna-max \
MANTIS_CONDUCTOR_DTYPE=float32 \
MANTIS_CONDUCTOR_MAX_NEW=128 \
docker compose up -d --force-recreate --no-deps openfugu

wait_for http://localhost:8088/health 180

echo "[run_all] conductor-old"
python3 eval/run_eval.py --config conductor-old --fixtures eval/fixtures.jsonl --output eval/results.jsonl --timeout 300

echo "[run_all] restarting openfugu for conductor-new..."
docker compose -f docker-compose.yml -f eval/docker-compose.conductor.yml up -d --force-recreate --no-deps openfugu

wait_for http://localhost:8088/health 180

echo "[run_all] conductor-new"
python3 eval/run_eval.py --config conductor-new --fixtures eval/fixtures.jsonl --output eval/results.jsonl --timeout 300

echo "[run_all] restarting openfugu for conductor-luna (LiteLLM planner)..."
MANTIS_LOCAL_CONDUCTOR= \
MANTIS_CONDUCTOR_MODEL=gpt-5.6-luna-max \
MANTIS_CONDUCTOR_DTYPE=float32 \
MANTIS_CONDUCTOR_MAX_NEW=128 \
docker compose up -d --force-recreate --no-deps openfugu

wait_for http://localhost:8088/health 180

echo "[run_all] conductor-luna"
python3 eval/run_eval.py --config conductor-luna --fixtures eval/fixtures.jsonl --output eval/results-luna.jsonl --timeout 300

echo "[run_all] scoring..."
python3 eval/score.py --fixtures eval/fixtures.jsonl --results eval/results.jsonl --output eval/report.md

echo "[run_all] generating conductor-luna report..."
python3 eval/report_luna.py

echo "[run_all] done. Reports: eval/report.md, eval/report-luna-conductor.md"
