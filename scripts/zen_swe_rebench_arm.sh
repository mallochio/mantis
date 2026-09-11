#!/usr/bin/env bash
# Zen-only SWE-rebench Phase 2 worker: run ONE fixed model arm across the full
# 40-task manifest. Each arm is its own SkyPilot cluster; results fan in to
# gs://your-eval-storage-bucket/<RUN_ID>/<ARM>/results.jsonl.
#
# Env:
#   ARM       (required) one of: deepseek mimo hy3 nemotron-ultra nemotron-lightning laguna
#   RUN_ID    (required) shared run prefix, e.g. job-<8hex>
#   MANIFEST  (optional) manifest path inside EVAL_ROOT, default eval/router_manifest.json
set -euo pipefail

ARM="${ARM:?ARM is required}"
RUN_ID="${RUN_ID:?RUN_ID is required}"
MANIFEST="${MANIFEST:-eval/router_manifest.json}"

case "$ARM" in
  deepseek)          MODEL="opencode-zen/deepseek-v4-flash-free" ;;
  mimo)              MODEL="opencode-zen/mimo-v2.5-free" ;;
  hy3)               MODEL="opencode-zen/hy3-free" ;;
  nemotron-ultra)    MODEL="opencode-zen/nemotron-3-ultra-free" ;;
  nemotron-lightning) MODEL="opencode-zen/nemotron-3.5-lightning-free" ;;
  laguna)            MODEL="opencode-zen/laguna-s-2.1-free" ;;
  *) echo "unknown ARM: $ARM" >&2; exit 2 ;;
esac

TS="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
EVAL_ROOT="/opt/mantis-eval"
RESULT_DIR="/tmp/zen-phase2-results"
LOG_DIR="${RESULT_DIR}/logs"
LITELLM_DIR="${LITELLM_DIR:-$HOME/.litellm-data}"
OUT_FILE="${RESULT_DIR}/results.jsonl"
mkdir -p "$LOG_DIR" "$LITELLM_DIR"
chmod 700 "$LITELLM_DIR" "$RESULT_DIR"

export EVAL_PROXY_RETRIES="${EVAL_PROXY_RETRIES:-8}"
export EVAL_PROXY_RETRY_BASE="${EVAL_PROXY_RETRY_BASE:-1.0}"
export EVAL_PROXY_RETRY_MAX_WAIT="${EVAL_PROXY_RETRY_MAX_WAIT:-60.0}"
export EVAL_PROXY_RETRY_ON_STATUS="${EVAL_PROXY_RETRY_ON_STATUS:-429,500,502,503,504}"

BUCKET_PREFIX="${GCS_EVAL_BUCKET:-gs://your-eval-storage-bucket}/${RUN_ID}/${ARM}"

fail() { echo "PHASE2_FAIL[$ARM]: $*" >&2; exit 1; }

if [ -z "${OPENCODE_API_KEY:-}" ]; then fail "OPENCODE_API_KEY not set"; fi
if [ -z "${LITELLM_API_KEY:-}" ]; then fail "LITELLM_API_KEY not set"; fi

# --- Zen-only LiteLLM proxy config (6 viable models; ling dropped) -------------------
cat > "$LITELLM_DIR/config.yaml" <<'EOF'
model_list:
  - model_name: "opencode-zen/deepseek-v4-flash-free"
    litellm_params:
      model: "openai/deepseek-v4-flash-free"
      api_base: "https://opencode.ai/zen/v1"
      api_key: "os.environ/OPENCODE_API_KEY"
  - model_name: "opencode-zen/mimo-v2.5-free"
    litellm_params:
      model: "openai/mimo-v2.5-free"
      api_base: "https://opencode.ai/zen/v1"
      api_key: "os.environ/OPENCODE_API_KEY"
  - model_name: "opencode-zen/hy3-free"
    litellm_params:
      model: "openai/hy3-free"
      api_base: "https://opencode.ai/zen/v1"
      api_key: "os.environ/OPENCODE_API_KEY"
  - model_name: "opencode-zen/nemotron-3-ultra-free"
    litellm_params:
      model: "openai/nemotron-3-ultra-free"
      api_base: "https://opencode.ai/zen/v1"
      api_key: "os.environ/OPENCODE_API_KEY"
  - model_name: "opencode-zen/nemotron-3.5-lightning-free"
    litellm_params:
      model: "openai/nemotron-3.5-lightning-free"
      api_base: "https://opencode.ai/zen/v1"
      api_key: "os.environ/OPENCODE_API_KEY"
  - model_name: "opencode-zen/laguna-s-2.1-free"
    litellm_params:
      model: "openai/laguna-s-2.1-free"
      api_base: "https://opencode.ai/zen/v1"
      api_key: "os.environ/OPENCODE_API_KEY"
general_settings:
  master_key: "os.environ/LITELLM_API_KEY"
litellm_settings:
  drop_params: true
EOF
chmod 600 "$LITELLM_DIR/config.yaml"

# --- Start LiteLLM proxy -------------------------------------------------------------
nohup litellm --config "$LITELLM_DIR/config.yaml" --host 127.0.0.1 --port 8080 \
  </dev/null >> "$LOG_DIR/litellm.out" 2>> "$LOG_DIR/litellm.err" &
echo "$!" > "$LITELLM_DIR/litellm.pid"

ready=0
for _ in $(seq 1 90); do
  if curl -sf -m 3 "http://127.0.0.1:8080/v1/models" -H "Authorization: Bearer ${LITELLM_API_KEY}" -o "$LOG_DIR/models.json" 2>/dev/null; then
    ready=1
    break
  fi
  sleep 2
done
[ "$ready" = "1" ] || fail "LiteLLM did not become ready; see $LOG_DIR/litellm.err"

python3 - "$LOG_DIR/models.json" <<'PY'
import json, sys
expected = {
    "opencode-zen/deepseek-v4-flash-free",
    "opencode-zen/mimo-v2.5-free",
    "opencode-zen/hy3-free",
    "opencode-zen/nemotron-3-ultra-free",
    "opencode-zen/nemotron-3.5-lightning-free",
    "opencode-zen/laguna-s-2.1-free",
}
data = json.load(open(sys.argv[1]))
ids = {m["id"] for m in data.get("data", [])}
if not expected <= ids:
    print(f"PHASE2_FAIL: allowlist missing {sorted(expected - ids)}")
    sys.exit(1)
print("model allowlist verified:", len(ids), "models")
PY

# --- Resume previous progress from the bucket ---------------------------------------
if gcloud storage ls "${BUCKET_PREFIX}/results.jsonl" >/dev/null 2>&1; then
  gcloud storage cp "${BUCKET_PREFIX}/results.jsonl" "$OUT_FILE" --quiet
  echo "resumed $(wc -l < "$OUT_FILE") result lines from ${BUCKET_PREFIX}/results.jsonl"
else
  echo "fresh arm run (no prior results at ${BUCKET_PREFIX}/results.jsonl)"
fi

# --- Heartbeat upload ---------------------------------------------------------------
(
  while true; do
    sleep 60
    [ -f "$OUT_FILE" ] && gcloud storage cp "$OUT_FILE" "${BUCKET_PREFIX}/results.jsonl" --quiet >/dev/null 2>&1 || true
  done
) &
HEARTBEAT_PID=$!
trap 'kill "$HEARTBEAT_PID" 2>/dev/null || true' EXIT

# --- Run the 40-task fixed-arm rollout ----------------------------------------------
cd "$EVAL_ROOT"
set +e
python3 eval/router_eval.py \
  --manifest "$MANIFEST" \
  --prices eval/model_prices.json \
  --shadow-prices eval/prices/zen-2026-08-13.json \
  --cost-mode shadow \
  --litellm-endpoint http://127.0.0.1:8080/v1/chat/completions \
  --tier-models cheap=opencode-zen/deepseek-v4-flash-free,middle=opencode-zen/hy3-free,expensive=opencode-zen/nemotron-3-ultra-free \
  --fixed-model "$ARM=$MODEL" \
  --arms "$ARM" \
  --per-instance-cost 5.0 \
  --resume \
  --timeout 1200 \
  --output-token-limit 4096 \
  --worktrees-root /tmp/worktrees \
  --keep-worktrees \
  --pi-executable pi \
  --output "$OUT_FILE" > "$LOG_DIR/arm.out" 2> "$LOG_DIR/arm.err"
ARM_RC=$?
set -e

kill "$HEARTBEAT_PID" 2>/dev/null || true
trap - EXIT

python3 - "$ARM" "$RUN_ID" "$TS" "$ARM_RC" "$OUT_FILE" > "$RESULT_DIR/arm-metadata.json" <<'PY'
import json, os, sys
arm, run_id, ts, rc, out_path = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
lines = [l for l in open(out_path) if l.strip()] if os.path.exists(out_path) else []
results = [json.loads(l) for l in lines if json.loads(l).get("record_type") == "result"]
resolved = sum(1 for r in results if r.get("resolved"))
json.dump({
    "arm": arm, "run_id": run_id, "phase": "p2-tier-selection", "timestamp": ts,
    "bucket": os.environ.get("GCS_EVAL_BUCKET", "gs://your-eval-storage-bucket"), "arm_exit_code": rc,
    "result_rows": len(results), "resolved_rows": resolved,
    "rate_limit_policy": {
        "retries": os.environ.get("EVAL_PROXY_RETRIES"),
        "base_s": os.environ.get("EVAL_PROXY_RETRY_BASE"),
        "max_wait_s": os.environ.get("EVAL_PROXY_RETRY_MAX_WAIT"),
        "statuses": os.environ.get("EVAL_PROXY_RETRY_ON_STATUS"),
    },
}, sys.stdout, indent=2)
PY

gcloud storage cp "$RESULT_DIR/arm-metadata.json" "${BUCKET_PREFIX}/arm-metadata.json" --quiet || true
gcloud storage cp "$OUT_FILE" "${BUCKET_PREFIX}/results.jsonl" --quiet || true
gcloud storage cp -r "$LOG_DIR" "${BUCKET_PREFIX}/logs" --quiet >/dev/null 2>&1 || true

echo "=== RESULTS_BEGIN ${RUN_ID}/${ARM} ==="
cat "$RESULT_DIR/arm-metadata.json"
echo "=== RESULTS_END ${RUN_ID}/${ARM} ==="
echo "ARM_EXIT_CODE=$ARM_RC"
echo "UPLOAD_PREFIX ${BUCKET_PREFIX}"
exit "$ARM_RC"
