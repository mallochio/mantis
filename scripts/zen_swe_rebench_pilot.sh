#!/usr/bin/env bash
# Zen-only SWE-rebench Phase 1 pilot: 12 tasks x 7 fixed Zen arms = 84 rollouts.
#
# Starts a private Zen-only Bifrost on 127.0.0.1:8080 (same worker), then runs
# eval/router_eval.py in fixed-arm shadow-cost mode with resume. A heartbeat
# loop uploads the incremental JSONL to gs://ih-storage-sid/<JOB_ID>/ so a
# spot preemption loses at most one heartbeat interval of work.
set -euo pipefail

JOB_ID="${JOB_ID:-job-$(openssl rand -hex 4 | tr -d '\n')}"
TS="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

EVAL_ROOT="/opt/mantis-eval"
RESULT_DIR="/tmp/zen-pilot-results"
LOG_DIR="${RESULT_DIR}/logs"
BIFROST_DIR="${BIFROST_DIR:-$HOME/.bifrost-data}"
PILOT_OUT="${RESULT_DIR}/pilot.jsonl"
mkdir -p "$LOG_DIR" "$BIFROST_DIR"
chmod 700 "$BIFROST_DIR" "$RESULT_DIR"

export EVAL_PROXY_RETRIES="${EVAL_PROXY_RETRIES:-8}"
export EVAL_PROXY_RETRY_BASE="${EVAL_PROXY_RETRY_BASE:-1.0}"
export EVAL_PROXY_RETRY_MAX_WAIT="${EVAL_PROXY_RETRY_MAX_WAIT:-60.0}"
export EVAL_PROXY_RETRY_ON_STATUS="${EVAL_PROXY_RETRY_ON_STATUS:-429,500,502,503,504}"

BUCKET_PREFIX="gs://ih-storage-sid/${JOB_ID}"

fail() { echo "PILOT_FAIL: $*" >&2; exit 1; }

if [ -z "${OPENCODE_API_KEY:-}" ]; then fail "OPENCODE_API_KEY not set"; fi
if [ -z "${BIFROST_API_KEY:-}" ]; then fail "BIFROST_API_KEY not set"; fi
if [ -z "${BIFROST_ENCRYPTION_KEY:-}" ]; then fail "BIFROST_ENCRYPTION_KEY not set"; fi

# --- Zen-only Bifrost config -------------------------------------------------------
cat > "$BIFROST_DIR/config.json" <<'EOF'
{
  "encryption_key": "env.BIFROST_ENCRYPTION_KEY",
  "client": {
    "enforce_auth_on_inference": true,
    "enable_logging": false,
    "initial_pool_size": 50,
    "allowed_origins": ["http://127.0.0.1:8080", "http://localhost:8080"],
    "disable_content_logging": true
  },
  "governance": {
    "virtual_keys": [
      {
        "id": "vk-zen-swe-rebench",
        "name": "zen swe-rebench benchmark",
        "value": "env.BIFROST_API_KEY",
        "is_active": true,
        "provider_configs": [
          {
            "provider": "opencode-zen",
            "weight": 1.0,
            "allowed_models": [
              "deepseek-v4-flash-free",
              "mimo-v2.5-free",
              "hy3-free",
              "ling-3.0-tiny-free",
              "nemotron-3-ultra-free",
              "nemotron-3.5-lightning-free",
              "laguna-s-2.1-free"
            ],
            "key_ids": ["*"]
          }
        ]
      }
    ]
  },
  "providers": {
    "opencode-zen": {
      "keys": [
        {
          "name": "opencode-zen-free-benchmark",
          "value": "env.OPENCODE_API_KEY",
          "models": [
            "deepseek-v4-flash-free",
            "mimo-v2.5-free",
            "hy3-free",
            "ling-3.0-tiny-free",
            "nemotron-3-ultra-free",
            "nemotron-3.5-lightning-free",
            "laguna-s-2.1-free"
          ],
          "weight": 1.0
        }
      ],
      "store_raw_request_response": false
    }
  },
  "source_of_truth": "config.json"
}
EOF
chmod 600 "$BIFROST_DIR/config.json"

# --- Start Bifrost ------------------------------------------------------------------
nohup npx -y @maximhq/bifrost -app-dir "$BIFROST_DIR" -host 127.0.0.1 -port 8080 -log-style pretty \
  </dev/null >> "$LOG_DIR/bifrost.out" 2>> "$LOG_DIR/bifrost.err" &
echo "$!" > "$BIFROST_DIR/bifrost.pid"

ready=0
for _ in $(seq 1 90); do
  if curl -sf -m 3 "http://127.0.0.1:8080/v1/models" -H "x-bf-vk: ${BIFROST_API_KEY}" -o "$LOG_DIR/models.json" 2>/dev/null; then
    ready=1
    break
  fi
  sleep 2
done
[ "$ready" = "1" ] || fail "Bifrost did not become ready; see $LOG_DIR/bifrost.err"

python3 - "$LOG_DIR/models.json" <<'PY'
import json, sys
expected = {
    "opencode-zen/deepseek-v4-flash-free",
    "opencode-zen/mimo-v2.5-free",
    "opencode-zen/hy3-free",
    "opencode-zen/ling-3.0-tiny-free",
    "opencode-zen/nemotron-3-ultra-free",
    "opencode-zen/nemotron-3.5-lightning-free",
    "opencode-zen/laguna-s-2.1-free",
}
data = json.load(open(sys.argv[1]))
ids = {m["id"] for m in data.get("data", [])}
missing = sorted(expected - ids)
extra = sorted(ids - expected)
if missing or extra:
    print(f"PILOT_FAIL: model allowlist mismatch missing={missing} extra={extra}")
    sys.exit(1)
print("model allowlist verified:", len(ids), "models")
PY

# --- Resume previous progress from the bucket (upload-only; never delete/move) -------
if gcloud storage ls "${BUCKET_PREFIX}/pilot.jsonl" >/dev/null 2>&1; then
  gcloud storage cp "${BUCKET_PREFIX}/pilot.jsonl" "$PILOT_OUT" --quiet
  echo "resumed $(wc -l < "$PILOT_OUT") result lines from ${BUCKET_PREFIX}/pilot.jsonl"
else
  echo "fresh pilot run (no prior results at ${BUCKET_PREFIX}/pilot.jsonl)"
fi

# --- Heartbeat upload of incremental results ----------------------------------------
(
  while true; do
    sleep 60
    if [ -f "$PILOT_OUT" ]; then
      gcloud storage cp "$PILOT_OUT" "${BUCKET_PREFIX}/pilot.jsonl" --quiet >/dev/null 2>&1 || true
    fi
    if [ -f "$LOG_DIR/bifrost.err" ]; then
      gcloud storage cp "$LOG_DIR/bifrost.err" "${BUCKET_PREFIX}/logs/bifrost.err" --quiet >/dev/null 2>&1 || true
    fi
  done
) &
HEARTBEAT_PID=$!
trap 'kill "$HEARTBEAT_PID" 2>/dev/null || true' EXIT

# --- Run the 84-rollout pilot ---------------------------------------------------------
cd "$EVAL_ROOT"
set +e
python3 eval/router_eval.py \
  --manifest eval/manifests/zen-pilot-12.json \
  --prices eval/model_prices.json \
  --shadow-prices eval/prices/zen-2026-08-13.json \
  --cost-mode shadow \
  --bifrost-endpoint http://127.0.0.1:8080/v1/chat/completions \
  --fixed-model deepseek=opencode-zen/deepseek-v4-flash-free \
  --fixed-model mimo=opencode-zen/mimo-v2.5-free \
  --fixed-model hy3=opencode-zen/hy3-free \
  --fixed-model ling=opencode-zen/ling-3.0-tiny-free \
  --fixed-model nemotron-ultra=opencode-zen/nemotron-3-ultra-free \
  --fixed-model nemotron-lightning=opencode-zen/nemotron-3.5-lightning-free \
  --fixed-model laguna=opencode-zen/laguna-s-2.1-free \
  --arms deepseek,mimo,hy3,ling,nemotron-ultra,nemotron-lightning,laguna \
  --per-instance-cost 5.0 \
  --resume \
  --timeout 1200 \
  --output-token-limit 4096 \
  --worktrees-root /tmp/worktrees \
  --keep-worktrees \
  --pi-executable pi \
  --output "$PILOT_OUT" > "$LOG_DIR/pilot.out" 2> "$LOG_DIR/pilot.err"
PILOT_RC=$?
set -e

kill "$HEARTBEAT_PID" 2>/dev/null || true
trap - EXIT

python3 - "$JOB_ID" "$TS" "$PILOT_RC" "$PILOT_OUT" > "$RESULT_DIR/pilot-metadata.json" <<'PY'
import json, os, sys
job_id, ts, rc, out_path = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
lines = [l for l in open(out_path) if l.strip()] if os.path.exists(out_path) else []
results = [json.loads(l) for l in lines if json.loads(l).get("record_type") == "result"]
resolved = sum(1 for r in results if r.get("resolved"))
json.dump({
    "job_id": job_id,
    "phase": "p1-pilot",
    "timestamp": ts,
    "bucket": "gs://ih-storage-sid",
    "pilot_exit_code": rc,
    "result_rows": len(results),
    "resolved_rows": resolved,
    "rate_limit_policy": {
        "retries": os.environ.get("EVAL_PROXY_RETRIES"),
        "base_s": os.environ.get("EVAL_PROXY_RETRY_BASE"),
        "max_wait_s": os.environ.get("EVAL_PROXY_RETRY_MAX_WAIT"),
        "statuses": os.environ.get("EVAL_PROXY_RETRY_ON_STATUS"),
    },
}, sys.stdout, indent=2)
PY

# --- Final upload (upload-only) ------------------------------------------------------
cp "$EVAL_ROOT/eval/manifests/zen-pilot-12.json" "$RESULT_DIR/zen-pilot-12.json"
cp "$EVAL_ROOT/eval/prices/zen-2026-08-13.json" "$RESULT_DIR/zen-2026-08-13.json"
gcloud storage cp -r "$RESULT_DIR" "$BUCKET_PREFIX" --quiet >/dev/null 2>&1 || true
gcloud storage cp "$PILOT_OUT" "${BUCKET_PREFIX}/pilot.jsonl" --quiet || true

echo "=== RESULTS_BEGIN $JOB_ID ==="
cat "$RESULT_DIR/pilot-metadata.json"
echo "=== RESULTS_END $JOB_ID ==="
echo "PILOT_EXIT_CODE=$PILOT_RC"
echo "UPLOAD_PREFIX $BUCKET_PREFIX"
exit "$PILOT_RC"
