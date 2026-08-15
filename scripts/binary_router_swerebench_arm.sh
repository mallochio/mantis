#!/usr/bin/env bash
# Binary direct-router development shard: one fixed arm over the frozen 100-task
# repository-disjoint manifest. All inference goes through private worker-local
# Bifrost using an explicit provider-qualified model ID.
set -euo pipefail
ARM="${ARM:?ARM is required (low or high)}"
RUN_ID="${RUN_ID:?RUN_ID is required}"
case "$ARM" in
  low) MODEL="opencode-zen/hy3-free" ;;
  high) MODEL="opencode-go/deepseek-v4-flash" ;;
  *) echo "unknown ARM: $ARM" >&2; exit 2 ;;
esac
EVAL_ROOT="/opt/mantis-eval"
RESULT_DIR="/tmp/binary-router-results"
LOG_DIR="$RESULT_DIR/logs"
BIFROST_DIR="${BIFROST_DIR:-$HOME/.bifrost-data}"
OUT_FILE="$RESULT_DIR/results.jsonl"
BUCKET_PREFIX="gs://ih-storage-sid/${RUN_ID}/${ARM}"
mkdir -p "$LOG_DIR" "$BIFROST_DIR"
chmod 700 "$RESULT_DIR" "$BIFROST_DIR"
export EVAL_PROXY_RETRIES="${EVAL_PROXY_RETRIES:-8}"
export EVAL_PROXY_RETRY_BASE="${EVAL_PROXY_RETRY_BASE:-1.0}"
export EVAL_PROXY_RETRY_MAX_WAIT="${EVAL_PROXY_RETRY_MAX_WAIT:-60.0}"
[ -n "${OPENCODE_API_KEY:-}" ] || { echo "OPENCODE_API_KEY missing" >&2; exit 1; }
[ -n "${BIFROST_API_KEY:-}" ] || { echo "BIFROST_API_KEY missing" >&2; exit 1; }
[ -n "${BIFROST_ENCRYPTION_KEY:-}" ] || { echo "BIFROST_ENCRYPTION_KEY missing" >&2; exit 1; }
cat > "$BIFROST_DIR/config.json" <<'EOF'
{
  "encryption_key": "env.BIFROST_ENCRYPTION_KEY",
  "client": {"enforce_auth_on_inference": true, "enable_logging": false, "initial_pool_size": 50, "disable_content_logging": true},
  "governance": {"virtual_keys": [{"id": "vk-binary-router", "name": "binary router development", "value": "env.BIFROST_API_KEY", "is_active": true, "provider_configs": [
    {"provider": "opencode-zen", "weight": 1.0, "allowed_models": ["hy3-free"], "key_ids": ["*"]},
    {"provider": "opencode-go", "weight": 1.0, "allowed_models": ["deepseek-v4-flash"], "key_ids": ["*"]}
  ]}]},
  "providers": {
    "opencode-zen": {"keys": [{"name": "zen", "value": "env.OPENCODE_API_KEY", "models": ["hy3-free"], "weight": 1.0}], "store_raw_request_response": false},
    "opencode-go": {"keys": [{"name": "go", "value": "env.OPENCODE_API_KEY", "models": ["deepseek-v4-flash"], "weight": 1.0}], "store_raw_request_response": false}
  },
  "source_of_truth": "config.json"
}
EOF
chmod 600 "$BIFROST_DIR/config.json"
nohup npx -y @maximhq/bifrost -app-dir "$BIFROST_DIR" -host 127.0.0.1 -port 8080 -log-style pretty </dev/null >>"$LOG_DIR/bifrost.out" 2>>"$LOG_DIR/bifrost.err" &
for _ in $(seq 1 90); do
  curl -sf -m 3 http://127.0.0.1:8080/v1/models -H "x-bf-vk: $BIFROST_API_KEY" -o "$LOG_DIR/models.json" && break
  sleep 2
done
python3 - "$LOG_DIR/models.json" <<'PY'
import json,sys
ids={x['id'] for x in json.load(open(sys.argv[1])).get('data',[])}
expected={'opencode-zen/hy3-free','opencode-go/deepseek-v4-flash'}
if not expected <= ids: raise SystemExit(f'missing models: {sorted(expected-ids)}')
PY
if gcloud storage ls "$BUCKET_PREFIX/results.jsonl" >/dev/null 2>&1; then
  gcloud storage cp "$BUCKET_PREFIX/results.jsonl" "$OUT_FILE" --quiet
fi
(while true; do sleep 60; [ -f "$OUT_FILE" ] && gcloud storage cp "$OUT_FILE" "$BUCKET_PREFIX/results.jsonl" --quiet >/dev/null 2>&1 || true; done) &
HEARTBEAT_PID=$!
trap 'kill "$HEARTBEAT_PID" 2>/dev/null || true' EXIT
cd "$EVAL_ROOT"
set +e
python3 eval/router_eval.py \
 --manifest eval/manifests/binary-dev-100.json \
 --prices eval/model_prices.json \
 --shadow-prices eval/prices/binary-hy3-deepseek-2026-08-15.json \
 --cost-mode shadow \
 --bifrost-endpoint http://127.0.0.1:8080/v1/chat/completions \
 --fixed-model "$ARM=$MODEL" --arms "$ARM" \
 --per-instance-cost 5.0 --budget-usd 500 --resume --timeout 1200 \
 --output-token-limit 4096 --worktrees-root /tmp/worktrees --keep-worktrees \
 --pi-executable pi --output "$OUT_FILE" >"$LOG_DIR/arm.out" 2>"$LOG_DIR/arm.err"
RC=$?
set -e
kill "$HEARTBEAT_PID" 2>/dev/null || true
trap - EXIT
gcloud storage cp "$OUT_FILE" "$BUCKET_PREFIX/results.jsonl" --quiet || true
gcloud storage cp -r "$LOG_DIR" "$BUCKET_PREFIX/logs" --quiet >/dev/null 2>&1 || true
echo "BINARY_ROUTER_ARM=$ARM RC=$RC PREFIX=$BUCKET_PREFIX"
exit "$RC"
