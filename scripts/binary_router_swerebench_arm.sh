#!/usr/bin/env bash
# Binary direct-router development shard: one fixed arm over the frozen 100-task
# repository-disjoint manifest. All inference goes through a private worker-local
# LiteLLM proxy using an explicit provider-qualified model ID.
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
LITELLM_DIR="${LITELLM_DIR:-$HOME/.litellm-data}"
OUT_FILE="$RESULT_DIR/results.jsonl"
BUCKET_PREFIX="${GCS_EVAL_BUCKET:-gs://your-eval-storage-bucket}/${RUN_ID}/${ARM}"
mkdir -p "$LOG_DIR" "$LITELLM_DIR"
chmod 700 "$RESULT_DIR" "$LITELLM_DIR"
# Single-writer guard: refuse to start if another shard is already running.
# A second concurrent worker would re-download OUT_FILE, replacing the inode
# while the live worker appends to the old (then-deleted) inode, silently
# stranding all subsequently written rows.
exec 9>"$RESULT_DIR/.shard.lock"
if ! flock -n 9; then
  echo "another binary-router shard holds $RESULT_DIR/.shard.lock; refusing to start" >&2
  exit 3
fi
export EVAL_PROXY_RETRIES="${EVAL_PROXY_RETRIES:-8}"
export EVAL_PROXY_RETRY_BASE="${EVAL_PROXY_RETRY_BASE:-1.0}"
export EVAL_PROXY_RETRY_MAX_WAIT="${EVAL_PROXY_RETRY_MAX_WAIT:-60.0}"
[ -n "${OPENCODE_API_KEY:-}" ] || { echo "OPENCODE_API_KEY missing" >&2; exit 1; }
[ -n "${LITELLM_API_KEY:-}" ] || { echo "LITELLM_API_KEY missing" >&2; exit 1; }
cat > "$LITELLM_DIR/config.yaml" <<'EOF'
model_list:
  - model_name: "opencode-zen/hy3-free"
    litellm_params:
      model: "openai/hy3-free"
      api_base: "https://opencode.ai/zen/v1"
      api_key: "os.environ/OPENCODE_API_KEY"
  - model_name: "opencode-go/deepseek-v4-flash"
    litellm_params:
      model: "openai/deepseek-v4-flash"
      api_base: "https://opencode.ai/zen/go/v1"
      api_key: "os.environ/OPENCODE_API_KEY"
general_settings:
  master_key: "os.environ/LITELLM_API_KEY"
litellm_settings:
  drop_params: true
EOF
chmod 600 "$LITELLM_DIR/config.yaml"
nohup litellm --config "$LITELLM_DIR/config.yaml" --host 127.0.0.1 --port 8080 </dev/null >>"$LOG_DIR/litellm.out" 2>>"$LOG_DIR/litellm.err" &
for _ in $(seq 1 90); do
  curl -sf -m 3 http://127.0.0.1:8080/v1/models -H "Authorization: Bearer $LITELLM_API_KEY" -o "$LOG_DIR/models.json" && break
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
 --litellm-endpoint http://127.0.0.1:8080/v1/chat/completions \
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
