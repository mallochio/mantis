#!/usr/bin/env bash
set -euo pipefail
# Download the latest trained TRINITY head and Conductor checkpoint from S3.
# Falls back to rebuilding the base vector from the committed
# artifacts/router_head.safetensors if S3 is not configured.
# Usage:
#   ./scripts/download_artifacts.sh [output_dir]
# Defaults output_dir to the repo root.

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DIR="${1:-.}"
mkdir -p "$DIR/artifacts" "$DIR/outputs/conductor_retrain/retrain-conductor-20260802_003213/checkpoint"

S3_SRC="s3://sid-llm-runs/retrain-fugu-router/retrain-router-20260801_214418"
S3_CONDUCTOR="s3://sid-llm-runs/retrain-fugu-conductor/retrain-conductor-20260802_003213/checkpoint"

# Try S3 first; if credentials or the bucket are unavailable, build the base vector locally.
if aws s3 ls "$S3_SRC/model_iter_60.npy" >/dev/null 2>&1; then
    echo "Downloading TRINITY router head to $DIR/artifacts/ ..."
    aws s3 cp "$S3_SRC/model_iter_60.npy" "$DIR/artifacts/model_iter_60.npy"
    aws s3 cp "$S3_SRC/router_head.npy" "$DIR/artifacts/router_head.npy"
else
    echo "S3 artifacts not reachable; rebuilding base vector from router_head.safetensors ..."
    FUGU_ROUTER_HEAD="${FUGU_ROUTER_HEAD:-$REPO_ROOT/artifacts/router_head.safetensors}" \
    FUGU_VECTOR_OUT="$DIR/artifacts/model_iter_60.npy" \
        python3 "$REPO_ROOT/scripts/make_vec.py"
fi

if aws s3 ls "$S3_CONDUCTOR/" >/dev/null 2>&1; then
    echo "Downloading Conductor checkpoint to $DIR/outputs/conductor_retrain/retrain-conductor-20260802_003213/checkpoint/ ..."
    aws s3 sync --quiet "$S3_CONDUCTOR/" \
                "$DIR/outputs/conductor_retrain/retrain-conductor-20260802_003213/checkpoint/"
else
    echo "WARN: S3 Conductor checkpoint not reachable; omitting local checkpoint." >&2
fi

echo "Done. Set these in .env before docker compose up:"
echo "  FUGU_VECTOR=$DIR/artifacts/model_iter_60.npy"
echo "  FUGU_HEAD=$DIR/artifacts/router_head.npy  (or $DIR/artifacts/router_head.safetensors)"
echo "  FUGU_LOCAL_CONDUCTOR=$DIR/outputs/conductor_retrain/retrain-conductor-20260802_003213/checkpoint"
