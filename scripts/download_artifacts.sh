#!/usr/bin/env bash
set -euo pipefail
# Download the latest trained TRINITY head and Conductor checkpoint from S3.
# Usage:
#   ./scripts/download_artifacts.sh [output_dir]
# Defaults output_dir to the repo root.

DIR="${1:-.}"
mkdir -p "$DIR/artifacts" "$DIR/outputs/conductor_retrain/retrain-conductor-20260802_003213/checkpoint"

echo "Downloading TRINITY router head to $DIR/artifacts/ ..."
aws s3 cp s3://sid-llm-runs/retrain-fugu-router/retrain-router-20260801_214418/model_iter_60.npy "$DIR/artifacts/model_iter_60.npy"
aws s3 cp s3://sid-llm-runs/retrain-fugu-router/retrain-router-20260801_214418/router_head.npy "$DIR/artifacts/router_head.npy"

echo "Downloading Conductor checkpoint to $DIR/outputs/conductor_retrain/retrain-conductor-20260802_003213/checkpoint/ ..."
aws s3 sync s3://sid-llm-runs/retrain-fugu-conductor/retrain-conductor-20260802_003213/checkpoint/ \
            "$DIR/outputs/conductor_retrain/retrain-conductor-20260802_003213/checkpoint/"

echo "Done. Set these in .env before docker compose up:"
echo "  FUGU_VECTOR=$DIR/artifacts/model_iter_60.npy"
echo "  FUGU_HEAD=$DIR/artifacts/router_head.npy"
echo "  FUGU_LOCAL_CONDUCTOR=$DIR/outputs/conductor_retrain/retrain-conductor-20260802_003213/checkpoint"
