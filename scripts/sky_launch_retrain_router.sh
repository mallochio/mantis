#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

CMD=(sky launch)
if [ "${1:-}" == "--job" ]; then
  CMD=(sky job submit)
  shift
fi

# Forward OpenRouter + HF credentials from the local shell.
ENV_ARGS=()
[ -n "${OPENROUTER_API_KEY:-}" ] && ENV_ARGS+=(--env "OPENROUTER_API_KEY=$OPENROUTER_API_KEY")
[ -n "${HF_TOKEN:-}" ] && ENV_ARGS+=(--env "HF_TOKEN=$HF_TOKEN")

exec "${CMD[@]}" "${ENV_ARGS[@]}" launch/sky/retrain_fugu_router.yaml "$@"
