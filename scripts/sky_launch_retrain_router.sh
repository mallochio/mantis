#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

CMD=(sky launch)
if [ "${1:-}" == "--job" ]; then
  CMD=(sky job submit)
  shift
fi

# Accept --dry-run as an alias for SkyPilot's --dryrun.
EXTRA_ARGS=("$@")
for i in "${!EXTRA_ARGS[@]}"; do
  if [ "${EXTRA_ARGS[$i]}" == "--dry-run" ]; then
    EXTRA_ARGS[$i]="--dryrun"
  fi
done

# Forward OpenRouter + HF credentials from the local shell.
ENV_ARGS=()
[ -n "${OPENROUTER_API_KEY:-}" ] && ENV_ARGS+=(--env "OPENROUTER_API_KEY=$OPENROUTER_API_KEY")
[ -n "${HF_TOKEN:-}" ] && ENV_ARGS+=(--env "HF_TOKEN=$HF_TOKEN")

exec "${CMD[@]}" "${ENV_ARGS[@]}" launch/sky/retrain_fugu_router.yaml "${EXTRA_ARGS[@]}"
