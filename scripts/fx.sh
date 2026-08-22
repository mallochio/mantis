#!/usr/bin/env bash
# fx.sh — headless multi-turn Fusion harness on OpenRouter :free models.
#
# Requires:
#   OPENROUTER_API_KEY  — OpenRouter API key (free-tier models use the :free suffix)
#
# Optional:
#   FX_BRIEF            — override the default multi-step brief
#   FX_FOLLOW_UP        — second user turn for chat/completions path
#   FX_MAIN_MODEL       — Fusion lead model (default stealth/ox-alpha)
#   FX_SIDEKICK_MODEL   — Fusion sidekick model (default stealth/ox-alpha)
#   FX_PORT             — local Mantis API port (default 5511)
#   MANTIS_API_KEY      — local API auth token (default sk-fx-headless)
#
# Example:
#   export OPENROUTER_API_KEY=sk-or-v1-...
#   ./scripts/fx.sh
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CATALOG="${FX_CATALOG:-$REPO_ROOT/config/catalog.fusion-openrouter-free.toml}"
PORT="${FX_PORT:-5511}"
URL="http://127.0.0.1:${PORT}"
TOKEN="${MANTIS_API_KEY:-sk-fx-headless}"
OUTPUT="${FX_OUTPUT:-$REPO_ROOT/artifacts/fx-session-report.json}"
BRIEF="${FX_BRIEF:-In this scratch directory, create hello.py that prints exactly HELLO-FX, run it with python3, then run python3 -m py_compile hello.py. Report whether both commands succeeded.}"
FOLLOW_UP="${FX_FOLLOW_UP:-Without rerunning everything, confirm hello.py still prints HELLO-FX.}"
MAIN_MODEL="${FX_MAIN_MODEL:-stealth/ox-alpha}"
SIDEKICK_MODEL="${FX_SIDEKICK_MODEL:-stealth/ox-alpha}"

[[ -f "$CATALOG" ]] || { echo "[fx] missing catalog: $CATALOG" >&2; exit 1; }
[[ -n "${OPENROUTER_API_KEY:-}" ]] || {
  echo "[fx] ERROR: export OPENROUTER_API_KEY before running" >&2
  exit 1
}

if [[ -f "$REPO_ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.env"
  set +a
fi

PYTHON="${FX_PYTHON:-}"
if [[ -z "$PYTHON" ]]; then
  if command -v uv >/dev/null 2>&1; then
    uv sync --locked --no-dev >/dev/null 2>&1 || true
    PYTHON="uv run --no-sync python"
  else
    PYTHON="python3"
  fi
fi

export PYTHONPATH="$REPO_ROOT/scripts:$REPO_ROOT/apps/api${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$(dirname "$OUTPUT")"

echo "[fx] validating catalog..."
$PYTHON - <<'PY' "$CATALOG"
import sys
import model_catalog
path = sys.argv[1]
catalog = model_catalog.load_mantis_catalog(path)
if catalog is None:
    raise SystemExit(f"catalog load failed: {path}")
main = catalog.bindings.workers["gpt-5_6-sol"]
side = catalog.bindings.workers["gpt-5_6-luna"]
print(f"[fx] lead  {main.provider} -> {main.upstream_model}")
print(f"[fx] side  {side.provider} -> {side.upstream_model}")
PY

echo "[fx] running multi-turn session (delegate + chat)..."
FX_ARGS=(
  --managed
  --url "$URL"
  --token "$TOKEN"
  --catalog "$CATALOG"
  --brief "$BRIEF"
  --follow-up "$FOLLOW_UP"
  --max-iterations "${FX_MAX_ITERATIONS:-12}"
  --output "$OUTPUT"
  --main-model "$MAIN_MODEL"
  --sidekick-model "$SIDEKICK_MODEL"
)
exec $PYTHON "$REPO_ROOT/scripts/fx_session.py" "${FX_ARGS[@]}"
