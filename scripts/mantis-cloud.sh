#!/usr/bin/env bash
# mantis-cloud.sh — Local CLI to manage, monitor, and connect to Mantis on Render.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_DIR="$REPO_ROOT/config"

usage() {
  cat << 'USAGE'
Mantis Cloud Control CLI

Usage:
  ./scripts/mantis-cloud.sh [command] [options]

Commands:
  status [URL] [API_KEY]     Check health and readiness of the cloud deployment
  test [URL] [API_KEY]       Run a test inference request through cloud Mantis
  sync                       Sync local catalog/bifrost configs and push to GitHub (triggers Render deploy)
  use-cloud <URL> <KEY>      Configure local shell (~/.zshrc) to route to Render & stop local servers
  use-local                  Revert local shell (~/.zshrc) to local loopback servers (127.0.0.1:8088)

Examples:
  ./scripts/mantis-cloud.sh status https://mantis-orchestrator.onrender.com
  ./scripts/mantis-cloud.sh test https://mantis-orchestrator.onrender.com sk-your-key
  ./scripts/mantis-cloud.sh use-cloud https://mantis-orchestrator.onrender.com/v1 sk-your-key
USAGE
  exit 1
}

cmd="${1:-}"
shift || true

case "$cmd" in
  status)
    URL="${1:-${MANTIS_CLOUD_URL:-}}"
    if [[ -z "$URL" ]]; then
      echo "Error: Supply the Render service URL (e.g. https://mantis-orchestrator.onrender.com)" >&2
      exit 1
    fi
    URL="${URL%/v1}"
    URL="${URL%/}"
    echo "Checking Mantis cloud readiness at $URL/ready..."
    curl -fsS --max-time 10 "$URL/ready" && echo "" || { echo "Failed to reach $URL/ready" >&2; exit 1; }
    ;;

  test)
    URL="${1:-${MANTIS_CLOUD_URL:-}}"
    KEY="${2:-${MANTIS_API_KEY:-}}"
    if [[ -z "$URL" || -z "$KEY" ]]; then
      echo "Error: Usage: ./scripts/mantis-cloud.sh test <URL> <MANTIS_API_KEY>" >&2
      exit 1
    fi
    URL="${URL%/}"
    [[ "$URL" =~ /v1$ ]] || URL="$URL/v1"
    echo "Testing chat completion on $URL via model 'mantis/base'..."
    curl -fsS -X POST "$URL/chat/completions" \
      -H "Content-Type: application/json" \
      -H "Authorization: Bearer $KEY" \
      -d '{
        "model": "mantis/base",
        "messages": [{"role": "user", "content": "Respond with the word SUCCESS and nothing else."}]
      }' | jq .
    ;;

  sync)
    echo "=== Syncing local routing configs to Mantis repo ==="
    if [[ -f "$HOME/.config/ai-routing/catalog.toml" ]]; then
      cp "$HOME/.config/ai-routing/catalog.toml" "$CONFIG_DIR/catalog.toml"
      echo "Updated config/catalog.toml"
    fi
    if [[ -f "$HOME/.config/ai-routing/bifrost.json" ]]; then
      cp "$HOME/.config/ai-routing/bifrost.json" "$CONFIG_DIR/bifrost.json"
      echo "Updated config/bifrost.json"
    fi
    cd "$REPO_ROOT"
    if [[ -n "$(git status -s)" ]]; then
      echo "Changes detected in repository. Committing and pushing..."
      git add config/ Dockerfile render.yaml scripts/
      git commit -m "Update Mantis cloud deployment configuration & routes"
      git push origin main
      echo "Pushed to GitHub! Render will automatically trigger a new deployment."
    else
      echo "No configuration changes to push. Everything up to date."
    fi
    ;;

  use-cloud)
    URL="${1:-}"
    KEY="${2:-}"
    if [[ -z "$URL" || -z "$KEY" ]]; then
      echo "Error: Usage: ./scripts/mantis-cloud.sh use-cloud <URL> <KEY>" >&2
      exit 1
    fi
    URL="${URL%/}"
    [[ "$URL" =~ /v1$ ]] || URL="$URL/v1"
    echo "Configuring local ~/.zshrc to use Cloud Mantis at $URL..."

    # Stop local background stack to save battery
    echo "Stopping local background Mantis stack..."
    "$REPO_ROOT/launch/host/llm-stack.sh" stop || true

    python3 - << PY
import re
zshrc_path = "$HOME/.zshrc"
with open(zshrc_path, "r") as f:
    text = f.read()

text = re.sub(r'export MANTIS_URL=.*', f'export MANTIS_URL="{URL}"', text)
text = re.sub(r'export MANTIS_API_KEY=.*', f'export MANTIS_API_KEY="{KEY}"', text)

with open(zshrc_path, "w") as f:
    f.write(text)
PY
    echo "Updated ~/.zshrc. Local battery drain eliminated!"
    ;;

  use-local)
    echo "Switching ~/.zshrc back to local Mantis stack (127.0.0.1:8088/v1)..."
    python3 - << PY
import re
zshrc_path = "$HOME/.zshrc"
with open(zshrc_path, "r") as f:
    text = f.read()

text = re.sub(r'export MANTIS_URL=.*', 'export MANTIS_URL="http://127.0.0.1:8088/v1"', text)
text = re.sub(r'export MANTIS_API_KEY=.*', 'export MANTIS_API_KEY="${MANTIS_API_KEY:-sk-mantis-local}"', text)

with open(zshrc_path, "w") as f:
    f.write(text)
PY
    echo "Restarting local Mantis stack..."
    "$REPO_ROOT/launch/host/llm-stack.sh" start
    echo "Local stack is up."
    ;;

  *)
    usage
    ;;
esac
