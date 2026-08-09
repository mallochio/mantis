#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck source=../endpoint-profile.sh
. ./endpoint-profile.sh

reset_case() {
  unset ROUTELLM_ENDPOINT_PROFILE ROUTELLM_GATEWAY_MODE AI_GATEWAY_API_KEY MANTIS_GATEWAY_API_KEY
  EXPENSIVE_BASE="https://openrouter.ai/api/v1"
  CHEAP_BASE="https://opencode.ai/zen/go/v1"
  EXPENSIVE_KEY="provider-expensive"
  CHEAP_KEY="provider-cheap"
}

reset_case
router_select_endpoint_keys
[ "$EXPENSIVE_KEY" = provider-expensive ]
[ "$CHEAP_KEY" = provider-cheap ]

reset_case
EXPENSIVE_BASE="https://unified-ai-gateway.siddsantham.workers.dev/v1"
AI_GATEWAY_API_KEY="gateway-token"
router_select_endpoint_keys
[ "$EXPENSIVE_KEY" = gateway-token ]
[ "$CHEAP_KEY" = provider-cheap ]

reset_case
CHEAP_BASE="https://UNIFIED-AI-GATEWAY.SIDDSANTHAM.WORKERS.DEV/path"
MANTIS_GATEWAY_API_KEY="fallback-token"
router_select_endpoint_keys
[ "$EXPENSIVE_KEY" = provider-expensive ]
[ "$CHEAP_KEY" = fallback-token ]

reset_case
ROUTELLM_ENDPOINT_PROFILE=cloudflare
AI_GATEWAY_API_KEY="profile-token"
router_select_endpoint_keys
[ "$EXPENSIVE_KEY" = profile-token ]
[ "$CHEAP_KEY" = profile-token ]

reset_case
ROUTELLM_GATEWAY_MODE=1
AI_GATEWAY_API_KEY="mode-token"
router_select_endpoint_keys
[ "$EXPENSIVE_KEY" = mode-token ]
[ "$CHEAP_KEY" = mode-token ]

reset_case
EXPENSIVE_BASE="https://unified-ai-gateway.siddsantham.workers.dev/v1"
if router_select_endpoint_keys >/dev/null 2>&1; then
  echo "missing gateway token was accepted" >&2
  exit 1
fi
[ "$EXPENSIVE_KEY" = provider-expensive ]

reset_case
EXPENSIVE_BASE="https://unified-ai-gateway.siddsantham.workers.dev/v1"
ROUTELLM_ENDPOINT_PROFILE=direct
router_select_endpoint_keys
[ "$EXPENSIVE_KEY" = provider-expensive ]

reset_case
ROUTELLM_ENDPOINT_PROFILE=invalid
if router_select_endpoint_keys >/dev/null 2>&1; then
  echo "invalid endpoint profile was accepted" >&2
  exit 1
fi

[ "$(router_url_host 'https://user:pass@unified-ai-gateway.siddsantham.workers.dev:443/v1')" = unified-ai-gateway.siddsantham.workers.dev ]
if router_url_host 'not-a-url' >/dev/null 2>&1; then
  echo "invalid URL was accepted" >&2
  exit 1
fi

echo "endpoint profile tests passed"
