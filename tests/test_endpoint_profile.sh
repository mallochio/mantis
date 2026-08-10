#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck source=../endpoint-profile.sh
. ./endpoint-profile.sh

reset_case() {
  unset ROUTELLM_ENDPOINT_PROFILE ROUTELLM_GATEWAY_MODE AI_GATEWAY_API_KEY MANTIS_GATEWAY_API_KEY MIDDLE_BASE MIDDLE_KEY
  EXPENSIVE_BASE="https://openrouter.ai/api/v1"
  CHEAP_BASE="https://opencode.ai/zen/go/v1"
  EXPENSIVE_KEY="provider-expensive"
  CHEAP_KEY="provider-cheap"
}

reset_case
router_select_endpoint_keys
[ "$EXPENSIVE_KEY" = provider-expensive ]
[ "$CHEAP_KEY" = provider-cheap ]

# Direct endpoints keep their own provider keys even when a profile is forced.
reset_case
ROUTELLM_ENDPOINT_PROFILE=direct
router_select_endpoint_keys
[ "$EXPENSIVE_KEY" = provider-expensive ]
[ "$CHEAP_KEY" = provider-cheap ]

# Direct bases must be valid HTTP(S) URLs.
reset_case
EXPENSIVE_BASE="not-a-url"
if router_select_endpoint_keys >/dev/null 2>&1; then
  echo "invalid direct base was accepted" >&2
  exit 1
fi

[ "$(router_url_host 'https://user:pass@openrouter.ai:443/v1')" = openrouter.ai ]
if router_url_host 'not-a-url' >/dev/null 2>&1; then
  echo "invalid URL was accepted" >&2
  exit 1
fi

echo "endpoint profile tests passed"
