#!/usr/bin/env bash
# Endpoint credential selection helpers. This file does not print credentials.

ROUTELLM_CLOUDFLARE_HOST="unified-ai-gateway.siddsantham.workers.dev"

router_url_host() {
  local url="${1:-}"
  uv run python - "$url" <<'PY'
import sys
from urllib.parse import urlsplit
value = urlsplit(sys.argv[1])
if value.scheme not in {"http", "https"} or not value.hostname:
    raise SystemExit(2)
print(value.hostname.lower())
PY
}

router_gateway_requested() {
  case "${ROUTELLM_GATEWAY_MODE:-}" in
    1|true|TRUE|yes|YES|on|ON|cloudflare) return 0 ;;
    ""|0|false|FALSE|no|NO|off|OFF) return 1 ;;
    *) echo "ERROR: invalid ROUTELLM_GATEWAY_MODE" >&2; return 2 ;;
  esac
}

router_backend_profile() {
  local base="$1" profile="${ROUTELLM_ENDPOINT_PROFILE:-}"
  case "$profile" in
    direct|cloudflare) printf '%s\n' "$profile"; return ;;
    "") ;;
    *) echo "ERROR: ROUTELLM_ENDPOINT_PROFILE must be direct or cloudflare" >&2; return 2 ;;
  esac
  if router_gateway_requested; then
    printf '%s\n' cloudflare
    return
  else
    local status=$?
    [ "$status" -eq 1 ] || return "$status"
  fi
  local host
  host=$(router_url_host "$base") || {
    echo "ERROR: backend base must be a valid HTTP(S) URL" >&2
    return 2
  }
  if [ "$host" = "$ROUTELLM_CLOUDFLARE_HOST" ]; then
    printf '%s\n' cloudflare
  else
    printf '%s\n' direct
  fi
}

router_select_endpoint_keys() {
  local gateway_key="${AI_GATEWAY_API_KEY:-${MANTIS_GATEWAY_API_KEY:-}}"
  local expensive_profile cheap_profile
  expensive_profile=$(router_backend_profile "$EXPENSIVE_BASE") || return
  cheap_profile=$(router_backend_profile "$CHEAP_BASE") || return
  if [ "$expensive_profile" = cloudflare ] || [ "$cheap_profile" = cloudflare ]; then
    if [ -z "$gateway_key" ]; then
      echo "ERROR: Cloudflare gateway endpoint requires AI_GATEWAY_API_KEY or MANTIS_GATEWAY_API_KEY" >&2
      return 2
    fi
  fi
  if [ "$expensive_profile" = cloudflare ]; then
    EXPENSIVE_KEY="$gateway_key"
    export EXPENSIVE_KEY
  fi
  if [ "$cheap_profile" = cloudflare ]; then
    CHEAP_KEY="$gateway_key"
    export CHEAP_KEY
  fi
}
