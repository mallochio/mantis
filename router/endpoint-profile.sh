#!/usr/bin/env bash
# Endpoint credential selection helpers. This file does not print credentials.
# The router runs direct catalog-configured endpoints only (Cloudflare
# gateway retired); these helpers validate bases and keep the launcher's
# env contract for the legacy no-catalog fallback.

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

router_backend_profile() {
  local base="$1"
  router_url_host "$base" >/dev/null || {
    echo "ERROR: backend base must be a valid HTTP(S) URL" >&2
    return 2
  }
  printf '%s\n' direct
}

router_select_endpoint_keys() {
  # Direct endpoints use their own provider keys (EXPENSIVE_KEY/CHEAP_KEY/
  # MIDDLE_KEY stay whatever the launcher exported); validate the bases and
  # do not remap credentials.
  router_backend_profile "$EXPENSIVE_BASE" >/dev/null || return 2
  router_backend_profile "$CHEAP_BASE" >/dev/null || return 2
  if [ -n "${MIDDLE_BASE:-}" ]; then
    router_backend_profile "$MIDDLE_BASE" >/dev/null || return 2
  fi
}
