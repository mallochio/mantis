#!/bin/sh
# Fail-closed catalog gate for the Mantis orchestrator container.
#
# When the catalog profile is selected (MANTIS_ENDPOINT_PROFILE=catalog, or
# any catalog environment is present), the container requires the mounted
# catalog file and the rendered bindings and validates the catalog against the
# trained ABI manifest before the server starts.  Any failure exits non-zero so
# `docker compose up` reports an error instead of silently serving legacy
# defaults.  Direct profiles are unchanged.
set -eu

profile="${MANTIS_ENDPOINT_PROFILE:-direct}"
catalog_mode=0
[ "$profile" = "catalog" ] && catalog_mode=1
[ -n "${AI_ROUTING_CONFIG:-}" ] && catalog_mode=1
[ -n "${MANTIS_CATALOG_PATH:-}" ] && catalog_mode=1
[ -n "${MANTIS_PROVIDER_BINDINGS:-}" ] && catalog_mode=1
[ -n "${MANTIS_IDENTITY_CONTRACT:-}" ] && catalog_mode=1

if [ "$catalog_mode" = 1 ]; then
    catalog_path="${MANTIS_CATALOG_PATH:-/app/catalog/catalog.toml}"
    if [ ! -f "$catalog_path" ] && [ -f /app/catalog/catalog.toml ]; then
        # The host MANTIS_CATALOG_PATH drives the mount source; inside the
        # container the file always lives at the fixed mount target.
        catalog_path="/app/catalog/catalog.toml"
    fi
    if [ ! -f "$catalog_path" ]; then
        if [ -d "$catalog_path" ]; then
            echo "ERROR: $catalog_path is a directory, not a catalog file" >&2
            echo "ERROR: docker created it for the bind mount because the file was" >&2
            echo "ERROR: missing; remove it and create the catalog file" >&2
        else
            echo "ERROR: catalog mode requires a mounted catalog file at $catalog_path" >&2
            echo "ERROR: set MANTIS_CATALOG_PATH and mount it (stack.py up does this)" >&2
        fi
        exit 1
    fi
    python /app/openfugu/model_catalog.py validate --catalog "$catalog_path" || {
        echo "ERROR: Mantis catalog validation failed; refusing to start" >&2
        exit 1
    }
    for var in MANTIS_PROVIDER_BINDINGS MANTIS_WORKER_BINDINGS \
        MANTIS_IDENTITY_CONTRACT MANTIS_PROVIDER_KEYS; do
        eval "value=\${$var:-}"
        if [ -z "$value" ]; then
            echo "ERROR: catalog mode requires $var" >&2
            exit 1
        fi
    done
fi

exec "$@"
