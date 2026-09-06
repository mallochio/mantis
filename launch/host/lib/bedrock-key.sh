#!/usr/bin/env bash
# Ensure BEDROCK_API_KEY for the Bedrock OpenAI-compatible endpoint.
#
# The Grok capable target rides bedrock-runtime .../openai/v1 with a bearer
# key, not IAM SigV4. Mint a short-term token (12h) from the IAM creds and
# persist it to a shared file so Switchyard and Mantis (separate envs) use
# the same key. Sourced by switchyard-local.sh and run_mantis_native.sh.
# Requires: REPO_ROOT, MANTIS_DATA_DIR (optional), uv on PATH.
ensure_bedrock_api_key() {
    local key_file="${MANTIS_DATA_DIR:-$HOME/.local/share/mantis}/bedrock_api_key"
    if [ -z "${BEDROCK_API_KEY:-}" ] && [ -f "$key_file" ] \
        && [ -z "$(find "$key_file" -mmin +600 2>/dev/null)" ]; then
        BEDROCK_API_KEY="$(cat "$key_file" 2>/dev/null || true)"
    fi
    if [ -z "${BEDROCK_API_KEY:-}" ]; then
        local token
        token="$(cd "$REPO_ROOT" && uv run --no-sync --with aws-bedrock-token-generator python -c "from aws_bedrock_token_generator import provide_token; print(provide_token())" 2>/dev/null || true)"
        if [ -n "$token" ]; then
            BEDROCK_API_KEY="$token"
            printf '%s' "$token" > "$key_file" 2>/dev/null || true
            chmod 600 "$key_file" 2>/dev/null || true
        else
            echo "WARNING: could not mint BEDROCK_API_KEY; Grok capable target will fail" >&2
        fi
    fi
    export BEDROCK_API_KEY
}
