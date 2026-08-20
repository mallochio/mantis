# Mantis on Render

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/mallochio/mantis)

This directory deploys the complete Mantis stack as one Render web service:

1. **Mantis API** — public, OpenAI-compatible HTTPS API at `https://<service>.onrender.com/v1`.
2. **Mantis direct router** — the internal Supra gateway on `127.0.0.1:5500`; it is never exposed publicly.
3. **Bifrost** — internal upstream gateway on `:8080`; its dashboard is private over Tailscale only.
4. **Tailscale** — optional private access to Bifrost at `:8080`.

The public service health check is `GET /ready`. A successful `/ready` response confirms that Bifrost, the Mantis direct router, and the Mantis API have all started.

## Deploy

1. In the [Render Dashboard](https://dashboard.render.com/), select **New + → Blueprint**, connect the repository, and apply `render.yaml` from `main`.
2. On an existing service, add new `sync: false` variables manually under **Environment**. Render prompts for them only on initial Blueprint creation.
3. Deploy the latest commit, then verify:

   ```bash
   curl -fsS https://mantis-orchestrator.onrender.com/ready
   ```

   A `502` with `x-render-routing: no-deploy` means Render has not yet completed a healthy deployment; it is not an API authentication error.

### Required configuration

`render.yaml` generates service-internal secrets automatically. Do not copy them into source control.

| Render setting | Purpose | Notes |
|---|---|---|
| `MANTIS_API_KEY` | Public Mantis API bearer key | Copy this value to a local runtime variable named `MANTIS_RENDER_API_KEY`; do not reuse a retired local key. |
| `GCP_SERVICE_ACCOUNT_JSON` **or** secret file `gcp-service-account.json` | Vertex AI service-account JSON | **Prefer Secret Files:** Render → service → Environment → Secret Files → add `gcp-service-account.json`. It is mounted at `/etc/secrets/gcp-service-account.json`. The entrypoint also accepts the JSON in `GCP_SERVICE_ACCOUNT_JSON`. Never commit either form. |
| `VERTEXAI_PROJECT`, `VERTEXAI_LOCATION` | Vertex target | Set these for the project/location permitted by the service account. |
| `AZURE_OPENAI_RESOURCE_URL`, `AZURE_OPENAI_API_KEY` | Medium/complex primary routes | Required for GPT-5.6 Terra and GPT-5.6 Sol. Replace the Blueprint URL placeholder with the real Azure resource endpoint. |
| `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_BEDROCK_REGION` | Reasoning primary and fallbacks | Required for Claude Opus 5 and Bedrock fallbacks. |
| `OPENROUTER_API_KEY` | Last-resort Bifrost fallbacks | Required if OpenRouter fallbacks should be usable. |
| `TAILSCALE_AUTHKEY` | Private Bifrost dashboard | Use a current reusable or ephemeral auth key from the [Tailscale admin console](https://login.tailscale.com/admin/settings/keys). |
| `BIFROST_ADMIN_USERNAME`, `BIFROST_ADMIN_PASSWORD` | Bifrost dashboard Basic Auth | Separate from inference/virtual keys. Default username is `admin`; use the exact generated/admin password for the dashboard. |
| `BIFROST_COMPLEXITY_PILOT_KEY` | Complexity-router pilot inference key | Use only for direct pilot requests to Bifrost; it is not the dashboard password. |

## Access Mantis

Mantis is the public endpoint. Authenticate with the Render service's `MANTIS_API_KEY`:

```bash
export MANTIS_RENDER_API_KEY='<Render MANTIS_API_KEY>'

curl https://mantis-orchestrator.onrender.com/v1/models \
  -H "Authorization: Bearer $MANTIS_RENDER_API_KEY"

curl https://mantis-orchestrator.onrender.com/v1/chat/completions \
  -H "Authorization: Bearer $MANTIS_RENDER_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "mantis/base",
    "messages": [{"role": "user", "content": "Reply with SUCCESS only."}]
  }'
```

Available Mantis model IDs are `mantis/base`, `mantis/trinity`, `mantis/ultra`, and `mantis/fusion`.

## Access Bifrost privately over Tailscale

Bifrost is intentionally not served from the public Render URL. After a successful Tailscale join, find the **online** Render node:

```bash
tailscale status
```

It is named `mantis-render` initially. Redeployments can create suffixes such as `mantis-render-3`; stale offline nodes may be removed from the Tailscale Machines console. Use the online node's Tailnet IP or its MagicDNS name:

```text
http://<online-tailscale-ip>:8080
# or, if MagicDNS is enabled:
http://mantis-render-<n>.<tailnet>.ts.net:8080
```

Log into the dashboard with:

```text
username: BIFROST_ADMIN_USERNAME  (normally admin)
password: BIFROST_ADMIN_PASSWORD
```

Do **not** use `BIFROST_API_KEY` or `BIFROST_COMPLEXITY_PILOT_KEY` as the dashboard password. Tailscale starts in the background so a Tailscale auth failure does not prevent the public Mantis service from passing `/ready`; inspect deploy logs for `Tailscale Background` if the node is absent.

## Complexity-router pilot

The Bifrost complexity ladder is deliberately scoped to virtual key `vk-interactive-complexity-pilot`; normal Mantis traffic continues to use Mantis/Supra routing. Send a direct OpenAI-compatible request to private Bifrost with `BIFROST_COMPLEXITY_PILOT_KEY` and model `auto`:

```bash
export BIFROST_URL='http://<online-tailscale-ip>:8080/v1'
export BIFROST_COMPLEXITY_PILOT_KEY='<Render BIFROST_COMPLEXITY_PILOT_KEY>'

curl "$BIFROST_URL/chat/completions" \
  -H "Authorization: Bearer $BIFROST_COMPLEXITY_PILOT_KEY" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "auto",
    "messages": [{"role": "user", "content": "Fix the failing test in src/auth.py"}]
  }'
```

| Tier | Primary | Fallback chain |
|---|---|---|
| SIMPLE | Vertex `google/gemini-3.7-flash` | OpenRouter Gemini 3.7 Flash |
| MEDIUM | Azure `gpt-5.6-terra` | Bedrock Terra → OpenRouter Terra |
| COMPLEX | Azure `gpt-5.6-sol` | Bedrock Sol → OpenRouter Sol |
| REASONING | Bedrock `anthropic/claude-opus-5` | Vertex Claude Opus 5 → OpenRouter Claude Opus 5 |

The analyzer is a fast, deterministic lexical classifier. It uses coding and systems vocabulary, conservative high-precision reasoning phrases, and boundaries `.10 / .35 / .60`. Requests without a supported text signal fall through to their existing route rather than being guessed. Use the dashboard's **Complexity Router** and **Routing Logs** pages to inspect `tier`, score, and selected model before broadening this pilot.

## Prime Agent

Prime can use the public Mantis endpoint and the private Bifrost pilot as two distinct OpenAI-compatible providers. Configure secrets as runtime environment variables, not in `~/.prime/agent/models.json`:

```bash
export MANTIS_RENDER_API_KEY='<Render MANTIS_API_KEY>'
export BIFROST_COMPLEXITY_PILOT_KEY='<Render BIFROST_COMPLEXITY_PILOT_KEY>'
```

Example provider definitions (all models use a 262,144-token context window):

```json
{
  "providers": {
    "mantis-render": {
      "name": "Mantis Router (Render)",
      "api": "openai-completions",
      "baseUrl": "https://mantis-orchestrator.onrender.com/v1",
      "apiKey": "!printenv MANTIS_RENDER_API_KEY",
      "models": [
        {"id": "base", "contextWindow": 262144, "maxTokens": 131072},
        {"id": "trinity", "contextWindow": 262144, "maxTokens": 32768},
        {"id": "ultra", "contextWindow": 262144, "maxTokens": 32768},
        {"id": "fusion", "contextWindow": 262144, "maxTokens": 32768}
      ]
    },
    "bifrost-complexity": {
      "name": "Bifrost Complexity Pilot (Render via Tailscale)",
      "api": "openai-completions",
      "baseUrl": "http://<online-tailscale-ip>:8080/v1",
      "apiKey": "!python3 -c 'import os; k=os.environ[\"BIFROST_COMPLEXITY_PILOT_KEY\"]; print(k if k.startswith(\"sk-bf-\") else \"sk-bf-\" + k)'",
      "models": [
        {"id": "auto", "contextWindow": 262144, "maxTokens": 131072}
      ]
    }
  }
}
```

Run a headless coding task through Mantis:

```bash
prime-agent --mode text --provider mantis-render --model base --no-session \
  -p 'Inspect this repository, implement the requested change, run its tests, and fix failures.'
```

The `bifrost-complexity/auto` provider is suitable for testing the four-tier pilot once the Tailnet node is online. Render-generated virtual-key values are normalized to the required `sk-bf-` prefix by the container; the Prime command above applies the same normalization client-side without storing a duplicate secret.

## Configuration synchronization

`deploy/render/bifrost.template.json` is the deployed Bifrost configuration. It is intentionally identical to `config/bifrost.template.json`; its `source_of_truth` is `config.json` so Git-defined analyzer keyword removals do not persist accidentally as UI/DB additions. Commit and push changes to trigger Render deployment. Do not copy a local, secret-bearing Bifrost runtime JSON over this template.

`deploy/scripts/mantis-cloud.sh status` and `test` remain lightweight helpers. The older `use-cloud`/`use-local` workflow targets retired local-provider naming; use the Prime configuration above for the supported cloud path.

## Development-only local stack

The repository still contains local development launchers and loopback addresses for contributors. They are not part of the supported cloud-control path and should not be used as public endpoints. See `apps/gateway/README.md` for internal gateway development details.
