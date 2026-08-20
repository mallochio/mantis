# Mantis Cloud Deployment (Render)

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/mallochio/mantis)

This directory contains the production container deployment configuration for Mantis + Bifrost on Render.

## Overview

The deployment packages the complete Mantis stack into a single private/public Docker service on Render:

1. **Bifrost AI Gateway (`:8080`)**: Unified upstream routing to OpenAI/Azure, AWS Bedrock, Google Vertex, OpenCode, OpenRouter, and Meta with rate limiting and fallback cascades.
2. **Mantis Router / Gateway (`:5500`)**: Supra classifier scoring and smart session-ratchet routing configured via `config/catalog.toml`.
3. **Mantis API (`:8088` mapped to `$PORT`)**: OpenAI-compatible API serving `mantis/base`, `mantis/trinity`, `mantis/ultra`, and `mantis/fusion`.
4. **Tailscale Daemon (Optional)**: Secure peer-to-peer overlay network enabling private browser access to the Bifrost Web UI (:8080) directly over your Tailnet.

All internal processes run secured in the container. The public endpoint is exposed with HTTPS and secured via `MANTIS_API_KEY`.

---

## Directory Structure

```text
deploy/
├── render/
│   ├── Dockerfile                 # Debian-based Python 3.13 + Node + uv + Bifrost + Tailscale runtime
│   ├── render.yaml                # Render Blueprint service definition
│   ├── render-entrypoint.sh       # Container supervisor & service orchestrator
│   └── bifrost.template.json      # Gateway upstream provider routing template
├── scripts/
│   └── mantis-cloud.sh            # Local CLI to manage, monitor, test, and switch to Cloud Mantis
└── README.md                      # This deployment guide
```

---

## 1. Quick Deploy on Render

1. In the **[Render Dashboard](https://dashboard.render.com/)**, click **New +** → **Blueprint**.
2. Connect your Git repository (`mallochio/mantis` on branch `main`).
3. Render automatically reads `render.yaml` and sets up the `mantis-orchestrator` Web Service.
4. Fill the required provider API keys under **Environment Variables**:
   - `AZURE_OPENAI_API_KEY`
   - `AWS_ACCESS_KEY_ID` & `AWS_SECRET_ACCESS_KEY`
   - `OPENCODE_API_KEY` / `OPENROUTER_API_KEY` / `META_API_KEY`
   - `MANTIS_API_KEY` *(the secret key used by your local machine / agents to authorize requests)*
   - `TAILSCALE_AUTHKEY` *(Optional: Generate a reusable or ephemeral auth key from [Tailscale Admin Console](https://login.tailscale.com/admin/settings/keys))*
5. Click **Apply**.

---

## 2. Accessing the Bifrost Web UI over Tailscale

When `TAILSCALE_AUTHKEY` is provided:
1. The container will automatically join your Tailnet as `mantis-render` (configurable via `TAILSCALE_HOSTNAME`).
2. Bifrost (`:8080`) is served directly over Tailscale.
3. Open your browser on any Tailscale-connected device and navigate to:
   - `http://mantis-render:8080` (or `http://<tailscale-ip>:8080`)
4. You get full access to the Bifrost dashboard, live request logs, provider metrics, and latency graphs securely without exposing port 8080 to the public internet!

---

## 3. Managing & Controlling from Local Machine

Use the local CLI script (`./scripts/mantis-cloud.sh` or `./deploy/scripts/mantis-cloud.sh`):

### Check Health / Status
```bash
./scripts/mantis-cloud.sh status https://mantis-orchestrator.onrender.com
```

### Test End-to-End Chat Completion
```bash
./scripts/mantis-cloud.sh test https://mantis-orchestrator.onrender.com <YOUR_MANTIS_API_KEY>
```

### Sync Local Routing Changes & Auto-Deploy
When you modify `config/catalog.toml` or `deploy/render/bifrost.template.json` locally:
```bash
./scripts/mantis-cloud.sh sync
```
*(Syncs configs, commits to git, and pushes to `origin main` to trigger a Render redeploy).*

### Switch Local Machine to Use Cloud (Save Mac Battery)
```bash
./scripts/mantis-cloud.sh use-cloud https://mantis-orchestrator.onrender.com/v1 <YOUR_MANTIS_API_KEY>
```
This stops local background Python/Go processes on your Mac and sets `MANTIS_URL` in `~/.zshrc` and `~/.prime/agent/models.json`.

### Revert to Local Stack
```bash
./scripts/mantis-cloud.sh use-local
```
