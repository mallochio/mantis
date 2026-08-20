# Mantis

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/mallochio/mantis)

Mantis is an intelligent AI model orchestration and routing platform providing four OpenAI-compatible modes.

## Modes

| Model | Mode | Use |
|---|---|---|
| `mantis/base` | Direct | The gateway scores one request with Supra classifier and sends it to a cheap, middle, or expensive model through Bifrost. |
| `mantis/trinity` | Trinity | Multi-agent coordination over Worker, Thinker, and Verifier roles. |
| `mantis/ultra` | Ultra | Conductor plans and executes a bounded workflow DAG over the worker pool. |
| `mantis/fusion` | Fusion | Lead/sidekick orchestration with tool-use follow-ups and a review loop. Works as `model: mantis/fusion` in chat or via `/v1/fusion/delegate`. |

Only these model IDs are accepted. Select `mantis/trinity`, `mantis/ultra`, or `mantis/fusion` manually; Mantis never selects between modes. All four paths send model calls through the Bifrost gateway.

---

## Architecture & Directory Layout

```text
mantis/
├── apps/
│   ├── api/                    # Mantis public API server (:8088 / $PORT)
│   └── gateway/                # Internal Direct-mode router & classifier (:5500)
├── config/
│   ├── catalog.toml            # Tracked model routing catalog (source of truth)
│   └── worker-costs.json       # Cost evaluation price tables
├── deploy/                     # Cloud container deployment (Cleanly isolated)
│   ├── render/
│   │   ├── Dockerfile          # Debian-based container definition
│   │   ├── render.yaml         # Render Blueprint service manifest
│   │   ├── render-entrypoint.sh# Multi-service container supervisor
│   │   └── bifrost.template.json # Bifrost upstream provider config template
│   ├── scripts/
│   │   └── mantis-cloud.sh     # Local CLI for cloud monitoring & switching
│   └── README.md               # Detailed cloud deployment guide
├── artifacts/                  # Trained router weights & manifests
├── eval/                       # Router evaluation & SWE-rebench harnesses
├── launch/
│   ├── host/                   # Local host system service launcher (llm-stack.sh)
│   └── sky/                    # SkyPilot cluster launch definitions
├── scripts/                    # Development, retraining, and verification scripts
└── tests/                      # Automated test suite (420+ tests)
```

---

## Cloud Deployment (Render)

Mantis runs as the `mantis-orchestrator` Render web service. The public OpenAI-compatible endpoint is:

```text
https://mantis-orchestrator.onrender.com/v1
```

Bifrost remains private on the Tailnet; it is not exposed through the public Render URL. Its pilot uses four classified tiers and a balanced Terra catch-all for unclassified `auto` requests. See the complete setup, credentials, Tailscale access, complexity-router pilot, and Prime Agent instructions in [`deploy/README.md`](deploy/README.md).

### Quick start

1. In Render, create/apply the Blueprint for `main` and configure the required provider credentials.
2. Add the Vertex service account as a Render Secret File named `gcp-service-account.json` (preferred) or `GCP_SERVICE_ACCOUNT_JSON`.
3. Verify deployment without credentials:

   ```bash
   curl -fsS https://mantis-orchestrator.onrender.com/ready
   ```

4. Set the public API key only in your local runtime environment:

   ```bash
   export MANTIS_RENDER_API_KEY='<Render MANTIS_API_KEY>'
   ```

5. Use `mantis-render/base` in Prime Agent or call Mantis directly. The supported Prime provider configuration is documented in [`deploy/README.md`](deploy/README.md#prime-agent).

### Cloud status helper

```bash
./scripts/mantis-cloud.sh status https://mantis-orchestrator.onrender.com
./scripts/mantis-cloud.sh test https://mantis-orchestrator.onrender.com "$MANTIS_RENDER_API_KEY"
```

The legacy local-stack switching workflow is retained only for development compatibility. It is not the supported way to configure Render-backed Prime providers.

---

## Local development only

The Render deployment is the supported operational path. Contributors who need
a local development stack can start it in the foreground:

```bash
./scripts/run_mantis_native.sh
```

This binds development-only loopback services (Bifrost `:8080`, gateway `:5500`,
and Mantis API `:8088`). Do not use those addresses for cloud clients or copy
local credentials into Render. See [`apps/gateway/README.md`](apps/gateway/README.md)
for internal gateway development details.

---

## Routing Ratchet and Cost Control

`mantis/base` uses a classifier (Supra) to pick a worker tier per request and a session ratchet to keep the warm prompt-cache prefix on one tier. The ratchet climbs freely but does not downgrade by default.

Two knobs relax this without breaking multi-turn tool loops:
- `MANTIS_ROUTER_DOWNGRADE_IDLE_S=N`: after `N` seconds of silence the next turn drops to the freshly scored tier.
- `MANTIS_ROUTER_RESCORE_EVERY_N=N`: every `N` completed turns the ratchet releases and the classifier's current tier wins.

---

## Calling the API (OpenAI Compatible)

```python
import os
from openai import OpenAI

client = OpenAI(
    base_url="https://mantis-orchestrator.onrender.com/v1",
    api_key=os.environ["MANTIS_RENDER_API_KEY"],
)

response = client.chat.completions.create(
    model="mantis/base",
    messages=[{"role": "user", "content": "Explain quicksort briefly."}],
)
print(response.choices[0].message.content)
```

`stream=True` returns standard `text/event-stream` chunks. Internal status frames are emitted under `delta.reasoning` / `mantis_event`. For local development, replace the endpoint and use the explicitly configured local credential; do not copy local credentials into the Render deployment.

---

## Maintenance, Hygiene & Cleanup

### Marked for Deletion / Cleaned:
1. **Empty directories**:
   - `openfugu/` (legacy empty directory — *removed*)
   - `eval/runs/worktrees/**/db_files` (stale empty test fixture dir — *removed*)
2. **Local runtime / Build artifacts (Ignored in Git)**:
   - `mantis.egg-info/` — generated build metadata.
   - `coverage.xml`, `.coverage` — generated test coverage files.
   - `outputs/conductor_retrain/` — local scratch training runs.
   - `runs/logs/` — legacy manual benchmark execution logs.
   - `.scratch/`, `apps/.scratch/` — transient tool caches.

### Verification & Testing
```bash
uv run pytest tests -q
uv run ruff check .
./scripts/verify.sh
```

---

## Security

Render provides TLS for the public endpoint. Keep `MANTIS_API_KEY`, provider credentials, GCP service-account JSON, Bifrost admin credentials, and virtual keys in Render Environment/Secret Files; never commit or embed them in client configuration. Use a local runtime variable such as `MANTIS_RENDER_API_KEY` for clients.
