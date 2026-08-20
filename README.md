# Mantis

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

Mantis can run in the cloud on **Render** to eliminate battery/memory drain on your local machine while remaining fully controllable from your local Git repo and CLI.

### Quick Start
1. Go to **[Render Dashboard](https://dashboard.render.com/)** → **New +** → **Blueprint**.
2. Connect `mallochio/mantis` (branch `main`).
3. Fill in your cloud provider keys (`AZURE_OPENAI_API_KEY`, `AWS_ACCESS_KEY_ID`, etc.) and deploy.

### Local Cloud Control CLI
Manage your cloud instance locally using `./scripts/mantis-cloud.sh`:

```bash
# Check cloud deployment status
./scripts/mantis-cloud.sh status https://mantis-orchestrator.onrender.com

# Test an end-to-end chat completion
./scripts/mantis-cloud.sh test https://mantis-orchestrator.onrender.com <MANTIS_API_KEY>

# Sync local config/catalog.toml changes & trigger auto-deploy
./scripts/mantis-cloud.sh sync

# Switch your Mac's environment (~/.zshrc) to the Cloud instance (stops local daemons to save battery)
./scripts/mantis-cloud.sh use-cloud https://mantis-orchestrator.onrender.com/v1 <MANTIS_API_KEY>

# Revert local machine to local host stack
./scripts/mantis-cloud.sh use-local
```

For detailed container specs and environment configurations, see [`deploy/README.md`](deploy/README.md).

---

## Running Locally on Host

The local launcher stack runs Bifrost, Mantis Router, and Mantis API as background processes:

```bash
~/Startup/llm-stack.sh start   # Bifrost :8080 -> gateway :5500 -> Mantis API :8088
~/Startup/llm-stack.sh status
~/Startup/llm-stack.sh stop
```

Or start directly in foreground:
```bash
./scripts/run_mantis_native.sh
```

---

## Routing Ratchet and Cost Control

`mantis/base` uses a classifier (Supra) to pick a worker tier per request and a session ratchet to keep the warm prompt-cache prefix on one tier. The ratchet climbs freely but does not downgrade by default.

Two knobs relax this without breaking multi-turn tool loops:
- `MANTIS_ROUTER_DOWNGRADE_IDLE_S=N`: after `N` seconds of silence the next turn drops to the freshly scored tier.
- `MANTIS_ROUTER_RESCORE_EVERY_N=N`: every `N` completed turns the ratchet releases and the classifier's current tier wins.

---

## Calling the API (OpenAI Compatible)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8088/v1",  # Or your Render URL
    api_key="sk-mantis-local",
)

response = client.chat.completions.create(
    model="mantis/base",
    messages=[{"role": "user", "content": "Explain quicksort briefly."}],
)
print(response.choices[0].message.content)
```

`stream=True` returns standard `text/event-stream` chunks. Internal status frames are emitted under `delta.reasoning` / `mantis_event`.

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

Use a strong `MANTIS_API_KEY`, put TLS in front of public endpoints, and keep provider keys server-side in your deployment environment variables or secret store.
