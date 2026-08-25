# Mantis

Mantis is a local AI model orchestration and routing platform with an OpenAI-compatible API. The supported runtime is a three-process loopback stack:

```text
Bifrost :8080 → Switchyard :5500 → Mantis API :8088
```

No service binds publicly by default. Provider credentials remain on this machine.

## Modes

| Model | Mode | Use |
|---|---|---|
| `mantis/base` | Direct | NVIDIA NeMo Switchyard stage-router picks an efficient or capable model through Bifrost. |
| `mantis/trinity` | Trinity | Multi-agent coordination over Worker, Thinker, and Verifier roles. |
| `mantis/ultra` | Ultra | Conductor plans and executes a bounded workflow DAG over the worker pool. |
| `mantis/fusion` | Fusion | Lead/sidekick orchestration with tool-use follow-ups and a review loop. |

Mantis never switches among these four modes automatically. Base and Fusion are available in normal mode; Trinity and Ultra are experimental and require starting the stack with `--experimental`.

## Layout

```text
mantis/
├── apps/
│   └── api/                    # Mantis API (:8088)
├── artifacts/                  # Trinity router vector/head
├── config/
│   ├── catalog.toml            # Shared model/provider routing catalog
│   ├── bifrost.template.json   # Canonical Bifrost providers, keys, and fallback rules
│   └── worker-costs.json       # Evaluation price tables
├── launch/host/
│   ├── llm-stack.sh            # Full local stack controller
│   └── lib/                    # Bifrost, Switchyard, and API launchers
├── scripts/                    # Development, retraining and verification
├── eval/                       # Router evaluations and SWE-rebench harnesses
└── tests/                      # Test suite
```

## Local setup

### Requirements

- macOS or Linux with `zsh`, `curl`, and `lsof`
- Node/npm (`npx` launches `@maximhq/bifrost`)
- Rust with Cargo, then `cargo install --locked switchyard-server`
- [`uv`](https://docs.astral.sh/uv/)
- Python 3.13
- Provider credentials exported in `~/.zshrc`

Install the Python environment:

```bash
cd ~/Personal/other/mantis
uv sync --locked --no-dev
```

Create private runtime directories and seed Bifrost from the tracked, secret-free template:

```bash
install -d -m 700 ~/.local/share/bifrost/logs ~/.local/share/mantis/switchyard ~/.local/share/llm-stack
install -m 600 config/bifrost.template.json ~/.local/share/bifrost/config.json
ln -sfn "$PWD/config/catalog.toml" ~/.config/ai-routing/catalog.toml
```

The Bifrost template references environment variables; never place secret values in the tracked JSON. At minimum configure:

```text
MANTIS_API_KEY
BIFROST_API_KEY                 # must start with sk-bf-
BIFROST_ENCRYPTION_KEY
BIFROST_ADMIN_USERNAME
BIFROST_ADMIN_PASSWORD
```

Provider routes additionally require the matching Azure, AWS/Bedrock, Vertex ADC, OpenRouter, or OpenCode credentials. For Vertex, export `GOOGLE_APPLICATION_CREDENTIALS`, `VERTEXAI_PROJECT`, and `VERTEXAI_LOCATION`.

## Start and stop

Install the optional StartupFolder-compatible controller link:

```bash
ln -sfn "$PWD/launch/host/llm-stack.sh" ~/Startup/llm-stack.sh
```

Control the stack:

```bash
# Normal mode: Bifrost + Switchyard + Mantis API; exposes Base and Fusion only.
~/Startup/llm-stack.sh start

# Experimental mode: additionally exposes Trinity and Ultra/Conductor.
~/Startup/llm-stack.sh start --experimental
# Equivalent shorthand:
~/Startup/llm-stack.sh --experimental

~/Startup/llm-stack.sh status
~/Startup/llm-stack.sh restart                 # return to normal mode
~/Startup/llm-stack.sh restart --experimental  # experimental mode
~/Startup/llm-stack.sh stop
```

A bare StartupFolder invocation defaults to `start` in normal mode. The controller starts Bifrost, Switchyard, and the API in dependency order and checks readiness. Trinity's Qwen coordinator remains lazy and is not loaded unless an experimental Trinity request is made; Ultra/Conductor also starts work only on an experimental request. Runtime state/logs live under `~/.local/share/bifrost` and `~/.local/share/mantis`.

You can also run only the Mantis API in the foreground for development:

```bash
./scripts/run_mantis_native.sh
```

That command expects Bifrost and Switchyard to be available separately.

## Call Mantis

```python
import os
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8088/v1",
    api_key=os.environ["MANTIS_API_KEY"],
)

response = client.chat.completions.create(
    model="mantis/base",
    messages=[{"role": "user", "content": "Explain quicksort briefly."}],
)
print(response.choices[0].message.content)
```

Health and catalog:

```bash
curl -fsS http://127.0.0.1:8088/ready
curl -fsS http://127.0.0.1:8088/v1/models \
  -H "Authorization: Bearer $MANTIS_API_KEY"
```

The private Bifrost dashboard is available only on this machine at `http://127.0.0.1:8080`; log in with `BIFROST_ADMIN_USERNAME` and `BIFROST_ADMIN_PASSWORD`.

## Prime Agent

Use distinct provider IDs to avoid stale credentials stored for previous provider names. A local configuration should expose:

```text
mantis/base
mantis/trinity
mantis/ultra
mantis/fusion
```

All are configured with a 262,144-token context window. `mantis/trinity` and `mantis/ultra` return an experimental-mode error unless the stack was started with `--experimental`; Base and Fusion remain available normally. Resolve API keys from `MANTIS_API_KEY` at runtime rather than embedding them in `~/.prime/agent/models.json`.

Example headless task:

```bash
prime-agent --mode text --provider mantis --model base --no-session \
  -p 'Inspect this repository, implement the requested change, run tests, and fix failures.'
```

## Switching models and providers

`config/catalog.toml` is the only place that names Base models and the provider they ride on. Edit `[base.targets.efficient]` / `[base.targets.capable]` (or the `[providers.*]` they reference), then restart the stack. Launch regenerates Switchyard's `routes.toml` from that catalog; the API sends `base.route_id` (exported as `MANTIS_BASE_ROUTE_ID`). Application code does not hard-code upstream IDs.

To change how Base decides, set `base.algorithm` to `stage_router` (default, tool-signal routing) or `escalation` (weak-first with a judge). Optional `[base.targets.judge]` selects the escalation judge model.

Trinity, Ultra, and Fusion keep using `[mantis.workers]` and `[fusion]` in the same file.

## Routing and cost control

`mantis/base` uses Switchyard's stage router. Turns start on the efficient catalog target and escalate to the capable target when tool-result signals (errors, spinning, exploration vs production) clear `confidence_threshold`. Pass `X-Route-Session` or `metadata.session_id` so session state can stick across a coding loop.

See the [Switchyard stage-router docs](https://github.com/NVIDIA-NeMo/Switchyard/blob/main/docs/routing_algorithms/stage_router_routing.md) for signal details.

## Checks

```bash
uv run pytest tests -q
uv run ruff check .
./scripts/verify.sh
```

## Security

All services bind to loopback by default. Keep local API keys and provider credentials out of Git, protect `~/.zshrc` and runtime config files with mode `0600`, and do not expose ports 8080, 5500, or 8088 without TLS and non-default credentials.
