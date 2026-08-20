# Mantis

Mantis is a local AI model orchestration and routing platform with an OpenAI-compatible API. The supported runtime is a three-process loopback stack:

```text
Bifrost :8080 → Mantis direct router :5500 → Mantis API :8088
```

No service binds publicly by default. Provider credentials remain on this machine.

## Modes

| Model | Mode | Use |
|---|---|---|
| `mantis/base` | Direct | Supra scores the request and chooses a cheap, middle, or expensive worker through Bifrost. |
| `mantis/trinity` | Trinity | Multi-agent coordination over Worker, Thinker, and Verifier roles. |
| `mantis/ultra` | Ultra | Conductor plans and executes a bounded workflow DAG over the worker pool. |
| `mantis/fusion` | Fusion | Lead/sidekick orchestration with tool-use follow-ups and a review loop. |

Mantis never switches among these four modes automatically. Base and Fusion are available in normal mode; Trinity and Ultra are experimental and require starting the stack with `--experimental`.

## Layout

```text
mantis/
├── apps/
│   ├── api/                    # Mantis API (:8088)
│   └── gateway/                # Internal Supra router (:5500)
├── artifacts/                  # Trinity router vector/head
├── config/
│   ├── catalog.toml            # Shared model/provider routing catalog
│   ├── bifrost.template.json   # Canonical Bifrost providers, keys, rules and complexity policy
│   └── worker-costs.json       # Evaluation price tables
├── launch/host/
│   ├── llm-stack.sh            # Full local stack controller
│   └── lib/                    # Bifrost, gateway and API launchers
├── scripts/                    # Development, retraining and verification
├── eval/                       # Router evaluations and SWE-rebench harnesses
└── tests/                      # Test suite
```

## Local setup

### Requirements

- macOS or Linux with `zsh`, `curl`, and `lsof`
- Node/npm (`npx` launches `@maximhq/bifrost`)
- [`uv`](https://docs.astral.sh/uv/)
- Python 3.13
- Provider credentials exported in `~/.zshrc`

Install the Python environments:

```bash
cd ~/Personal/other/mantis
uv sync --locked --no-dev
cd apps/gateway && uv sync --dev
```

Create private runtime directories and seed Bifrost from the tracked, secret-free template:

```bash
install -d -m 700 ~/.local/share/bifrost/logs ~/.local/share/mantis/router ~/.local/share/llm-stack
install -m 600 config/bifrost.template.json ~/.local/share/bifrost/config.json
ln -sfn "$PWD/config/catalog.toml" ~/.config/ai-routing/catalog.toml
```

The Bifrost template references environment variables; never place secret values in the tracked JSON. At minimum configure:

```text
MANTIS_API_KEY
MANTIS_ROUTER_KEY
BIFROST_API_KEY                 # must start with sk-bf-
BIFROST_ENCRYPTION_KEY
BIFROST_ADMIN_USERNAME
BIFROST_ADMIN_PASSWORD
BIFROST_COMPLEXITY_PILOT_KEY    # must start with sk-bf-
```

Provider routes additionally require the matching Azure, AWS/Bedrock, Vertex ADC, OpenRouter, or OpenCode credentials. For Vertex, export `GOOGLE_APPLICATION_CREDENTIALS`, `VERTEXAI_PROJECT`, and `VERTEXAI_LOCATION`.

## Start and stop

Install the optional StartupFolder-compatible controller link:

```bash
ln -sfn "$PWD/launch/host/llm-stack.sh" ~/Startup/llm-stack.sh
```

Control the stack:

```bash
# Normal mode: Bifrost + Supra + Mantis API; exposes Base and Fusion only.
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

A bare StartupFolder invocation defaults to `start` in normal mode. The controller starts Bifrost, the gateway, and the API in dependency order and checks readiness. Trinity's Qwen coordinator remains lazy and is not loaded unless an experimental Trinity request is made; Ultra/Conductor also starts work only on an experimental request. Runtime state/logs live under `~/.local/share/bifrost` and `~/.local/share/mantis`.

You can also run only the Mantis API in the foreground for development:

```bash
./scripts/run_mantis_native.sh
```

That command expects Bifrost and the internal gateway to be available separately.

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

## Bifrost complexity router

The complexity ladder is scoped to virtual key `vk-interactive-complexity-pilot`; normal Mantis traffic continues to use the Supra router. Send direct pilot requests to local Bifrost with model `auto`:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
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
| Unclassified `auto` | Azure `gpt-5.6-terra` | Bedrock Terra → OpenRouter Terra |

The analyzer uses coding/systems vocabulary, conservative multiword Reasoning triggers, and `.10 / .35 / .60` boundaries. The default priority-5 route prevents unclassified `auto` prompts from failing provider resolution.

The private dashboard is available only on this machine at `http://127.0.0.1:8080`; log in with `BIFROST_ADMIN_USERNAME` and `BIFROST_ADMIN_PASSWORD`.

## Prime Agent

Use distinct provider IDs to avoid stale credentials stored for previous provider names. A local configuration should expose:

```text
mantis-local/base
mantis-local/trinity
mantis-local/ultra
mantis-local/fusion
bifrost-local-complexity/auto
```

All are configured with a 262,144-token context window. `mantis-local/trinity` and `mantis-local/ultra` return an experimental-mode error unless the stack was started with `--experimental`; Base, Fusion, and Bifrost Auto remain available normally. API keys should resolve from `MANTIS_API_KEY` and `BIFROST_COMPLEXITY_PILOT_KEY` at runtime rather than being embedded in `~/.prime/agent/models.json`.

Example headless task:

```bash
prime-agent --mode text --provider mantis-local --model base --no-session \
  -p 'Inspect this repository, implement the requested change, run tests, and fix failures.'
```

## Routing and cost control

`mantis/base` uses Supra to select one worker tier per request and a session ratchet to keep a warm prompt-cache prefix on one tier. It climbs freely but does not downgrade by default.

- `MANTIS_ROUTER_DOWNGRADE_IDLE_S=N`: after `N` seconds of silence, the next turn can drop to the freshly scored tier.
- `MANTIS_ROUTER_RESCORE_EVERY_N=N`: every `N` completed turns, release the ratchet and use the classifier's current tier.

See [`apps/gateway/README.md`](apps/gateway/README.md) for routing behavior and internal protocol details.

## Checks

```bash
uv run pytest tests -q
uv run ruff check .
./scripts/verify.sh
```

## Security

All services bind to loopback by default. Keep local API keys and provider credentials out of Git, protect `~/.zshrc` and runtime config files with mode `0600`, and do not expose ports 8080, 5500, or 8088 without TLS and non-default credentials.
