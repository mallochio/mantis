# Mantis

Mantis provides three OpenAI-compatible routing modes. Choose the mode
explicitly.

## Modes

| Model | Mode | Use |
|---|---|---|
| `mantis` | Direct | The gateway scores one request with Supra and sends it to a cheap, middle, or expensive model through Bifrost. |
| `mantis-trinity` | Trinity | Multi-agent coordination over Worker, Thinker, and Verifier roles. |
| `mantis-ultra` | Ultra | Conductor plans and executes a bounded workflow DAG over the worker pool. |

Only these model IDs are accepted. Select `mantis-trinity` or `mantis-ultra`
manually; Mantis never selects between modes. All three paths send model calls
through the local Bifrost gateway. The OpenCode, Pi, and Prime harness catalogs
advertise a 256k-token context limit for every configured local model.

## Repository layout

`apps/api/` serves the three public modes at :8088. `apps/gateway/` is the
internal direct-mode gateway at :5500. They share the routing catalog but keep
separate dependency locks because the API includes orchestration and training
dependencies that the gateway does not need.

`mantis` responses carry `x-route-decision`, `x-route-reason`,
`x-route-sticky`, `x-route-model`, `x-route-attempts`, and
`x-route-fallback`. Send `X-Route-Session` to retain routing affinity across
turns. Internal orchestration steps remain server-side. Only real tools
provided by the calling harness are returned as standard OpenAI `tool_calls`.

## Run

The canonical host deployment is the local launcher stack, versioned at
`launch/host/` with `~/Startup/` symlinking to it (StartupFolder runs it at
login):

```bash
~/Startup/llm-stack.sh start   # Bifrost :8080 -> gateway :5500 -> Mantis API :8088
```

`config/catalog.toml` is the tracked, versioned source of truth for the
gateway and API worker pool; `~/.config/ai-routing/catalog.toml` is a symlink to it. `bifrost.json`
(contains the gateway encryption key) stays local and untracked.

## Portable host installation

Mantis has no container runtime requirement. Install [uv](https://docs.astral.sh/uv/),
clone this repository, create `.env` (or configure the shared routing catalog), then run:

```bash
./scripts/run_mantis_native.sh
```

The launcher uses the locked uv environment, prepares the router vector when
needed, renders the catalog bindings for the host, and starts the endpoint at
`127.0.0.1:8088`. It works on macOS, Linux, and WSL. Torch chooses MPS, CUDA,
or CPU automatically. Set `MANTIS_HOST=0.0.0.0` only when remote access is
required.

## Call it like any OpenAI model

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8088/v1",
    api_key="sk-mantis-local",
)

response = client.chat.completions.create(
    model="mantis",
    messages=[{"role": "user", "content": "Explain quicksort briefly."}],
)
print(response.choices[0].message.content)
```

`stream=True` returns standard `text/event-stream` Chat Completions chunks.
During orchestration it streams safe model, role, order, retry, and verification
status through `delta.reasoning` plus a versioned `mantis_event`. Answer content
remains buffered until verification and output validation succeed, then it is
replayed as paced content chunks before `data: [DONE]`. Set
`X-Mantis-Events: none` to suppress status
frames. Responses include
`X-Mantis-Streaming: live-status,verified-buffered-content`.

## Agent tools

Send normal OpenAI function tools. Mantis supports `tool_choice` values `auto`,
`none`, `required`, and a named function, and may return
`finish_reason: "tool_calls"`.
The harness executes those calls and sends the assistant tool-call message plus
`role: "tool"` results back to the same `/v1/chat/completions` endpoint. The
opaque tool-call IDs carry the temporary run identity, so no custom continuation
API or client adapter is required.

```python
response = client.chat.completions.create(
    model="mantis",
    messages=[{"role": "user", "content": "Read pyproject.toml."}],
    tools=[{
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from the active workspace",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    }],
)
```

The calling harness owns tool execution and its filesystem/network permissions.
Mantis only chooses and orchestrates models.

## Images and structured output

User messages may contain standard OpenAI `text` and `image_url` parts. Mantis
routes on text and preserves images for worker calls. Workers must support the
image format you send.

`response_format` accepts `json_object` or `json_schema`. Mantis forwards the
format to answer-producing workers and validates the final JSON with
`jsonschema`; invalid output fails instead of being returned as structured data.

Standard `max_tokens`/`max_completion_tokens`, `reasoning`/`reasoning_effort`,
and `web_search_options` controls are passed to answer workers. Support depends
on the configured provider and worker model; Mantis does not implement its own
reasoning engine or search crawler.

## Configuration

| Variable | Purpose | Default |
|---|---|---|
| `MANTIS_API_KEY` | Bearer token required by `/v1/*` | required |
| `OPENROUTER_API_KEY` | OpenRouter worker credentials | optional by pool |
| `OPENCODE_API_KEY` | OpenCode Go worker credentials | optional by pool |
| `MANTIS_MODEL` | TRINITY router backbone | `Qwen/Qwen3-0.6B` |
| `MANTIS_VECTOR` | Trained TRINITY vector | `artifacts/model_iter_60.npy` |
| `MANTIS_HEAD` | Optional head override | unset |
| `MANTIS_WORKER_MODELS` | Ordered provider/model[effort] pool | see `.env.example` |
| `MANTIS_CONDUCTOR_MODEL` | Conductor planner model spec | first worker |
| `MANTIS_LOCAL_MODELS` | Optional local HF worker pool | unset |
| `MANTIS_LOCAL_CONDUCTOR` | Optional local Conductor checkpoint | unset |
| `MANTIS_MAX_TURNS` | TRINITY turn cap | `5` |
| `MANTIS_WORKER_TIMEOUT` | Downstream timeout in seconds | `240` |
| `MANTIS_RUN_TTL` | Idle tool-run lifetime in seconds | `600` |
| `MANTIS_RUN_STORE` | Run state backend: `memory` or optional `redis` | `memory` |
| `MANTIS_REDIS_URL` | Redis URL when `MANTIS_RUN_STORE=redis` | unset |
| `MANTIS_REDIS_PREFIX` | Key namespace for shared Redis state | `mantis:run:` |
| `MANTIS_ALLOW_DEBUG_TRACE` | Allow `X-Mantis-Details: debug` responses | `0` |
| `MANTIS_MAX_CONCURRENT_RUNS` | Bounded tool-run count per backend | `32` |
| `MANTIS_MAX_CONCURRENT_REQUESTS` | Concurrent HTTP request limit; excess receives `429` | `32` |
| `MANTIS_SSE_KEEPALIVE_SECONDS` | SSE keep-alive interval during orchestration | `10` |
| `MANTIS_STREAM_EVENTS` | Stream safe orchestration status and model-order events | `1` |
| `MANTIS_FINAL_CHUNK_DELAY_MS` | Delay between verified final-answer chunks | `5` |
| `MANTIS_MAX_BODY_BYTES` | Maximum request body size | `52428800` |
| `MANTIS_UPSTREAM_STREAM` | Stream provider responses upstream (SSE) instead of buffering | `1` |
| `MANTIS_CACHE_BREAKPOINTS` | Add prompt-cache breakpoints to Claude-family requests | `1` |
| `MANTIS_ROUTER_URL` | Base URL of the in-repo Mantis router for `mantis` | `http://127.0.0.1:5500/v1` |
| `MANTIS_ROUTER_TIMEOUT_S` | Upstream timeout for `mantis` calls | `300` |
| `ROUTELLM_KEY` | Compatibility name for the router bearer token | required |

Supported hosted model prefixes are currently `openrouter/` and `opencode-go/`.
OpenRouter `openai/*` workers use the stateless Responses API with stable,
privacy-safe cache keys, sticky session routing, and automatic prompt-cache
breakpoints; all other hosted workers use Chat Completions. Provider base URLs are overridable with
`OPENROUTER_BASE_URL` and `OPENCODE_GO_ENDPOINT_URL`, so the whole pool can be
pointed at a pass-through proxy (e.g. a Cloudflare Worker gateway) by setting the
matching API key to the gateway secret. An OpenRouter proxy must forward both
`/v1/chat/completions` and `/v1/responses`. Upstream streaming keeps long
generations alive through such proxies. Cached prompt tokens are accounted per
model in the `mantis` details and billed at the model's cache-read price when known.
Reasoning effort is appended with `|`, for example
`openrouter/openai/gpt-5.6-luna|max`.

Tool runs use bounded process memory by default until completion or TTL expiry.
For multi-replica deployment, set `MANTIS_RUN_STORE=redis` and configure
`MANTIS_REDIS_URL` for a trusted, private Redis deployment. Redis stores
serialized run state and locks each advance, so replicas can share tool loops.
Run one replica with the memory backend.

## API

- `GET /health` — public liveness check
- `GET /ready` — public configuration readiness check
- `GET /v1/models` — authenticated model list
- `POST /v1/chat/completions` — authenticated OpenAI-compatible completion

By default the response is a plain OpenAI-compatible completion; internal
routing and worker metadata are not returned. Send the opt-in header
`X-Mantis-Details: summary` to add a deterministic orchestration summary in a
`mantis` response object and matching `X-Mantis-Run-Id` / `X-Mantis-Mode` /
`X-Mantis-Outcome` / `X-Mantis-Duration-Ms` / `X-Mantis-Cost-Usd` response
headers:

```json
{
  "mantis": {
    "run_id": "...",
    "mode": "trinity",
    "outcome": "verifier_accept",
    "duration_ms": 18420.0,
    "activity": [
      {"type": "step", "role": "Worker", "model": "...", "status": "completed", "summary": "Drafted the answer"},
      {"type": "complete", "status": "completed", "summary": "Run completed"}
    ],
    "usage": {
      "total": 0.0142,
      "known": true,
      "source": "price_table",
      "models": [{"model": "...", "prompt_tokens": 5, "completion_tokens": 7, "cost": 0.0142, "source": "price_table"}]
    }
  }
}
```

Activity summaries are derived deterministically from orchestration events
(model calls, tool calls/results, retries/failovers, verification outcomes).
They are not model-generated reasoning and contain no prompts, completions, or
tool payloads. `X-Mantis-Details: debug` additionally exposes per-step timing
and failover attempts; it is gated behind `MANTIS_ALLOW_DEBUG_TRACE=1`.
Streaming requests with `stream_options.include_usage` emit a final `mantis`
SSE frame before `[DONE]` when details are requested.

`/v1/models` descriptors report `context_length` and `max_completion_tokens`, both env-configurable via `MANTIS_CONTEXT_LENGTH` and `MANTIS_MAX_COMPLETION_TOKENS`; downstream cost is reported per request in `usage.cost`.

## Artifacts

The small baseline head is tracked at `artifacts/router_head.safetensors`.
Build or download the runtime vector with:

```bash
./scripts/download_artifacts.sh
# or
python3 scripts/make_vec.py
```

Large checkpoints and generated outputs remain outside Git.

## Verify

```bash
uv run pytest tests -q
uv run ruff check .
uv run mypy apps/api scripts --exclude outputs
./scripts/verify.sh
uv sync --directory apps/gateway --locked --all-groups
uv run --directory apps/gateway pytest tests -q
```

`tests/test_api.py` and `tests/test_serve.py` check completions, tools, images,
structured output, usage, authentication, request limits, and SSE framing. The
gateway suite lives under `apps/gateway/tests/` and keeps its own lockfile and
venv.

## Security

Use a strong `MANTIS_API_KEY`, put TLS and rate limits in front of public
deployments, and remember that one request may make several paid downstream
calls. Provider keys stay server-side and are never forwarded to clients.

Mantis is an independent OpenFugu-based implementation and is not affiliated
with Sakana AI. See `NOTICE` and `LICENSE`.
