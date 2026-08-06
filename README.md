# Mantis

Mantis serves the trained TRINITY router and Fugu-style Conductor as ordinary
OpenAI Chat Completions models. Any agent harness that supports OpenAI function
tools can use it; Mantis has no harness-specific integration.

## Models

| Model | Behavior |
|---|---|
| `mantis` | Default TRINITY orchestration |
| `mantis-trinity` | Force TRINITY |
| `mantis-ultra` | Force Conductor workflow orchestration |

Legacy aliases `trinity`, `fugu`, `conductor`, and `ultra` remain accepted.
Unknown model IDs are rejected.

TRINITY selects a worker and role (Worker, Thinker, or Verifier) each turn.
Conductor plans a bounded DAG, then executes its nodes against the configured
worker pool. Internal orchestration steps remain server-side. Only real tools
provided by the calling harness are returned as standard OpenAI `tool_calls`.

## Run

```bash
cp .env.example .env
# Set MANTIS_API_KEY and the provider keys used by your worker pool.

docker compose up --build -d
curl http://127.0.0.1:8088/health
```

For native MPS/CUDA execution:

```bash
./scripts/run_mantis_native.sh
```

The native script creates `.venv-mantis`, installs this project, prepares the
router vector if needed, and starts the same endpoint on port 8088.

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

`stream=True` returns standard `text/event-stream` Chat Completions chunks,
sends SSE keep-alive comments during hidden orchestration, and ends with
`data: [DONE]`. Answer content is buffered until orchestration verifies the
final response; responses include `X-Mantis-Streaming: buffered`.

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
| `MANTIS_WORKER_MODELS` | Ordered `provider/model[|effort]` pool | see `.env.example` |
| `MANTIS_CONDUCTOR_MODEL` | Conductor planner model spec | first worker |
| `MANTIS_LOCAL_MODELS` | Optional local HF worker pool | unset |
| `MANTIS_LOCAL_CONDUCTOR` | Optional local Conductor checkpoint | unset |
| `MANTIS_MAX_TURNS` | TRINITY turn cap | `5` |
| `MANTIS_WORKER_TIMEOUT` | Downstream timeout in seconds | `240` |
| `MANTIS_RUN_TTL` | Idle tool-run lifetime in seconds | `600` |
| `MANTIS_MAX_CONCURRENT_RUNS` | Bounded in-memory tool-run count | `32` |
| `MANTIS_MAX_CONCURRENT_REQUESTS` | Concurrent HTTP request limit; excess receives `429` | `32` |
| `MANTIS_SSE_KEEPALIVE_SECONDS` | SSE keep-alive interval during orchestration | `10` |
| `MANTIS_MAX_BODY_BYTES` | Maximum request body size | `5242880` |

Supported hosted model prefixes are currently `openrouter/` and `opencode-go/`.
Reasoning effort is appended with `|`, for example
`openrouter/openai/gpt-5.6-luna|max`.

Tool runs are held in bounded process memory until completion or TTL expiry.
Run one server replica unless you add shared state; a load balancer must use
sticky sessions for in-flight tool loops.

## API

- `GET /health` — public liveness check
- `GET /ready` — public configuration readiness check
- `GET /v1/models` — authenticated model list
- `POST /v1/chat/completions` — authenticated OpenAI-compatible completion

Internal routing and worker metadata are not returned to clients.

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
uv run mypy openfugu-patch scripts --exclude outputs
./scripts/verify.sh
```

`tests/test_api.py` and `tests/test_serve.py` check completions, tools, images,
structured output, usage, authentication, request limits, and SSE framing.

## Security

Use a strong `MANTIS_API_KEY`, put TLS and rate limits in front of public
deployments, and remember that one request may make several paid downstream
calls. Provider keys stay server-side and are never forwarded to clients.

Mantis is an independent OpenFugu-based implementation and is not affiliated
with Sakana AI. See `NOTICE` and `LICENSE`.
