# llm-router

An OpenAI-compatible FastAPI router for `/v1/chat/completions`. RouteLLM MF and
Supra select one of two provider backends. The server calls each configured
provider directly with one lifespan-owned asynchronous HTTP pool.

## Setup and run

Use `uv`; do not put credentials in this repository.

```bash
uv sync --dev
./llm-router.sh
```

The default listener is `http://127.0.0.1:5500/v1`, model `auto`, with the
loopback-only development credential `sk-route-local`. A non-loopback bind
requires an externally supplied `ROUTELLM_KEY` that is not the default. The
launcher refuses to stop a port owner unless its recorded PID, working
directory, command, and listening socket all identify this checkout.

Required provider configuration is `EXPENSIVE_BASE`, `EXPENSIVE_KEY`,
`EXPENSIVE_MODEL`, `CHEAP_BASE`, `CHEAP_KEY`, and `CHEAP_MODEL`.
`OPENAI_API_KEY` is required by MF scoring. See `server.py` for optional limits.

## Behavior

- Request JSON and supported Chat Completions field types are validated.
  Reviewed unknown provider extensions are preserved.
- The body limit is enforced while bytes are read, even without a valid
  `Content-Length`.
- A request has at most two provider attempts. Transport errors, timeouts, 429,
  and selected 5xx statuses can fail over. Both attempts share one deadline.
- Streams require an upstream finish reason and `[DONE]`. Abrupt EOF produces
  an error event and never a synthetic success marker. Data after `[DONE]` is
  discarded.
- Response replay is off by default. To opt in, send `Idempotency-Key`.
  Tool-bearing requests and responses, refusals, and incomplete responses are
  never cached. Entry count and total bytes are bounded. Cache counters appear
  in `/healthz`.

Every routed response includes `x-request-id`, `x-route-decision`,
`x-route-model`, `x-route-score`, `x-route-attempts`, and `x-route-fallback`.
Optional Supra headers are also returned.

## Logs and training

Operational logs contain prompt hashes and routing metadata, not prompt text.
Set `ROUTELLM_TRAINING_LOG=1` only when full-prompt training collection is
explicitly required. Directories use mode `0700`; private files use `0600`.
Request and outcome occurrence IDs support exact evaluation joins. Old outcome
rows without IDs use the legacy prompt-hash join.

`pseudo_label.py` validates closed label enums/schema. It fsyncs both result
files before atomically advancing its state checkpoint.

## Checks

One command runs collected tests, static checks, syntax checks, and local
self-tests. Tests install a strict socket guard and use ASGI plus mock upstreams;
they make no real or billed calls.

```bash
./checks.sh
```
