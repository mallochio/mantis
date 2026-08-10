# llm-router

An OpenAI-compatible FastAPI router for `/v1/chat/completions`. The router
selects one of two provider backends. The server calls each configured provider
directly with one lifespan-owned asynchronous HTTP pool.

Two routing modes are available (`ROUTELLM_ROUTER`):

- `supra` (default): the Supra-Router-51M complexity gate is the primary
  signal. Prompts with complexity >= `ROUTELLM_SUPRA_THRESHOLD` (3) go to the
  expensive backend. MF scoring is off by default; set
  `ROUTELLM_SCORE_WITH_MF=1` for observability (it never influences the
  decision). This mode was chosen because on the Aug 5-10 workload the RouteLLM
  MF score had essentially zero separation against the Gemini difficulty
  labels (AUC 0.52 vs 0.66 for Supra), and the MF gate on top of Supra only
  added wasted expensive calls.
- `mf` (legacy): RouteLLM MF score >= `ROUTELLM_THRESHOLD`, with Supra as a
  secondary gate below the threshold. `bert` also selects the old checkpoint.

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

### Endpoint credential profiles

The launcher keeps provider keys for direct provider URLs. If either base URL's
parsed host is `unified-ai-gateway.siddsantham.workers.dev`, it automatically
uses `AI_GATEWAY_API_KEY` (or `MANTIS_GATEWAY_API_KEY`) for that backend. It
fails closed when the gateway token is absent. URL hosts are parsed and
compared without displaying credentials.

Set `ROUTELLM_ENDPOINT_PROFILE=direct` or `cloudflare` to override base-host
auto-detection for both backends. The legacy `ROUTELLM_GATEWAY_MODE=1` also
selects the Cloudflare profile when no explicit profile is set. Direct bases
continue to use `EXPENSIVE_KEY` and `CHEAP_KEY`.

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
  never cached. Entry count, buffered streams, and total bytes are bounded.
  Concurrent identical keyed requests share one bounded in-flight result.
  Cache counters appear in `/healthz`.

Every routed response includes `x-request-id`, `x-route-decision`,
`x-route-model`, `x-route-score`, `x-route-attempts`, and `x-route-fallback`.
Optional Supra headers are also returned. In supra mode `x-route-score` is
`n/a` unless `ROUTELLM_SCORE_WITH_MF=1`.

### Learned per-prompt routing

This workload is dominated by repeated prompts (top 25 prompts were ~38% of
calls in the Aug 5-10 log), so the router keeps a persistent per-prompt
decision store (`decision-state.jsonl`, append-only, mode 0600) that survives
restarts:

- A prompt that completes cleanly on the cheap backend `ROUTELLM_PIN_CHEAP_AFTER`
  (default 5) times is pinned cheap and skips all scoring (no embedding call,
  no Supra inference) until `ROUTELLM_PIN_TTL_S` (default 7 days) elapses.
- A prompt whose cheap attempt refuses (or is retried by the client)
  `ROUTELLM_PIN_EXPENSIVE_AFTER` (default 2) times is pinned expensive, so
  later requests skip the doomed cheap attempt entirely.
- Stats reset after 24h without a new note, so changed prompt behavior
  re-learns. Pinned responses carry `x-route-pinned: true`.
- Supra generation stops as soon as the `Complexity:` digit is emitted
  (greedy decode is deterministic, so the parsed value is unchanged); median
  scoring latency drops from ~480ms to ~60ms.

## Logs and training

Operational logs contain prompt hashes and routing metadata, not prompt text.
Set `ROUTELLM_TRAINING_LOG=1` only when full-prompt training collection is
explicitly required. Directories use mode `0700`; private files use `0600`.
Request and outcome occurrence IDs support exact evaluation joins. Old outcome
rows without IDs use the legacy prompt-hash join.

`pseudo_label.py` validates closed label enums/schema. A fsynced append-only
journal is canonical; both output projections are rebuilt atomically from its
committed offset before the state checkpoint advances. Recovery truncates an
uncommitted journal tail after a crash.

## Checks

One command runs collected tests, static checks, syntax checks, and local
self-tests. Tests install a strict socket guard and use ASGI plus mock upstreams;
they make no real or billed calls.

```bash
./checks.sh
```
