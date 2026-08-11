# llm-router

> Part of the Mantis monorepo (`router/`). Full history: `git log 33b0ccd -- router/`. Run with `uv run --directory router ...`; the router serves :5500 and is exposed to clients as `mantis-basic` through Mantis :8088.

An OpenAI-compatible FastAPI router for explicit `/v1/chat/completions` and
`/v1/responses` endpoints. The router selects cheap, middle, and expensive tiers. In the default gateway profile,
all tiers use the Cloudflare gateway; its model catalog routes to OpenCode Go,
Modal, or OpenRouter. Direct two-backend deployments remain supported. The
server uses one lifespan-owned asynchronous HTTP pool.

Two routing modes are available (`ROUTELLM_ROUTER`):

- `supra` (default): the Supra-Router-51M complexity gate is the primary signal. With the gateway middle tier enabled, complexity 1–2 goes cheap,
  complexity 3–4 goes middle (Terra by default), and complexity 5+ goes
  expensive (Sol by default). Direct two-tier mode retains the legacy
  `ROUTELLM_SUPRA_THRESHOLD` cutoff. Override the expensive boundary with
  `ROUTELLM_EXPENSIVE_MIN_COMPLEXITY` when deliberately testing another policy. MF scoring is off by default; set
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

The launcher defaults to the Cloudflare gateway (`unified-ai-gateway.siddsantham.workers.dev`) and requires `AI_GATEWAY_API_KEY` (or `MANTIS_GATEWAY_API_KEY`). Default models are `deepseek-v4-flash` (cheap), `openai/gpt-5.6-terra` at maximum reasoning (middle), and `openai/gpt-5.6-sol` (expensive). Set `ROUTELLM_ENDPOINT_PROFILE=direct` to retain the direct two-backend contract with `EXPENSIVE_BASE`, `EXPENSIVE_KEY`, `CHEAP_BASE`, `CHEAP_KEY`, and their model variables. A direct middle tier can be enabled with `MIDDLE_BASE`, `MIDDLE_KEY`, and `MIDDLE_MODEL`. `user` is not treated as a session identifier unless `ROUTELLM_SESSION_FROM_USER=1` is set; prefer `X-Route-Session` for conversation affinity. `OPENAI_API_KEY` is required by MF scoring. See `server.py` for optional limits.

### Endpoint credential profiles

The launcher keeps provider keys for direct provider URLs. If either base URL's
parsed host is `unified-ai-gateway.siddsantham.workers.dev`, it automatically
uses `AI_GATEWAY_API_KEY` (or `MANTIS_GATEWAY_API_KEY`) for that backend. It
fails closed when the gateway token is absent. URL hosts are parsed and
compared without displaying credentials.

Set `ROUTELLM_ENDPOINT_PROFILE=direct` or `cloudflare` to override base-host
auto-detection for all tiers. `ROUTELLM_GATEWAY_MODE=1` also selects the
Cloudflare profile. The legacy `ROUTELLM_GATEWAY_MODE=1` also
selects the Cloudflare profile when no explicit profile is set. Direct bases
continue to use `EXPENSIVE_KEY` and `CHEAP_KEY`.

## Behavior

### Responses API

Clients that want Responses must call `POST /v1/responses` explicitly. The
router preserves the Responses request and response shapes and sends the request
to `/responses` through OpenRouter or the Cloudflare gateway. It never translates
Chat Completions to Responses or vice versa.

Only `openai/*` models on OpenRouter or the configured Cloudflare gateway are
Responses-capable. If a selected cheap or middle tier is incompatible, the
request is promoted to the nearest higher compatible tier and returns
`x-route-reason: responses_protocol_upgrade`. Failover also skips incompatible
tiers. With the default gateway models, Terra and Sol can serve Responses directly;
an explicitly configured `openai/gpt-5.6-luna` cheap tier may also serve
Responses directly. A non-OpenAI middle model such as `kimi-k3` is promoted to
Sol instead. The Terra middle tier uses maximum reasoning (`max`) for every
Responses request routed to it, overriding only the native `reasoning.effort`
field while preserving other reasoning fields such as `summary`. Cheap and
expensive tiers preserve the caller's explicit reasoning settings.

Responses streaming preserves provider SSE event frames and terminates on
`response.completed`, `response.failed`, `response.incomplete`, `error`, or a
provider terminal marker. Incomplete streams receive a Responses-native error
event; no synthetic success is emitted. Local Responses caching and coalescing
are disabled initially. Responses carry `x-route-api: responses` and
`x-route-upstream-path: /responses`; Chat carries the corresponding chat values.
- Encrypted reasoning, compaction items, provider-side `previous_response_id`,
  and `conversation` state are model/provider-origin-bound. The router keeps a
  short in-memory origin index (never the ciphertext) and binds such
  continuations to the exact target, base URL, upstream model, and target
  revision that produced them. Continuations with unknown, stale, or
  cross-origin state are rejected with HTTP 409
  (`responses_continuation_affinity_required`) instead of being rerouted or
  failovered to another model. Completed non-stream responses return an
  `x-route-responses-affinity` capability header as an extra origin check;
  streamed completions are indexed automatically from their native frames.

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
`n/a` unless `ROUTELLM_SCORE_WITH_MF=1`. Requests may provide `X-Route-Session`,
`metadata.session_id`, or `user`; the router stores only an HMAC digest. Session
affinity keeps short continuations on the current tier and requires an explicit
new-task signal to downgrade. It expires after one hour and is not persisted.
Responses expose `x-route-reason`, `x-route-sticky`, and the opaque
`x-route-session` digest. Logs include normalized prompt/cache token metrics when
upstream usage provides them.

### Learned per-prompt routing

This workload is dominated by repeated prompts (top 25 prompts were ~38% of
calls in the Aug 5-10 log), so the router keeps a persistent per-prompt
decision store (`decision-state.jsonl`, append-only, mode 0600) that survives
restarts:

- A prompt that completes cleanly on the cheap backend `ROUTELLM_PIN_CHEAP_AFTER`
  (default 5) times is pinned cheap and skips all scoring (no embedding call,
  no Supra inference) until `ROUTELLM_PIN_TTL_S` (default 7 days) elapses.
- A prompt whose cheap attempt explicitly refuses
  `ROUTELLM_PIN_EXPENSIVE_AFTER` (default 2) times is pinned expensive, so
  later requests skip the doomed cheap attempt entirely. Repeated identical
  requests are deliberately not escalation signals because agent loops often
  repeat prompts such as `Proceed` and polling instructions.
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
