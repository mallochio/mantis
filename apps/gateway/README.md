# Mantis gateway

> Internal direct-mode gateway. Run with `uv run --directory apps/gateway ...`; it serves :5500 and is exposed to clients as `mantis` through the Mantis API at :8088.

The gateway implements Mantis's direct/Base mode: it scores one request and sends it
to one cheap, middle, or expensive model through Bifrost. Normal local startup exposes
Base and Fusion; Trinity and Ultra/Conductor require `llm-stack.sh start --experimental`. It exposes
OpenAI-compatible `/v1/chat/completions` and `/v1/responses` endpoints. All
tiers route through the local Bifrost gateway per the shared routing catalog.
The server uses one lifespan-owned asynchronous HTTP pool.

Supra-Router-51M is the routing signal. Complexity 1–2 goes cheap, 3–4 goes middle when configured, and 5 goes expensive.


## Setup and run

Use `uv`; do not put credentials in this repository.

```bash
uv sync --dev
./llm-router.sh
```

The default listener is `http://127.0.0.1:5500/v1`, model `auto`, with the
loopback-only development credential `sk-route-local`. Clients normally use
`mantis` at `http://127.0.0.1:8088/v1`; :5500 is the internal gateway endpoint.
A non-loopback bind requires an externally supplied `MANTIS_ROUTER_KEY` that
is not the default. The launcher refuses to stop a port owner unless its
recorded PID, working directory, command, and listening socket all identify
this checkout.

Targets, provider bindings, and the Supra complexity policy come from one
explicit source: the shared routing catalog (`[gateway]` section of
`~/.config/ai-routing/catalog.toml`) or `MANTIS_ROUTER_TARGETS_JSON`. This
internal cheap/middle/expensive Supra policy is distinct from Bifrost's
separately scoped `vk-interactive-complexity-pilot` lexical four-tier pilot.
There is no environment-variable fallback contract; `user` is not treated as a session
identifier unless `MANTIS_ROUTER_SESSION_FROM_USER=1` is set; prefer
`X-Route-Session` for conversation affinity.

## Behavior

### Responses API

Clients that want Responses must call `POST /v1/responses` explicitly. The
router preserves the Responses request and response shapes and sends the request
to `/responses`. It never translates Chat Completions to Responses or vice versa.

A target is Responses-capable when its configured `protocols` include
`responses`. If a selected tier is incompatible, the request is promoted to the
nearest higher compatible tier and returns
`x-route-reason: responses_protocol_upgrade`. Failover also skips incompatible
tiers. In the default deployment the middle and expensive tiers serve Responses
directly; a chat-only middle model is promoted to
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
Supra headers are returned and `x-route-score` is `n/a`. Requests may provide `X-Route-Session`,
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

- A prompt that completes cleanly on the cheap backend `MANTIS_ROUTER_PIN_CHEAP_AFTER`
  (default 5) times is pinned cheap and skips all scoring (no embedding call,
  no Supra inference) until `MANTIS_ROUTER_PIN_TTL_S` (default 7 days) elapses.
- A prompt whose cheap attempt explicitly refuses
  `MANTIS_ROUTER_PIN_EXPENSIVE_AFTER` (default 2) times is pinned expensive, so
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
Set `MANTIS_ROUTER_TRAINING_LOG=1` only when full-prompt training collection is
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
