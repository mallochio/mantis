# 004 — Mantis routing cost control

## Context

`mantis/base` routes each user request through a small local classifier (Supra)
that scores complexity 1–5 and maps it to a gateway backend tier (`cheap`,
`middle`, `expensive`, ...). Once a session is assigned a tier, a cache-oriented
ratchet keeps it at that tier or higher. The ratchet climbs freely but does not
downgrade by default. This is good for prompt-cache hit rates and bad for cost
when the user sends a mix of hard and easy turns, or when an early hard turn
pushes the session onto an expensive tier it never leaves.

Phase 1 (idle-time downgrade and N-turn rescoring) has been implemented.
`MANTIS_ROUTER_DOWNGRADE_IDLE_S` is opt-in and defaults to `0`.
`MANTIS_ROUTER_RESCORE_EVERY_N` defaults to `0` in the code but is set to `4`
in `launch/host/lib/llm-router.sh` (and the underlying
`apps/gateway/llm-router.sh`), which re-evaluates after a typical
"hard task + a few follow-up turns" block without churning the cache.

This plan records the next two cost-control ideas without implementing them.

## Phase 2 — Deterministic cascade on failures

### Idea

Try the cheapest proposed tier first. Only escalate to a stronger tier if the
call produces a deterministic failure signal. This is the FrugalGPT "LLM
cascade" pattern, but using free failure detectors instead of a learned quality
predictor.

### Failure signals to treat as escalation triggers

- Native refusal or `content_filter` finish reason.
- Empty or unparseable response body.
- Tool-call parse error or tool execution failure.
- `finish_reason` == `length` with an unfinished tool call.
- Structured-output `response_format` validation failure.

### Behavior

1. `_decide` returns the cheap tier as before.
2. `_open_with_failover` / `_open_responses_stream_with_failover` makes the
   call.
3. If the response matches a failure signal, retry with the next fallback in
   `_fallback_routes(decision)`.
4. Stop after two attempts (existing failover bound) or after success.

### Cost/latency trade-off

- Latency only increases when the cheap call fails; most simple queries pay the
  cheap price.
- The failure detector is rule-based (`_is_refusal`, validator, tool result
  `isError`) and adds no extra model cost.
- Risk: a cheap model may produce a plausible-but-wrong answer that passes all
  deterministic checks. That requires a quality verifier (Phase 3 or a separate
  improvement).

### Implementation sketch

- Extend `_is_refusal` or add `_is_cascade_failure(response, expected_format)`.
- In `_open_with_failover`, after the first response, inspect it before returning
  to the client. If it is a cascade failure and a fallback exists, call the
  fallback with the same request and append the attempt.
- Add `MANTIS_ROUTER_CASCADE=1` env to enable the behavior. Default off.

## Phase 3 — Budget pacing

### Idea

Track per-session or per-window spend and bias routing toward cheaper tiers as
spend grows. This is the ParetoBandit / WISERouter / SLARouter budget-pacing
pattern.

### Data sources

- `config/worker-costs.json` gives estimated per-call shadow prices.
- `usage.prompt_tokens`, `usage.completion_tokens`, and
  `usage.prompt_tokens_details.cached_tokens` from `_session_note` give actual
  token volumes.

### Behavior

1. Maintain a rolling cost window in session state (e.g., 10 minutes or last N
   calls).
2. Compute `spent` and `budget` from `MANTIS_ROUTER_BUDGET_USD` and
   `MANTIS_ROUTER_BUDGET_WINDOW_S`.
3. Apply a cost penalty that shifts `proposed` down by one rank when the window
   is, for example, 80% consumed, and down to the cheapest tier at 95%.

### Implementation sketch

- In `_session_note`, accumulate `session_cost` using `worker-costs.json`.
- In `_session_route` or `_target_for_complexity`, add a
   `_cost_pressure(state)` factor.
- Map the final decision through `_ranked_compatible_targets` starting from the
  pressure-adjusted floor instead of the raw `proposed`.

### Complexity and risks

- Requires tuning the budget and window per deployment.
- Can starve hard tasks if the budget is too tight.
- Needs a fallback escape hatch (`x-route-new-task: true` or `x-route-override`).

## When to consider each

- Phase 1 already solves "stuck on expensive after one hard turn".
- Phase 2 should come next if you see the cheap tier failing often enough that
  you still over-pay, but the failures are detectable (refusals, parse errors).
- Phase 3 is the right move if you have a hard cost cap per user/session and are
  willing to trade some quality for enforceable spend limits.
