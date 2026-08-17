# 004 — Mantis routing cost control

**Status:** Partial — Phase 1 implemented; Phases 2 and 3 deferred
**Priority:** P2

## Current state

Phase 1 (idle-time downgrade and periodic rescoring) is implemented in
`apps/gateway/server.py`. Both router launchers configure
`MANTIS_ROUTER_RESCORE_EVERY_N=4`:

- `launch/host/lib/llm-router.sh`
- `apps/gateway/llm-router.sh`

The remaining phases are deliberately not implementation commitments.

## Phase 2 — failure-triggered cascade

Try a cheaper tier first and escalate only on deterministic failures such as a
refusal, empty/unparseable output, tool-call failure, or structured-output
validation failure.

**Implement when:** logs show cheap-tier deterministic failures materially
affecting coding-agent completion. First establish the failure rate and whether
those failures are reliably detectable; do not add a cascade based on cost
intuition alone.

## Phase 3 — budget pacing

Bias routing toward cheaper tiers as a real per-session or rolling-window spend
limit is approached.

**Implement when:** a real per-session/window cost cap is required. Before then,
do not add speculative budget state or tuning knobs.
