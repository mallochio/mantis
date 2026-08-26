# 005 — Fusion hardening

**Status:** Active remediation plan  
**Priority:** P0–P2, with deferred follow-on work

## Confirmed scope

- **Headless execution (P0):** `scripts/fusion_headless.py` runs model-supplied
  `bash -c` commands in `TemporaryDirectory`. That directory is not a sandbox or
  Git worktree, and generated work is discarded. Remove/deprecate this harness,
  or require a real isolated worktree plus an explicit unsafe opt-in. If it
  remains, use an existing container/VM/OS platform boundary; do not invent a
  bespoke sandbox.
- **Usage accounting (P1):** `apps/api/fusion.py` calls `add_usage`, while
  `apps/api/providers.py` also accounts usage against `active_run`. Add a focused
  provider-path test to determine whether the same response is counted twice,
  then retain exactly one accounting owner.
- **Review parsing (P1):** `FusionRun._parse_main_review` in
  `apps/api/fusion.py` accepts any word-boundary `ACCEPT` and defaults malformed
  output to acceptance. Require an exact, typed/structured decision; reject or
  safely handle malformed output.
- **Tool-result identity (P1):** `FusionRun._validate_tool_results` accepts
  unknown IDs and preserves client ordering. Require exact pending-call identity,
  reject duplicates/unknowns, and canonicalize results to pending-call order.
- **Bounded results (P1):** Fusion tool results are not capped at the API/state
  boundary. Apply `RUN_MAX_MSG_BYTES` (with an explicit truncation policy) before
  persistence and provider replay, not only in native-run paths.
- **Admission and errors (P2):** Native `/v1/fusion/*` routes in
  `apps/api/api.py` bypass `_capacity`, while chat-completions Fusion uses normal
  admission. Add Fusion admission limits and typed HTTP errors for validation,
  capacity, and provider/transient failures.
- **Follow-up contract:** Chat textual follow-up is wired via `advance(..., message=)`
  and `X-Mantis-Run-Id` / encoded tool-call resume. Native `/v1/fusion/follow_up`
  remains tool-results-only; document or extend native textual follow-up only when
  callers need it. The threaded `message` argument is no longer unused on the
  chat path.
- **At-least-once behavior:** Provider-boundary crash recovery can repeat calls;
  document this clearly and add recovery tests before attempting stronger
  journaling.
- **Delegate lifecycle:** Delegate creation is non-idempotent. Document the
  duplicate-run behavior now; add idempotency only when callers need retry-safe
  creation.

## Deferred (add only when)

- **Journaling:** add when duplicate provider spend or externally visible
  repeated transitions are observed in production and stronger recovery is
  justified.
- **Advanced locking:** `apps/api/runs.py` confirms memory has no outer run lock,
  file locks are process-local, and Redis locks use fixed leases. Add topology-
  appropriate locking/CAS when multi-process file storage or long-running Redis
  turns are supported; otherwise enforce/document supported deployment topology.
- **Delegate idempotency:** add when clients need safe retry after ambiguous
  delegate responses.
- **Model affinity:** add when failover/session cache or provider-bound state
  causes observed quality or cost regressions.
- **Comparative evaluation:** add a checked-in Fusion quality/cost comparison
  when a real product decision depends on Fusion’s quality/cost trade-off; none
  is currently checked in.

## Order of work

1. Disable or gate unsafe headless execution and define the artifact/worktree
   contract (P0).
2. Add focused accounting, review-parser, tool-identity/order, and bounded-result
   tests/fixes (P1).
3. Add native Fusion admission control and typed error behavior (P2).
4. Reassess deferred items from observed failures, deployment topology, and
   measured cost/quality data.

## Verification targets

- Focused Fusion tests exercise the real provider accounting path, not only a
  mocked `_call_worker`.
- Tests cover ambiguous review text, malformed review output, extra/duplicate/
  reordered tool results, and oversized results.
- `scripts/fusion_headless.py` cannot be used unattended without an explicit
  unsafe choice and a real isolated worktree/platform boundary.
