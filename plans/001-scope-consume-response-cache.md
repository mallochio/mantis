# Plan 001: Scope and consume Pi response cache entries

> **Executor**: Follow each step and verification gate. Update `plans/README.md` when done.
>
> **Drift check**: `git diff --stat 5874163..HEAD -- extensions/mantis.ts extensions/tests/stream-test.ts`. STOP if the cache flow has materially changed.

## Status
- **Priority**: P1
- **Effort**: S
- **Risk**: LOW
- **Depends on**: none
- **Category**: bug
- **Planned at**: `5874163`, 2026-08-03

## Why this matters
`sessionCache` exists only to bridge Pi's first provider call (which emits `mantis_step` tool calls) to the immediate follow-up call containing their tool results. Its key currently contains only messages, entries persist after use, and no-step responses are cached. Identical branches or mode switches can therefore receive stale output from another coordinator.

## Current state
- `extensions/mantis.ts:225` builds `key = JSON.stringify(messages)` without `model.id`.
- `extensions/mantis.ts:456-463` returns cached output without deleting it.
- `extensions/mantis.ts:506-516` caches before checking whether tool steps exist.
- Tests follow the provider/tool/provider bridge in `extensions/tests/stream-test.ts:182-287`.

## Scope
**In**: `extensions/mantis.ts`, `extensions/tests/stream-test.ts`.
**Out**: backend coordinator behavior, Pi core, Slipstream.

## Steps
1. Make cache identity include the Mantis provider model/mode while retaining the message-derived identity.
2. Cache only responses that emitted tool calls and need the second provider call.
3. Consume/delete an entry on successful cache retrieval. Do not turn this into a general response cache.
4. Add regression checks for: tool-call IDs bridge the immediate tool-result call to the final response despite repository-context mutation; replayed/consumed tool results fail closed instead of invoking the backend; mode switches do not collide; no-step responses are never cached.

## Verification
```bash
cd extensions && npm run typecheck && bun run tests/stream-test.ts
```
Expected: exit 0 and all cache regression checks pass.

## Done criteria
- Pending final responses are correlated by native `mantis_step` tool-call IDs, bridge-only, and single-use.
- Volatile repository context cannot cause a second orchestration after acceptance.
- No response without native tool steps is cached.
- No files outside Scope change, except `plans/README.md` status.

## STOP conditions
- Pi invokes more than one post-tool provider call for the same assistant turn in the installed runtime; report the observed event sequence before changing cache lifetime.
- Correctness requires modifying Pi or Slipstream.

## Maintenance
Keep this cache an internal Pi protocol bridge. Do not add TTL/config/general memoization without a demonstrated need.
