# Plan 004: Stream worker events incrementally and reject corrupt streams

> **Drift check**: `git diff --stat 5874163..HEAD -- extensions/mantis.ts extensions/tests/stream-test.ts openfugu-patch/serve.py tests/test_serve.py`. STOP on material transport drift.

## Status
- **Priority**: P1
- **Effort**: M
- **Risk**: MED
- **Depends on**: plans 001–003
- **Category**: bug
- **Planned at**: `5874163`, 2026-08-03

## Why this matters
The backend flushes NDJSON per step, but the extension calls `collectStream` and waits for completion before emitting Pi tool calls. It also ignores malformed JSON and accepts a stream with no terminal result, which can look like dropped turns or a successful empty answer.

## Current state
- `extensions/mantis.ts:350-380` parses NDJSON but silently ignores malformed lines.
- `extensions/mantis.ts:386-389` buffers generators.
- `extensions/mantis.ts:470-504` processes only after buffering and does not require exactly one result.
- Backend event emission is `openfugu-patch/serve.py:637-676`.

## Scope
**In**: `extensions/mantis.ts`, `extensions/tests/stream-test.ts`; backend/tests only for protocol contract assertions.
**Out**: SSE/WebSockets, custom Pi message types, parallel tool execution.

## Steps
1. Process `streamOrchestrate()` with `for await` and preserve event order; remove `collectStream` if unused.
2. Emit native `mantis_step` tool-call events as each `step-end` arrives while retaining the final result for the single-use bridge from plan 001.
3. Treat malformed nonblank NDJSON, duplicate terminal results, step-end without matching start where applicable, and EOF without result/error as provider errors.
4. Preserve legacy JSON completion fallback.
5. Add chunk-boundary, malformed-line, truncated-stream, backend-error, empty-reply, and cancellation tests. Assert tool events precede terminal backend completion in the mock.

## Verification
```bash
cd extensions && npm run typecheck && npm test
uv run pytest -q --no-cov tests/test_serve.py
```
Expected: all pass; no `collectStream(` remains if no caller needs it.

## Done criteria
- Pi receives turns incrementally in backend order.
- Corrupt/truncated streams fail, never become `(no response)` success.
- Legacy JSON and tool-result bridge still work.

## STOP conditions
- Pi's provider contract rejects toolcall events before backend completion; capture the actual Pi error/event trace and report before inventing custom UI.
