# Plan 005: Calibrate context accounting and remove duplicate history

> **Drift check**: `git diff --stat 5874163..HEAD -- extensions/mantis.ts openfugu-patch/serve.py tests/test_serve.py extensions/tests/stream-test.ts README.md .env.example`. STOP on material context-flow drift.

## Status
- **Priority**: P2
- **Effort**: M
- **Risk**: MED
- **Depends on**: plan 004
- **Category**: perf
- **Planned at**: `5874163`, 2026-08-03

## Why this matters
`HistoryWorker._combine` already includes prior assistant messages, then `_enhance_worker_messages` inserts the latest assistant response again. This inflates prompts and can bias workers. Separately, Pi advertises a fixed 256K window without a calibration knob tied to the smallest configured worker.

## Current state
- `openfugu-patch/serve.py:113-130` prepends history.
- `openfugu-patch/serve.py:152-195` duplicates the latest assistant in the current user prompt.
- `extensions/mantis.ts:448-453` estimates envelope usage at chars/4.
- `extensions/mantis.ts:577-604` hardcodes 256000 for all provider modes.
- LiteLLM config declares no explicit context limits.

## Scope
**In**: `openfugu-patch/serve.py`, `tests/test_serve.py`, `extensions/mantis.ts`, `extensions/tests/stream-test.ts`, `README.md`, `.env.example`.
**Out**: provider tokenizer dependencies, automatic OpenRouter metadata discovery, changing Slipstream defaults.

## Steps
1. Remove the duplicate prior-answer wrapper; rely on ordered history plus the current query. Add a multi-turn assertion that each prior assistant text occurs once per worker request.
2. Add one calibration environment variable for the advertised Mantis context window, defaulting to the current 256000. Validate it as a sensible positive integer and use it for all three modes.
3. Document that operators must set it to the smallest effective worker window, leaving output reserve intact. Keep chars/4 usage estimation; do not add tokenizer dependencies.
4. Ensure repository/system context is counted once and oversized input still raises a Pi-recognizable overflow error.
5. Test default, override, invalid value, multi-turn prompt shape, and overflow boundary.

## Verification
```bash
uv run pytest -q --no-cov tests/test_serve.py
cd extensions && npm run typecheck && npm test
```
Expected: all pass; prior assistant content has no duplicate occurrence.

## Done criteria
- No duplicate history injection.
- Context window has one documented calibration knob and defaults to 256K.
- Pi/Slipstream usage and overflow tests pass.

## STOP conditions
- A configured worker has no discoverable/documented context limit; preserve the operator knob and document uncertainty rather than guessing.
