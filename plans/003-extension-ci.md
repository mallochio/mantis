# Plan 003: Gate Pi extension behavior in CI

> **Drift check**: `git diff --stat 5874163..HEAD -- .github/workflows/ci.yml extensions/package.json extensions/tests/stream-test.ts`. STOP if CI/package management changed materially.

## Status
- **Priority**: P1
- **Effort**: S
- **Risk**: LOW
- **Depends on**: plans 001–002
- **Category**: tests
- **Planned at**: `5874163`, 2026-08-03

## Why this matters
CI runs Python checks only. The executable TypeScript integration check is omitted, and `npm test` falsely prints that no tests exist. Pi regressions can merge despite the most relevant check existing.

## Current state
- `.github/workflows/ci.yml:18-35` installs Python and runs ruff, mypy, pytest.
- `extensions/package.json:6-10` has typecheck but a placeholder test script.
- `extensions/tests/stream-test.ts` is the existing no-network provider integration check.

## Scope
**In**: `.github/workflows/ci.yml`, `extensions/package.json`, lockfile only if package scripts alter it, extension tests needed by plans 001–002.
**Out**: new test frameworks, release automation, Docker integration in hosted CI.

## Steps
1. Make `npm test` run the existing stream test using an already-supported runtime. Prefer the repository's existing Bun invocation; if CI cannot install Bun with an official action, use the smallest supported Node execution path rather than adding a framework.
2. Add CI setup for Node 20, dependency installation from `extensions/package-lock.json`, typecheck, and extension test.
3. Keep Python gates unchanged.
4. Confirm no test performs real network calls or reads user credentials.

## Verification
```bash
cd extensions && npm ci && npm run typecheck && npm test
uv run pytest -q --no-cov tests/test_serve.py
```
Expected: exit 0; `npm test` executes assertions rather than echoing a placeholder.

## Done criteria
- Pull requests cannot pass when extension typecheck or stream integration fails.
- Lockfile remains synchronized.
- No new test dependency unless the installed runtime cannot execute the current test.

## STOP conditions
- The chosen runtime requires an unpinned curl-pipe-shell installer.
- CI secrets or live Mantis services become necessary.
