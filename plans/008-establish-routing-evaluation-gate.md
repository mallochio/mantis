# Plan 008: Establish a representative routing evaluation and promotion gate

> **Executor instructions**: Follow this plan step by step. Run every verification command. Update `plans/README.md` when done. Do not make paid API calls without operator approval.
>
> **Drift check**: `git diff --stat 3cccee1..HEAD -- eval configs/worker-costs.json README.md`. If the evaluation formats changed materially, STOP and reconcile this plan first.

## Status
- **Priority**: P1
- **Effort**: S
- **Risk**: LOW
- **Depends on**: none
- **Category**: tests / direction
- **Planned at**: commit `3cccee1`, 2026-08-04

## Why this matters
Every later routing experiment needs the same quality/cost/latency gate. The current evidence has only 16 keyword-scored prompts: `eval/fixtures.jsonl:1-16`; `eval/score.py:24-30` scores literal substring coverage. It is enough to motivate direct-vs-TRINITY routing, but not enough to promote a learned policy.

## Current state
- `eval/report-native-v2.md:5-10` records direct, TRINITY, and local Conductor quality, latency, and estimated cost.
- `eval/report-native-v2.md:37-50` shows direct winning the simple cost regime and TRINITY winning hard tasks.
- `eval/run_eval.py:8-22` estimates cost from fixed 2K-input/1K-output assumptions rather than observed usage.
- `eval/run_eval.py:235-236` appends results, so rerunning can duplicate task/config rows.
- Verification conventions are `uv run pytest tests -q`, `uv run ruff check .`, and `uv run mypy openfugu-patch scripts --exclude outputs`.

## Scope
**In scope**: `eval/fixtures*.jsonl`, `eval/run_eval.py`, `eval/score.py`, focused tests under `tests/`, and a generated-report schema/documentation file under `eval/`.

**Out of scope**: runtime routing, model retraining, paid benchmark execution, new evaluation frameworks, Conductor promotion.

## Steps
1. Define a versioned result schema with task ID, task family, risk class, route, success/quality, actual or estimated input/output tokens, USD cost, latency, failure, and repetition seed. Keep old result loading compatible only where trivial.
   - **Verify**: focused unit tests parse one old row and one new row; malformed and duplicate rows fail clearly.
2. Expand fixtures to at least 100 representative tasks before any production promotion: simple Q&A, single-file edits, multi-file edits, debugging, test repair, research, tool-heavy work, and explicitly marked security/high-risk tasks. Prefer replayable repository fixtures and executable checks over keyword scoring. Do not invent 100 near-duplicates merely to meet the count.
   - **Verify**: a test asserts unique IDs, required fields, family coverage, and at least 20% held out from tuning.
3. Report quality, failure rate, mean/p50/p95 latency, total/mean cost, and Pareto dominance by route and task family. Add paired bootstrap confidence intervals or another stdlib implementation suitable for paired task outcomes.
   - **Verify**: a deterministic synthetic result set produces known dominance and confidence results.
4. Add an offline utility sweep over configurable cost and latency weights. This is reporting, not a runtime config system.
   - **Verify**: utility rankings change on a synthetic fixture when the supplied weights change.
5. Record a promotion rule: no statistically credible quality regression beyond an operator-selected tolerance; lower cost or latency on the held-out set; no increase in security/high-risk failures.
   - **Verify**: `uv run pytest tests -q` passes without network access.

## Test plan
Model tests after the pure-function style in `tests/test_retrain_router_pool.py`. Cover schema validation, duplicate rejection, executable-vs-keyword scoring precedence, paired metrics, utility sweep, and promotion pass/fail.

## Done criteria
- [ ] A fresh offline report can be generated from checked-in results with no network access.
- [ ] At least 100 non-duplicate representative fixtures exist before this plan is marked DONE; otherwise mark BLOCKED with the achieved count.
- [ ] Actual token/cost fields are preferred when present; estimates are visibly labeled.
- [ ] `uv run pytest tests -q`, `uv run ruff check .`, and `git diff --check` pass.
- [ ] No files outside scope changed except `plans/README.md`.

## STOP conditions
- Fixture expansion requires copying private task text into Git; use sanitized/replayable tasks or stop.
- A scoring method needs an external paid judge by default; keep it optional and retain a deterministic gate.
- Existing raw result provenance cannot be established; label it legacy rather than silently mixing it with new runs.

## Maintenance notes
Version fixture and result schemas. Never compare policies on different task sets without stating it. Keep raw run data out of plans and committed reports when it may contain project text.
