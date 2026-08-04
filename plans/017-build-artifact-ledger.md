# Plan 017: Add a compact artifact ledger for cross-agent memory

> **Executor instructions**: Start with deterministic extraction from existing events. Do not add a summarizing model call.
>
> **Drift check**: `git diff --stat 3cccee1..HEAD -- openfugu-patch/serve.py tests/test_serve.py eval README.md`. STOP on message-history or tool-observation drift.

## Status
- **Priority**: P2
- **Effort**: M
- **Risk**: MED
- **Depends on**: plans 008 and 012
- **Category**: perf / direction
- **Planned at**: commit `3cccee1`, 2026-08-04

## Why this matters
Agents can repeat file reads, tests, and failed hypotheses across turns. Current tool telemetry retains only tool name/error/test flags (`openfugu-patch/serve.py:1693-1714`), while complete tool content stays in each local model transcript. A bounded artifact ledger can pass forward useful state without replaying every transcript.

## Current state
- TRINITY builds each role message from conversation history and role-specific content at `openfugu-patch/serve.py:1802-1821`.
- Conductor passes selected prior node outputs through access lists at `openfugu-patch/serve.py:2030-2050`.
- Tool result content is bounded when returned to the same model at `openfugu-patch/serve.py:1903-1909` and `2149-2156`.

## Scope
**In scope**: in-memory per-run ledger, deterministic entries, relevance/bounds, tests/evaluation.

**Out of scope**: cross-user persistence, embeddings/vector DB, external memory service, raw repository snapshots, model-generated summaries.

## Steps
1. Define bounded entries for file/symbol inspected, command/test status, failed operation, changed path, and unresolved verifier finding. Store only concise metadata and hashes where content is sensitive.
2. Populate entries from validated tool calls/results and orchestration events. Deduplicate by stable identity and cap count/bytes.
3. Inject only relevant entries into later worker prompts. Preserve Conductor intra-workflow access isolation; do not leak sibling node trajectories outside declared dependencies.
4. Measure repeated reads/tests, input size, cost, and quality on tool-heavy held-out tasks.
5. Keep the feature disabled if it does not reduce repeated work without quality loss.

## Test plan
Cover deduplication, bounds, no raw tool output, relevance filtering, verifier findings, Conductor isolation, cancellation cleanup, and prompt-size accounting.

## Done criteria
- [ ] Ledger memory is per-run and bounded.
- [ ] No secret-shaped value or raw tool output is persisted.
- [ ] Conductor access-list isolation remains intact.
- [ ] Held-out tool repetition or input tokens improve without quality regression.
- [ ] Full verification passes.

## STOP conditions
- Useful entries require storing source contents or arbitrary command output.
- Ledger injection breaks worker isolation or context accounting.
- No measurable reduction appears on the representative tool-heavy set.

## Maintenance notes
Keep entry types boring and finite. Add a new type only after a repeated-work case demonstrates value.
