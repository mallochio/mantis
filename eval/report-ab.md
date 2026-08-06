# A/B: mantis vs sakana/fugu-ultra

Fixtures: 16 tasks x 2 targets. Scoring: keyword hit rate on `expect` terms (eval/score.py).

| target | success | score | cost $ | score/$ | latency s (med/p95) | completion tokens (med) |
|---|---|---|---|---|---|---|
| mantis | 16/16 | 0.92 | 0.3672 | 2.5 | 61.8/300.5 | 5183 |
| openrouter-fugu | 0/16 | 0.00 | 0.0000 | nan | 0.0/0.0 | 0 |

## Per tier (mean score / total cost $)

### debug

| target | score | cost $ | latency med s |
|---|---|---|---|
| mantis | 0.79 | 0.0842 | 34.7 |

### hard

| target | score | cost $ | latency med s |
|---|---|---|---|
| mantis | 0.96 | 0.1525 | 65.6 |

### medium

| target | score | cost $ | latency med s |
|---|---|---|---|
| mantis | 0.95 | 0.0293 | 63.3 |

### simple

| target | score | cost $ | latency med s |
|---|---|---|---|
| mantis | 1.00 | 0.1011 | 72.9 |


## Notes

- **fugu-ultra leg could not run from this machine.** OpenRouter accepts the
  key (control call to `openai/gpt-5.6-luna` returned 200), but Sakana AI
  blocks fugu-ultra at the provider level for this region:
  `403 "sakana/fugu-ultra is not available in your region: Sakana AI blocks
  requests originating from your location."` Both slugs
  (`sakana/fugu-ultra`, `sakana/fugu-ultra-20260615`) are affected. Re-run
  from an allowed region (US/EU) to complete the A/B:
  `uv run --extra eval python eval/ab_compare.py --targets mantis,openrouter-fugu`
- **Method:** identical fixtures (`eval/fixtures.jsonl`), non-streaming calls,
  keyword hit-rate scoring (`eval/score.py`), wall-clock latency, and the real
  per-request USD cost reported in each response's `usage.cost`.
- **Empty-final bug found and fixed during this run:** 5/16 mantis answers
  were empty (workers with reasoning effort consumed the budget and returned
  `content: null`). Fix: `_final()` walks back to the last non-empty reply
  (TrinityRun and ConductorRun); regression tests in
  `tests/test_empty_final.py`. After the fix the six affected fixtures all
  return content; overall score rose 0.56 -> 0.92.
- **Latency:** median 61.8 s includes orchestration (TRINITY routing +
  verifier + reasoning-effort workers at 4096 internal max tokens). p95 300 s
  is dominated by one slow cold run (t04, 300 s). Token spend (median 5183
  completion tokens per task) reflects reasoning-heavy workers.
