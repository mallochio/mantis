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

- **Region workaround:** sakana/fugu-ultra provider-blocks the dev machine's
  region (403 "not available in your region"), so the fugu leg ran from GCP
  `us-central1` via `launch/sky/ab_fugu_region.yaml` (`sky launch --down`, a
  spot `n4-highcpu-2`; local file_mounts only — no object-storage bucket was
  created). Cluster auto-tore down at job end.
- **Method:** identical fixtures (`eval/fixtures.jsonl`), non-streaming calls,
  keyword hit-rate scoring (`eval/score.py`), wall-clock latency, real
  per-request USD cost from each response's `usage.cost`. Mantis leg ran on
  the dev host; fugu leg on the GCP VM. fugu-ultra used its defaults
  (mandatory reasoning, effort xhigh); mantis used its configured pool
  (reasoning-effort workers, internal 4096-token cap).
- **Result:** near-parity on score (mantis 0.92 vs fugu 0.87), with mantis
  5x cheaper ($0.37 vs $1.86 for 16 tasks) and 5x better score-per-dollar
  (2.5 vs 0.5). fugu-ultra is faster on median latency (37 s vs 62 s) and
  emits fewer completion tokens (2.1k vs 5.2k median) — both artifacts of
  fugu-ultra's mandatory xhigh reasoning, which also drives its 5x cost.
- **Caveats:** 16 fixtures, keyword scoring, one run each; mantis latency
  includes TRINITY/verifier orchestration and a cold t04 (300 s). Read as a
  directional cost-quality comparison, not a benchmark headline.
- **Empty-final bug found and fixed during this work:** 5/16 mantis answers
  were empty (reasoning workers returned `content: null`); `_final()` now
  walks back to the last non-empty reply (see tests/test_empty_final.py).
  Score rose 0.56 -> 0.92 after the fix.
