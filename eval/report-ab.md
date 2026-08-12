## Provenance
- config/arm: `mantis,openrouter-fugu`
- git SHA: `4af8134b85851387c78b18c8a2cb3275b7a2f69e`
- catalog revision: `bifrost-2026-08-12`
- fixtures: `eval/fixtures.jsonl` (sha256 `3fdd2ae72173ee1f91b1eae60edd223a8e0a433c32d1c1d2d6c81c8afdeee366`)
- generated at: `2026-08-12T18:58:55.848163+00:00`
- cost method: `provider-reported usage.cost (real USD when supplied)`

# A/B: mantis vs sakana/fugu-ultra

Fixtures: 16 tasks x 2 targets. Scoring: keyword hit rate on `expect` terms (eval/score.py).

| target | success | errors | score | cost_usd | score/$ | latency s (med/p95) | completion tokens (med) |
|---|---:|---:|---:|---:|---:|---|---:|
| mantis | 16/16 | 0 | 0.92 | 0.3672 | 2.5 | 61.8/300.5 | 5183 |
| openrouter-fugu | 16/16 | 0 | 0.87 | 1.8603 | 0.5 | 37.1/215.3 | 2141 |

## Per tier (mean score / total cost $)

### debug

| target | success | errors | score | cost_usd | latency med s |
|---|---:|---:|---:|---:|---:|
| mantis | 4/4 | 0 | 0.79 | 0.0842 | 34.7 |
| openrouter-fugu | 4/4 | 0 | 0.82 | 0.2052 | 13.1 |

### hard

| target | success | errors | score | cost_usd | latency med s |
|---|---:|---:|---:|---:|---:|
| mantis | 4/4 | 0 | 0.96 | 0.1525 | 65.6 |
| openrouter-fugu | 4/4 | 0 | 0.71 | 0.9687 | 77.6 |

### medium

| target | success | errors | score | cost_usd | latency med s |
|---|---:|---:|---:|---:|---:|
| mantis | 4/4 | 0 | 0.95 | 0.0293 | 63.3 |
| openrouter-fugu | 4/4 | 0 | 0.95 | 0.5375 | 45.5 |

### simple

| target | success | errors | score | cost_usd | latency med s |
|---|---:|---:|---:|---:|---:|
| mantis | 4/4 | 0 | 1.00 | 0.1011 | 72.9 |
| openrouter-fugu | 4/4 | 0 | 1.00 | 0.1488 | 10.6 |
