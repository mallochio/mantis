## Provenance
- config/arm: `direct,trinity,conductor-luna`
- git SHA: `1bf1909dea0bd5b66e3a3a43cd514fd54c8dd79e`
- catalog revision: `bifrost-2026-08-12`
- fixtures: `eval/fixtures.jsonl` (sha256 `3fdd2ae72173ee1f91b1eae60edd223a8e0a433c32d1c1d2d6c81c8afdeee366`)
- generated at: `2026-08-12T19:08:07.764372+00:00`
- cost method: `native estimated prices; 2K prompt + 1K completion assumption`

# Comparative Eval: direct vs TRINITY vs Conductor-Luna

Scoring rule: report a mean only with at least 4 successful rows; comparisons use successful-item intersections.

Scoring note: an empty response without an error scores 0.0 under the current keyword scorer; row output does not distinguish an empty answer from a wrong answer.

Conductor-Luna uses the LiteLLM planner `gpt-5.6-luna-max` instead of a local 3B checkpoint.

## Summary

| config | success | errors | auto_score_mean | latency_sum_s | est_cost_sum_usd |
|---|---:|---:|---:|---:|---:|
| direct | 16/16 | 0 | 0.568 | 125.5 | $0.0128 |
| trinity | 16/16 | 0 | 0.890 | 1023.0 | $0.4563 |
| conductor-luna | 16/16 | 0 | 0.626 | 1331.2 | $0.8628 |

## Results matrix (auto_score, latency, cost)

| id | tier | direct | trinity | conductor-luna |
|---|---|---|---|---|
| t01 | simple | 1.00 / 12.0s / $0.0008 | 1.00 / 96.4s / $0.0007 | 0.00 / 56.7s / $0.0508 |
| t02 | simple | 0.83 / 7.6s / $0.0008 | 0.67 / 181.4s / $0.0036 | 0.67 / 78.7s / $0.0508 |
| t03 | simple | 1.00 / 2.5s / $0.0008 | 1.00 / 10.3s / $0.0007 | 1.00 / 6.2s / $0.0175 |
| t04 | simple | 1.00 / 7.8s / $0.0008 | 1.00 / 23.5s / $0.0012 | 0.80 / 34.2s / $0.0508 |
| t05 | medium | 0.00 / 8.9s / $0.0008 | 1.00 / 76.3s / $0.0917 | 1.00 / 76.9s / $0.0508 |
| t06 | medium | 0.80 / 7.0s / $0.0008 | 1.00 / 85.2s / $0.0015 | 1.00 / 91.2s / $0.0508 |
| t07 | medium | 1.00 / 6.9s / $0.0008 | 1.00 / 49.7s / $0.1018 | 1.00 / 60.8s / $0.0508 |
| t08 | medium | 1.00 / 4.5s / $0.0008 | 1.00 / 19.3s / $0.0113 | 1.00 / 31.6s / $0.0508 |
| t09 | hard | 0.00 / 10.0s / $0.0008 | 0.33 / 72.1s / $0.0018 | 1.00 / 202.7s / $0.0841 |
| t10 | hard | 0.00 / 10.3s / $0.0008 | 0.67 / 65.2s / $0.0221 | 0.00 / 197.1s / $0.0841 |
| t11 | hard | 0.00 / 9.3s / $0.0008 | 1.00 / 107.9s / $0.0366 | 0.00 / 102.8s / $0.0508 |
| t12 | hard | 0.00 / 9.0s / $0.0008 | 0.86 / 103.1s / $0.0137 | 0.00 / 150.3s / $0.0675 |
| t13 | debug | 0.86 / 3.4s / $0.0008 | 0.86 / 31.3s / $0.0407 | 0.71 / 23.9s / $0.0508 |
| t14 | debug | 0.00 / 10.6s / $0.0008 | 0.86 / 17.0s / $0.0012 | 0.43 / 90.3s / $0.0508 |
| t15 | debug | 0.60 / 4.8s / $0.0008 | 1.00 / 15.8s / $0.0909 | 0.40 / 36.6s / $0.0508 |
| t16 | debug | 1.00 / 10.8s / $0.0008 | 1.00 / 68.3s / $0.0370 | 1.00 / 91.0s / $0.0508 |

## Per-tier averages

| tier | config | success | errors | auto_score_mean | latency_sum_s | est_cost_sum_usd |
|---|---|---:|---:|---:|---:|---:|
| simple | direct | 4/4 | 0 | 0.958 | 29.9 | $0.0032 |
| simple | trinity | 4/4 | 0 | 0.917 | 311.7 | $0.0062 |
| simple | conductor-luna | 4/4 | 0 | 0.617 | 175.9 | $0.1699 |
| medium | direct | 4/4 | 0 | 0.700 | 27.4 | $0.0032 |
| medium | trinity | 4/4 | 0 | 1.000 | 230.6 | $0.2063 |
| medium | conductor-luna | 4/4 | 0 | 1.000 | 260.5 | $0.2032 |
| hard | direct | 4/4 | 0 | 0.000 | 38.7 | $0.0032 |
| hard | trinity | 4/4 | 0 | 0.714 | 348.2 | $0.0742 |
| hard | conductor-luna | 4/4 | 0 | 0.250 | 653.0 | $0.2865 |
| debug | direct | 4/4 | 0 | 0.614 | 29.6 | $0.0032 |
| debug | trinity | 4/4 | 0 | 0.928 | 132.5 | $0.1697 |
| debug | conductor-luna | 4/4 | 0 | 0.636 | 241.8 | $0.2032 |

## Headline comparisons

### a) conductor-luna vs trinity

- trinity overall mean: 0.890
- conductor-luna overall mean: 0.626
- overall quality delta (paired n=16): -0.264
- hard-tier trinity mean: 0.714
- hard-tier conductor-luna mean: 0.250
- hard-tier quality delta (paired n=4): -0.464

### b) cost-per-quality-point

- overall: conductor-luna is not a quality win (delta -0.264) despite $+0.4064 cost delta
- hard tier: conductor-luna is not a quality win (delta -0.464) on hard tasks

### c) failure rate

- conductor-luna HTTP 500/timeout count: 0 / 16
- failure rate: 0.0%

## Decision table

- Decision branch: conductor-luna hard-tier mean < trinity - 0.10
- Recommended Conductor routing threshold: 6
- Reasoning: TRINITY remains the better orchestrator for hard tasks; keep Conductor out of auto mode.

## Errors / timeouts

No errors or timeouts recorded for conductor-luna.

## Total spend

- conductor-luna estimated API spend: **$0.8628**.
- direct baseline (for reference): **$0.0128**.
- trinity baseline (for reference): **$0.4563**.
Costs are per-call estimates from `config/worker-costs.json` (2K prompt + 1K completion); actual OpenRouter spend may differ.

## Fixture list

- `t01` (simple): what does git rerere do?...
- `t02` (simple): write a bash one-liner to find duplicate filenames in a directory tree...
- `t03` (simple): explain the difference between 'git merge' and 'git rebase' in one sentence...
- `t04` (simple): how do I list all environment variables starting with 'FUGU_' in bash?...
- `t05` (medium): write a Python retry decorator with exponential backoff and jitter, including a ...
- `t06` (medium): Implement a CLI argument parser in Python that accepts --config and --verbose an...
- `t07` (medium): write a Python function that reads a CSV and returns the most common value in a ...
- `t08` (medium): Create a Python snippet that recursively counts all .py files under a directory ...
- `t09` (hard): design and implement a small rate limiter module (token bucket) with tests and a...
- `t10` (hard): Refactor a multi-file distributed lock manager across services into a single Pyt...
- `t11` (hard): Implement a simple LRU cache with O(1) get/put in Python and explain the design...
- `t12` (hard): Create a minimal HTTP client class in Python supporting GET, POST, retries, and ...
- `t13` (debug): This function is supposed to return the sum of squares but returns 0. Fix it and...
- `t14` (debug): Bug: the script 'for f in $(ls *.txt); do cat $f; done' fails on filenames with ...
- `t15` (debug): The following Python code throws NameError: name 'json' is not defined at runtim...
- `t16` (debug): A Docker build fails with 'COPY failed: no such file or directory'. What are thr...
