## Provenance
- config/arm: `direct,trinity,conductor-old,conductor-new`
- git SHA: `4af8134b85851387c78b18c8a2cb3275b7a2f69e`
- catalog revision: `bifrost-2026-08-12`
- fixtures: `eval/fixtures.jsonl` (sha256 `3fdd2ae72173ee1f91b1eae60edd223a8e0a433c32d1c1d2d6c81c8afdeee366`)
- generated at: `2026-08-12T18:58:55.591865+00:00`
- cost method: `native estimated prices; 2K prompt + 1K completion assumption`

# Comparative Eval: direct vs TRINITY vs Conductor-old vs Conductor-new

## Summary

| config | success | errors | auto_score_mean | latency_mean_s | est_cost_sum_usd |
|---|---:|---:|---:|---:|---:|
| direct | 16/16 | 0 | 0.568 | 7.8 | $0.0128 |
| trinity | 16/16 | 0 | 0.890 | 63.9 | $0.4563 |
| conductor-old | 1/16 | 15 | 0.833 | 56.6 | $0.1183 |
| conductor-new | 1/16 | 15 | 1.000 | 103.0 | $0.1183 |

## Results matrix (auto_score, latency, cost)

| id | tier | direct | trinity | conductor-old | conductor-new |
|---|---|---|---|---|---|
| t01 | simple | 1.00 / 12.0s / $0.0008 | 1.00 / 96.4s / $0.0007 | error: HTTPError: 500 Server Error: Internal Server Error for url: http://localhost:8088/v1/chat/completions | error: HTTPError: 500 Server Error: Internal Server Error for url: http://localhost:8088/v1/chat/completions |
| t02 | simple | 0.83 / 7.6s / $0.0008 | 0.67 / 181.4s / $0.0036 | error: HTTPError: 500 Server Error: Internal Server Error for url: http://localhost:8088/v1/chat/completions | error: HTTPError: 500 Server Error: Internal Server Error for url: http://localhost:8088/v1/chat/completions |
| t03 | simple | 1.00 / 2.5s / $0.0008 | 1.00 / 10.3s / $0.0007 | error: HTTPError: 500 Server Error: Internal Server Error for url: http://localhost:8088/v1/chat/completions | 1.00 / 103.0s / $0.1183 |
| t04 | simple | 1.00 / 7.8s / $0.0008 | 1.00 / 23.5s / $0.0012 | error: HTTPError: 500 Server Error: Internal Server Error for url: http://localhost:8088/v1/chat/completions | error: HTTPError: 500 Server Error: Internal Server Error for url: http://localhost:8088/v1/chat/completions |
| t05 | medium | 0.00 / 8.9s / $0.0008 | 1.00 / 76.3s / $0.0917 | error: HTTPError: 500 Server Error: Internal Server Error for url: http://localhost:8088/v1/chat/completions | error: HTTPError: 500 Server Error: Internal Server Error for url: http://localhost:8088/v1/chat/completions |
| t06 | medium | 0.80 / 7.0s / $0.0008 | 1.00 / 85.2s / $0.0015 | error: HTTPError: 500 Server Error: Internal Server Error for url: http://localhost:8088/v1/chat/completions | error: HTTPError: 500 Server Error: Internal Server Error for url: http://localhost:8088/v1/chat/completions |
| t07 | medium | 1.00 / 6.9s / $0.0008 | 1.00 / 49.7s / $0.1018 | error: HTTPError: 500 Server Error: Internal Server Error for url: http://localhost:8088/v1/chat/completions | error: HTTPError: 500 Server Error: Internal Server Error for url: http://localhost:8088/v1/chat/completions |
| t08 | medium | 1.00 / 4.5s / $0.0008 | 1.00 / 19.3s / $0.0113 | error: HTTPError: 500 Server Error: Internal Server Error for url: http://localhost:8088/v1/chat/completions | error: HTTPError: 500 Server Error: Internal Server Error for url: http://localhost:8088/v1/chat/completions |
| t09 | hard | 0.00 / 10.0s / $0.0008 | 0.33 / 72.1s / $0.0018 | error: HTTPError: 500 Server Error: Internal Server Error for url: http://localhost:8088/v1/chat/completions | error: HTTPError: 500 Server Error: Internal Server Error for url: http://localhost:8088/v1/chat/completions |
| t10 | hard | 0.00 / 10.3s / $0.0008 | 0.67 / 65.2s / $0.0221 | error: HTTPError: 500 Server Error: Internal Server Error for url: http://localhost:8088/v1/chat/completions | error: HTTPError: 500 Server Error: Internal Server Error for url: http://localhost:8088/v1/chat/completions |
| t11 | hard | 0.00 / 9.3s / $0.0008 | 1.00 / 107.9s / $0.0366 | error: HTTPError: 500 Server Error: Internal Server Error for url: http://localhost:8088/v1/chat/completions | error: HTTPError: 500 Server Error: Internal Server Error for url: http://localhost:8088/v1/chat/completions |
| t12 | hard | 0.00 / 9.0s / $0.0008 | 0.86 / 103.1s / $0.0137 | error: HTTPError: 500 Server Error: Internal Server Error for url: http://localhost:8088/v1/chat/completions | error: HTTPError: 500 Server Error: Internal Server Error for url: http://localhost:8088/v1/chat/completions |
| t13 | debug | 0.86 / 3.4s / $0.0008 | 0.86 / 31.3s / $0.0407 | error: HTTPError: 500 Server Error: Internal Server Error for url: http://localhost:8088/v1/chat/completions | error: HTTPError: 500 Server Error: Internal Server Error for url: http://localhost:8088/v1/chat/completions |
| t14 | debug | 0.00 / 10.6s / $0.0008 | 0.86 / 17.0s / $0.0012 | error: HTTPError: 500 Server Error: Internal Server Error for url: http://localhost:8088/v1/chat/completions | error: HTTPError: 500 Server Error: Internal Server Error for url: http://localhost:8088/v1/chat/completions |
| t15 | debug | 0.60 / 4.8s / $0.0008 | 1.00 / 15.8s / $0.0909 | error: HTTPError: 500 Server Error: Internal Server Error for url: http://localhost:8088/v1/chat/completions | error: HTTPError: 500 Server Error: Internal Server Error for url: http://localhost:8088/v1/chat/completions |
| t16 | debug | 1.00 / 10.8s / $0.0008 | 1.00 / 68.3s / $0.0370 | 0.83 / 56.6s / $0.1183 | error: HTTPError: 500 Server Error: Internal Server Error for url: http://localhost:8088/v1/chat/completions |

## Per-tier averages

| tier | config | success | errors | auto_score_mean | latency_sum_s | est_cost_sum_usd |
|---|---|---:|---:|---:|---:|---:|
| simple | direct | 4/4 | 0 | 0.958 | 29.9 | $0.0032 |
| simple | trinity | 4/4 | 0 | 0.917 | 311.7 | $0.0062 |
| simple | conductor-old | 0/4 | 4 | 0.000 | 0.0 | $0.0000 |
| simple | conductor-new | 1/4 | 3 | 1.000 | 103.0 | $0.1183 |
| medium | direct | 4/4 | 0 | 0.700 | 27.4 | $0.0032 |
| medium | trinity | 4/4 | 0 | 1.000 | 230.6 | $0.2063 |
| medium | conductor-old | 0/4 | 4 | 0.000 | 0.0 | $0.0000 |
| medium | conductor-new | 0/4 | 4 | 0.000 | 0.0 | $0.0000 |
| hard | direct | 4/4 | 0 | 0.000 | 38.7 | $0.0032 |
| hard | trinity | 4/4 | 0 | 0.714 | 348.2 | $0.0742 |
| hard | conductor-old | 0/4 | 4 | 0.000 | 0.0 | $0.0000 |
| hard | conductor-new | 0/4 | 4 | 0.000 | 0.0 | $0.0000 |
| debug | direct | 4/4 | 0 | 0.614 | 29.6 | $0.0032 |
| debug | trinity | 4/4 | 0 | 0.929 | 132.5 | $0.1697 |
| debug | conductor-old | 1/4 | 3 | 0.833 | 56.6 | $0.1183 |
| debug | conductor-new | 0/4 | 4 | 0.000 | 0.0 | $0.0000 |

## Headline comparisons

### a) trinity vs direct

- direct mean auto_score (16 successful, 0 errors): 0.568
- trinity mean auto_score (16 successful, 0 errors): 0.890
- quality delta: +0.322 (+56.6% vs direct)
- direct total cost: $0.0128, latency: 125.5s
- trinity total cost: $0.4563, latency: 1023.0s
- cost delta: $+0.4435 (3465.0% vs direct)

### b) conductor-new vs conductor-old

- conductor-old mean auto_score (1 successful, 15 errors): 0.833
- conductor-new mean auto_score (1 successful, 15 errors): 1.000
- quality delta: +0.167 (+20.0% vs old)
- conductor-old total cost: $0.1183, latency: 56.6s
- conductor-new total cost: $0.1183, latency: 103.0s

### c) conductor-new vs trinity

- trinity mean auto_score (16 successful, 0 errors): 0.890
- conductor-new mean auto_score (1 successful, 15 errors): 1.000
- quality delta: +0.110 (+12.4% vs trinity)
- trinity total cost: $0.4563, latency: 1023.0s
- conductor-new total cost: $0.1183, latency: 103.0s

### Hard-tier comparison

- direct: mean auto_score (4 successful, 0 errors) = 0.000, cost = $0.0032
- trinity: mean auto_score (4 successful, 0 errors) = 0.714, cost = $0.0742
- conductor-old: mean auto_score (0 successful, 4 errors) = 0.000, cost = $0.0000
- conductor-new: mean auto_score (0 successful, 4 errors) = 0.000, cost = $0.0000

## Recommendation

Results are mixed. Recommendation: run a targeted hard-tier eval with more prompts and human scoring before committing to more Conductor training.

## Errors / timeouts

- `conductor-old` / `t01`: HTTPError: 500 Server Error: Internal Server Error for url: http://localhost:8088/v1/chat/completions
- `conductor-old` / `t02`: HTTPError: 500 Server Error: Internal Server Error for url: http://localhost:8088/v1/chat/completions
- `conductor-old` / `t03`: HTTPError: 500 Server Error: Internal Server Error for url: http://localhost:8088/v1/chat/completions
- `conductor-old` / `t04`: HTTPError: 500 Server Error: Internal Server Error for url: http://localhost:8088/v1/chat/completions
- `conductor-old` / `t05`: HTTPError: 500 Server Error: Internal Server Error for url: http://localhost:8088/v1/chat/completions
- `conductor-old` / `t06`: HTTPError: 500 Server Error: Internal Server Error for url: http://localhost:8088/v1/chat/completions
- `conductor-old` / `t07`: HTTPError: 500 Server Error: Internal Server Error for url: http://localhost:8088/v1/chat/completions
- `conductor-old` / `t08`: HTTPError: 500 Server Error: Internal Server Error for url: http://localhost:8088/v1/chat/completions
- `conductor-old` / `t09`: HTTPError: 500 Server Error: Internal Server Error for url: http://localhost:8088/v1/chat/completions
- `conductor-old` / `t10`: HTTPError: 500 Server Error: Internal Server Error for url: http://localhost:8088/v1/chat/completions
- `conductor-old` / `t11`: HTTPError: 500 Server Error: Internal Server Error for url: http://localhost:8088/v1/chat/completions
- `conductor-old` / `t12`: HTTPError: 500 Server Error: Internal Server Error for url: http://localhost:8088/v1/chat/completions
- `conductor-old` / `t13`: HTTPError: 500 Server Error: Internal Server Error for url: http://localhost:8088/v1/chat/completions
- `conductor-old` / `t14`: HTTPError: 500 Server Error: Internal Server Error for url: http://localhost:8088/v1/chat/completions
- `conductor-old` / `t15`: HTTPError: 500 Server Error: Internal Server Error for url: http://localhost:8088/v1/chat/completions
- `conductor-new` / `t01`: HTTPError: 500 Server Error: Internal Server Error for url: http://localhost:8088/v1/chat/completions
- `conductor-new` / `t02`: HTTPError: 500 Server Error: Internal Server Error for url: http://localhost:8088/v1/chat/completions
- `conductor-new` / `t04`: HTTPError: 500 Server Error: Internal Server Error for url: http://localhost:8088/v1/chat/completions
- `conductor-new` / `t05`: HTTPError: 500 Server Error: Internal Server Error for url: http://localhost:8088/v1/chat/completions
- `conductor-new` / `t06`: HTTPError: 500 Server Error: Internal Server Error for url: http://localhost:8088/v1/chat/completions
- `conductor-new` / `t07`: HTTPError: 500 Server Error: Internal Server Error for url: http://localhost:8088/v1/chat/completions
- `conductor-new` / `t08`: HTTPError: 500 Server Error: Internal Server Error for url: http://localhost:8088/v1/chat/completions
- `conductor-new` / `t09`: HTTPError: 500 Server Error: Internal Server Error for url: http://localhost:8088/v1/chat/completions
- `conductor-new` / `t10`: HTTPError: 500 Server Error: Internal Server Error for url: http://localhost:8088/v1/chat/completions
- `conductor-new` / `t11`: HTTPError: 500 Server Error: Internal Server Error for url: http://localhost:8088/v1/chat/completions
- `conductor-new` / `t12`: HTTPError: 500 Server Error: Internal Server Error for url: http://localhost:8088/v1/chat/completions
- `conductor-new` / `t13`: HTTPError: 500 Server Error: Internal Server Error for url: http://localhost:8088/v1/chat/completions
- `conductor-new` / `t14`: HTTPError: 500 Server Error: Internal Server Error for url: http://localhost:8088/v1/chat/completions
- `conductor-new` / `t15`: HTTPError: 500 Server Error: Internal Server Error for url: http://localhost:8088/v1/chat/completions
- `conductor-new` / `t16`: HTTPError: 500 Server Error: Internal Server Error for url: http://localhost:8088/v1/chat/completions

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

## Total spend

Estimated total API spend across all configs: **$0.7058**.
This is a per-call estimate based on `config/worker-costs.json`; actual OpenRouter spend may differ.
