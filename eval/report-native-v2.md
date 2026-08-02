# Comparative Eval: direct vs TRINITY vs Conductor-old vs Conductor-new

## Summary

| config | auto_score_mean | latency_mean_s | cost_sum_usd |
|---|---|---|---|
| direct | 0.568 | 7.8 | $0.0128 |
| trinity | 0.890 | 63.9 | $0.4563 |
| conductor-old | 0.188 | 94.0 | $0.2648 |
| conductor-new | 0.073 | 95.7 | $0.2298 |

## Results matrix (auto_score, latency, cost)

| id | tier | direct | trinity | conductor-old | conductor-new |
|---|---|---|---|---|---|
| t01 | simple | 1.00 / 12.0s / $0.0008 | 1.00 / 96.4s / $0.0007 | 1.00 / 93.3s / $0.0883 | error: HTTPError: 500 Server Error: Internal Server Error for url: http://localhost:8088/v1/chat/completions |
| t02 | simple | 0.83 / 7.6s / $0.0008 | 0.67 / 181.4s / $0.0036 | error: HTTPError: 500 Server Error: Internal Server Error for url: http://localhost:8088/v1/chat/completions | 0.67 / 82.9s / $0.0533 |
| t03 | simple | 1.00 / 2.5s / $0.0008 | 1.00 / 10.3s / $0.0007 | 1.00 / 86.3s / $0.0883 | error: HTTPError: 500 Server Error: Internal Server Error for url: http://localhost:8088/v1/chat/completions |
| t04 | simple | 1.00 / 7.8s / $0.0008 | 1.00 / 23.5s / $0.0012 | error: HTTPError: 500 Server Error: Internal Server Error for url: http://localhost:8088/v1/chat/completions | 0.00 / 125.8s / $0.0883 |
| t05 | medium | 0.00 / 8.9s / $0.0008 | 1.00 / 76.3s / $0.0917 | error: HTTPError: 500 Server Error: Internal Server Error for url: http://localhost:8088/v1/chat/completions | error: HTTPError: 500 Server Error: Internal Server Error for url: http://localhost:8088/v1/chat/completions |
| t06 | medium | 0.80 / 7.0s / $0.0008 | 1.00 / 85.2s / $0.0015 | error: HTTPError: 500 Server Error: Internal Server Error for url: http://localhost:8088/v1/chat/completions | error: HTTPError: 500 Server Error: Internal Server Error for url: http://localhost:8088/v1/chat/completions |
| t07 | medium | 1.00 / 6.9s / $0.0008 | 1.00 / 49.7s / $0.1018 | error: HTTPError: 500 Server Error: Internal Server Error for url: http://localhost:8088/v1/chat/completions | error: HTTPError: 500 Server Error: Internal Server Error for url: http://localhost:8088/v1/chat/completions |
| t08 | medium | 1.00 / 4.5s / $0.0008 | 1.00 / 19.3s / $0.0113 | 1.00 / 142.6s / $0.0883 | error: HTTPError: 500 Server Error: Internal Server Error for url: http://localhost:8088/v1/chat/completions |
| t09 | hard | 0.00 / 10.0s / $0.0008 | 0.33 / 72.1s / $0.0018 | error: HTTPError: 500 Server Error: Internal Server Error for url: http://localhost:8088/v1/chat/completions | error: HTTPError: 500 Server Error: Internal Server Error for url: http://localhost:8088/v1/chat/completions |
| t10 | hard | 0.00 / 10.3s / $0.0008 | 0.67 / 65.2s / $0.0221 | error: HTTPError: 500 Server Error: Internal Server Error for url: http://localhost:8088/v1/chat/completions | 0.50 / 121.1s / $0.0883 |
| t11 | hard | 0.00 / 9.3s / $0.0008 | 1.00 / 107.9s / $0.0366 | error: HTTPError: 500 Server Error: Internal Server Error for url: http://localhost:8088/v1/chat/completions | error: HTTPError: 500 Server Error: Internal Server Error for url: http://localhost:8088/v1/chat/completions |
| t12 | hard | 0.00 / 9.0s / $0.0008 | 0.86 / 103.1s / $0.0137 | error: HTTPError: 500 Server Error: Internal Server Error for url: http://localhost:8088/v1/chat/completions | error: HTTPError: 500 Server Error: Internal Server Error for url: http://localhost:8088/v1/chat/completions |
| t13 | debug | 0.86 / 3.4s / $0.0008 | 0.86 / 31.3s / $0.0407 | error: HTTPError: 500 Server Error: Internal Server Error for url: http://localhost:8088/v1/chat/completions | error: HTTPError: 500 Server Error: Internal Server Error for url: http://localhost:8088/v1/chat/completions |
| t14 | debug | 0.00 / 10.6s / $0.0008 | 0.86 / 17.0s / $0.0012 | error: HTTPError: 500 Server Error: Internal Server Error for url: http://localhost:8088/v1/chat/completions | error: HTTPError: 500 Server Error: Internal Server Error for url: http://localhost:8088/v1/chat/completions |
| t15 | debug | 0.60 / 4.8s / $0.0008 | 1.00 / 15.8s / $0.0909 | error: HTTPError: 500 Server Error: Internal Server Error for url: http://localhost:8088/v1/chat/completions | error: HTTPError: 500 Server Error: Internal Server Error for url: http://localhost:8088/v1/chat/completions |
| t16 | debug | 1.00 / 10.8s / $0.0008 | 1.00 / 68.3s / $0.0370 | error: HTTPError: 500 Server Error: Internal Server Error for url: http://localhost:8088/v1/chat/completions | error: HTTPError: 500 Server Error: Internal Server Error for url: http://localhost:8088/v1/chat/completions |

## Per-tier averages

| tier | config | auto_score_mean | latency_sum_s | cost_sum_usd |
|---|---|---|---|---|
| simple | direct | 0.958 | 29.9 | $0.0032 |
| simple | trinity | 0.917 | 311.7 | $0.0062 |
| simple | conductor-old | 0.500 | 320.8 | $0.1765 |
| simple | conductor-new | 0.167 | 403.6 | $0.1416 |
| medium | direct | 0.700 | 27.4 | $0.0032 |
| medium | trinity | 1.000 | 230.6 | $0.2063 |
| medium | conductor-old | 0.250 | 498.7 | $0.0883 |
| medium | conductor-new | 0.000 | 370.4 | $0.0000 |
| hard | direct | 0.000 | 38.7 | $0.0032 |
| hard | trinity | 0.714 | 348.2 | $0.0742 |
| hard | conductor-old | 0.000 | 498.5 | $0.0000 |
| hard | conductor-new | 0.125 | 444.7 | $0.0883 |
| debug | direct | 0.614 | 29.6 | $0.0032 |
| debug | trinity | 0.929 | 132.5 | $0.1697 |
| debug | conductor-old | 0.000 | 185.3 | $0.0000 |
| debug | conductor-new | 0.000 | 313.0 | $0.0000 |

## Headline comparisons

### a) trinity vs direct

- direct mean auto_score: 0.568
- trinity mean auto_score: 0.890
- quality delta: +0.322 (+56.6% vs direct)
- direct total cost: $0.0128, latency: 125.5s
- trinity total cost: $0.4563, latency: 1023.0s
- cost delta: $+0.4435 (3465.0% vs direct)

### b) conductor-new vs conductor-old

- conductor-old mean auto_score: 0.188
- conductor-new mean auto_score: 0.073
- quality delta: -0.115 (-61.1% vs old)
- conductor-old total cost: $0.2648, latency: 1503.3s
- conductor-new total cost: $0.2298, latency: 1531.7s

### c) conductor-new vs trinity

- trinity mean auto_score: 0.890
- conductor-new mean auto_score: 0.073
- quality delta: -0.817 (-91.8% vs trinity)
- trinity total cost: $0.4563, latency: 1023.0s
- conductor-new total cost: $0.2298, latency: 1531.7s

### Hard-tier comparison

- direct: mean auto_score = 0.000, cost = $0.0032
- trinity: mean auto_score = 0.714, cost = $0.0742
- conductor-old: mean auto_score = 0.000, cost = $0.0000
- conductor-new: mean auto_score = 0.125, cost = $0.0883

## Recommendation

Conductor-new scores well below direct (0.073 vs 0.568) and fails on 81% of prompts. Recommendation: do not deploy the local Conductor checkpoints as-is; use TRINITY in `/fugu auto` mode, and do not spend more on Conductor training until the checkpoint reliably emits valid DAGs on CPU/float32 serving.

## Errors / timeouts

- `conductor-old` / `t02`: HTTPError: 500 Server Error: Internal Server Error for url: http://localhost:8088/v1/chat/completions
- `conductor-old` / `t04`: HTTPError: 500 Server Error: Internal Server Error for url: http://localhost:8088/v1/chat/completions
- `conductor-old` / `t05`: HTTPError: 500 Server Error: Internal Server Error for url: http://localhost:8088/v1/chat/completions
- `conductor-old` / `t06`: HTTPError: 500 Server Error: Internal Server Error for url: http://localhost:8088/v1/chat/completions
- `conductor-old` / `t07`: HTTPError: 500 Server Error: Internal Server Error for url: http://localhost:8088/v1/chat/completions
- `conductor-old` / `t09`: HTTPError: 500 Server Error: Internal Server Error for url: http://localhost:8088/v1/chat/completions
- `conductor-old` / `t10`: HTTPError: 500 Server Error: Internal Server Error for url: http://localhost:8088/v1/chat/completions
- `conductor-old` / `t11`: HTTPError: 500 Server Error: Internal Server Error for url: http://localhost:8088/v1/chat/completions
- `conductor-old` / `t12`: HTTPError: 500 Server Error: Internal Server Error for url: http://localhost:8088/v1/chat/completions
- `conductor-old` / `t13`: HTTPError: 500 Server Error: Internal Server Error for url: http://localhost:8088/v1/chat/completions
- `conductor-old` / `t14`: HTTPError: 500 Server Error: Internal Server Error for url: http://localhost:8088/v1/chat/completions
- `conductor-old` / `t15`: HTTPError: 500 Server Error: Internal Server Error for url: http://localhost:8088/v1/chat/completions
- `conductor-old` / `t16`: HTTPError: 500 Server Error: Internal Server Error for url: http://localhost:8088/v1/chat/completions
- `conductor-new` / `t01`: HTTPError: 500 Server Error: Internal Server Error for url: http://localhost:8088/v1/chat/completions
- `conductor-new` / `t03`: HTTPError: 500 Server Error: Internal Server Error for url: http://localhost:8088/v1/chat/completions
- `conductor-new` / `t05`: HTTPError: 500 Server Error: Internal Server Error for url: http://localhost:8088/v1/chat/completions
- `conductor-new` / `t06`: HTTPError: 500 Server Error: Internal Server Error for url: http://localhost:8088/v1/chat/completions
- `conductor-new` / `t07`: HTTPError: 500 Server Error: Internal Server Error for url: http://localhost:8088/v1/chat/completions
- `conductor-new` / `t08`: HTTPError: 500 Server Error: Internal Server Error for url: http://localhost:8088/v1/chat/completions
- `conductor-new` / `t09`: HTTPError: 500 Server Error: Internal Server Error for url: http://localhost:8088/v1/chat/completions
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

Estimated total API spend across all configs: **$0.9638**.
This is a per-call estimate based on `configs/worker-costs.json`; actual OpenRouter spend may differ.
