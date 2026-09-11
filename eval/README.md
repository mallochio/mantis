# eval/

Evaluation harness and results for Mantis.

## Router harness (W1/W2)

`router_manifest.json` freezes 40 stratified instances from the `2026_03`
split of `nebius/SWE-rebench-leaderboard` at dataset revision
`34d5a58864acf91613740a09ec5d205228dcfa39`. Its split, seed,
`created_at_window`, repository distribution, and exact instance records are
the reproducibility boundary; runs belong in the ignored `eval/runs/`
directory.

`router_eval.py` supports these arms:

- `cheap-only`, `middle-only`, `expensive-only`: direct LiteLLM calls pinned to
  the corresponding catalog `upstream_model`;
- `mantis-direct`: the routed `model=mantis/base` endpoint;
- `heuristic`: client-side prompt complexity selection;
- `random-matched`: tier sampling from a frozen distribution measured after
  the complete `mantis-direct` phase;
- `trinity`: independently selectable `model=mantis/trinity`.

Every routed request carries `X-Route-Session` and records route decision
headers. `pi` (headless `--mode json`, from the `@earendil-works/pi-coding-agent`
CLI) drives each task in a host worktree checked out at the instance's
`base_commit`; its chat-completions calls go through a local header-recording
proxy to LiteLLM or Switchyard (no litellm SDK or mini-SWE-agent involved).
The submitted `git diff` is graded in a fresh SWE-rebench container; `resolved`
is true only when every `FAIL_TO_PASS` test passes and every `PASS_TO_PASS`
test remains passing. Gold patches are never used by an arm. Each arm selects
its own endpoint, model, session header, and request budget; `pi` must be
installed and on `PATH`.

`--budget-usd`, `--per-instance-cost`, `--timeout`, and
`--output-token-limit` are recorded in run metadata. Cumulative and
per-instance ceilings are enforced by the recording proxy before every model
request; an aborted instance is not emitted, so output remains a complete
prefix with all arms present. Budget exhaustion preserves prior JSONL output
and writes `aborted_on_budget: true`. JSONL records are explicitly tagged as
`metadata` or `result`; `route_metrics.py` analyzes all result attempts, while
reporting completed, resolved, and error counts separately. `--dry-run` makes no model calls and
reports worst-case per-instance-cap spend:

```text
uv run python eval/router_eval.py --dry-run --include-trinity
```

Costs use response `usage.cost` when supplied, otherwise token counts multiplied
by the tracked `model_prices.json` OpenRouter snapshot, which has separate
input/output prices. Missing usage and missing price data remain `unknown`;
there is no estimate fallback. Budget caps and prices are independent. Every
request and result row records its cost method.
`route_metrics.py` derives the cheapest resolving tier oracle, routing accuracy,
under/over-routing regret, endpoint interpolation, and an injected-decider
complexity confusion matrix. It never treats an error row as a successful
resolution.

## Fixed arms and shadow cost (free-model Zen study)

For the free-model studies the harness can run **fixed arms** against explicit
provider-qualified model IDs and account cost on a frozen **shadow** price
snapshot instead of the `$0` free-tier price:

```bash
uv run python eval/router_eval.py   --manifest eval/router_manifest.json   --prices eval/model_prices.json   --shadow-prices eval/prices/zen-2026-08-13.json   --cost-mode shadow   --fixed-model hy3=opencode-zen/hy3-free   --arms hy3   --resume   --timeout 1200 --output-token-limit 4096   --per-instance-cost 5.0
```

- `--fixed-model NAME=PROVIDER/MODEL` (repeatable) declares a fixed arm pinned
  to one LiteLLM model; `--arms` then lists only those names. Fixed arms never
  resolve to `opencode-go/*` and never touch the live Mantis router.
- `--cost-mode shadow` ignores `usage.cost` and prices token usage against the
  dated snapshot; `actual_cost_usd`, `actual_cost_method`, `shadow_cost_usd`,
  and `shadow_cost_method` are written per request and per row. Missing token
  counts or prices stay `unknown`, never `$0`.
- `--resume` skips `(instance_id, arm)` pairs already present in `--output`, so
  a spot-preempted worker can continue from the last uploaded JSONL.
- `--per-instance-cost` / `--arm-cap ARM=USD` bound the shadow spend per
  episode; only a **global** budget abort stops the whole campaign, while
  timeouts and per-instance cap hits are recorded as per-episode failures.

The recording proxy retries transient upstream failures (429/5xx and
rate-limit error envelopes surfaced as a `400` body) with exponential backoff plus jitter,
honoring `Retry-After`. Tune it with `EVAL_PROXY_RETRIES` (default 8),
`EVAL_PROXY_RETRY_BASE` (1.0s), `EVAL_PROXY_RETRY_MAX_WAIT` (60s), and
`EVAL_PROXY_RETRY_ON_STATUS` (comma list; default `429,500,502,503,504`).

`manifests/zen-pilot-12.json` is the frozen 12-task compatibility pilot; the
40-task `router_manifest.json` is the tier-selection manifest. The shadow
snapshot is `prices/zen-2026-08-13.json`. `ling-3.0-tiny-free` is not served
by the Zen gateway (`ModelError`) and is excluded from cost/quality tiers.

The follow-on binary direct-router protocol is
`plans/direct-binary-router-xroutebench.md`: it pairs Zen `hy3-free` with the
subscription-backed `opencode-go/deepseek-v4-flash`, sends both explicit IDs
through LiteLLM, and trains a calibrated probability rather than forcing a
three-tier label. xRouteBench is used to compare offline router algorithms;
its public test split is not a hyperparameter hill-climbing target.

## Cloud benchmark runs

`launch/sky/zen-swe-rebench-*.yaml` define the GCP/SkyPilot tasks; the worker
scripts under `scripts/zen_swe_rebench_*.sh` host a private Zen-only LiteLLM proxy on
`127.0.0.1:8080` on each worker, verify the model allowlist, and upload results
to `gs://your-eval-storage-bucket/<job-id>/` (upload-only; no GCS bucket is created).
Phase 2 shards one cluster per fixed arm and fans results in to
`gs://your-eval-storage-bucket/<run-id>/<arm>/results.jsonl`.

```bash
sky launch -y -d --cluster zen-phase2-hy3 launch/sky/zen-swe-rebench-phase2.yaml   --env ARM=hy3 --env RUN_ID=job-<8hex>   --env OPENCODE_API_KEY=... --env LITELLM_API_KEY=...
```

Secrets are passed via `--env` and are never committed.

## Files

- `router_eval.py` — the graded SWE-rebench router/fixed-arm harness.
- `route_metrics.py` — oracle, regret, interpolation, and confusion-matrix metrics.
- `run_eval.py` — runs a config against fixtures and writes raw results.
- `score.py` — scores raw results and writes a scored projection plus `report.md`.
- `ab_compare.py` — compares two result sets for an A/B report.
- `report_luna.py` — builds the luna-conductor report against the native-v2 baseline.
- `fixtures.jsonl` — the fixed set of eval prompts/cases used by all configs.
- `router_manifest.json` — frozen 40-task SWE-rebench tier-selection manifest.
- `manifests/zen-pilot-12.json` — frozen 12-task compatibility pilot manifest.
- `prices/zen-2026-08-13.json` — frozen free-model shadow price snapshot.

## Regeneration commands

```
# run a config through the local LiteLLM or Mantis endpoint
uv run python eval/run_eval.py --config direct --fixtures eval/fixtures.jsonl --output eval/results.jsonl
# score raw results; writes <stem>-scored.jsonl and eval/report.md
python3 eval/score.py --results eval/results.jsonl
# luna report; reads tracked native-v2 raw results as baseline
python3 eval/report_luna.py
```

## Boundary rule

Commit fixtures, code, and canonical reports. Do not commit raw run output or
derived `-scored` projections; raw runs belong in `runs/` (gitignored) and
derived projections are regenerated on demand. Reports must be reproducible
from tracked fixtures and raw results.

Reports display a quality or latency mean only when an arm has at least four
successful rows. Comparisons use the intersection of successful item IDs and
require that paired intersection to contain at least four items; otherwise the
report omits derived deltas and percentages.
