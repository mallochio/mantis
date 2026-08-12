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

- `cheap-only`, `middle-only`, `expensive-only`: direct Bifrost calls pinned to
  the corresponding catalog `upstream_model`;
- `mantis-direct`: the routed `model=mantis` endpoint;
- `heuristic`: client-side prompt complexity selection;
- `random-matched`: tier sampling from a frozen distribution measured after
  the complete `mantis-direct` phase;
- `trinity`: independently selectable `model=mantis-trinity`.

Every routed request carries `X-Route-Session` and records route decision
headers. `pi` (headless `--mode json`, from the `@earendil-works/pi-coding-agent`
CLI) drives each task in a host worktree checked out at the instance's
`base_commit`; its chat-completions calls go through a local header-recording
proxy to Bifrost or the Mantis gateway (no litellm or mini-SWE-agent involved).
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

## Files

- `run_eval.py` — runs a config against fixtures and writes raw results.
- `score.py` — scores raw results and writes a scored projection plus `report.md`.
- `ab_compare.py` — compares two result sets for an A/B report.
- `report_luna.py` — builds the luna-conductor report against the native-v2 baseline.
- `fixtures.jsonl` — the fixed set of eval prompts/cases used by all configs.

## Regeneration commands

```
# run a config through the local Bifrost or Mantis endpoint
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
