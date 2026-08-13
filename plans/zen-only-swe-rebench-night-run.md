# Zen-only SWE-rebench router experiment

**Status:** planned; do not launch until the gates below pass.

**Goal:** establish whether three free OpenCode Zen models create a measurable routing problem on public SWE-rebench tasks, then collect public multi-step traces suitable for a later router-training decision.

**Non-goals:** production routing changes, private prompts/sessions/repositories, Zen fine-tuning, or retraining tonight.

## Scope and privacy boundary

| Allowed | Not allowed |
|---|---|
| Existing public SWE-rebench task text, public repository revisions, public test patches, and generated evaluation traces | Mantis production traffic, user prompts, local worktrees containing non-benchmark code, retraining data, secrets, or manually supplied private tasks |
| `eval/router_manifest.json` and later public datasets frozen by revision | Any task not pinned to a public dataset revision before execution |

The current manifest is 40 public tasks from `nebius/SWE-rebench-leaderboard`, revision `34d5a58864acf91613740a09ec5d205228dcfa39`. It is a smoke/tier-selection set, not training data.

## Models and identities

Use only the Bifrost Zen provider IDs below. Do not substitute same-named `opencode-go` models.

```text
opencode-zen/deepseek-v4-flash-free
opencode-zen/mimo-v2.5-free
opencode-zen/hy3-free
opencode-zen/ling-3.0-tiny-free
opencode-zen/nemotron-3-ultra-free
opencode-zen/nemotron-3.5-lightning-free
opencode-zen/laguna-s-2.1-free
```

The existing paid/subscription control remains distinct and excluded:

```text
opencode-go/deepseek-v4-flash
```

All Zen API costs must be recorded as observed cost, expected to be `$0`. That must never become the routing cost signal.

## Required implementation before any benchmark

The present `eval/router_eval.py` is insufficient for this experiment:

1. It supports only three named tier arms plus routed controls, not seven independently named fixed-model arms.
2. It prefers returned `usage.cost`; a free Zen response can therefore make every arm cost `$0`.
3. `route_metrics.py` consumes `cost_usd` for the frontier, oracle regret, and interpolation.
4. The tracked catalog maps production targets to paid models. Do not repoint it to Zen merely to evaluate Zen.
5. The current local Bifrost endpoint is loopback-only. A SkyPilot worker cannot use it unless a deliberately authenticated private endpoint is made reachable. Do not expose `:8080` publicly or copy long-lived user credentials into task specs.

Implement these changes in a separate commit before launch:

### A. Fixed-arm support

Add an explicit `--fixed-model NAME=PROVIDER/MODEL` repeatable option, or an equivalent fixed-arms manifest. It must:

- run exactly the listed models directly through Bifrost;
- generate one opaque `X-Route-Session` for each `task × arm`;
- retain the served model returned by Bifrost, request trace, tool trajectory, grader result, timeout, and error;
- keep an endpoint or grader error as a failed result, not silently retry it as another attempt;
- permit only network/Spot-preemption resume retries that reuse the same task and arm identity and are explicitly recorded.

Do **not** overload `cheap`, `middle`, and `expensive` during tier selection. Tier labels are selected only after all seven direct arms complete.

### B. Frozen Zen shadow-price snapshot

Add `--cost-mode actual|shadow`, defaulting to `actual` for existing behavior. For `shadow`:

- ignore `usage.cost` when computing evaluation cost;
- write `actual_cost_usd`, `actual_cost_method`, `shadow_cost_usd`, and `shadow_cost_method` to every request trace and result;
- calculate shadow cost only from returned input/output/cached token counts and a versioned local JSON snapshot;
- fail that request's shadow accounting as `unknown` when a required token component or exact model price is absent; never invent a token count;
- use `shadow_cost_usd` for caps, fixed-arm summaries, frontier, oracle regret, and interpolation;
- retain actual Zen cost separately for operations reporting.

Create a dated snapshot such as `eval/prices/zen-2026-08-13.json` before the pilot. Each entry must contain:

```json
{
  "served_model": "opencode-zen/deepseek-v4-flash-free",
  "source_model_version": "exact version if disclosed; otherwise unknown",
  "input_per_token": 0.0,
  "output_per_token": 0.0,
  "cached_input_per_token": 0.0,
  "source_url": "https://...",
  "retrieved_at": "2026-08-13T...Z",
  "method": "manufacturer price or one consistent paid-provider proxy"
}
```

`0.0` is invalid for shadow prices. Zen free access is not a shadow price. If one exact model has no defensible comparable paid price, mark it unavailable and do not put it on a cost frontier.

### C. Tests and dry-run

Add focused tests proving:

- a Zen response with `usage.cost = 0` still has nonzero shadow cost when token usage and prices exist;
- actual and shadow cost remain distinct;
- cached-token pricing is included;
- missing price or usage produces `unknown`, not zero;
- exact model lookup rejects an ambiguous basename;
- all router metrics use the selected evaluation cost field;
- fixed arms cannot accidentally call `opencode-go`.

Add a SkyPilot dry-run only after the harness changes pass locally. No model calls in a dry-run.

## Cloud execution design

Use one self-contained, CPU-only SkyPilot managed job per rollout shard. Model inference is remote; GPUs are unnecessary for evaluation.

### Network and credentials

Preferred: deploy a short-lived Bifrost instance in the same private GCP project/VPC as the SkyPilot workers, configured only with:

- `opencode-zen`;
- the seven allowlisted model IDs;
- a dedicated, scoped Bifrost virtual key;
- request/content logging disabled;
- no production provider keys;
- no Mantis API/gateway access.

Provide the Zen credential and Bifrost virtual key through the cloud secret manager or SkyPilot secret mechanism. Never commit credentials, paste them into a YAML file, emit them in logs, or expose the Bifrost listener publicly. Destroy the short-lived gateway/key after the campaign.

If a private GCP Bifrost gateway is not ready before tonight, do not launch cloud workers. Run only the local 12-task compatibility pilot after the harness gates are complete.

### Worker behavior

- CPU-only Spot instances, Docker enabled, automatic resume after preemption.
- One task/arm per worker process; no shared worktree.
- Read a checked-in public manifest shard by task ID.
- Write append-only JSONL to a unique Cloud Storage prefix.
- Upload after each result; a resume must skip an already completed `run_id/task_id/arm` tuple.
- Use a fixed container image digest, pinned `pi` version, pinned repository commit, and pinned Docker grader images.
- Configure a maximum parallelism only after pilot rate-limit results. Start at 4 workers; raise to 8, then 16 only if Zen 429/5xx rates remain below 2%.
- Add a 15-minute startup/image-pull allowance and a 20-minute task wall-clock cap for pilot. Tune only once, from pilot evidence, before the 40-task run.

### Public artifact layout

```text
gs://<private-eval-bucket>/zen-swe-rebench/
  manifests/<manifest-sha256>.json
  pricing/<price-snapshot-sha256>.json
  runs/<run-id>/metadata.json
  runs/<run-id>/shards/<task-id>--<arm>.jsonl
  runs/<run-id>/logs/
```

The bucket must be private, have a lifecycle rule, and contain no credentials. Raw results stay out of git. Commit only harness code, manifests, price snapshots, and reproducible report generators.

## Phase 0: preflight tonight

Do this before any paid cloud time or broad rollout:

1. Confirm the seven `opencode-zen/*` IDs from the **cloud-side** Bifrost `/v1/models` response.
2. Send one minimal public, non-sensitive completion to every model; record exact returned model ID, usage fields, and response behavior.
3. Test a minimal tool-call and JSON response only if the `pi` harness requires it.
4. Verify Bifrost returns no `developer`-role incompatibility and that the model can complete a one-tool agent loop.
5. Run `pytest` for the new shadow-cost and fixed-arm tests.
6. Run `router_eval.py --dry-run` with the final command and preserve its metadata.
7. Run `sky launch --dryrun <task.yaml>` and inspect chosen region, CPU, disk, Spot configuration, and private networking.

**Stop immediately** if any model has no usage metadata, cannot execute the required agent flow, returns a different undisclosed model identity, or sends data outside the declared Zen endpoint.

## Phase 1: 12-task compatibility pilot

Select exactly 12 task IDs from the existing 40-task frozen manifest before launching. Stratify by repository and task characteristics, and write the IDs plus rationale to a committed pilot manifest. Do not choose them after seeing model results.

```text
12 tasks × 7 fixed Zen models = 84 rollouts
```

Frozen settings for every arm:

```text
agent:                 pi headless JSON mode
agent tools:           read,bash,edit,write
attempts:              one
output token limit:    4096 (only tune after pilot evidence)
wall timeout:          20 minutes (only tune after pilot evidence)
grading:               fresh official SWE-rebench container
network during agent:  only the model endpoint; no general internet
```

Pilot success gates, per model:

| Gate | Required result |
|---|---|
| Identity | 12/12 responses report the intended `opencode-zen/*` model ID, or an explicitly documented alias |
| Privacy | no private input; only frozen public task data reaches Zen |
| Tool flow | no systematic malformed tool/JSON failure |
| Accounting | at least 95% of model calls have complete token usage and non-unknown shadow price |
| Reliability | fewer than 10% provider/transport failures, excluding recorded Spot preemptions |
| Grading | grader containers start and yield a result for at least 11/12 tasks |

Publish a pilot table with resolved count, provider errors, grader errors, median/p95 wall time, request count, token counts, actual cost, and shadow cost. Do not rank models from 12 tasks.

## Phase 2: fixed-arm tier selection

Run the full existing manifest once after pilot gates pass:

```text
40 tasks × 7 fixed Zen models = 280 rollouts
```

Use direct fixed arms only. Do not call `mantis`, `heuristic`, `random-matched`, `trinity`, retraining code, or production catalog targets in this phase.

Before examining outcomes, freeze the tier-selection rule:

1. Exclude models with >=10% provider/agent errors or incomplete shadow accounting on >=5% of task rows.
2. `cheap` is the viable model with the lowest median shadow cost.
3. `middle` is the lowest-shadow-cost viable model with a paired resolve-rate improvement over cheap whose 95% paired-bootstrap interval lower bound is greater than zero.
4. `strong` is the viable model with the highest resolve rate; if several overlap, choose the lower-shadow-cost one.
5. If no model qualifies for a distinct middle or strong tier, stop. Do not fabricate a three-tier routing experiment.
6. Record the exact selected models, their price snapshot hash, and this decision before training-data collection.

This phase can identify dominated models and candidate tiers. It cannot establish that the router works statistically.

## Phase 3: public training-data collection

This phase is conditional on Phase 2 producing three distinct viable tiers.

Build a new, public, revision-pinned dataset of 300–500 tasks. Split by repository, never by individual task:

```text
training repositories:    about 70%
validation repositories:  about 15%
final-holdout repositories: about 15%
```

No repository appears in more than one partition. The existing 40-task manifest remains tier-selection only and is excluded from this collection.

For training and validation tasks run only the frozen selected three arms:

```text
300–500 tasks × 3 fixed tiers = 900–1,500 rollouts
```

Retain public trajectories sufficient for later label construction:

```text
public task ID and dataset revision
selected fixed arm / served model
request-level route and usage traces
ordered tool trajectory
patch hash and grader outcome
actual and shadow cost
provider, agent, timeout, and grader error categories
```

Do not train immediately. First calculate label distribution:

```text
cheap is cheapest resolver
middle is cheapest resolver
strong is cheapest resolver
none resolves
```

Proceed to router retraining only if there are at least 300 independent successful task sessions and at least 50 examples in each viable selected-tier label. Otherwise, keep the existing router and report insufficient boundary data.

## Phase 4: untouched router validation

This phase is conditional on a retrained router and an untouched, public, repository-disjoint holdout of 150–250 tasks.

Run each holdout task against:

```text
cheap-only
middle-only
strong-only
mantis-direct
random-matched
heuristic
```

```text
150–250 tasks × 6 arms = 900–1,500 rollouts
```

`random-matched` uses the tier-frequency distribution from completed `mantis-direct` rows. It is the primary control because it tests task-level routing value at the router's own tier mix.

Pre-register:

```text
Primary metric:
  paired resolved-rate difference: mantis-direct minus random-matched

Success criteria:
  - paired 95% bootstrap confidence interval lower bound > 0
  - shadow cost per resolved task no worse than random-matched
  - provider/agent error rate no worse by more than 2 percentage points
  - result reproduced on the untouched repository-disjoint holdout

Secondary metrics:
  - cheapest-resolving-tier oracle accuracy
  - under-routing quality loss
  - over-routing shadow-cost waste
  - quality/shadow-cost frontier and fixed-arm interpolation
  - median/p95 elapsed time and token use
```

Bootstrap by task ID, never individual model request: requests within a task are correlated.

## Overnight launch order

Tonight's safe stopping point is Phase 0 plus, at most, Phase 1. Do not start a 300–500 task collection unattended before inspecting pilot outputs.

```text
18:00  Complete harness change, tests, pricing snapshot, and fixed pilot manifest.
19:00  Cloud gateway/private-network and SkyPilot dry-run.
20:00  Run 84-rollout Phase 1 pilot at parallelism 4.
22:00  Check identities, rate limits, usage metadata, errors, and grader logs.
23:00  If every pilot gate passes, launch 280-rollout Phase 2 at parallelism 8.
Morning Review Phase 2 results; freeze tier decision or stop.
```

The 300–500 task trace collection and 150–250 task holdout require an explicit next-day approval after the Phase 2 tier-selection report. They are intentionally not included in an unattended overnight job.

## Commands to run only after implementation

The exact CLI will change once fixed arms and shadow accounting exist. The intended shape is:

```bash
# No model calls; confirms manifest, arm list, and shadow price snapshot.
uv run python eval/router_eval.py \
  --manifest eval/manifests/zen-pilot-12.json \
  --fixed-model deepseek=opencode-zen/deepseek-v4-flash-free \
  --fixed-model mimo=opencode-zen/mimo-v2.5-free \
  --fixed-model hy3=opencode-zen/hy3-free \
  --fixed-model ling=opencode-zen/ling-3.0-tiny-free \
  --fixed-model nemotron-ultra=opencode-zen/nemotron-3-ultra-free \
  --fixed-model nemotron-lightning=opencode-zen/nemotron-3.5-lightning-free \
  --fixed-model laguna=opencode-zen/laguna-s-2.1-free \
  --cost-mode shadow \
  --prices eval/prices/zen-2026-08-13.json \
  --dry-run

# Preview SkyPilot placement/cost without provisioning.
sky launch --dryrun launch/zen-swe-rebench-pilot.yaml
```

Do not run a command against the existing `router_eval.py`: it does not yet implement `--fixed-model` or valid Zen shadow accounting.

## Completion checklist

- [ ] Seven exact Zen IDs verified from cloud-side Bifrost.
- [ ] Bifrost is private, short-lived, allowlisted, and has no production credentials.
- [ ] Fixed-arm and shadow-cost changes implemented, tested, committed.
- [ ] Frozen pricing snapshot committed and hashes recorded in run metadata.
- [ ] Pilot task IDs committed before model calls.
- [ ] SkyPilot dry-run inspected.
- [ ] Pilot gates pass and pilot report saved.
- [ ] Full 40-task fixed-arm report selects three tiers or stops.
- [ ] Separate public repository-disjoint training and holdout manifests created before any retraining.
- [ ] Final claim uses the pre-registered paired holdout test.
