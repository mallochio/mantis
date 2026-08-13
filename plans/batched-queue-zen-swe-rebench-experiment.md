# Batched Queue × Zen SWE-rebench experiment

**Status:** design only. Do not run until the implementation and preflight gates pass.

## Question

Does making the `batch_queue` Pi extension available improve a coding agent's **verified SWE-rebench resolution**, compared with the same Pi agent and the same Zen-served model using native tools alone?

This explicitly asks whether `batch_queue` makes the fixed-model **Pi agent harness more capable**—operationally, “smarter”—at verified coding work. In this plan, “smarter” has one testable meaning: with the same model, task, prompt, native-tool authority, fresh environment, and grading, does adding `batch_queue` increase official SWE-rebench resolution?

It does **not** ask whether the underlying Zen model's weights, training, or general intelligence changed. If the preregistered holdout passes, the supported conclusion is: **“`batch_queue` made this fixed-model coding-agent system/harness smarter at this benchmark under these controls.”** It is not: “the Zen model itself became smarter.”

## Execution environment

The benchmark campaign runs on **GCP through SkyPilot**. SkyPilot provisions and resumes the isolated worker VMs, mounts only public benchmark artifacts and scoped secrets, and runs one resumable shard per task/model/condition. Dockerized SWE-rebench grading runs on those workers. The private Zen-only Bifrost gateway runs in the same GCP project/VPC; workers must reach it only through its private endpoint.

Local execution is limited to adapter development and no-model checks. Do not run P1, P2, or P3 from a developer laptop or against a local Bifrost instance. Record the SkyPilot YAML/image digest, GCP region/zone, machine type, worker image, Bifrost deployment hash, and shard manifest in each phase artifact.

## Why a new experiment is needed

The sibling `batched-queue` repository already has useful mechanism tests and an incomplete Terminal-Bench study, but it is not evidence for this question:

- its custom fixture suite is intentionally mechanism-oriented and has documented oracle/aggregation issues;
- the Terminal-Bench prefix is incomplete and currently null-to-negative under neutral tool availability;
- its existing mode-router classifier learns from the same Terminal-Bench outcome ledger it evaluates, so it is not evidence that batch availability improves a new benchmark;
- its Terminal-Bench agent currently supports Azure/OpenAI/etc. credentials, not Bifrost;
- its paper draft contains claimed completed-run numbers inconsistent with the live readiness/runbook documents. Do not reuse those numbers.

SWE-rebench gives a different, repository-level task distribution with Docker grading. It must be treated as a new preregistered study, not as a rescue experiment for an expected positive result.

## Privacy and data boundary

Only public, revision-pinned benchmark inputs may reach Zen:

```text
public task instructions
public repository commits
public test patches and Docker images
model-generated diffs and tool traces from those public tasks
```

Never send production Mantis prompts, user sessions, local/private repositories, credentials, or retraining data. Disable Bifrost raw request/response and content logging for the benchmark gateway.

SWE-rebench is a fresh/public benchmark source, but not automatically proof of no model contamination. Before publication, record each Zen endpoint's exact served identity, release date, disclosed training cutoff (if any), task creation window, and dataset retrieval time. If a model's training cutoff is unknown, call the benchmark **public, recently collected**, not “uncontaminated” or “non-saturated.”

## Models

All model calls, including any optional objective-planner call, route through a private Bifrost gateway. Do not use `opencode-go` for this study.

### Discovery pool

```text
opencode-zen/deepseek-v4-flash-free
opencode-zen/mimo-v2.5-free
opencode-zen/hy3-free
opencode-zen/ling-3.0-tiny-free
opencode-zen/nemotron-3-ultra-free
opencode-zen/nemotron-3.5-lightning-free
opencode-zen/laguna-s-2.1-free
```

### Primary-model selection rule

Use the separately planned 12-task compatibility pilot and 40-task direct fixed-arm Mantis evaluation to select three models before this extension study:

```text
cheap:  lowest shadow-cost viable Zen model
middle: lowest shadow-cost model with a preregistered quality improvement over cheap
strong: highest-resolution viable Zen model
```

Freeze the three exact Bifrost IDs, returned served IDs, Bifrost config hash, endpoint version, Pi version, and price snapshot hash before any extension comparison. The extension study does not reroute among these tiers; it runs one fixed model per episode.

If no three viable Zen tiers exist, run the extension experiment on the best two viable models and label it a two-model replication. Do not invent a tier to preserve the plan.

## Experimental conditions

The primary treatment is **tool availability**, not a different prompt, model, planner, or tool authority.

| ID | Name | Pi setup | Use in claim |
|---|---|---|---|
| N | Native-only | Native `read`, `grep`, `find`, `ls`, `bash`, `edit`, and `write`; no extension loaded | Primary control |
| A | Batch-available | Exact same native tools and permissions plus `batch_queue`; neutral task prompt; model may ignore it | **Primary treatment** |
| F | Forced explicit batch | Same authority, but prompt requires an explicit `actions` batch when a safe batch is applicable | Mechanism/adoption ablation only |
| O | Objective batch | Same authority, `batch_queue` objective mode; planner usage separately logged | Separate secondary study only |

Do not compare native-only against a batch condition that removes native `bash`, `edit`, or `write`; that tests reduced authority, not batching. Do not include `F` or `O` in the primary headline: forcing a new protocol or adding another model-mediated planner is a different intervention.

The primary prompt must be byte-identical in N and A. The only difference is that A's tool registry contains `batch_queue` and its tool description. Record the tool manifest and hashes in each run artifact.

## Hypotheses and endpoints

### Primary hypothesis

For a fixed model and benchmark task, batch availability changes official grader pass probability:

```text
H0: P(pass | A) - P(pass | N) = 0
H1: P(pass | A) - P(pass | N) > 0
```

This is intentionally one-sided only if preregistered before data collection. Also report the two-sided paired result and every negative effect.

### Primary endpoint

```text
official SWE-rebench resolved status
```

A timeout, provider failure, malformed-tool event, invalid run, empty patch, or grader failure counts as not resolved. Do not drop it from the operational denominator.

### Secondary endpoints

- batch adoption rate in A;
- batch invocations, requested/completed actions, halts, and replans;
- agent/model turns and tool calls;
- elapsed wall time, including model waiting and grading, with median/p95;
- input/output/cache token counts;
- actual API spend (expected `$0` for Zen) and frozen shadow cost;
- provider/tool/agent/grader error taxonomy;
- patch size and final verification status;
- paired N/A conditional outcomes: both pass, A-only pass, N-only pass, neither pass.

A lower tool-call count without higher verified resolution is an interface-efficiency result, not evidence that the agent harness became smarter under this plan's definition.

## Dataset design

Use only an exact public SWE-rebench dataset revision. Materialize and commit small manifests containing only task identifiers, source revision, repository, base commit, task timestamp, Docker image digest, and split.

Split **by repository**, not individual task, before model calls:

| Partition | Intended size | Purpose |
|---|---:|---|
| Compatibility/tuning | existing 40 frozen tasks | Bifrost/model compatibility and Zen-tier selection; never used for paper headline or extension tuning |
| Development | 80 tasks | validate Bifrost Pi adapter, tune timeout/output cap, inspect tool-manifest parity, choose no more than one final fixed protocol |
| Primary holdout | 200 tasks | preregistered N vs A result, never used to revise prompts, extension behavior, model selection, or routing thresholds |
| Robustness holdout | 60 tasks | one additional run per N/A condition for stochasticity and order sensitivity |
| Mechanism subset | 50 tasks sampled from primary-holdout strata before runs | F and optional O ablations; not merged into primary N/A estimate |

If the selected SWE-rebench revision cannot supply 340 repository-disjoint public tasks, stop and report the available count. Do not mix repositories across partitions to reach the target.

Stratify each partition by repository ecosystem, task date, test/runtime band, and issue category when those fields exist. Save task selection seed and code. Do not select tasks after observing model performance.

## Sample size and statistical analysis

The independent unit is the **benchmark task**, not a model request, tool call, or repeat rollout.

Primary workload:

```text
200 tasks × 3 frozen Zen models × 2 conditions (N, A) = 1,200 agent episodes
```

This gives 200 task-level paired comparisons per model and a three-model replication. It is suitable for detecting a practically meaningful effect around 8–10 percentage points if the effect is consistent; it is not evidence for tiny gains. If only two models survive tier selection, run 800 primary episodes and state the lower replication breadth.

Analysis is locked before the primary holdout launches:

1. For each model, produce the 2×2 paired outcome table and exact McNemar p-value.
2. Calculate N/A difference in resolved rate with a paired task bootstrap (10,000 resamples, seed 42).
3. Calculate a pooled estimate by resampling **tasks**, retaining all selected-model outcomes for each sampled task. This avoids pretending three correlated model attempts are three independent benchmark tasks.
4. Report model-specific results, pooled result, and treatment × model interaction. Do not hide a harmful model behind a pooled mean.
5. Treat missing/invalid/provider-error episodes as failures for the primary operational result; additionally report a valid-run sensitivity analysis labeled secondary.
6. Correct only the three predeclared model-specific primary tests (Holm correction). All other slices and adoption-conditioned comparisons are exploratory.

The claim **“`batch_queue` made the fixed-model Pi agent harness smarter at verified SWE-rebench resolution”** is allowed only if the pooled paired 95% interval is above zero **and** at least two of the three model-specific point estimates are positive without materially worse operational error rates. Otherwise the result is null/mixed: `batch_queue` did not demonstrate a general harness-capability gain under this protocol.

## Order, independence, and stopping rules

For every `task × model` pair, generate a deterministic order from seed 42:

```text
N then A, or A then N
```

Run each condition in a fresh checkout/container and a fresh Pi session. Do not carry shell state, Bifrost conversation state, extension state, cache keys, or tool output between conditions. Use unique opaque `X-Route-Session` values per `study/task/model/condition/repetition`.

Do not peek at primary-holdout results to alter protocol. Stop only for:

- credentials or private data exposure;
- Bifrost reporting a model identity outside the frozen allowlist;
- tool-authority mismatch between N and A;
- provider outage/rate limit affecting more than 10% of scheduled episodes;
- missing token/trace accounting above 5% of calls;
- a confirmed grading harness defect.

Preserve all completed artifacts and restart only the affected phase after documenting the failure and a protocol amendment.

## Bifrost/Pi implementation work

The sibling `batched-queue` Terminal-Bench adapter currently invokes Pi's provider directly and its environment allowlist does not include Bifrost. Implement a separate SWE-rebench adapter rather than altering the historical Terminal-Bench result.

### Required adapter behavior

1. Create a benchmark-only Pi provider extension, modeled on `mantis/eval/router_eval.py::_render_provider_extension`:
   - `baseUrl` is the private Bifrost OpenAI-compatible endpoint;
   - `apiKey` is a short-lived Bifrost virtual key;
   - model IDs are the provider-qualified `opencode-zen/...` IDs;
   - model prices in the Pi provider declaration are zero only to prevent client-side billing; official accounting comes from the Bifrost recording proxy;
   - preserve the exact generated TypeScript file/hash as an artifact.
2. Load the provider extension in **both** N and A. A loads `batch_queue` in addition; N does not.
3. Use a local per-episode recording proxy between Pi and Bifrost, modeled on Mantis `eval/router_eval.py`:
   - forward only to private Bifrost;
   - attach `X-Route-Session`;
   - record request model, Bifrost returned model, route headers, usage, latency, and error status;
   - enforce the timeout and shadow-cost ceiling;
   - never log task text or raw assistant content outside the private run artifact bucket.
4. Assert at startup that every requested model is `opencode-zen/*`; reject `opencode-go/*`, direct OpenCode URLs, OpenRouter, Azure, Meta, and all unqualified model IDs.
5. Expose the identical native tool allowlist in N and A. The batch tool may call the same filesystem/shell operations, but it must not get extra network or process permission.
6. Pin the `batched-queue` commit as a tarball copied into each container. Validate its source hash before Pi starts.
7. Preserve Pi JSON events, compact batch metadata, grader result, and normalized cost/usage rows. Do not use final assistant prose as the success oracle.

### Gateway deployment

Use a short-lived Bifrost deployment inside the GCP project/VPC with only the seven Zen models allowlisted. Give SkyPilot workers a scoped benchmark virtual key from secret management. Do not make the local `127.0.0.1:8080` gateway public and do not place user credentials in SkyPilot YAML, shell history, logs, or git.

The cloud Bifrost `/v1/models` response must be archived before every phase. It must list the exact model IDs and no paid fallback route. Destroy the gateway virtual key and temporary cloud resources after the campaign.

## Phased execution

### Phase P0 — no-model validation

Run locally and in a disposable cloud worker:

```text
- verify pinned benchmark manifest/dataset revision and Docker image digests;
- verify N/A prompt equality and tool-authority equality;
- verify generated Pi provider extension points only at Bifrost;
- verify all seven Zen IDs from the cloud Bifrost `/v1/models` response;
- run Bifrost/Pi smoke with a public harmless prompt for each model;
- run extension unit/type checks and adapter tests;
- execute one no-model fake-provider N/A parity test;
- run `sky launch --dryrun`.
```

P0 fails closed on any direct-provider route or tool mismatch.

### Phase P1 — development pilot

```text
80 development tasks × 3 models × N/A = 480 episodes
```

Start at parallelism 4. Increase to 8 then 16 only when 429/5xx stays below 2%, episodes are resumable, and all trace fields appear. Use this phase only to validate the fixed protocol and measurement pipeline. It is not a paper result and must not be reused as the primary holdout.

P1 gate:

```text
>=95% complete accounting
<10% provider/agent error rate per model/condition
no systematic batch extension load failure
N and A have identical native tool manifest/permission hash
```

Freeze all settings after P1: model IDs, extension commit, Pi/Bun versions, task manifests, prompts, timeout, output limit, concurrency, proxy, and analysis script commit.

### Phase P2 — primary preregistered holdout

```text
200 tasks × 3 models × N/A = 1,200 episodes
```

Launch only after a written preregistration file is committed. It contains hypotheses, seed, task IDs, exclusion policy, cost/timeout policy, metrics, and the exact analysis command. Do not enable F/O in this phase.

### Phase P3 — robustness and mechanism

After P2 completes without changing P2 settings:

```text
60 tasks × 3 models × N/A × one additional repetition = 720 episodes
50 tasks × 3 models × F = 150 episodes
optional: 50 tasks × 3 models × O = 150 episodes
```

Report F as forced-interface behavior. Report O's driver and planner costs/tokens separately from deterministic queue execution; do not combine it with N/A.

## Cost accounting

Zen actual API cost is expected to be zero, but record it separately from shadow cost:

```json
{
  "actual_cost_usd": 0.0,
  "actual_cost_method": "bifrost_usage.cost",
  "shadow_cost_usd": 0.0,
  "shadow_cost_method": "frozen_zen_price_snapshot",
  "served_model": "opencode-zen/..."
}
```

`shadow_cost_usd` must use the same frozen exact-model/proxy-pricing rules as the Mantis router study. It is an analytic counterfactual, never a charge. The extension paper's primary endpoint does not depend on shadow cost, so missing defensible prices must not block the correctness study; it blocks only cost-efficiency conclusions.

Cloud compute is intentionally outside the current monetary gate, but set a separate operational ceiling for maximum VM-hours, Cloud Storage lifecycle, task timeout, and concurrent workers. Use SkyPilot Spot resume and shard-level idempotency.

## Reporting and paper boundaries

Report a CONSORT-style flow:

```text
scheduled task/model pairs
completed N/A pairs
excluded/predeclared invalid pairs
provider/agent/grader errors by condition
resolved N and A pairs
batch adoption in A
```

The paper may claim one of the following, in descending order of strength:

1. **Harness-capability (“smarter”) gain:** only if P2 primary criteria pass. State the operational definition and controls in the same sentence.
2. **Interaction compression without a demonstrated smarter-harness gain:** if turns/tool calls fall with non-inferior verified resolution.
3. **Workload-boundary/null result:** if A is neutral or harmful overall but helps a preregistered task category.
4. **No observed benefit:** if neither resolution nor efficiency improves.

Never claim that the base Zen model's weights, training, intrinsic reasoning ability, or general intelligence changed. Do not claim general tool-use superiority or benchmark decontamination from this study alone.

## Completion checklist

- [ ] P1, P2, and P3 ran on GCP through pinned SkyPilot workers, with per-shard resumability.
- [ ] Cloud Bifrost is private, Zen-only, scoped, and destroyed after the run.
- [ ] Bifrost/Pi adapter forbids non-Zen model IDs and direct provider endpoints.
- [ ] N/A native tool manifest and prompt hashes match.
- [ ] All selected model identities are frozen from Bifrost responses.
- [ ] Repository-disjoint development/primary/robustness manifests are committed before execution.
- [ ] P0/P1 gates pass and protocol is frozen.
- [ ] P2 has exactly the preregistered 1,200 N/A episodes or documented predeclared failures.
- [ ] Analysis uses task-level paired bootstrap and McNemar, with model-specific outcomes.
- [ ] Raw artifacts remain private; no secrets or private prompts are retained.
- [ ] Paper language matches the strongest supported claim only.
