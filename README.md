# mantis

Local orchestration stack for multi-tiered LLM routing:
- **LiteLLM proxy** (:3001) — translates LiteLLM aliases to OpenRouter models.
- **RouteLLM router with Supra complexity header** (:5500) — scores prompt complexity.
- **Mantis orchestrator** (:8088) — runs TRINITY and Conductor modes.
- **Pi `/mantis` extension** — switches modes and logs routing decisions (alias: `/fugu`).

## How it works

**TRINITY** is a tiny per-turn dispatcher (Qwen3-0.6B with a learned SVF+head). For each turn it picks one worker from the 7-slot pool plus a role — `Worker` (answer), `Thinker` (reason), or `Verifier` (check) — then returns the verifier-approved response.

**Conductor** is a planner that emits an entire multi-step workflow up front as three Python lists: `model_id`, `subtasks`, and `access_list`. The access list is a DAG — later steps may only read strictly earlier steps — and the last step's output becomes the final answer. There are two ways to power the planner today:
- **LiteLLM planner (default, works today):** set `MANTIS_CONDUCTOR_MODEL=gpt-5.6-luna-max` and do **not** set `MANTIS_LOCAL_CONDUCTOR`. The planner call goes through the LiteLLM proxy, and the orchestrator executes the generated DAG against the worker pool.
- **Local 3B planner (archived):** `MANTIS_LOCAL_CONDUCTOR` loads a Llama-3.2-3B checkpoint. Both the public `di-zhang-fdu/openfugu-conductor-3b` base checkpoint and the retrained `outputs/conductor_retrain/retrain-conductor-20260802_003213/checkpoint` failed format validation in this repo: one emits invalid DAG topology (self/forward references), the other collapses to plain text instead of the required three-list format. These are model-overfit / format-collapse issues, not infra/memory issues. The checkpoints remain in S3 for future SFT+GRPO work but are not wired in by default.

**Auto mode** asks the router for a complexity score and picks the cheapest coordinator that should still succeed:
- simple / low-complexity prompts → direct single-call (`/mantis direct`)
- moderate complexity prompts → TRINITY (`/mantis trinity`)
- high complexity prompts → Conductor (`/mantis conductor`)

`MANTIS_AUTO_THRESHOLD` defaults to **6** because the current LiteLLM-planned Conductor still underperforms TRINITY on hard prompts; a Supra complexity score of 6 is never emitted by the router in practice, so auto mode stays in TRINITY/direct.

```text
                    ┌──────────────────┐
      user query →  │  Pi /mantis auto │
                    └────────┬─────────┘
                             │
          ┌──────────────────┼──────────────────┐
          ▼                  ▼                  ▼
    ┌──────────┐      ┌──────────┐       ┌────────────┐
    │  direct  │      │  trinity │       │  conductor │
    │ 1 call   │      │ 1 worker │       │ 1 plan +   │
    │ cheap    │      │ per turn │       │ N workers  │
    └──────────┘      └──────────┘       └────────────┘
```

## Running the stack

### A. All-Docker (default, no local GPU conductor)

Best for Linux with no local GPU or when you only want TRINITY/direct calls. The 3B Conductor runs on CPU inside the container and is slow.

```bash
cp .env.example .env          # add OPENROUTER_API_KEY, OPENAI_API_KEY, LITELLM_KEY; optional HF_TOKEN
docker compose build
docker compose up -d
./scripts/verify.sh
```

### B. Hybrid native-GPU (recommended on Apple Silicon / any GPU host)

Keep LiteLLM and `llm-router` in Docker, but run the Mantis orchestrator directly on the host so PyTorch can use MPS (Mac), CUDA (Linux), or a less-slow CPU fallback. The native process reaches LiteLLM on `localhost:3001` and the router on `localhost:5500` via the published Docker ports.

```bash
# 1. Start the Docker side of the stack (no openfugu container)
docker compose -f docker-compose.yml -f docker-compose.native-openfugu.yml up -d litellm router

# 2. Run the Mantis orchestrator natively
./scripts/run_mantis_native.sh

# 3. In another terminal, verify
./scripts/verify.sh
```

`run_mantis_native.sh` creates a dedicated venv, installs the core orchestrator package (`pip install -e .`), builds the TRINITY base vector from `artifacts/router_head.safetensors`, auto-detects the best PyTorch backend, and launches `openfugu-patch/serve.py` on `0.0.0.0:8088`.

## Platform / device table

| Host platform | Conductor device | Default dtype | Notes |
|---|---|---|---|
| macOS (Apple Silicon) | `mps` | `bfloat16` | Docker cannot access MPS; native mode is required for GPU conductor. |
| Linux + NVIDIA GPU | `cuda:0` | `bfloat16` | Native or Docker `--gpus all` both work; native avoids Docker GPU setup. |
| Linux / no GPU | `cpu` | `float32` | 3B model still loads, but generation is slow and may need >=12 GB RAM. |
| WSL / others | `cpu` | `float32` | Same as Linux CPU. |

Override with env vars:
```bash
FUGU_CONDUCTOR_DEVICE=mps               # or cuda:0 / cpu
FUGU_CONDUCTOR_DTYPE=bfloat16           # or float32
FUGU_CONDUCTOR_MAX_NEW=512
```

## Model knobs (set in `.env` or your shell, e.g. `.zshrc`)

All model selection is env-driven. Export the variables before `docker compose up` or before `run_mantis_native.sh`.

| Variable | What it controls | Default |
|---|---|---|
| `OPENROUTER_API_KEY` | API key LiteLLM uses to call OpenRouter | required |
| `OPENCODE_GO_API_KEY` | API key for optional opencode-* LiteLLM aliases | optional |
| `OPENAI_API_KEY` | OpenAI key for `llm-router` embeddings only | required for router |
| `LITELLM_KEY` | Shared internal bearer token for router/mantis | `change-me` |
| `FUGU_API_KEY` | Same token, used by the pi extension and `serve.py` auth | `${LITELLM_KEY}` |
| `EXPENSIVE_MODEL` / `CHEAP_MODEL` | Router cheap/expensive targets (LiteLLM aliases) | `gpt-5.6-sol-medium` / `gpt-5.6-luna-max` |
| `FUGU_MODEL` | TRINITY router backbone (Qwen3-0.6B) | `Qwen/Qwen3-0.6B` |
| `FUGU_VECTOR` | TRINITY SVF+head vector (built from `router_head.safetensors` at build time) | `/app/artifacts/model_iter_60.npy` in Docker; `artifacts/model_iter_60.npy` in native |
| `FUGU_HEAD` | Optional per-step head override | unset |
| `FUGU_WORKER_MODEL` / `FUGU_WORKER_MODELS` | Worker pool CSV for TRINITY/Conductor (LiteLLM aliases) | `gemini-3.6-flash-high,gpt-5.6-luna-max,gpt-5.6-sol-medium,deepseek-v4-flash-0731-xhigh,claude-opus-5-medium,claude-sonnet-5-medium,gemini-3.1-pro-preview-high` |
| `FUGU_LOCAL_MODELS` | Local HF worker models CSV (overrides LiteLLM pool) | unset |
| `FUGU_CONDUCTOR_MODEL` | Conductor planning model via LiteLLM. `gpt-5.6-luna-max` follows the three-list DAG format; `claude-opus-5-medium` tends to answer directly instead. | `gpt-5.6-luna-max` |
| `FUGU_LOCAL_CONDUCTOR` | HF id/path to a local Llama-3.2-3B Conductor. **Archived/experimental** — both the public base and retrained checkpoints failed format validation in this repo. | unset |
| `FUGU_CONDUCTOR_DEVICE` | Device for local Conductor (`cpu`, `mps`, `cuda:0`) | auto-detected; `cpu` fallback |
| `FUGU_CONDUCTOR_DTYPE` | Torch dtype for local Conductor | `bfloat16` on mps/cuda, `float32` on cpu |
| `FUGU_CONDUCTOR_MAX_NEW` | Max new tokens for local Conductor | `512` |
| `FUGU_MAX_TURNS` | TRINITY loop limit | `5` |
| `MANTIS_WORKER_TIMEOUT` / `FUGU_WORKER_TIMEOUT` | LiteLLM worker completion call timeout in seconds | `240` |
| `FUGU_AUTO_THRESHOLD` | Pi `/fugu auto` gate (score >= threshold -> conductor) | `6` |
| `MANTIS_CONTEXT_WINDOW` / `FUGU_CONTEXT_WINDOW` | Advertised context window for all provider modes in tokens. Operators should set this to the smallest effective worker window leaving output reserve intact. | `256000` |

### Example `.zshrc` snippet

```zsh
export OPENROUTER_API_KEY="sk-or-v1-..."
export OPENAI_API_KEY="sk-..."            # only for llm-router embeddings
export HF_TOKEN="hf-..."                  # optional, helps avoid HF rate limits

# 7-slot worker pool: must match aliases in configs/litellm.yaml
export FUGU_WORKER_MODELS="gemini-3.6-flash-high,gpt-5.6-luna-max,gpt-5.6-sol-medium,deepseek-v4-flash-0731-xhigh,claude-opus-5-medium,claude-sonnet-5-medium,gemini-3.1-pro-preview-high"

# Optional: swap deepseek/glm to the OpenCode Go endpoint by setting OPENCODE_GO_API_KEY
# export FUGU_WORKER_MODELS="gemini-3.6-flash-high,gpt-5.6-luna-max,gpt-5.6-sol-medium,opencode-deepseek-v4-flash,claude-opus-5-medium,claude-sonnet-5-medium,gemini-3.1-pro-preview-high"

# Local 3B Conductor (archived/experimental; both base and retrained checkpoints
# failed DAG-format validation in this repo — see eval/conductor-500-diagnosis.md)
# export FUGU_LOCAL_CONDUCTOR="di-zhang-fdu/openfugu-conductor-3b"
```

## Artifacts: where the trained TRINITY/Conductor models live

The repo does not store large model binaries in Git. The only tracked model artifact
is the small baseline `artifacts/router_head.safetensors` (~44 KB). The full
`model_iter_60.npy` TRINITY vector is built from it at Docker/build time by
`scripts/make_vec.py`.

| Artifact | Local path | S3 path |
|---|---|---|
| TRINITY `router_head.safetensors` (baseline, tracked) | `artifacts/router_head.safetensors` | `s3://sid-llm-runs/retrain-fugu-router/retrain-router-20260801_214418/router_head.safetensors` |
| TRINITY `model_iter_60.npy` (generated) | `artifacts/model_iter_60.npy` | `s3://sid-llm-runs/retrain-fugu-router/retrain-router-20260801_214418/model_iter_60.npy` |
| TRINITY `router_head.npy` (generated/downloaded) | `artifacts/router_head.npy` | `s3://sid-llm-runs/retrain-fugu-router/retrain-router-20260801_214418/router_head.npy` |
| Conductor checkpoint | `outputs/conductor_retrain/retrain-conductor-20260802_003213/checkpoint/` | `s3://sid-llm-runs/retrain-fugu-conductor/retrain-conductor-20260802_003213/checkpoint/` |

### Download and use them

```bash
# One-shot download (or fall back to building the baseline vector locally)
./scripts/download_artifacts.sh

# Or manually from S3:
mkdir -p artifacts
aws s3 cp s3://sid-llm-runs/retrain-fugu-router/retrain-router-20260801_214418/router_head.safetensors artifacts/
python3 scripts/make_vec.py

# Conductor checkpoint (~34 GB)
mkdir -p outputs/conductor_retrain/retrain-conductor-20260802_003213
aws s3 sync s3://sid-llm-runs/retrain-fugu-conductor/retrain-conductor-20260802_003213/checkpoint/ \
            outputs/conductor_retrain/retrain-conductor-20260802_003213/checkpoint/
```

Set the env vars before `docker compose up` or `run_mantis_native.sh`:

```bash
export FUGU_VECTOR="/app/artifacts/model_iter_60.npy"        # Docker path
export FUGU_HEAD="/app/artifacts/router_head.safetensors"    # optional override
# To use the retrained Conductor instead of the base HF checkpoint:
export FUGU_LOCAL_CONDUCTOR="/app/outputs/conductor_retrain/retrain-conductor-20260802_003213/checkpoint"
```

In the `openfugu` container, `/app` is the repo root, so the paths above map to the local files if you copy them into `artifacts/` / `outputs/` before `docker compose build` (or use a Docker bind/volume). In native mode, `run_mantis_native.sh` automatically rewrites the Docker `/app/...` paths to repo-local paths if they do not exist.

## Eval results

### TRINITY vs direct

On 16 realistic coding-agent prompts (`eval/fixtures.jsonl`):

| config | auto_score_mean | latency_mean_s | cost_sum_usd |
|---|---|---|---|
| direct | 0.568 | 7.8 | $0.0128 |
| trinity | 0.890 | 63.9 | $0.4563 |

TRINITY beats direct by **+0.322 (+56.6%)** on the auto-score rubric, at roughly **35× the cost and 8× the latency**. This matches the TRINITY retrain validation signal: the budgeted router label averaged **0.0437** reward per task vs an average of **0.0170** across all workers individually, and the head matched the budgeted gold label **77.8%** of the time on the held-out validation set.

### Conductor re-eval (native CPU, no one-shot example)

A second run using native `openfugu-patch/serve.py` (no Docker openfugu) with `FUGU_CONDUCTOR_DEVICE=cpu` and an assistant-prefill `Plan:\n` prompt:

| config | auto_score_mean | latency_mean_s | cost_sum_usd |
|---|---|---|---|
| conductor-old | 0.188 | 94.0 | $0.2648 |
| conductor-new | 0.073 | 95.7 | $0.2298 |

Both local Conductor checkpoints still fail to parse/execute on most prompts under CPU/float32 serving (13/16 for old, 13/16 for new). The retrained checkpoint is not yet a net win over the base public checkpoint. Native GPU (MPS/CUDA) with `bfloat16` may change latency and success rate, but the parse/execution failures are primarily model-output issues, not memory.

### Conductor with LiteLLM planner (`gpt-5.6-luna-max`)

A third run using the hosted `gpt-5.6-luna-max` as the Conductor planner (no local 3B checkpoint, no Docker openfugu rebuild needed):

| config | auto_score_mean | latency_sum_s | cost_sum_usd |
|---|---|---|---|
| direct | 0.568 | 125.5 | $0.0128 |
| trinity | 0.890 | 1023.0 | $0.4563 |
| conductor-luna | 0.626 | 1331.2 | $0.8628 |

Conductor-luna had a **0% HTTP failure rate**, but its overall auto-score (**0.626**) was well below TRINITY (**0.890**) and its hard-tier score (**0.250**) was far below TRINITY's **0.714**. It was also roughly **2× the cost** of TRINITY. Per the decision rule in `eval/report-luna-conductor.md`, `FUGU_AUTO_THRESHOLD` stays at **6** — the Supra router never emits a score that high in practice, so `/fugu auto` remains on TRINITY/direct. Users can still invoke `/fugu conductor` manually for experimentation.

## Zero-touch router learning

Set one flag, then use Trinity normally:

```bash
# .env
MANTIS_LEARNING=1
```

Mantis writes one append-only `runs-<hostname>.jsonl` file per machine under
`~/.local/share/mantis/learning` (a persistent Docker volume in all-Docker mode).
Records contain redacted task text, route/model ids, test/verifier outcomes, and timing;
tool outputs are not stored. Task text itself may contain sensitive project details, so review it
before sharing the directory outside your team. A high-confidence pseudo-label requires
a verifier acceptance and a final successful recognized test command. Ambiguous runs are kept
for diagnostics but never used for training.

The native and Docker launchers automatically run the local trainer. After 50 distinct
high-confidence tasks it trains a candidate, evaluates a hash-separated held-out split, and
promotes only when worker-label accuracy improves by at least 0.02. A promoted router is used
on the next Mantis restart. Change the thresholds with `MANTIS_LEARNING_MIN_RUNS` and
`MANTIS_LEARNING_MIN_IMPROVEMENT`.

For a team, point `MANTIS_LEARNING_DIR` at the same access-controlled synced directory in
native mode, or set `MANTIS_LEARNING_HOST_DIR` to that directory in Docker. Per-host filenames avoid append collisions; each trainer deduplicates tasks
before training. Only share this directory with colleagues allowed to see task descriptions.
Learning is off by default because task text may still be sensitive after secret redaction.

Run a one-off status/training check with:

```bash
python3 scripts/learn_router.py --promote
```

## Retraining the router head

The included TRINITY router head was trained on the current 7-slot pool defined in `configs/litellm.yaml`:

1. `gemini-3.6-flash-high`
2. `gpt-5.6-luna-max`
3. `gpt-5.6-sol-medium`
4. `deepseek-v4-flash-0731-xhigh`
5. `claude-opus-5-medium`
6. `claude-sonnet-5-medium`
7. `gemini-3.1-pro-preview-high`

To retrain it on a new pool, launch a SkyPilot job:

```bash
export OPENROUTER_API_KEY="sk-or-v1-..."
export HF_TOKEN="hf-..."
./scripts/sky_launch_retrain_router.sh --dry-run
```

Each retraining entry can append `|reasoning_effort` (e.g. `openai/gpt-5.6-luna|max`) so the labels match the runtime LiteLLM aliases. The script:

1. Loads `nvidia/ToolScale` or `s3://external-datasets-archive/terminal-bench-2.1/` tasks.
2. Calls each worker in the pool through OpenRouter and scores each response.
3. Extracts Qwen3-0.6B hidden states.
4. Fine-tunes the 10×1024 TRINITY head (worker + role logits) with L2 regularization toward the original head.
5. Writes `model_iter_60.npy` and `router_head.npy` to the S3 mount at `s3://sid-llm-runs/retrain-fugu-router/<timestamp>/`.

After you approve the shortlist and cost estimate, run the same command without `--dry-run`.

### Cost-aware router labels

By default the retrain picks the highest-scoring worker per task (`--label-mode quality`). For a cost-quality Pareto objective, use `--label-mode cost` and a `configs/worker-costs.json` table (OpenRouter prompt/completion prices, 2K-in/1K-out estimates). In cost mode the gold worker is `argmax(score_i / cost_i)`, so equal scores resolve to the cheaper model while a much better worker can still win despite a higher price.

## Retraining for a new pool (cheap, two commands)

When you change the worker pool, run both retrains. The router head is cheap; the Conductor acceptance smoke is also cheap and proves the real 3B checkpoint can retrain on the new pool. Skip the full Conductor run until acceptance passes.

```bash
# 1. Update aliases, cost table, and SkyPilot YAML pool defaults.
#    --fetch-costs requires OPENROUTER_API_KEY and contacts the OpenRouter pricing API.
python3 scripts/update_pool.py \
  --pool "google/gemini-3.6-flash|high,openai/gpt-5.6-luna|max,openai/gpt-5.6-sol|medium,deepseek/deepseek-v4-flash-0731|xhigh,anthropic/claude-opus-5|medium,anthropic/claude-sonnet-5|medium,google/gemini-3.1-pro-preview|high" \
  --fetch-costs \
  --update-yamls \
  --env-file .env

# 2. Retrain the TRINITY router head (cheap, ~$5-15 incl. worker API calls).
./scripts/sky_launch_retrain_router.sh

# 3. One-step acceptance test for the real 3B Conductor (cheap, ~$0.15 GCP L4 spot).
#    The run needs AWS keys because the TerminalBench mirror is in S3.
export OPENROUTER_API_KEY="sk-or-v1-..."
export HF_TOKEN="hf-..."
export AWS_ACCESS_KEY_ID="..."
export AWS_SECRET_ACCESS_KEY="..."
export AWS_DEFAULT_REGION="us-east-1"
sky launch -y --detach-run \
  --env OPENROUTER_API_KEY --env HF_TOKEN \
  --env AWS_ACCESS_KEY_ID --env AWS_SECRET_ACCESS_KEY --env AWS_DEFAULT_REGION \
  launch/sky/retrain_fugu_conductor_real_3b_smoke_gcp.yaml

# 4. (Optional, expensive) Full Conductor GRPO run on A100 spot.
# sky launch -y --env OPENROUTER_API_KEY --env HF_TOKEN \
#   launch/sky/retrain_fugu_conductor.yaml
```

`update_pool.py` edits `configs/litellm.yaml` (LiteLLM aliases for runtime), `configs/worker-costs.json` (used by `--label-mode cost`), the `RETRAIN_WORKER_MODELS` default in the three SkyPilot YAMLs, and optionally your `.env` file. If you skip `--fetch-costs`, the cost table is left untouched and you must add missing entries before using `--label-mode cost`.

## Retraining the Conductor on a new pool

The Conductor is a GRPO-fine-tuned `Llama-3.2-3B-Instruct` policy (checkpoint `di-zhang-fdu/openfugu-conductor-3b`) that writes a workflow DAG (`model_id`, `subtasks`, `access_list`) over the 7-slot worker pool. The base checkpoint was trained on an older pool, so it should be retrained whenever the pool changes.

**Rule:** retrain the TRINITY head first (cheap, minutes on an L4), evaluate the new `model_iter_60.npy`, and only retrain the Conductor if the router retrain shows routing gains. The Conductor is far more expensive because each rollout executes the generated DAG against live workers.

### Smoke test (20 steps, 8 tasks)

The fastest way to validate the pipeline on a CPU-only machine is a local
script run with a small instruction-tuned proxy. `di-zhang-fdu/openfugu-conductor-3b`
does not fit on CPU, so this smoke uses `HuggingFaceTB/SmolLM2-135M-Instruct` as
a stand-in base; it still exercises the same prompt template, DAG parser,
reward functions, and GRPO loop:

```bash
export OPENROUTER_API_KEY="sk-or-v1-..."
export HF_TOKEN="hf-..."
python3 scripts/retrain_conductor.py \
  --pool "google/gemini-3.6-flash|high,openai/gpt-5.6-luna|max,openai/gpt-5.6-sol|medium,deepseek/deepseek-v4-flash-0731|xhigh,anthropic/claude-opus-5|medium,anthropic/claude-sonnet-5|medium,google/gemini-3.1-pro-preview|high" \
  --dataset s3://external-datasets-archive/terminal-bench-2.1/ \
  --base HuggingFaceTB/SmolLM2-135M-Instruct \
  --steps 20 --limit 8 \
  --num-generations 2 --per-device-batch 2 \
  --max-completion-length 128 \
  --temperature 0.7 \
  --mock-worker \
  --output-dir outputs/conductor_retrain/$(date +%Y%m%d-%H%M%S)
```

On GPU, launch the SkyPilot YAML with the real base model:

```bash
export OPENROUTER_API_KEY="sk-or-v1-..."
export HF_TOKEN="hf-..."
sky launch --env OPENROUTER_API_KEY --env HF_TOKEN \
  launch/sky/retrain_fugu_conductor.yaml
```

### Real 3B checkpoint one-step acceptance smoke

Before a full retrain, validate that `di-zhang-fdu/openfugu-conductor-3b` loads,
trains for one GRPO step, saves a checkpoint, reloads it, and emits a parseable
Conductor DAG. The smoke fits on a single L4/A10G (24 GB) by using 8-bit AdamW and
a short `max_completion_length`:

```bash
# GCP: authenticate and point GOOGLE_APPLICATION_CREDENTIALS at a file.
# AWS keys are only needed because the TerminalBench mirror lives in S3.
export OPENROUTER_API_KEY="sk-or-v1-..."
export HF_TOKEN="hf-..."
export AWS_ACCESS_KEY_ID="..."
export AWS_SECRET_ACCESS_KEY="..."
export AWS_DEFAULT_REGION="us-east-1"

# Try spot first (cheapest).
sky launch -y --detach-run \
  --env OPENROUTER_API_KEY --env HF_TOKEN \
  --env AWS_ACCESS_KEY_ID --env AWS_SECRET_ACCESS_KEY --env AWS_DEFAULT_REGION \
  launch/sky/retrain_fugu_conductor_real_3b_smoke_gcp.yaml

# If spot is unavailable/exhausted, use on-demand:
sky launch -y --no-use-spot --detach-run \
  --env OPENROUTER_API_KEY --env HF_TOKEN \
  --env AWS_ACCESS_KEY_ID --env AWS_SECRET_ACCESS_KEY --env AWS_DEFAULT_REGION \
  launch/sky/retrain_fugu_conductor_real_3b_smoke_gcp.yaml
```

`--real-checkpoint-smoke` aborts immediately on CPU/MPS, never substitutes a
smaller model, and writes `manifest.json` with `valid_for_runtime=true` only when
the real 3B checkpoint runs on GPU and the acceptance generation is parseable.
A real-3B acceptance smoke on a GCP L4 spot completed successfully (see
`runs/reports/20260801-real-3b-gcp.md`).

`launch/sky/retrain_fugu_conductor.yaml` defaults to a smoke:
- `RETRAIN_STEPS=20`, `RETRAIN_LIMIT=8`, `RETRAIN_NUM_GENERATIONS=2`, `RETRAIN_PER_DEVICE_BATCH=2`
- `FUGU_WORKER_TIMEOUT=30`, `FUGU_WORKER_MAX_TOKENS=256` to keep wall-clock/cost bounded
- 2x A100-80GB spot (minimum; use 4x for full runs)
- `FUGU_BASE_MODEL=di-zhang-fdu/openfugu-conductor-3b`

Outputs are saved to `s3://sid-llm-runs/retrain-fugu-conductor/<timestamp>/` with `pool.json`, `dataset.json`, `metrics.jsonl`, and the checkpoint.

### Full run

Update the same YAML (or pass `--steps` / `--limit` / `--num-generations`) and relaunch. A full run with 200–500 TerminalBench tasks, `num_generations=8`, and `per_device_batch=2` on 4x A100 is expected to take on the order of 1–6 wall-clock hours and cost roughly **$20–80 in GPU spot time + worker API calls**. Worker calls dominate; the smoke below averaged ~10 s/step on CPU with a 135 M parameter proxy, so a 3 B model on A100 with larger generation groups will be slower but still bounded by spot pricing.

### Quarterly-retrain precedent

When the worker pool changes:
1. Update `RETRAIN_WORKER_MODELS` in **both** `launch/sky/retrain_fugu_router.yaml` and `launch/sky/retrain_fugu_conductor.yaml` (or use `update_pool.py`).
2. Run the router retrain, evaluate the new head.
3. If routing improves, run the Conductor smoke (`--steps 20 --limit 8`).
4. If smoke metrics show parseable-workflow format reward stable/non-zero, launch the full Conductor retrain.

### License note

The base `meta-llama/Llama-3.2-3B-Instruct` checkpoint and the derived `di-zhang-fdu/openfugu-conductor-3b` adapter are subject to the **Llama 3.2 Community License**. Ensure your use complies before downloading or redistributing the trained checkpoint.

## Security / exposure model

`openfugu-patch/serve.py` binds `0.0.0.0:8088` by default and **requires a Bearer token** on `/v1/models` and `/v1/chat/completions`. The token is read from `FUGU_API_KEY` (or `LITELLM_KEY` for backward compatibility). `serve.py` refuses to start if neither is set, rejects requests with an incorrect or missing `Authorization` header (HTTP 401), and rejects request bodies larger than `FUGU_MAX_BODY_BYTES` (default 5 MiB, HTTP 413). Health endpoints (`/health`, `/`) remain public for Docker/container probes.

All inter-service traffic in the Docker stack uses the same `LITELLM_KEY` value. Treat `0.0.0.0:8088` as an internal service: do not expose it to untrusted networks without an additional reverse proxy/mTLS layer.

## Building and type checking the pi extension

The TypeScript extension is in `extensions/`:

```bash
cd extensions
npm install
npm run typecheck
```

See `extensions/README.md` for how to load `mantis.ts` into pi and smoke test `/mantis auto` (`/fugu` is an alias).

## Troubleshooting

### `serve.py` exits with "FATAL: set MANTIS_API_KEY"

The mantis orchestrator now refuses to start without a bearer token. Copy `.env.example` to `.env` and set both `LITELLM_KEY` and `MANTIS_API_KEY` to the same value, or just set `LITELLM_KEY` and use `MANTIS_API_KEY=${LITELLM_KEY}`. `scripts/verify.sh` and the pi extension also read `MANTIS_API_KEY` (with `FUGU_API_KEY` as a fallback).

### `verify.sh` fails with HTTP 401

The orchestrator requires an `Authorization: Bearer <token>` header. `scripts/verify.sh` sources `.env` and uses `MANTIS_API_KEY`/`FUGU_API_KEY`/`LITELLM_KEY`. Make sure the token you pass to `curl` matches the value set in the mantis container/process.

### Conductor returns HTTP 500 or empty response

Common causes:

1. **Docker Desktop memory limit (Mac)** — the Llama-3.2-3B Conductor checkpoint needs ~12 GB of RAM at `float32`. If Docker Desktop's VM is capped at ~7.7 GB the container may OOM during load or generation. Increase the VM memory limit or switch to **hybrid native-GPU mode** (`./scripts/run_mantis_native.sh`), which runs PyTorch directly on the host.
2. **Invalid Conductor DAG** — the local checkpoint sometimes emits workflows with self/forward references, unequal-length lists, or direct answers instead of the three required lists. This is a model-output issue. Native path with `bfloat16`/GPU and an assistant `Plan:\n` prefill helps, but a checkpoint that reliably emits valid DAGs is required. See `eval/conductor-500-diagnosis.md` for the exact errors observed.
3. **Conductor returns a plain answer instead of a DAG** — the planner model is not following the workflow format. If `FUGU_CONDUCTOR_MODEL=claude-opus-5-medium`, switch it to `gpt-5.6-luna-max`, which reliably emits the three-list `model_id / subtasks / access_list` structure. Do not set `FUGU_LOCAL_CONDUCTOR` unless you are testing the archived 3B checkpoints.
4. **Litellm proxy not reachable** — in hybrid mode, the native mantis process needs `MANTIS_BASE_URL=http://127.0.0.1:3001/v1` (set by `run_mantis_native.sh` automatically). Confirm `curl http://localhost:3001/health` responds.

### TRINITY is slow on first call

The Qwen3-0.6B router and any local worker models download from HuggingFace on first use and are cached in `hf-cache` (Docker) or `~/.cache/huggingface` (native). Subsequent calls are much faster.

### `verify.sh` fails on the mantis check

`scripts/verify.sh` curls `http://localhost:8088/health`. If you are running hybrid mode, make sure `run_mantis_native.sh` is still running. If you are running all-Docker, make sure `docker compose up -d` included the `openfugu` service.

## Deviation notes

- The upstream `trotsky1997/OpenFugu` `fetch_artifacts.py` cannot locate the `model_iter_60.npy` vector. `mantis` includes the public `router_head.safetensors` from `nshkrdotcom/trinity-coordinator-adapted-qwen3-0.6b` and `scripts/make_vec.py` builds `artifacts/model_iter_60.npy` from it (zero SVF offsets + real head).
- `configs/litellm.yaml` and `docker-compose.yml` route all backend LLM calls through OpenRouter. `llm-router` still consumes `OPENAI_API_KEY` only for its internal `mf` embedding scorer.
- `openfugu-patch/serve.py` wraps the OpenFugu `LiteLLMWorker` classes to pass `custom_llm_provider="openai"` so LiteLLM dispatches proxy aliases correctly.
- `serve.py` was patched to select the TRINITY vs Conductor coordinator from the request `model` field, lazy-load the requested coordinator on first use, optionally load a local `transformers`-based Conductor checkpoint, auto-detect `mps`/`cuda`/`cpu`, log device/dtype at startup, and add an assistant `Plan:\n` prefill to nudge local Conductor checkpoints into the required three-list format.

## References

Papers and checkpoints this repo relies on:

- **Fugu / Fugu-Ultra** — *Sakana Fugu Technical Report* (arXiv:2606.21228). Describes the Fugu/Fugu-Ultra orchestration stack, latency-vs-quality routing, and the TRINITY/Conductor split.
- **TRINITY** — Jinglue Xu et al., *"TRINITY: An Evolved LLM Coordinator"* (arXiv:2512.04695). Introduces the compact 0.6 B coordinator with a lightweight SVF+head that routes a pool of workers across Worker/Thinker/Verifier turns.
- **Conductor** — Stefan Nielsen et al., *"Learning to Orchestrate Agents in Natural Language with the Conductor"* (arXiv:2512.04388). Describes the RL-trained Conductor LM that writes multi-step agent workflows (`model_id`, `subtasks`, `access_list`).
- **Qwen3-0.6B** — `Qwen/Qwen3-0.6B`: TRINITY router backbone.
- **TRINITY adapted head** — `nshkrdotcom/trinity-coordinator-adapted-qwen3-0.6b`: provides `router_head.safetensors` used by `scripts/make_vec.py` to build `artifacts/model_iter_60.npy`.
- **OpenFugu Conductor** — `di-zhang-fdu/openfugu-conductor-3b`: the GRPO-fine-tuned Llama-3.2-3B-Instruct conductor checkpoint; set `FUGU_LOCAL_CONDUCTOR` to load it locally.
- **Llama-3.2-3B-Instruct** — `meta-llama/Llama-3.2-3B-Instruct`: base model for the Conductor checkpoint.
- **Supra complexity router** — `SupraLabs/Supra-Router-51M`: the 51 M-parameter complexity/scoring model used inside `llm-router`.
- **TerminalBench 2.1** — `zai-org/terminal-bench-2-verified`: the benchmark used for router retraining/evaluation.
- **ToolScale** — `nvidia/ToolScale`: the tool-call planning dataset used as an alternative retraining signal.
- **OpenAI embedding** — `text-embedding-3-small`: used by `llm-router`’s `mf` scorer; requires `OPENAI_API_KEY`.
