# fugu-local

Local orchestration stack wiring:
- LiteLLM proxy (:3001)
- RouteLLM router with Supra complexity header (:5500)
- OpenFugu coordinator (:8088) with TRINITY and Conductor modes
- Pi `/fugu` mode-switching extension

## Quickstart

```bash
cp .env.example .env          # add OPENROUTER_API_KEY and OPENAI_API_KEY; optional HF_TOKEN
docker compose build
docker compose up -d
./scripts/verify.sh
```

The backend LLM calls are routed through **OpenRouter** via the LiteLLM proxy. `llm-router`'s internal `mf` scorer still needs an OpenAI key for `text-embedding-3-small`; `OPENAI_API_KEY` is only used for that embedding call.

The Pi extension logs every routing decision to `~/.config/fugu/routing-log.jsonl` (one JSON line per turn with `score`, `coordinator`, and a `task_prefix`), and the OpenFugu response `usage` object now includes `fugu_trace` — a compact string such as `Worker(4)→Thinker(1)→Verifier(1):verifier_accept` for TRINITY or `steps:5:conductor` for Conductor.

## Model knobs (set in `.env` or your shell, e.g. `.zshrc`)

All model selection is env-driven. Export the variables before `docker compose up` (or put them in `.zshrc`/`.bashrc` and run `set -a; source <file>; set +a` before compose).

| Variable | What it controls | Default |
|---|---|---|
| `OPENROUTER_API_KEY` | API key LiteLLM uses to call OpenRouter | required |
| `OPENCODE_GO_API_KEY` | API key for optional opencode-* LiteLLM aliases | optional |
| `OPENAI_API_KEY` | OpenAI key for `llm-router` embeddings only | required for router |
| `LITELLM_KEY` | Internal bearer token for router/openfugu | `sk-fugu-local` |
| `EXPENSIVE_MODEL` / `CHEAP_MODEL` | Router cheap/expensive targets (LiteLLM aliases) | `claude-opus-5` / `gpt-5.6-luna-max` |
| `FUGU_MODEL` | TRINITY router backbone (Qwen3-0.6B) | `Qwen/Qwen3-0.6B` |
| `FUGU_VECTOR` | TRINITY SVF+head vector | `/app/artifacts/model_iter_60.npy` |
| `FUGU_HEAD` | Optional per-step head override | unset |
| `FUGU_WORKER_MODEL` / `FUGU_WORKER_MODELS` | Worker pool CSV for TRINITY/Conductor (LiteLLM aliases) | `claude-sonnet-5,claude-opus-5,gpt-5.6-sol-medium,gpt-5.6-luna-max,gpt-5.6-terra-xhigh,deepseek-v4-flash,glm-5.2` |
| `FUGU_LOCAL_MODELS` | Local HF worker models CSV (overrides LiteLLM pool) | unset |
| `FUGU_CONDUCTOR_MODEL` | Conductor planning model via LiteLLM | `claude-opus-5` |
| `FUGU_LOCAL_CONDUCTOR` | HF id/path to load a local Conductor (e.g. `di-zhang-fdu/openfugu-conductor-3b`) | unset |
| `FUGU_CONDUCTOR_DEVICE` | Device for local Conductor (`cpu`, `mps`, `cuda:0`) | `cpu` |
| `FUGU_CONDUCTOR_DTYPE` | Torch dtype for local Conductor | `float32` |
| `FUGU_CONDUCTOR_MAX_NEW` | Max new tokens for local Conductor | `512` |
| `FUGU_MAX_TURNS` | TRINITY loop limit | `5` |
| `FUGU_AUTO_THRESHOLD` | Pi `/fugu auto` gate (score >= threshold -> conductor) | `4` |

### Example `.zshrc` snippet

```zsh
export OPENROUTER_API_KEY="sk-or-v1-..."
export OPENAI_API_KEY="sk-..."            # only for llm-router embeddings
export HF_TOKEN="hf-..."                  # optional, helps avoid HF rate limits

# 7-slot worker pool: must match aliases in configs/litellm.yaml
export FUGU_WORKER_MODELS="claude-sonnet-5,claude-opus-5,gpt-5.6-sol-medium,gpt-5.6-luna-max,gpt-5.6-terra-xhigh,deepseek-v4-flash,glm-5.2"

# Optional: swap deepseek/glm to the OpenCode Go endpoint by setting OPENCODE_GO_API_KEY
# export FUGU_WORKER_MODELS="claude-sonnet-5,claude-opus-5,gpt-5.6-sol-medium,gpt-5.6-luna-max,gpt-5.6-terra-xhigh,opencode-deepseek-v4-flash,opencode-glm-5.2"

# Use the real OpenFugu Llama-3.2-3B Conductor inside Docker (CPU)
# export FUGU_LOCAL_CONDUCTOR="di-zhang-fdu/openfugu-conductor-3b"
```

## Retraining the router head on a new model pool

The included TRINITY router head was trained on an older 7-slot pool. To retrain it on the current Pareto-frontier pool, launch a SkyPilot job:

```bash
export OPENROUTER_API_KEY="sk-or-v1-..."
export HF_TOKEN="hf-..."
./scripts/sky_launch_retrain_router.sh --dry-run
```

The default pool in `launch/sky/retrain_fugu_router.yaml` is the 7 requested frontier models: Anthropic Sonnet/Opus 5 (medium thinking), GPT-5.6 Sol/Luna/Terra with graded reasoning effort, and low-cost DeepSeek V4 Flash / GLM-5.2. Each retraining entry can append `|reasoning_effort` (e.g. `openai/gpt-5.6-terra|xhigh`) so the labels match the runtime LiteLLM aliases. The script:

1. Loads `nvidia/ToolScale` or `s3://external-datasets-archive/terminal-bench-2.1/` tasks.
2. Calls each worker in the pool through OpenRouter and scores each response.
3. Extracts Qwen3-0.6B hidden states.
4. Fine-tunes the 10x1024 TRINITY head (worker + role logits) with L2 regularization toward the original head.
5. Writes `model_iter_60.npy` and `router_head.npy` to the S3 mount at `s3://sid-llm-runs/retrain-fugu-router/<timestamp>/`.

After you approve the shortlist and cost estimate, run the same command without `--dry-run`.

### Cost-aware router labels

By default the retrain picks the highest-scoring worker per task (`--label-mode quality`). For a cost-quality Pareto objective, use `--label-mode cost` and a `configs/worker-costs.json` table (OpenRouter prompt/completion prices, 2K-in/1K-out estimates). In cost mode the gold worker is `argmax(score_i / cost_i)`, so equal scores resolve to the cheaper model while a much better worker can still win despite a higher price.

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
  --pool "anthropic/claude-sonnet-5|medium,anthropic/claude-opus-5|medium,openai/gpt-5.6-sol|medium,openai/gpt-5.6-luna|max,openai/gpt-5.6-terra|xhigh,deepseek/deepseek-v4-flash|none,z-ai/glm-5.2|none" \
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
1. Update `RETRAIN_WORKER_MODELS` in **both** `launch/sky/retrain_fugu_router.yaml` and `launch/sky/retrain_fugu_conductor.yaml`.
2. Run the router retrain, evaluate the new head.
3. If routing improves, run the Conductor smoke (`--steps 20 --limit 8`).
4. If smoke metrics show parseable-workflow format reward stable/non-zero, launch the full Conductor retrain.

### License note

The base `meta-llama/Llama-3.2-3B-Instruct` checkpoint and the derived `di-zhang-fdu/openfugu-conductor-3b` adapter are subject to the **Llama 3.2 Community License**. Ensure your use complies before downloading or redistributing the trained checkpoint.

## Mac M5 2025 / Apple Silicon notes

- Docker Desktop for Mac does **not** expose MPS or Metal to Linux containers, so PyTorch runs on CPU inside the `openfugu` container. The default `FUGU_CONDUCTOR_DEVICE=cpu` and `FUGU_CONDUCTOR_DTYPE=float32` are correct.
- The Qwen3-0.6B router (~1.5 GB) and Llama-3.2-3B Conductor (~6–7 GB) will download on first run and be cached in the `hf-cache` Docker volume. Give Docker enough memory (>=10 GB recommended if using the 3B Conductor).
- If you want to use `mps` or Metal, run `openfugu/serve.py` natively outside Docker with `FUGU_CONDUCTOR_DEVICE=mps` and the rest of the stack still in Docker.

## Deviation notes

- The upstream `trotsky1997/OpenFugu` `fetch_artifacts.py` cannot locate the `model_iter_60.npy` vector. `fugu-local` includes the public `router_head.safetensors` from `nshkrdotcom/trinity-coordinator-adapted-qwen3-0.6b` and `scripts/make_vec.py` builds `artifacts/model_iter_60.npy` from it (zero SVF offsets + real head).
- `configs/litellm.yaml` and `docker-compose.yml` route all backend LLM calls through OpenRouter. `llm-router` still consumes `OPENAI_API_KEY` only for its internal `mf` embedding scorer.
- `openfugu-patch/serve.py` wraps the OpenFugu `LiteLLMWorker` classes to pass `custom_llm_provider="openai"` so LiteLLM dispatches proxy aliases correctly.
- `serve.py` was patched to select the TRINITY vs Conductor coordinator from the request `model` field, lazy-load the requested coordinator on first use, optionally load a local `transformers`-based Conductor checkpoint, and log each request's routed model/coordinator.

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
