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

1. Loads `nvidia/ToolScale` tasks.
2. Calls each worker in the pool through OpenRouter and scores each response against the expected tool-call plan.
3. Extracts Qwen3-0.6B hidden states.
4. Fine-tunes the 10x1024 TRINITY head (worker + role logits) with L2 regularization toward the original head.
5. Writes `model_iter_60.npy` and `router_head.npy` to the S3 mount at `s3://sid-llm-runs/retrain-fugu-router/<timestamp>/`.

After you approve the shortlist and cost estimate, run the same command without `--dry-run`.

## Mac M5 2025 / Apple Silicon notes

- Docker Desktop for Mac does **not** expose MPS or Metal to Linux containers, so PyTorch runs on CPU inside the `openfugu` container. The default `FUGU_CONDUCTOR_DEVICE=cpu` and `FUGU_CONDUCTOR_DTYPE=float32` are correct.
- The Qwen3-0.6B router (~1.5 GB) and Llama-3.2-3B Conductor (~6–7 GB) will download on first run and be cached in the `hf-cache` Docker volume. Give Docker enough memory (>=10 GB recommended if using the 3B Conductor).
- If you want to use `mps` or Metal, run `openfugu/serve.py` natively outside Docker with `FUGU_CONDUCTOR_DEVICE=mps` and the rest of the stack still in Docker.

## Deviation notes

- The upstream `trotsky1997/OpenFugu` `fetch_artifacts.py` cannot locate the `model_iter_60.npy` vector. `fugu-local` includes the public `router_head.safetensors` from `nshkrdotcom/trinity-coordinator-adapted-qwen3-0.6b` and `scripts/make_vec.py` builds `artifacts/model_iter_60.npy` from it (zero SVF offsets + real head).
- `configs/litellm.yaml` and `docker-compose.yml` route all backend LLM calls through OpenRouter. `llm-router` still consumes `OPENAI_API_KEY` only for its internal `mf` embedding scorer.
- `openfugu-patch/serve.py` wraps the OpenFugu `LiteLLMWorker` classes to pass `custom_llm_provider="openai"` so LiteLLM dispatches proxy aliases correctly.
- `serve.py` was patched to select the TRINITY vs Conductor coordinator from the request `model` field, lazy-load the requested coordinator on first use, optionally load a local `transformers`-based Conductor checkpoint, and log each request's routed model/coordinator.
