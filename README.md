# fugu-local

Local orchestration stack wiring:
- LiteLLM proxy (:3001)
- RouteLLM router with Supra complexity header (:5500)
- OpenFugu coordinator (:8088) with TRINITY and Conductor modes
- Pi `/fugu` mode-switching extension

## Quickstart

```bash
cp .env.example .env          # add a real OPENAI_API_KEY and optional HF_TOKEN
docker compose build
docker compose up -d
./scripts/verify.sh
```

## Model knobs (set in `.env` or your shell, e.g. `.zshrc`)

All model selection is env-driven. Export the variables before `docker compose up` (or put them in `.zshrc`/`.bashrc` and run `set -a; source <file>; set +a` before compose).

| Variable | What it controls | Default |
|---|---|---|
| `OPENAI_API_KEY` | API key the LiteLLM proxy uses for workers | required |
| `LITELLM_KEY` | Internal bearer token for router/openfugu | `sk-fugu-local` |
| `EXPENSIVE_MODEL` / `CHEAP_MODEL` | Router cheap/expensive targets | `gpt-4o` / `gpt-4o-mini` |
| `FUGU_MODEL` | TRINITY router backbone (Qwen3-0.6B) | `Qwen/Qwen3-0.6B` |
| `FUGU_VECTOR` | TRINITY SVF+head vector | `/app/artifacts/model_iter_60.npy` |
| `FUGU_HEAD` | Optional per-step head override | unset |
| `FUGU_WORKER_MODEL` / `FUGU_WORKER_MODELS` | Worker pool CSV for TRINITY/Conductor | `openai/gpt-4o-mini` |
| `FUGU_LOCAL_MODELS` | Local HF worker models CSV (overrides LiteLLM pool) | unset |
| `FUGU_CONDUCTOR_MODEL` | Conductor planning model via LiteLLM | `openai/gpt-4o-mini` |
| `FUGU_LOCAL_CONDUCTOR` | HF id/path to load a local Conductor (e.g. `di-zhang-fdu/openfugu-conductor-3b`) | unset |
| `FUGU_CONDUCTOR_DEVICE` | Device for local Conductor (`cpu`, `mps`, `cuda:0`) | `cpu` |
| `FUGU_CONDUCTOR_DTYPE` | Torch dtype for local Conductor | `float32` |
| `FUGU_CONDUCTOR_MAX_NEW` | Max new tokens for local Conductor | `512` |
| `FUGU_MAX_TURNS` | TRINITY loop limit | `5` |
| `FUGU_AUTO_THRESHOLD` | Pi `/fugu auto` gate (score >= threshold -> conductor) | `4` |

### Example `.zshrc` snippet

```zsh
export OPENAI_API_KEY="sk-..."
export HF_TOKEN="hf-..."  # optional, helps avoid HF rate limits

# Use the real OpenFugu Llama-3.2-3B Conductor inside Docker (CPU)
export FUGU_LOCAL_CONDUCTOR="di-zhang-fdu/openfugu-conductor-3b"
export FUGU_CONDUCTOR_DEVICE="cpu"
export FUGU_CONDUCTOR_DTYPE="float32"

# Worker pool: first slot used for quick TRINITY/Conductor steps when no CSV is given
export FUGU_WORKER_MODELS="openai/gpt-4o-mini"
```

## Mac M5 2025 / Apple Silicon notes

- Docker Desktop for Mac does **not** expose MPS or Metal to Linux containers, so PyTorch runs on CPU inside the `openfugu` container. The default `FUGU_CONDUCTOR_DEVICE=cpu` and `FUGU_CONDUCTOR_DTYPE=float32` are correct.
- The Qwen3-0.6B router (~1.5 GB) and Llama-3.2-3B Conductor (~6–7 GB) will download on first run and be cached in the `hf-cache` Docker volume. Give Docker enough memory (>=10 GB recommended if using the 3B Conductor).
- If you want to use `mps` or Metal, run `openfugu/serve.py` natively outside Docker with `FUGU_CONDUCTOR_DEVICE=mps` and the rest of the stack still in Docker.

## Deviation notes

- The upstream `trotsky1997/OpenFugu` `fetch_artifacts.py` cannot locate the `model_iter_60.npy` vector. `fugu-local` includes the public `router_head.safetensors` from `nshkrdotcom/trinity-coordinator-adapted-qwen3-0.6b` and `scripts/make_vec.py` builds `artifacts/model_iter_60.npy` from it (zero SVF offsets + real head).
- `litellm` and `llm-router` required small env/auth alignment tweaks in `docker-compose.yml` and `configs/litellm.yaml`.
- `serve.py` was patched to select the TRINITY vs Conductor coordinator from the request `model` field, lazy-load the requested coordinator on first use, optionally load a local `transformers`-based Conductor checkpoint, and log each request's routed model/coordinator.
