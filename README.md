# fugu-local

Local orchestration stack wiring:
- LiteLLM proxy (:3001)
- RouteLLM router with Supra complexity header (:5500)
- OpenFugu coordinator (:8088)
- Pi `/fugu` mode-switching extension

## Quickstart

```bash
cp .env.example .env          # add a real OPENAI_API_KEY
docker compose build
docker compose up -d
./scripts/verify.sh
```

## Deviation notes

- The upstream `trotsky1997/OpenFugu` `fetch_artifacts.py` cannot locate the `model_iter_60.npy` vector. The `openfugu` image was bootstrapped with `scripts/make_vec.py` using the public `router_head.safetensors` from `nshkrdotcom/trinity-coordinator-adapted-qwen3-0.6b` plus zero SVF offsets.
- `litellm` and `llm-router` required small env/auth alignment tweaks in `docker-compose.yml` and `configs/litellm.yaml` (see diff).
- `serve.py` was patched to select the TRINITY vs Conductor coordinator from the request `model` field and lazy-load the requested coordinator on first use.
