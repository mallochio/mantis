# llm-router

OpenAI-compatible proxy that routes each request to a cheap or expensive model
based on prompt complexity, using an ensemble of two routers:

1. **RouteLLM MF** — matrix factorization scorer trained on Chatbot Arena
   preference data. Returns a 0-1 "strong model win rate" estimate. Uses
   OpenAI `text-embedding-3-small` for embeddings (~$0.02/1M tokens).
2. **Supra-Router-51M** — a 51M-param micro-LLM that classifies prompt
   complexity on a 1-5 scale. Runs locally on CPU (~400ms, no API cost).

A request goes to the **expensive** model if:

```
MF score >= threshold  OR  Supra complexity >= supra_threshold
```

This ensemble catches hard prompts (distributed systems, proofs, infrastructure
debugging) that the MF scorer alone underweights, while avoiding Supra-Router's
tendency to route all code queries as expensive.

## Files

- `server.py` — FastAPI server exposing `/v1/chat/completions` (OpenAI-compatible)
- `llm-router.sh` — launch script (proxy start-if-down, router stop-and-restart
  if running, detach + print status). Designed for StartupFolder or manual use.
- `test_server_helpers.py` — unit tests for message normalization and Supra output parsing
- `_test_scores.sh` — smoke test that routes sample prompts and prints scores/decisions

## Setup

```bash
# Python deps
python -m venv .venv
. .venv/bin/activate
pip install routellm fastapi uvicorn httpx transformers torch

# Env (e.g. in ~/.zshrc)
export OPENAI_API_KEY="sk-..."           # for MF embeddings
export CHEAP_BASE="https://your-cheap-endpoint/v1"
export CHEAP_KEY="your-key"
export CHEAP_MODEL="deepseek-v4-flash"
export EXPENSIVE_BASE="https://your-expensive-endpoint/v1"
export EXPENSIVE_KEY="your-key"
export EXPENSIVE_MODEL="glm-5.2"
export ROUTELLM_THRESHOLD="0.156"        # MF score >= this -> expensive
export ROUTELLM_USE_SUPRA="1"            # 1=ensemble, 0=MF only
export ROUTELLM_SUPRA_THRESHOLD="3"      # Supra complexity >= this -> expensive
export ROUTELLM_KEY="sk-route-local"     # bearer token clients must present
```

## Run

```bash
./llm-router.sh
# proxy already running on :41437
# router running pid 12345 on :5500
```

Clients connect to `http://127.0.0.1:5500/v1` with `Authorization: Bearer sk-route-local`
and use model `auto`.

## Architecture

```
Client (Pi, OpenCode, etc.)
  │
  ▼
server.py (:5500)
  ├── MF scorer (OpenAI embeddings API)
  ├── Supra-Router-51M (local CPU, thread-pooled)
  └── routes to:
        ├── cheap backend  (e.g. deepseek-v4-flash)
        └── expensive backend (e.g. glm-5.2)
```

The Supra-Router model is downloaded once to `~/.cache/huggingface/` and loaded
from disk on startup (~1s). Inference runs in `asyncio.to_thread()` so it
doesn't block the event loop.

## Response headers

Every response includes:

- `x-route-decision: expensive|cheap`
- `x-route-score: 0.1234` (MF win-rate)
- `x-route-supra-complexity: 3` (1-5, only when Supra is enabled)
- `x-route-model: glm-5.2` (actual backend model used)
