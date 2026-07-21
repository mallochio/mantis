# llm-router

OpenAI-compatible proxy for Pi and other clients. It uses RouteLLM MF +
Supra-Router to choose cheap or expensive, then sends both through a local
LiteLLM proxy. LiteLLM handles provider normalization, retries/fallbacks, and
the Responses API bridge needed by GPT-5.6 function-tool requests.

LiteLLM owns the backend model definitions and fallback behavior; MF+Supra
only decides whether a request uses the cheap or expensive model group.

## Files

- `server.py` — FastAPI server exposing `/v1/chat/completions` (OpenAI-compatible)
- `llm-router.sh` — launch script (router stop-and-restart if running, detach +
  print status). Designed for StartupFolder or manual use.
- `test_server_helpers.py` — unit tests for message normalization and Supra output parsing
- `_test_scores.sh` — smoke test that routes sample prompts and prints scores/decisions

## Setup

```bash
# Python deps
python -m venv .venv
. .venv/bin/activate
pip install routellm fastapi uvicorn httpx transformers torch

# Minimal router config in ~/.zshrc
export EXPENSIVE_MODEL="gpt-5.6-luna"
export EXPENSIVE_REASONING_EFFORT="xhigh"
export CHEAP_MODEL="deepseek-v4-pro"
export CHEAP_REASONING_EFFORT="xhigh"

# OPENAI_API_KEY is only needed for the local MF scorer.
```

## Run

```bash
./llm-router.sh
# LiteLLM proxy and router running
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
  └── LiteLLM proxy (:3001)
        ├── cheap model      (configurable in ~/.zshrc)
        └── expensive model  (configurable in ~/.zshrc)
```

`llm-router.sh` starts the LiteLLM Docker Compose stack at
`http://127.0.0.1:3001` when it is not already healthy, then starts the
RouteLLM compatibility endpoint at `:5500`.

## Response headers

Every response includes:

- `x-route-decision: expensive|cheap`
- `x-route-score: 0.1234` (MF win-rate)
- `x-route-supra-complexity: 3` (1-5, when Supra is enabled)
- `x-route-model: deepseek-v4-pro|gpt-5.6-luna` (LiteLLM model group selected)
