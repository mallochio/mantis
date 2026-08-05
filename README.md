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
- `requirements.txt` — pinned Python dependencies

## Setup

```bash
# Python deps (versions pinned in requirements.txt)
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt

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

## Training data

Full router prompts are collected in
`~/.local/share/mantis/router/training.jsonl` only when
`ROUTELLM_TRAINING_LOG=1`. Change it to `0` and restart `llm-router.sh` to
stop collection. `~/.local/share/mantis/router/decisions.log` remains the
operational log and stores only the first 200 prompt characters.

Run `./.venv/bin/python pseudo_label.py` to write deduplicated Supra labels to
`~/.local/share/mantis/router/pseudo-labels.jsonl`. Training files are local,
mode `0600`, and outside the repository.

## Response headers

Every response includes:

- `x-route-decision: expensive|cheap`
- `x-route-score: 0.1234` (MF win-rate)
- `x-route-supra-complexity: 3` (1-5, when Supra is enabled)
- `x-route-model: deepseek-v4-pro|gpt-5.6-luna` (LiteLLM model group selected)

## Request handling

Before forwarding a client request to LiteLLM, the router applies several deliberate compatibility mutations required for LiteLLM's Responses API bridge:

- `model` is always overwritten with the chosen backend model (`gpt-5.6-luna` / `deepseek-v4-pro`); the client's value is ignored.
- `max_tokens` is renamed to `max_completion_tokens` and clamped to the backend's `CHEAP_MAX_TOKENS` (default 131072).
- `stop` sequences are dropped.
- `temperature` is dropped for gpt-5.6 models unless it is `1`.
- `reasoning_effort` is injected from `EXPENSIVE_REASONING_EFFORT` / `CHEAP_REASONING_EFFORT`.
- `developer`-role messages are rewritten to `system`.

