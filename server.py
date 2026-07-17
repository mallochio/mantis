"""RouteLLM coding-router server.

Exposes one OpenAI-compatible model ("auto"). For each request it scores the
last user message with the RouteLLM `mf` router (matrix factorization) and
optionally ensembles with Supra-Router-51M (a 51M-param complexity classifier)
to catch hard prompts the MF scorer underweights. A request goes expensive if
MF score >= threshold OR Supra-Router complexity >= supra threshold.

The mf router calls OpenAI text-embedding-3-small for each prompt (~$0.02/1M
tokens). Supra-Router runs locally (~400ms CPU, no API). OPENAI_API_KEY must
be set before the router loads. ~/Startup/llm-router.sh retrieves it from
macOS Keychain.

Config via env:
  ROUTELLM_HOST=127.0.0.1
  ROUTELLM_PORT=5500
  ROUTELLM_KEY=sk-route-local          # bearer token clients must present
  ROUTELLM_THRESHOLD=0.156             # mf score >= threshold -> expensive
  ROUTELLM_ROUTER=mf                   # mf|bert (mf needs OpenAI embeds)
  ROUTELLM_USE_SUPRA=1                 # 1=ensemble MF+Supra, 0=MF only
  ROUTELLM_SUPRA_THRESHOLD=3          # supra complexity >= threshold -> expensive
  EXPENSIVE_BASE=http://127.0.0.1:41437/v1
  EXPENSIVE_KEY=dummy
  EXPENSIVE_MODEL=gpt-5.6-luna
  EXPENSIVE_REASONING_EFFORT=ultra
  CHEAP_BASE=https://opencode.ai/zen/go/v1
  CHEAP_KEY=$OPENCODE_GO_API_KEY
  CHEAP_MODEL=glm-5.2
  LOG_FILE=~/.config/llm-router/logs/decisions.log
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path

import httpx
from fastapi import FastAPI, Header, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

HOST = os.environ.get("ROUTELLM_HOST", "127.0.0.1")
PORT = int(os.environ.get("ROUTELLM_PORT", "5500"))
SERVER_KEY = os.environ.get("ROUTELLM_KEY", "sk-route-local")
THRESHOLD = float(os.environ.get("ROUTELLM_THRESHOLD", "0.45"))
ROUTER_NAME = os.environ.get("ROUTELLM_ROUTER", "mf")
SUPRA_ENABLED = os.environ.get("ROUTELLM_USE_SUPRA", "1") != "0"
SUPRA_THRESHOLD = int(os.environ.get("ROUTELLM_SUPRA_THRESHOLD", "3"))
MODEL_ID = "auto"

EXPENSIVE = {
    "base": os.environ.get("EXPENSIVE_BASE", "http://127.0.0.1:41437/v1"),
    "key": os.environ.get("EXPENSIVE_KEY", "dummy"),
    "model": os.environ.get("EXPENSIVE_MODEL", "gpt-5.6-luna"),
    "effort": os.environ.get("EXPENSIVE_REASONING_EFFORT", "ultra"),
}
CHEAP = {
    "base": os.environ.get("CHEAP_BASE", "https://opencode.ai/zen/go/v1"),
    "key": os.environ.get("CHEAP_KEY", os.environ.get("OPENCODE_GO_API_KEY", "")),
    "model": os.environ.get("CHEAP_MODEL", "glm-5.2"),
    "effort": os.environ.get("CHEAP_REASONING_EFFORT"),  # None = don't send
    "max_tokens": int(os.environ.get("CHEAP_MAX_TOKENS", "64000")),
}

LOG_PATH = Path(os.environ.get("LOG_FILE", str(Path.home()/".config/llm-router/logs/decisions.log")))
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

_router = None  # lazy global


def _load_router():
    global _router
    if _router is not None:
        return _router
    # mf router calls OpenAI text-embedding-3-small at scoring time; the OpenAI()
    # client is instantiated at import of routellm.routers.similarity_weighted.utils,
    # so the key must be in env before this import runs.
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY required for mf router (embeddings)")
    from routellm.routers.routers import ROUTER_CLS
    cfg = {"checkpoint_path": "routellm/mf_gpt4_augmented"}
    if ROUTER_NAME == "bert":
        cfg = {"checkpoint_path": "routellm/bert_gpt4_augmented"}
    _router = ROUTER_CLS[ROUTER_NAME](**cfg)
    return _router


def _extract_prompt(body: dict) -> str:
    for m in reversed(body.get("messages") or []):
        if m.get("role") == "user":
            c = m.get("content")
            if isinstance(c, str):
                return c
            if isinstance(c, list):
                # text parts only
                return " ".join(p.get("text", "") for p in c if isinstance(p, dict) and p.get("type") == "text")
            return str(c)
    return ""


_supra_model = None
_supra_tokenizer = None


def _load_supra():
    global _supra_model, _supra_tokenizer
    if _supra_model is not None:
        return _supra_model, _supra_tokenizer
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    _supra_tokenizer = AutoTokenizer.from_pretrained("SupraLabs/Supra-Router-51M")
    _supra_model = AutoModelForCausalLM.from_pretrained(
        "SupraLabs/Supra-Router-51M", dtype=torch.float32,
    )
    _supra_model.eval()
    return _supra_model, _supra_tokenizer


def _parse_supra_complexity(text: str) -> int:
    for part in text.split("|"):
        part = part.strip()
        if part.lower().startswith("complexity:"):
            try:
                return int(part.split(":", 1)[1].strip())
            except (ValueError, IndexError):
                return 0
    return 0


def _supra_complexity(prompt: str) -> int:
    model, tokenizer = _load_supra()
    fmt = f"Task: {prompt}\nAnalysis: "
    inputs = tokenizer(fmt, return_tensors="pt")
    import torch
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=128, do_sample=False,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
        )
    gen = tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True,
    ).strip()
    return _parse_supra_complexity(gen)


def _decide(prompt: str) -> tuple[str, float, int | None]:
    r = _load_router()
    score = float(r.calculate_strong_win_rate(prompt))
    supra_complexity = None
    if score >= THRESHOLD:
        return "expensive", score, supra_complexity
    if SUPRA_ENABLED:
        supra_complexity = _supra_complexity(prompt)
        if supra_complexity >= SUPRA_THRESHOLD:
            return "expensive", score, supra_complexity
    return "cheap", score, supra_complexity


def _backend_for(decision: str) -> dict:
    return EXPENSIVE if decision == "expensive" else CHEAP


def _normalize_messages_for_backend(messages):
    out = []
    changed = False
    for message in messages:
        if isinstance(message, dict) and message.get("role") == "developer":
            message = {**message, "role": "system"}
            changed = True
        out.append(message)
    return out if changed else messages


def _authorize(authorization: str | None) -> bool:
    if not authorization:
        return False
    if not authorization.startswith("Bearer "):
        return False
    return authorization[7:] == SERVER_KEY


def _log(decision: str, score: float, backend_model: str, prompt: str, ttfb_ms: int | None, supra_complexity: int | None = None):
    row = {
        "ts": time.time(),
        "router": ROUTER_NAME,
        "threshold": THRESHOLD,
        "score": round(score, 4),
        "supra_complexity": supra_complexity,
        "decision": decision,
        "model": backend_model,
        "ttfb_ms": ttfb_ms,
        "prompt": prompt[:200],
    }
    with LOG_PATH.open("a") as f:
        f.write(json.dumps(row) + "\n")


app = FastAPI(title="RouteLLM coding-router")


@app.get("/healthz")
async def healthz():
    return {"ok": True, "router": ROUTER_NAME, "threshold": THRESHOLD}


@app.get("/v1/models")
async def list_models():
    return {"object": "list", "data": [{
        "id": MODEL_ID, "object": "model", "owned_by": "routellm",
        "context_window": 262144, "max_tokens": 64000,
    }]}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request, authorization: str | None = Header(default=None)):
    if not _authorize(authorization):
        return JSONResponse({"error": {"message": "Invalid API key", "type": "auth_error"}}, status_code=401)

    body = await request.json()
    prompt = _extract_prompt(body)
    if prompt.strip():
        decision, score, supra_complexity = await asyncio.to_thread(_decide, prompt)
    else:
        decision, score, supra_complexity = "cheap", 0.0, None
    backend = _backend_for(decision)

    out_body = dict(body)
    if isinstance(out_body.get("messages"), list):
        out_body["messages"] = _normalize_messages_for_backend(out_body["messages"])
    out_body["model"] = backend["model"]
    if backend.get("max_tokens"):
        for key in ("max_tokens", "max_completion_tokens"):
            if isinstance(out_body.get(key), int):
                out_body[key] = min(out_body[key], backend["max_tokens"])
    if backend["effort"]:
        out_body.setdefault("reasoning_effort", backend["effort"])
        out_body["reasoning_effort"] = backend["effort"]

    want_stream = bool(body.get("stream"))
    headers = {"Authorization": f"Bearer {backend['key']}", "Content-Type": "application/json"}
    url = backend["base"].rstrip("/") + "/chat/completions"
    route_hdr = {
        "x-route-decision": decision,
        "x-route-score": f"{score:.4f}",
        "x-route-model": backend["model"],
        "x-route-router": ROUTER_NAME,
    }
    if supra_complexity is not None:
        route_hdr["x-route-supra-complexity"] = str(supra_complexity)

    if want_stream:
        req = httpx.stream("POST", url, json=out_body, headers=headers, timeout=None)
        client = httpx.Client(timeout=None)

        def gen():
            t0 = time.time()
            ttfb = None
            try:
                with client.stream("POST", url, json=out_body, headers=headers) as resp:
                    if resp.status_code != 200:
                        err = resp.read()
                        yield b'data: ' + json.dumps({"error": {"status": resp.status_code, "message": err.decode(errors="replace")[:500]}}).encode() + b'\n\ndata: [DONE]\n\n'
                        return
                    for line in resp.iter_lines():
                        if ttfb is None:
                            ttfb = int((time.time() - t0) * 1000)
                        if line:
                            yield (line + "\n").encode()
                        yield b"\n"
                    yield b"data: [DONE]\n\n"
            finally:
                client.close()
            _log(decision, score, backend["model"], prompt, ttfb, supra_complexity)

        return StreamingResponse(gen(), media_type="text/event-stream", headers=route_hdr)

    t0 = time.time()
    with httpx.Client(timeout=None) as c:
        resp = c.post(url, json=out_body, headers=headers)
    ttfb = int((time.time() - t0) * 1000)
    _log(decision, score, backend["model"], prompt, ttfb, supra_complexity)
    return Response(content=resp.content, status_code=resp.status_code,
                    media_type="application/json", headers=route_hdr)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")