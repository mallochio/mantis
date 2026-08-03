"""RouteLLM coding-router server.

Exposes one OpenAI-compatible model ("auto"). Requests are scored by the
RouteLLM MF+Supra router and forwarded through a local LiteLLM proxy, which
handles provider normalization, Responses API translation, and retries.

Config via env:
  ROUTELLM_HOST=127.0.0.1
  ROUTELLM_PORT=5500
  ROUTELLM_KEY=sk-route-local          # bearer token clients must present
  LITELLM_BASE=http://127.0.0.1:3001/v1
  LITELLM_KEY=sk-mundial
  ROUTELLM_THRESHOLD=0.156
  ROUTELLM_ROUTER=mf
  ROUTELLM_USE_SUPRA=1
  ROUTELLM_SUPRA_THRESHOLD=3
  EXPENSIVE_MODEL=gpt-5.6-luna
  CHEAP_MODEL=deepseek-v4-pro
  LOG_FILE=~/.config/llm-router/logs/decisions.log
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path

import httpx
from fastapi import FastAPI, Header, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

# Defaults mirror llm-router.sh (canonical source, calibrated there) —
# keep in sync so bare `python server.py` behaves identically to the launcher.
HOST = os.environ.get("ROUTELLM_HOST", "127.0.0.1")
PORT = int(os.environ.get("ROUTELLM_PORT", "5500"))
SERVER_KEY = os.environ.get("ROUTELLM_KEY", "sk-route-local")
THRESHOLD = float(os.environ.get("ROUTELLM_THRESHOLD", "0.156"))
ROUTER_NAME = os.environ.get("ROUTELLM_ROUTER", "mf")
SUPRA_ENABLED = os.environ.get("ROUTELLM_USE_SUPRA", "1") != "0"
SUPRA_THRESHOLD = int(os.environ.get("ROUTELLM_SUPRA_THRESHOLD", "3"))
SUPRA_MIN_SCORE = float(os.environ.get("ROUTELLM_SUPRA_MIN_SCORE", "0"))
ROUTELLM_CONTEXT_WINDOW = os.environ.get("ROUTELLM_CONTEXT_WINDOW", "auto")
ROUTELLM_MAX_TOKENS = int(os.environ.get("ROUTELLM_MAX_TOKENS", "131072"))
MODEL_ID = "auto"

LITELLM_BASE = os.environ.get("LITELLM_BASE", "http://127.0.0.1:3001/v1")
LITELLM_KEY = os.environ.get("LITELLM_KEY", "sk-mundial")

EXPENSIVE = {
    "base": os.environ.get("EXPENSIVE_BASE", LITELLM_BASE),
    "key": os.environ.get("EXPENSIVE_KEY", LITELLM_KEY),
    "model": os.environ.get("EXPENSIVE_MODEL", "gpt-5.6-luna"),
    "effort": os.environ.get("EXPENSIVE_REASONING_EFFORT", "xhigh"),
}
CHEAP = {
    "base": os.environ.get("CHEAP_BASE", LITELLM_BASE),
    "key": os.environ.get("CHEAP_KEY", LITELLM_KEY),
    "model": os.environ.get("CHEAP_MODEL", "deepseek-v4-pro"),
    "effort": os.environ.get("CHEAP_REASONING_EFFORT", "xhigh"),
    "max_tokens": int(os.environ.get("CHEAP_MAX_TOKENS", str(ROUTELLM_MAX_TOKENS))),
}

LOG_PATH = Path(os.environ.get("LOG_FILE", str(Path.home()/".config/llm-router/logs/decisions.log")))
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

_cached_context_window = None


def _fetch_model_context_length(base: str, key: str, model_id: str) -> int | None:
    try:
        if "openrouter.ai" in base:
            url = "https://openrouter.ai/api/v1/models"
            resp = httpx.get(url, timeout=5.0)
            if resp.status_code == 200:
                for item in resp.json().get("data", []):
                    if item.get("id") == model_id:
                        return item.get("context_length")
        else:
            url = base.rstrip("/") + "/models"
            headers = {"Authorization": f"Bearer {key}"} if key else {}
            resp = httpx.get(url, headers=headers, timeout=5.0)
            if resp.status_code == 200:
                for item in resp.json().get("data", []):
                    if item.get("id") == model_id or item.get("model_name") == model_id:
                        return item.get("context_window") or item.get("context_length")
    except Exception:
        pass
    return None


def _get_context_window() -> int:
    global _cached_context_window
    if _cached_context_window is not None:
        return _cached_context_window

    env_val = os.environ.get("ROUTELLM_CONTEXT_WINDOW")
    if env_val and env_val.isdigit() and int(env_val) > 0:
        _cached_context_window = int(env_val)
        return _cached_context_window

    ctx_exp = _fetch_model_context_length(EXPENSIVE["base"], EXPENSIVE["key"], EXPENSIVE["model"])
    ctx_cheap = _fetch_model_context_length(CHEAP["base"], CHEAP["key"], CHEAP["model"])

    valid = [c for c in (ctx_exp, ctx_cheap) if isinstance(c, int) and c > 0]
    if valid:
        _cached_context_window = min(valid)
        return _cached_context_window

    _cached_context_window = 1000000
    print(
        "WARNING: could not determine context window from model lists; "
        "falling back to 1000000. Set ROUTELLM_CONTEXT_WINDOW to override.",
        flush=True,
    )
    return _cached_context_window

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


def _supra_complexity(prompt: str) -> tuple[int, int]:
    model, tokenizer = _load_supra()
    fmt = f"Task: {prompt}\nAnalysis: "
    inputs = tokenizer(fmt, return_tensors="pt")
    import torch
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=128, do_sample=False,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
        )
    supra_ms = int((time.time() - t0) * 1000)
    gen = tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True,
    ).strip()
    return _parse_supra_complexity(gen), supra_ms


def _decide_uncached(trimmed_prompt: str) -> tuple[str, float, int | None, int | None]:
    try:
        r = _load_router()
        score = float(r.calculate_strong_win_rate(trimmed_prompt))
        supra_complexity = None
        supra_ms = None
        if score >= THRESHOLD:
            return "expensive", score, supra_complexity, supra_ms
        if SUPRA_ENABLED and score >= SUPRA_MIN_SCORE:
            supra_complexity, supra_ms = _supra_complexity(trimmed_prompt)
            if supra_complexity >= SUPRA_THRESHOLD:
                return "expensive", score, supra_complexity, supra_ms
        return "cheap", score, supra_complexity, supra_ms
    except Exception as err:
        print(f"Router decision failed ({err}); defaulting to expensive", flush=True)
        return tuple(["expensive", 1.0, None, None])  # type: ignore


@lru_cache(maxsize=256)
def _decide_cached(trimmed_prompt: str) -> tuple[str, float, int | None, int | None]:
    return _decide_uncached(trimmed_prompt)


def _decide(prompt: str) -> tuple[str, float, int | None, int | None]:
    # Take tail of prompt (~15k chars) so routing evaluates the latest user request & context
    trimmed_prompt = prompt[-15000:] if len(prompt) > 15000 else prompt
    return _decide_cached(trimmed_prompt)


def _backend_for(decision: str) -> dict:
    return EXPENSIVE if decision == "expensive" else CHEAP


def _build_outgoing_body(body: dict, backend: dict) -> dict:
    out_body = dict(body)
    if isinstance(out_body.get("messages"), list):
        out_body["messages"] = _normalize_messages_for_backend(out_body["messages"])
    out_body["model"] = backend["model"]
    if isinstance(out_body.get("max_tokens"), int):
        out_body["max_completion_tokens"] = out_body.pop("max_tokens")
    if isinstance(out_body.get("max_completion_tokens"), int) and backend.get("max_tokens"):
        out_body["max_completion_tokens"] = min(out_body["max_completion_tokens"], backend["max_tokens"])
    out_body.pop("stop", None)
    if backend["model"].startswith("gpt-5.6-") and out_body.get("temperature") not in (None, 1):
        out_body.pop("temperature")
    if backend["effort"]:
        out_body["reasoning_effort"] = backend["effort"]
    return out_body


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


def _log(
    decision: str,
    score: float,
    backend_model: str,
    prompt: str,
    ttfb_ms: int | None,
    supra_complexity: int | None = None,
    supra_ms: int | None = None,
):
    row = {
        "ts": time.time(),
        "router": ROUTER_NAME,
        "threshold": THRESHOLD,
        "score": round(score, 4),
        "supra_complexity": supra_complexity,
        "supra_ms": supra_ms,
        "decision": decision,
        "model": backend_model,
        "ttfb_ms": ttfb_ms,
        "prompt": prompt[:200],
    }
    with LOG_PATH.open("a") as f:
        f.write(json.dumps(row) + "\n")


_READY = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load both scoring models before serving so the first request is fast and
    # load failures (missing OPENAI_API_KEY, HF unreachable) fail startup
    # instead of silently degrading routing mid-request.
    try:
        _load_router()
        if SUPRA_ENABLED:
            _load_supra()
    finally:
        global _READY
        _READY = True
    yield


app = FastAPI(title="RouteLLM coding-router", lifespan=lifespan)


@app.get("/healthz")
async def healthz():
    return {"ok": _READY, "router": ROUTER_NAME, "threshold": THRESHOLD,
            "backend": "litellm", "ready": _READY}


@app.get("/v1/models")
async def list_models():
    return {"object": "list", "data": [{
        "id": MODEL_ID, "object": "model", "owned_by": "routellm",
        "context_window": _get_context_window(), "max_tokens": ROUTELLM_MAX_TOKENS,
    }]}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request, authorization: str | None = Header(default=None)):
    if not _authorize(authorization):
        return JSONResponse({"error": {"message": "Invalid API key", "type": "auth_error"}}, status_code=401)

    body = await request.json()
    prompt = _extract_prompt(body)
    if prompt.strip():
        decision, score, supra_complexity, supra_ms = await asyncio.to_thread(_decide, prompt)
    else:
        decision, score, supra_complexity, supra_ms = "cheap", 0.0, None, None
    backend = _backend_for(decision)

    out_body = _build_outgoing_body(body, backend)

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
        client = httpx.Client(timeout=None)

        def gen():
            t0 = time.time()
            ttfb = None
            saw_finish_reason = False
            saw_done = False
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
                            sline = line.strip()
                            if sline == "data: [DONE]":
                                saw_done = True
                            elif sline.startswith("data: "):
                                try:
                                    payload = json.loads(sline[6:])
                                    choices = payload.get("choices")
                                    if isinstance(choices, list) and choices and choices[0].get("finish_reason") is not None:
                                        saw_finish_reason = True
                                except Exception:
                                    pass
                            yield (line + "\n").encode()
                        else:
                            yield b"\n"
                    if not saw_finish_reason:
                        finish_chunk = {
                            "id": "gen-finish",
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": backend["model"],
                            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                        }
                        yield f"data: {json.dumps(finish_chunk)}\n\n".encode()
                    if not saw_done:
                        yield b"data: [DONE]\n\n"
            finally:
                client.close()
            _log(decision, score, backend["model"], prompt, ttfb, supra_complexity, supra_ms)

        return StreamingResponse(gen(), media_type="text/event-stream", headers=route_hdr)

    t0 = time.time()
    with httpx.Client(timeout=None) as c:
        resp = c.post(url, json=out_body, headers=headers)
    ttfb = int((time.time() - t0) * 1000)
    _log(decision, score, backend["model"], prompt, ttfb, supra_complexity, supra_ms)
    return Response(content=resp.content, status_code=resp.status_code,
                    media_type="application/json", headers=route_hdr)


if __name__ == "__main__":
    import uvicorn
    print(
        "effective config: "
        f"router={ROUTER_NAME} threshold={THRESHOLD} supra={SUPRA_ENABLED} "
        f"supra_threshold={SUPRA_THRESHOLD} supra_min_score={SUPRA_MIN_SCORE} "
        f"expensive={EXPENSIVE['model']} cheap={CHEAP['model']} port={PORT}",
        flush=True,
    )
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")