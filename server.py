"""RouteLLM coding-router server.

Exposes one OpenAI-compatible model ("auto"). Requests are scored by the
RouteLLM MF+Supra router and forwarded directly to OpenRouter or OpenCode Go.

Config via env:
  ROUTELLM_HOST=127.0.0.1
  ROUTELLM_PORT=5500
  ROUTELLM_KEY=sk-route-local          # bearer token clients must present
  EXPENSIVE_BASE=https://openrouter.ai/api/v1
  EXPENSIVE_KEY=...
  CHEAP_BASE=https://opencode.ai/zen/go/v1
  CHEAP_KEY=...
  ROUTELLM_THRESHOLD=0.2
  ROUTELLM_ROUTER=mf
  ROUTELLM_USE_SUPRA=1
  ROUTELLM_SUPRA_THRESHOLD=3
  EXPENSIVE_MODEL=openai/gpt-5.6-sol
  CHEAP_MODEL=deepseek-v4-flash
  LOG_FILE=~/.config/llm-router/logs/decisions.log
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
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
THRESHOLD = float(os.environ.get("ROUTELLM_THRESHOLD", "0.2"))
ROUTER_NAME = os.environ.get("ROUTELLM_ROUTER", "mf")
SUPRA_ENABLED = os.environ.get("ROUTELLM_USE_SUPRA", "1") != "0"
SUPRA_THRESHOLD = int(os.environ.get("ROUTELLM_SUPRA_THRESHOLD", "3"))
SUPRA_MIN_SCORE = float(os.environ.get("ROUTELLM_SUPRA_MIN_SCORE", "0"))
ROUTELLM_CONTEXT_WINDOW = os.environ.get("ROUTELLM_CONTEXT_WINDOW", "auto")
ROUTELLM_MAX_TOKENS = int(os.environ.get("ROUTELLM_MAX_TOKENS", "131072"))
MODEL_ID = "auto"

EXPENSIVE = {
    "base": os.environ.get("EXPENSIVE_BASE", "https://openrouter.ai/api/v1"),
    "key": os.environ.get("EXPENSIVE_KEY", ""),
    "model": os.environ.get("EXPENSIVE_MODEL", "openai/gpt-5.6-sol"),
    "effort": os.environ.get("EXPENSIVE_REASONING_EFFORT", "medium"),
}
CHEAP = {
    "base": os.environ.get("CHEAP_BASE", "https://opencode.ai/zen/go/v1"),
    "key": os.environ.get("CHEAP_KEY", ""),
    "model": os.environ.get("CHEAP_MODEL", "deepseek-v4-flash"),
    "effort": os.environ.get("CHEAP_REASONING_EFFORT", "none"),
    "max_tokens": int(os.environ.get("CHEAP_MAX_TOKENS", str(ROUTELLM_MAX_TOKENS))),
}

# Shared connection pool: reuse TCP/TLS to upstream providers instead of
# handshaking per request (sync Client is thread-safe). A total timeout bounds
# stalled upstreams so a hung provider cannot pin the router forever.
TIMEOUT_S = float(os.environ.get("ROUTELLM_TIMEOUT_S", "600"))
_client = httpx.Client(timeout=httpx.Timeout(TIMEOUT_S, connect=10.0))

# Reject oversized bodies before they are buffered into memory (413).
MAX_BODY_BYTES = int(os.environ.get("ROUTELLM_MAX_BODY_BYTES", str(50 * 1024 * 1024)))

for _b in (EXPENSIVE, CHEAP):
    # OpenRouter reports provider-billed cost in usage.cost when asked.
    _b["usage_include"] = "openrouter.ai" in _b["base"]

DATA_DIR = Path(os.environ.get("MANTIS_DATA_DIR", str(Path.home()/".local/share/mantis")))
LOG_PATH = Path(os.environ.get("LOG_FILE", str(DATA_DIR/"router/decisions.log")))
TRAINING_LOG_ENABLED = os.environ.get("ROUTELLM_TRAINING_LOG", "0").lower() in {"1", "true", "yes", "on"}
TRAINING_LOG_PATH = Path(os.environ.get("TRAINING_LOG_FILE", str(DATA_DIR/"router/training.jsonl")))
OUTCOME_LOG_PATH = Path(os.environ.get("OUTCOME_LOG_FILE", str(DATA_DIR/"router/outcomes.jsonl")))
# Same prompt re-sent within this window usually means the previous route failed.
RETRY_WINDOW_S = float(os.environ.get("ROUTELLM_RETRY_WINDOW_S", "900"))
# Short prompts are mostly trivial/ack traffic; MF embeddings on tiny text are
# noisy and over-route (27% of <=120-char prompts went expensive in the
# Aug 5-8 log, labeler called them cheap). Force cheap unless MF is very sure.
SHORT_PROMPT_MAX_CHARS = int(os.environ.get("ROUTELLM_SHORT_PROMPT_MAX_CHARS", "120"))
SHORT_PROMPT_FORCE_CHEAP_SCORE = float(os.environ.get("ROUTELLM_SHORT_PROMPT_SCORE", "0.25"))
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
if TRAINING_LOG_ENABLED:
    TRAINING_LOG_PATH.touch(mode=0o600, exist_ok=True)
    TRAINING_LOG_PATH.chmod(0o600)

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
        import subprocess
        try:
            k = subprocess.check_output(
                ["security", "find-generic-password", "-s", "AI.Playground.openai.apiKey", "-w"],
                text=True, stderr=subprocess.DEVNULL,
            ).strip()
            if k:
                os.environ["OPENAI_API_KEY"] = k
        except Exception:
            pass
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
    inputs = tokenizer(fmt, return_tensors="pt", truncation=True,
                       max_length=tokenizer.model_max_length)
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
        if len(trimmed_prompt) <= SHORT_PROMPT_MAX_CHARS and score < SHORT_PROMPT_FORCE_CHEAP_SCORE:
            return "cheap", score, None, None
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


_REFUSAL_RE = re.compile(
    r"cannot (assist|help|comply)|i('| a)?m sorry|not (able|allowed) to (assist|help)|can('| no)t (assist|help)",
    re.I,
)


def _safe_json(content: bytes) -> dict:
    try:
        data = json.loads(content)
        return data if isinstance(data, dict) else {"error": {"message": content.decode(errors="replace")[:500]}}
    except Exception:
        return {"error": {"message": content.decode(errors="replace")[:500]}}


def _is_refusal(status: int, data: dict) -> bool:
    """True when an upstream response is a refusal we should retry on the other model."""
    s = ""
    err = data.get("error")
    if isinstance(err, dict):
        s += f"{err.get('message','')} {err.get('code','')} {err.get('type','')}"
    elif err:
        s += str(err)
    choices = data.get("choices") or []
    if choices:
        fr = choices[0].get("finish_reason")
        if fr == "content_filter":
            return True
        msg = choices[0].get("message")
        if isinstance(msg, dict):
            s += f" {msg.get('content') or ''} {msg.get('reasoning_content') or ''}"
    if status != 200 and re.search(r"content[\s_-]?filter", s, re.I):
        return True
    return bool(_REFUSAL_RE.search(s))


def _do_post(backend: dict, out_body: dict):
    url = backend["base"].rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {backend['key']}", "Content-Type": "application/json"}
    return _client.post(url, json=out_body, headers=headers)


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
    if backend.get("usage_include") and "usage" not in out_body:
        out_body["usage"] = {"include": True}
    return out_body


def _extract_cost(data: dict) -> float | None:
    try:
        cost = (data.get("usage") or {}).get("cost")
        return float(cost) if isinstance(cost, (int, float)) else None
    except Exception:
        return None


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
    cost_usd: float | None = None,
    usage: dict | None = None,
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
        "prompt_hash": _prompt_hash(prompt),
    }
    if cost_usd is not None:
        row["cost_usd"] = cost_usd
    if usage:
        row["usage"] = usage
    with LOG_PATH.open("a") as f:
        f.write(json.dumps(row) + "\n")
    if TRAINING_LOG_ENABLED:
        row["prompt"] = prompt
        with TRAINING_LOG_PATH.open("a") as f:
            f.write(json.dumps(row) + "\n")


def _prompt_hash(prompt: str) -> str:
    return hashlib.sha1(prompt.encode("utf-8", errors="replace")).hexdigest()[:16]


def _log_outcome(prompt_hash: str, outcome: str, **detail) -> None:
    """Outcome feedback for retraining, joined to decisions via prompt_hash."""
    row = {"ts": time.time(), "prompt_hash": prompt_hash, "outcome": outcome, **detail}
    try:
        OUTCOME_LOG_PATH.touch(mode=0o600, exist_ok=True)
        OUTCOME_LOG_PATH.chmod(0o600)
        with OUTCOME_LOG_PATH.open("a") as f:
            f.write(json.dumps(row) + "\n")
    except Exception:
        pass


_recent_prompts: dict[str, tuple[str, str, float]] = {}  # request_hash -> (decision, model, ts)

# ---- response cache: identical resends (timeout retries) served without
# hitting upstream. Keyed on the full request body so agent-loop iterations
# (which append tool results) never match. ----
RESP_CACHE_TTL_S = float(os.environ.get("ROUTELLM_RESP_CACHE_TTL_S", "120"))
_RESP_CACHE_MAX = 512
_resp_cache: dict[str, tuple[float, list[bytes]]] = {}  # body hash -> (ts, chunks)


def _request_body_hash(body: dict) -> str:
    return hashlib.sha1(
        json.dumps(body, sort_keys=True, ensure_ascii=False, default=str).encode()
    ).hexdigest()[:16]


def _cache_get(req_hash: str) -> list[bytes] | None:
    hit = _resp_cache.get(req_hash)
    if not hit:
        return None
    ts, chunks = hit
    if time.time() - ts > RESP_CACHE_TTL_S:
        _resp_cache.pop(req_hash, None)
        return None
    return chunks


def _cache_put(req_hash: str, chunks: list[bytes]) -> None:
    if len(_resp_cache) >= _RESP_CACHE_MAX:
        oldest = min(_resp_cache, key=lambda k: _resp_cache[k][0])
        _resp_cache.pop(oldest, None)
    _resp_cache[req_hash] = (time.time(), chunks)


def _request_hash(body: dict) -> str:
    """Hash of the tail of the conversation. Agent loops append tool results
    between calls (hash changes); a true retry resends an identical body
    (hash stable) — hashing just the last user message misfires on loops."""
    msgs = body.get("messages") or []
    return hashlib.sha1(
        json.dumps(msgs[-6:], sort_keys=True, ensure_ascii=False, default=str).encode()
    ).hexdigest()[:16]


def _record_and_detect_retry(req_hash: str, decision: str, model: str, prompt_hash: str) -> None:
    now = time.time()
    for k, (_, _, ts) in list(_recent_prompts.items()):
        if now - ts > RETRY_WINDOW_S:
            del _recent_prompts[k]
    prev = _recent_prompts.pop(req_hash, None)
    if prev and now - prev[2] <= RETRY_WINDOW_S:
        _log_outcome(prompt_hash, "retried", decision=prev[0], model=prev[1],
                     request_hash=req_hash, retry_after_s=round(now - prev[2], 1))
    _recent_prompts[req_hash] = (decision, model, now)


def _length_truncated(data: dict) -> bool:
    ch = data.get("choices")
    return bool(isinstance(ch, list) and ch and ch[0].get("finish_reason") == "length")


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
    _client.close()


app = FastAPI(title="RouteLLM coding-router", lifespan=lifespan)


@app.middleware("http")
async def _body_limit(request: Request, call_next):
    length = request.headers.get("content-length")
    if _body_too_large(length):
        return JSONResponse(
            {"error": {"message": "request body too large", "type": "request_too_large"}},
            status_code=413,
        )
    return await call_next(request)


def _body_too_large(content_length: str | None) -> bool:
    return bool(content_length and content_length.isdigit() and int(content_length) > MAX_BODY_BYTES)


@app.get("/healthz")
async def healthz():
    return {"ok": _READY, "router": ROUTER_NAME, "threshold": THRESHOLD,
            "backend": "direct", "ready": _READY}


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
    cache_key = _request_body_hash(body)
    cached_chunks = _cache_get(cache_key)
    if cached_chunks is not None:
        if bool(body.get("stream")):
            return StreamingResponse(iter(cached_chunks), media_type="text/event-stream",
                                     headers={"x-route-cache": "hit"})
        return Response(content=b"".join(cached_chunks), status_code=200,
                        media_type="application/json", headers={"x-route-cache": "hit"})
    prompt = _extract_prompt(body)
    if prompt.strip():
        decision, score, supra_complexity, supra_ms = await asyncio.to_thread(_decide, prompt)
    else:
        decision, score, supra_complexity, supra_ms = "cheap", 0.0, None, None
    backend = _backend_for(decision)
    _record_and_detect_retry(_request_hash(body), decision, backend["model"], _prompt_hash(prompt))

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
    if supra_ms is not None:
        route_hdr["x-route-supra-ms"] = str(supra_ms)

    if want_stream:
        def gen():
            t0 = time.time()
            ttfb = None
            cache_chunks: list[bytes] = []
            saw_finish_reason = False
            saw_done = False
            saw_length = False
            usage_seen: dict = {}

            def track(payload: dict) -> None:
                nonlocal saw_finish_reason, saw_length
                ch = payload.get("choices")
                if isinstance(ch, list) and ch and ch[0].get("finish_reason") is not None:
                    saw_finish_reason = True
                    if ch[0].get("finish_reason") == "length":
                        saw_length = True
                u = payload.get("usage")
                if isinstance(u, dict):
                    usage_seen.update(u)
            with _client.stream("POST", url, json=out_body, headers=headers) as resp:
                if resp.status_code != 200:
                    # Refusal/content-filter surfaced as an HTTP error: retry the other model.
                    err = resp.read()
                    if _is_refusal(resp.status_code, _safe_json(err)):
                        _log_outcome(_prompt_hash(prompt), "refused", decision=decision,
                                     model=backend["model"], status=resp.status_code)
                        flip = "cheap" if decision == "expensive" else "expensive"
                        fb = _backend_for(flip)
                        fb_url = fb["base"].rstrip("/") + "/chat/completions"
                        fb_headers = {"Authorization": f"Bearer {fb['key']}", "Content-Type": "application/json"}
                        with _client.stream("POST", fb_url, json=_build_outgoing_body(body, fb), headers=fb_headers) as resp:
                            if resp.status_code == 200:
                                # route_hdr was already sealed by StreamingResponse;
                                # the fallback is only visible in decisions.log.
                                for line in resp.iter_lines():
                                    if ttfb is None:
                                        ttfb = int((time.time() - t0) * 1000)
                                    if line:
                                        sline = line.strip()
                                        if sline == "data: [DONE]":
                                            saw_done = True
                                            yield (line + "\n").encode()
                                            break
                                        elif sline.startswith("data: "):
                                            try:
                                                track(json.loads(sline[6:]))
                                            except Exception:
                                                pass
                                        yield (line + "\n").encode()
                                    else:
                                        yield b"\n"
                                _log(flip, score, fb["model"], prompt, ttfb, supra_complexity, supra_ms,
                                     cost_usd=_extract_cost({"usage": usage_seen}), usage=usage_seen or None)
                                return
                    _log_outcome(_prompt_hash(prompt), "upstream_error", decision=decision,
                                 model=backend["model"], status=resp.status_code)
                    yield b'data: ' + json.dumps({"error": {"status": resp.status_code, "message": err.decode(errors="replace")[:500]}}).encode() + b'\n\ndata: [DONE]\n\n'
                    return
                for line in resp.iter_lines():
                    if ttfb is None:
                        ttfb = int((time.time() - t0) * 1000)
                    if line:
                        sline = line.strip()
                        if sline == "data: [DONE]":
                            saw_done = True
                            cache_chunks.append((line + "\n").encode())
                            yield (line + "\n").encode()
                            break
                        elif sline.startswith("data: "):
                            try:
                                track(json.loads(sline[6:]))
                            except Exception:
                                pass
                        cache_chunks.append((line + "\n").encode())
                        yield (line + "\n").encode()
                    else:
                        cache_chunks.append(b"\n")
                        yield b"\n"
                if not saw_finish_reason:
                    finish_chunk = {
                        "id": "gen-finish",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": backend["model"],
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    }
                    cache_chunks.append(f"data: {json.dumps(finish_chunk)}\n\n".encode())
                    yield f"data: {json.dumps(finish_chunk)}\n\n".encode()
                if not saw_done:
                    cache_chunks.append(b"data: [DONE]\n\n")
                    yield b"data: [DONE]\n\n"
            if saw_done and not saw_length:
                _cache_put(cache_key, cache_chunks)
            if saw_length:
                _log_outcome(_prompt_hash(prompt), "truncated", decision=decision,
                             model=backend["model"])
            _log(decision, score, backend["model"], prompt, ttfb, supra_complexity, supra_ms,
                 cost_usd=_extract_cost({"usage": usage_seen}), usage=usage_seen or None)

        return StreamingResponse(gen(), media_type="text/event-stream", headers=route_hdr)

    t0 = time.time()
    resp = _do_post(backend, out_body)
    ttfb = int((time.time() - t0) * 1000)
    data = _safe_json(resp.content)
    if _is_refusal(resp.status_code, data):
        _log_outcome(_prompt_hash(prompt), "refused", decision=decision,
                     model=backend["model"], status=resp.status_code)
        flip = "cheap" if decision == "expensive" else "expensive"
        fb = _backend_for(flip)
        t0 = time.time()
        resp = _do_post(fb, _build_outgoing_body(body, fb))
        ttfb = int((time.time() - t0) * 1000)
        data = _safe_json(resp.content)
        decision = flip
        route_hdr["x-route-fallback"] = "true"
        route_hdr["x-route-model"] = fb["model"]
    if resp.status_code != 200:
        _log_outcome(_prompt_hash(prompt), "upstream_error", decision=decision,
                     model=route_hdr["x-route-model"], status=resp.status_code)
    elif _length_truncated(data):
        _log_outcome(_prompt_hash(prompt), "truncated", decision=decision,
                     model=route_hdr["x-route-model"])
    cost = _extract_cost(data)
    route_hdr["x-route-ttfb-ms"] = str(ttfb)
    if cost is not None:
        route_hdr["x-route-cost-usd"] = f"{cost:.6f}"
    usage = data.get("usage") if isinstance(data.get("usage"), dict) else None
    _log(decision, score, route_hdr["x-route-model"], prompt, ttfb, supra_complexity, supra_ms,
         cost_usd=cost, usage=usage)
    if resp.status_code == 200 and "x-route-fallback" not in route_hdr:
        _cache_put(cache_key, [resp.content])
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