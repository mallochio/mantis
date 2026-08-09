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
import secrets
import time
import uuid
from collections import OrderedDict
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

# One async pool is created and closed by the ASGI lifespan.
TIMEOUT_S = float(os.environ.get("ROUTELLM_TIMEOUT_S", "600"))
_client: httpx.AsyncClient | None = None
RETRY_STATUSES = {429, 500, 502, 503, 504}

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
LOG_PATH.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
LOG_PATH.parent.chmod(0o700)
if TRAINING_LOG_ENABLED:
    TRAINING_LOG_PATH.touch(mode=0o600, exist_ok=True)
    TRAINING_LOG_PATH.chmod(0o600)

_cached_context_window = None


async def _fetch_model_context_length(base: str, key: str, model_id: str) -> int | None:
    """Discover capability without blocking the event loop."""
    if _client is None:
        return None
    try:
        url = ("https://openrouter.ai/api/v1/models" if "openrouter.ai" in base
               else base.rstrip("/") + "/models")
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        resp = await _client.get(url, headers=headers, timeout=5.0)
        if resp.status_code != 200:
            return None
        for item in resp.json().get("data", []):
            item_id = item.get("id") or item.get("model_name")
            # Providers can return either a bare name or provider/name.
            if item_id == model_id or str(item_id).rsplit("/", 1)[-1] == model_id.rsplit("/", 1)[-1]:
                value = item.get("context_window") or item.get("context_length")
                if isinstance(value, int) and value > 0:
                    return value
    except (httpx.HTTPError, ValueError, TypeError):
        return None
    return None


async def _get_context_window() -> int:
    global _cached_context_window
    if _cached_context_window is not None:
        return _cached_context_window
    env_val = os.environ.get("ROUTELLM_CONTEXT_WINDOW")
    if env_val and env_val.isdigit() and int(env_val) > 0:
        _cached_context_window = int(env_val)
        return _cached_context_window
    values = await asyncio.gather(
        _fetch_model_context_length(EXPENSIVE["base"], EXPENSIVE["key"], EXPENSIVE["model"]),
        _fetch_model_context_length(CHEAP["base"], CHEAP["key"], CHEAP["model"]),
    )
    valid = [value for value in values if isinstance(value, int) and value > 0]
    _cached_context_window = min(valid) if valid else 1_000_000
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
    if backend["model"].rsplit("/", 1)[-1].startswith("gpt-5.6-") and out_body.get("temperature") not in (None, 1):
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
    return secrets.compare_digest(authorization[7:].encode(), SERVER_KEY.encode())


def _secure_append(path: Path, row: dict) -> None:
    try:
        path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
        path.parent.chmod(0o700)
        fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        with os.fdopen(fd, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        path.chmod(0o600)
    except OSError:
        # Telemetry must never fail a user request.
        return


def _log(
    decision: str, score: float, backend_model: str, prompt: str,
    ttfb_ms: int | None, supra_complexity: int | None = None,
    supra_ms: int | None = None, cost_usd: float | None = None,
    usage: dict | None = None, request_id: str | None = None,
    occurrence_id: str | None = None, **detail,
):
    row = {
        "ts": time.time(), "request_id": request_id,
        "occurrence_id": occurrence_id or uuid.uuid4().hex,
        "router": ROUTER_NAME, "threshold": THRESHOLD, "score": round(score, 4),
        "supra_complexity": supra_complexity, "supra_ms": supra_ms,
        "decision": decision, "model": backend_model, "ttfb_ms": ttfb_ms,
        "prompt_hash": _prompt_hash(prompt), **detail,
    }
    if cost_usd is not None:
        row["cost_usd"] = cost_usd
    if usage:
        row["usage"] = usage
    _secure_append(LOG_PATH, row)
    if TRAINING_LOG_ENABLED:
        _secure_append(TRAINING_LOG_PATH, {**row, "prompt": prompt})


def _prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8", errors="replace")).hexdigest()[:24]


def _log_outcome(prompt_hash: str, outcome: str, **detail) -> None:
    row = {
        "ts": time.time(), "occurrence_id": uuid.uuid4().hex,
        "prompt_hash": prompt_hash, "outcome": outcome, **detail,
    }
    _secure_append(OUTCOME_LOG_PATH, row)


_recent_prompts: dict[str, tuple[str, str, float, str]] = {}

RESP_CACHE_TTL_S = float(os.environ.get("ROUTELLM_RESP_CACHE_TTL_S", "120"))
RESP_CACHE_MAX_ENTRIES = int(os.environ.get("ROUTELLM_RESP_CACHE_MAX_ENTRIES", "128"))
RESP_CACHE_MAX_BYTES = int(os.environ.get("ROUTELLM_RESP_CACHE_MAX_BYTES", str(8 * 1024 * 1024)))
_resp_cache: OrderedDict[str, tuple[float, bytes]] = OrderedDict()
_inflight: dict[str, asyncio.Future] = {}
_inflight_lock = asyncio.Lock()
_cache_bytes = 0
_cache_metrics = {"hits": 0, "misses": 0, "stores": 0, "evictions": 0}


def _request_body_hash(body: dict) -> str:
    return hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


def _cache_key(body: dict, idempotency_key: str | None) -> str | None:
    if not idempotency_key or len(idempotency_key) > 200:
        return None
    # Tool traffic can have external side effects and is never replay-safe.
    if body.get("tools") or any(isinstance(m, dict) and m.get("role") == "tool" for m in body.get("messages", [])):
        return None
    return hashlib.sha256((idempotency_key + ":" + _request_body_hash(body)).encode()).hexdigest()


def _cache_get(key: str | None) -> bytes | None:
    global _cache_bytes
    if key is None:
        return None
    hit = _resp_cache.get(key)
    if hit is None:
        _cache_metrics["misses"] += 1
        return None
    ts, content = hit
    if time.monotonic() - ts > RESP_CACHE_TTL_S:
        _cache_bytes -= len(content)
        del _resp_cache[key]
        _cache_metrics["misses"] += 1
        return None
    _resp_cache.move_to_end(key)
    _cache_metrics["hits"] += 1
    return content


def _response_replay_safe(body: dict, content: bytes) -> bool:
    if body.get("stream"):
        text = content.decode("utf-8", errors="replace")
        return (content.rstrip().endswith(b"data: [DONE]")
                and '"tool_calls"' not in text
                and not re.search(r'"finish_reason"\s*:\s*"content_filter"', text)
                and not _REFUSAL_RE.search(text))
    data = _safe_json(content)
    choices = data.get("choices") or []
    return (bool(choices) and not _is_refusal(200, data)
            and all(choice.get("finish_reason") != "content_filter"
                    and not (choice.get("message") or {}).get("tool_calls") for choice in choices))


def _cache_put(key: str | None, body: dict, content: bytes) -> None:
    global _cache_bytes
    if key is None or len(content) > RESP_CACHE_MAX_BYTES or not _response_replay_safe(body, content):
        return
    existing = _resp_cache.pop(key, None)
    if existing is not None:
        _cache_bytes -= len(existing[1])
    while _resp_cache and (len(_resp_cache) >= RESP_CACHE_MAX_ENTRIES or _cache_bytes + len(content) > RESP_CACHE_MAX_BYTES):
        _, (_, old) = _resp_cache.popitem(last=False)
        _cache_bytes -= len(old)
        _cache_metrics["evictions"] += 1
    if _cache_bytes + len(content) <= RESP_CACHE_MAX_BYTES:
        _resp_cache[key] = (time.monotonic(), content)
        _cache_bytes += len(content)
        _cache_metrics["stores"] += 1


async def _claim_inflight(key: str | None):
    if key is None:
        return True, None
    async with _inflight_lock:
        future = _inflight.get(key)
        if future is None:
            future = asyncio.get_running_loop().create_future()
            _inflight[key] = future
            return True, future
        return False, future


async def _finish_inflight(key: str | None, future, result) -> None:
    if key is None or future is None:
        return
    async with _inflight_lock:
        if _inflight.get(key) is future:
            _inflight.pop(key, None)
        if not future.done():
            future.set_result(result)


def _finish_inflight_nowait(key: str | None, future, result) -> None:
    """Release ownership synchronously from cancellation cleanup."""
    if key is None or future is None:
        return
    if _inflight.get(key) is future:
        _inflight.pop(key, None)
    if not future.done():
        future.set_result(result)


def _replayed_response(result, request_id: str):
    content, status, media, headers = result
    replay_headers = {**headers, "x-route-coalesced": "true", "x-request-id": request_id}
    if media == "text/event-stream":
        return StreamingResponse(iter([content]), status_code=status, media_type=media, headers=replay_headers)
    return Response(content=content, status_code=status, media_type=media, headers=replay_headers)


def _request_hash(body: dict) -> str:
    """Hash of the tail of the conversation. Agent loops append tool results
    between calls (hash changes); a true retry resends an identical body
    (hash stable) — hashing just the last user message misfires on loops."""
    msgs = body.get("messages") or []
    return hashlib.sha1(
        json.dumps(msgs[-6:], sort_keys=True, ensure_ascii=False, default=str).encode()
    ).hexdigest()[:16]


def _record_and_detect_retry(req_hash: str, decision: str, model: str, prompt_hash: str,
                             request_id: str, occurrence_id: str) -> None:
    now = time.time()
    for key, value in list(_recent_prompts.items()):
        if now - value[2] > RETRY_WINDOW_S:
            del _recent_prompts[key]
    prev = _recent_prompts.pop(req_hash, None)
    if prev and now - prev[2] <= RETRY_WINDOW_S:
        _log_outcome(prompt_hash, "retried", decision=prev[0], model=prev[1],
                     request_hash=req_hash, request_id=request_id,
                     decision_occurrence_id=prev[3], retry_after_s=round(now - prev[2], 1))
    _recent_prompts[req_hash] = (decision, model, now, occurrence_id)


def _length_truncated(data: dict) -> bool:
    ch = data.get("choices")
    return bool(isinstance(ch, list) and ch and ch[0].get("finish_reason") == "length")


_READY = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _READY, _client
    _client = httpx.AsyncClient(timeout=httpx.Timeout(TIMEOUT_S, connect=min(10.0, TIMEOUT_S)))
    try:
        # Model initialization is CPU/blocking work and must not block the loop.
        await asyncio.to_thread(_load_router)
        if SUPRA_ENABLED:
            await asyncio.to_thread(_load_supra)
        _READY = True
        yield
    finally:
        _READY = False
        if _client is not None:
            await _client.aclose()
        _client = None


app = FastAPI(title="RouteLLM coding-router", lifespan=lifespan)


def _openai_error(message: str, status: int, *, error_type: str = "invalid_request_error", param=None, code=None):
    return JSONResponse(
        {"error": {"message": message, "type": error_type, "param": param, "code": code}},
        status_code=status,
    )


@app.middleware("http")
async def _body_limit(request: Request, call_next):
    length = request.headers.get("content-length")
    if _body_too_large(length):
        return _openai_error("Request body is too large", 413, code="request_too_large")
    return await call_next(request)


def _body_too_large(content_length: str | None) -> bool:
    return bool(content_length and content_length.isdigit() and int(content_length) > MAX_BODY_BYTES)


async def _read_json_body(request: Request) -> dict:
    chunks, size = [], 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > MAX_BODY_BYTES:
            raise OverflowError
        chunks.append(chunk)
    try:
        data = json.loads(b"".join(chunks))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError("Invalid JSON body") from exc
    if not isinstance(data, dict):
        raise ValueError("Request body must be a JSON object")
    return data


def _validate_request(body: dict) -> tuple[str, str | None] | None:
    if body.get("model") != MODEL_ID:
        return "Only model 'auto' is supported", "model"
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        return "'messages' must be a non-empty array", "messages"
    roles = {"system", "developer", "user", "assistant", "tool", "function"}
    for index, message in enumerate(messages):
        if not isinstance(message, dict) or message.get("role") not in roles:
            return "Each message must contain a supported role", f"messages.{index}.role"
        content = message.get("content")
        if content is not None and not isinstance(content, (str, list)):
            return "Message content must be text, an array, or null", f"messages.{index}.content"
    checks = {
        "stream": bool, "temperature": (int, float), "top_p": (int, float),
        "n": int, "max_tokens": int, "max_completion_tokens": int,
        "presence_penalty": (int, float), "frequency_penalty": (int, float),
        "tools": list, "tool_choice": (str, dict), "response_format": dict,
        "stream_options": dict, "seed": int, "stop": (str, list),
    }
    for name, expected in checks.items():
        if name in body and (isinstance(body[name], bool) and expected is not bool or not isinstance(body[name], expected)):
            return f"'{name}' has an invalid type", name
    for name in ("n", "max_tokens", "max_completion_tokens"):
        if name in body and body[name] <= 0:
            return f"'{name}' must be greater than zero", name
    # Unknown keys are intentionally preserved as reviewed provider extensions.
    return None


def _route_headers(decision, score, backend, request_id, supra_complexity, supra_ms):
    headers = {
        "x-request-id": request_id, "x-route-decision": decision,
        "x-route-score": f"{score:.4f}", "x-route-model": backend["model"],
        "x-route-router": ROUTER_NAME, "x-route-fallback": "false",
        "x-route-attempts": "1",
    }
    if supra_complexity is not None:
        headers["x-route-supra-complexity"] = str(supra_complexity)
    if supra_ms is not None:
        headers["x-route-supra-ms"] = str(supra_ms)
    return headers


def _upstream_request(backend: dict, body: dict) -> httpx.Request:
    assert _client is not None
    return _client.build_request(
        "POST", backend["base"].rstrip("/") + "/chat/completions",
        json=_build_outgoing_body(body, backend),
        headers={"Authorization": f"Bearer {backend['key']}", "Content-Type": "application/json"},
    )


async def _send(backend: dict, body: dict, *, stream: bool) -> httpx.Response:
    if _client is None:
        raise RuntimeError("router is not ready")
    return await _client.send(_upstream_request(backend, body), stream=stream)


def _retryable(status: int) -> bool:
    return status in RETRY_STATUSES


async def _open_with_failover(body: dict, decision: str, deadline: float, *, stream: bool):
    attempts = []
    routes = (decision, "cheap" if decision == "expensive" else "expensive")
    for index, current in enumerate(routes):
        backend = _backend_for(current)
        try:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError
            async with asyncio.timeout(remaining):
                response = await _send(backend, body, stream=stream)
                data = None
                if not stream or response.status_code != 200:
                    data = _safe_json(await response.aread())
            refusal = data is not None and _is_refusal(response.status_code, data)
            attempts.append((current, backend, response.status_code, "refusal" if refusal else None))
            retry = refusal or _retryable(response.status_code)
            if not retry or index == len(routes) - 1:
                return current, backend, response, attempts
            await response.aclose()
        except (httpx.TransportError, TimeoutError, asyncio.TimeoutError) as exc:
            attempts.append((current, backend, None, type(exc).__name__))
    return attempts[-1][0], attempts[-1][1], None, attempts


def _log_attempts(attempts, prompt: str, score: float, request_id: str, occurrence_id: str) -> None:
    for index, (decision, backend, status, error) in enumerate(attempts, 1):
        _log(decision, score, backend["model"], prompt, None, request_id=request_id,
             occurrence_id=occurrence_id, record_type="attempt", attempt=index,
             status=status, error=error)


@app.get("/healthz")
async def healthz():
    return {"ok": _READY, "router": ROUTER_NAME, "threshold": THRESHOLD,
            "backend": "direct", "ready": _READY, "cache": {**_cache_metrics, "entries": len(_resp_cache), "bytes": _cache_bytes}}


@app.get("/v1/models")
async def list_models():
    return {"object": "list", "data": [{
        "id": MODEL_ID, "object": "model", "owned_by": "routellm",
        "context_window": await _get_context_window(), "max_tokens": ROUTELLM_MAX_TOKENS,
    }]}


MAX_SSE_EVENT_BYTES = int(os.environ.get("ROUTELLM_MAX_SSE_EVENT_BYTES", str(1024 * 1024)))


def _sse_boundary(buffer: bytearray) -> int | None:
    """Return the end of the first blank SSE line for CR, LF, or CRLF."""
    line_start = 0
    index = 0
    while index < len(buffer):
        if buffer[index] not in (10, 13):
            index += 1
            continue
        if buffer[index] == 13 and index + 1 == len(buffer):
            return None  # wait to distinguish CR from a split CRLF
        end = index + (2 if buffer[index:index + 2] == b"\r\n" else 1)
        if index == line_start:
            return end
        line_start = end
        index = end
    return None


async def _iter_sse_events(response: httpx.Response):
    """Yield exact SSE event frames while bounding one provider event."""
    buffer = bytearray()
    async for chunk in response.aiter_bytes():
        buffer.extend(chunk)
        while (end := _sse_boundary(buffer)) is not None:
            yield bytes(buffer[:end])
            del buffer[:end]
        if len(buffer) > MAX_SSE_EVENT_BYTES:
            raise ValueError("upstream SSE event is too large")
    if buffer:
        if len(buffer) > MAX_SSE_EVENT_BYTES:
            raise ValueError("upstream SSE event is too large")
        yield bytes(buffer)


def _sse_data(event: bytes) -> str | None:
    values = []
    for raw_line in event.splitlines():
        if raw_line.startswith(b"data:"):
            value = raw_line[5:]
            if value.startswith(b" "):
                value = value[1:]
            values.append(value.decode("utf-8", errors="replace"))
    return "\n".join(values) if values else None


def _possible_refusal_prefix(text: str) -> bool:
    value = text.strip().lower()
    leads = ("i", "i'", "i’m", "i am", "i cannot", "i can't", "i’m sorry",
             "i'm sorry", "sorry", "unfortunately", "as an ai", "cannot")
    return any(lead.startswith(value) or value.startswith(lead) for lead in leads)


async def _prefetch_sse(events, deadline: float):
    """Inspect a small prefix for standard delta-based refusals before release."""
    prefix, text_parts = [], []
    prefix_bytes = data_events = 0
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError
        try:
            async with asyncio.timeout(remaining):
                event = await anext(events)
        except StopAsyncIteration:
            return prefix, False
        prefix.append(event)
        prefix_bytes += len(event)
        if prefix_bytes >= MAX_SSE_EVENT_BYTES or data_events >= 8:
            return prefix, False
        data = _sse_data(event)
        if data is None:
            continue
        data_events += 1
        if data.strip() == "[DONE]":
            return prefix, False
        try:
            payload = json.loads(data)
        except json.JSONDecodeError:
            return prefix, False
        if _is_refusal(200, payload):
            return prefix, True
        choices = payload.get("choices") or []
        finish = None
        if choices and isinstance(choices[0], dict):
            choice = choices[0]
            finish = choice.get("finish_reason")
            delta = choice.get("delta") or {}
            if isinstance(delta, dict):
                for field in ("content", "reasoning_content"):
                    value = delta.get(field)
                    if isinstance(value, str):
                        text_parts.append(value)
        combined = "".join(text_parts)
        if _REFUSAL_RE.search(combined) or finish == "content_filter":
            return prefix, True
        if finish is not None:
            return prefix, False
        if combined and not _possible_refusal_prefix(combined):
            return prefix, False


def _stream_error(message: str, code: str, request_id: str) -> bytes:
    return ("data: " + json.dumps({"error": {"message": message, "type": "upstream_error", "code": code},
                                   "request_id": request_id}) + "\n\n").encode()


@app.post("/v1/chat/completions")
async def chat_completions(
    request: Request, authorization: str | None = Header(default=None),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
):
    request_id = request.headers.get("x-request-id") or f"req_{uuid.uuid4().hex}"
    if not _authorize(authorization):
        return _openai_error("Invalid API key", 401, error_type="authentication_error", code="invalid_api_key")
    try:
        body = await _read_json_body(request)
    except OverflowError:
        return _openai_error("Request body is too large", 413, code="request_too_large")
    except ValueError as exc:
        return _openai_error(str(exc), 400, code="invalid_json")
    invalid = _validate_request(body)
    if invalid:
        return _openai_error(invalid[0], 400, param=invalid[1], code="invalid_request")

    cache_key = _cache_key(body, idempotency_key)
    cached = _cache_get(cache_key)
    if cached is not None:
        media = "text/event-stream" if body.get("stream") else "application/json"
        result = (cached, 200, media, {"x-route-cache": "hit"})
        return _replayed_response(result, request_id)
    leader, inflight = await _claim_inflight(cache_key)
    if not leader:
        result = await asyncio.shield(inflight)
        if result is not None:
            return _replayed_response(result, request_id)
        leader, inflight = await _claim_inflight(cache_key)

    upstream = None
    try:
        occurrence_id = uuid.uuid4().hex
        prompt = _extract_prompt(body)
        if prompt.strip():
            decision, score, supra_complexity, supra_ms = await asyncio.to_thread(_decide, prompt)
        else:
            decision, score, supra_complexity, supra_ms = "cheap", 0.0, None, None
        backend = _backend_for(decision)
        _record_and_detect_retry(_request_hash(body), decision, backend["model"], _prompt_hash(prompt),
                                 request_id, occurrence_id)
        headers = _route_headers(decision, score, backend, request_id, supra_complexity, supra_ms)
        deadline = time.monotonic() + TIMEOUT_S
        selected, backend, upstream, attempts = await _open_with_failover(
            body, decision, deadline, stream=bool(body.get("stream")))

        prefix = []
        events = None
        if upstream is not None and body.get("stream") and upstream.status_code == 200:
            try:
                events = _iter_sse_events(upstream)
                prefix, refusal = await _prefetch_sse(events, deadline)
                if refusal and len(attempts) < 2:
                    await upstream.aclose()
                    selected = "cheap" if selected == "expensive" else "expensive"
                    backend = _backend_for(selected)
                    remaining = deadline - time.monotonic()
                    async with asyncio.timeout(max(0, remaining)):
                        upstream = await _send(backend, body, stream=True)
                    attempts.append((selected, backend, upstream.status_code, "refusal_fallback"))
                    events = _iter_sse_events(upstream) if upstream.status_code == 200 else None
                    prefix, _ = await _prefetch_sse(events, deadline) if events is not None else ([], False)
            except (httpx.TransportError, TimeoutError, asyncio.TimeoutError, ValueError) as exc:
                if upstream is not None:
                    await upstream.aclose()
                attempts.append((selected, backend, None, type(exc).__name__))
                upstream = None

    except BaseException:
        _finish_inflight_nowait(cache_key, inflight, None)
        if upstream is not None:
            await asyncio.shield(upstream.aclose())
        raise

    headers.update({"x-route-decision": selected, "x-route-model": backend["model"],
                    "x-route-fallback": str(selected != decision).lower(), "x-route-attempts": str(len(attempts))})
    _log_attempts(attempts, prompt, score, request_id, occurrence_id)
    if upstream is None:
        _log(selected, score, backend["model"], prompt, None, request_id=request_id,
             occurrence_id=occurrence_id, record_type="decision", status=None,
             error="upstream_unavailable", attempts=len(attempts))
        _log_outcome(_prompt_hash(prompt), "upstream_error", request_id=request_id,
                     decision_occurrence_id=occurrence_id, attempts=len(attempts))
        await _finish_inflight(cache_key, inflight, None)
        return _openai_error("Upstream providers were unavailable", 502, error_type="upstream_error", code="upstream_unavailable")

    if not body.get("stream"):
        try:
            remaining = deadline - time.monotonic()
            async with asyncio.timeout(max(0, remaining)):
                content = await upstream.aread()
        except (httpx.TransportError, TimeoutError, asyncio.TimeoutError):
            await upstream.aclose()
            _log_outcome(_prompt_hash(prompt), "upstream_error", request_id=request_id,
                         decision_occurrence_id=occurrence_id, model=backend["model"])
            _log(selected, score, backend["model"], prompt, None, request_id=request_id,
                 occurrence_id=occurrence_id, record_type="decision", status=504,
                 error="upstream_timeout", attempts=len(attempts))
            await _finish_inflight(cache_key, inflight, None)
            return _openai_error("Upstream response timed out", 504, error_type="upstream_error", code="upstream_timeout")
        finally:
            await upstream.aclose()
        data = _safe_json(content)
        if upstream.status_code != 200:
            _log_outcome(_prompt_hash(prompt), "upstream_error", request_id=request_id,
                         decision_occurrence_id=occurrence_id, model=backend["model"], status=upstream.status_code)
        if _length_truncated(data):
            _log_outcome(_prompt_hash(prompt), "truncated", request_id=request_id,
                         decision_occurrence_id=occurrence_id, model=backend["model"])
        if upstream.status_code == 200:
            _cache_put(cache_key, body, content)
        _log(selected, score, backend["model"], prompt, None, supra_complexity, supra_ms,
             cost_usd=_extract_cost(data), usage=data.get("usage"), request_id=request_id,
             occurrence_id=occurrence_id, record_type="decision", attempts=len(attempts), status=upstream.status_code)
        result = ((content, upstream.status_code, "application/json", headers)
                  if (cache_key and upstream.status_code == 200 and len(content) <= RESP_CACHE_MAX_BYTES
                      and _response_replay_safe(body, content)) else None)
        await _finish_inflight(cache_key, inflight, result)
        return Response(content=content, status_code=upstream.status_code, media_type="application/json", headers=headers)

    if upstream.status_code != 200 or events is None:
        content = await upstream.aread()
        await upstream.aclose()
        _log_outcome(_prompt_hash(prompt), "upstream_error", request_id=request_id,
                     decision_occurrence_id=occurrence_id, model=backend["model"], status=upstream.status_code)
        _log(selected, score, backend["model"], prompt, None, request_id=request_id,
             occurrence_id=occurrence_id, record_type="decision", status=upstream.status_code)
        await _finish_inflight(cache_key, inflight, None)
        return Response(content=content, status_code=upstream.status_code, media_type="application/json", headers=headers)

    async def event_stream():
        cache_parts: list[bytes] | None = [] if cache_key is not None else None
        cache_size = 0
        saw_done = saw_finish = saw_length = False
        emitted_error = False
        usage: dict = {}
        started = time.monotonic()

        def remember(event: bytes) -> None:
            nonlocal cache_parts, cache_size
            if cache_parts is None:
                return
            cache_size += len(event)
            if cache_size > RESP_CACHE_MAX_BYTES:
                cache_parts = None
            else:
                cache_parts.append(event)

        def track(event: bytes) -> bool:
            nonlocal saw_done, saw_finish, saw_length
            data_text = _sse_data(event)
            if data_text is None:
                return False
            if data_text.strip() == "[DONE]":
                saw_done = True
                return True
            try:
                payload = json.loads(data_text)
            except json.JSONDecodeError:
                return False
            choices = payload.get("choices") or []
            if choices and choices[0].get("finish_reason") is not None:
                saw_finish = True
                saw_length = choices[0].get("finish_reason") == "length"
            if isinstance(payload.get("usage"), dict):
                usage.update(payload["usage"])
            return False

        try:
            remaining = deadline - time.monotonic()
            async with asyncio.timeout(max(0, remaining)):
                async def all_events():
                    for event in prefix:
                        yield event
                    async for event in events:
                        yield event
                async for event in all_events():
                    if await request.is_disconnected():
                        raise asyncio.CancelledError
                    done = track(event)
                    if done and not saw_finish:
                        saw_done = False
                        emitted_error = True
                        yield _stream_error("Upstream stream ended without a finish reason", "upstream_truncated", request_id)
                        break
                    remember(event)
                    yield event
                    if done:
                        break
            if not saw_done or not saw_finish:
                _log_outcome(_prompt_hash(prompt), "truncated", request_id=request_id,
                             decision_occurrence_id=occurrence_id, model=backend["model"], abrupt_eof=True)
                if not saw_done and not emitted_error:
                    yield _stream_error("Upstream stream ended before completion", "upstream_truncated", request_id)
            elif saw_length:
                _log_outcome(_prompt_hash(prompt), "truncated", request_id=request_id,
                             decision_occurrence_id=occurrence_id, model=backend["model"])
            elif cache_parts is not None:
                content = b"".join(cache_parts)
                _cache_put(cache_key, body, content)
        except asyncio.CancelledError:
            _log_outcome(_prompt_hash(prompt), "disconnected", request_id=request_id,
                         decision_occurrence_id=occurrence_id, model=backend["model"])
            raise
        except (httpx.TransportError, TimeoutError, asyncio.TimeoutError, ValueError):
            _log_outcome(_prompt_hash(prompt), "upstream_error", request_id=request_id,
                         decision_occurrence_id=occurrence_id, model=backend["model"])
            yield _stream_error("Upstream stream failed", "upstream_transport_error", request_id)
        finally:
            await upstream.aclose()
            _log(selected, score, backend["model"], prompt, int((time.monotonic() - started) * 1000),
                 supra_complexity, supra_ms, usage=usage or None, request_id=request_id,
                 occurrence_id=occurrence_id, record_type="decision",
                 attempts=len(attempts), completed=saw_done and saw_finish)
            result = None
            if cache_parts is not None and saw_done and saw_finish:
                content = b"".join(cache_parts)
                if _response_replay_safe(body, content):
                    result = (content, 200, "text/event-stream", headers)
            await _finish_inflight(cache_key, inflight, result)

    return StreamingResponse(event_stream(), media_type="text/event-stream", headers=headers)


def _bind_is_loopback(host: str) -> bool:
    return host in {"127.0.0.1", "::1", "localhost"}


def _validate_bind_security() -> None:
    if not _bind_is_loopback(HOST) and (not os.environ.get("ROUTELLM_KEY") or SERVER_KEY == "sk-route-local"):
        raise RuntimeError("non-loopback binding requires an externally supplied, non-default ROUTELLM_KEY")


if __name__ == "__main__":
    import uvicorn
    _validate_bind_security()
    print(
        "effective config: "
        f"router={ROUTER_NAME} threshold={THRESHOLD} supra={SUPRA_ENABLED} "
        f"supra_threshold={SUPRA_THRESHOLD} supra_min_score={SUPRA_MIN_SCORE} "
        f"expensive={EXPENSIVE['model']} cheap={CHEAP['model']} port={PORT}",
        flush=True,
    )
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")