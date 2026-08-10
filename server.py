"""LLM coding-router server.

Exposes one OpenAI-compatible model ("auto") on explicit Chat Completions
and Responses endpoints. Requests are routed by the Supra-Router-51M complexity
gate (default; see ROUTELLM_ROUTER below for the legacy RouteLLM MF mode).
Responses requests are restricted to OpenAI models via OpenRouter or the
Cloudflare gateway; Chat Completions behavior remains independent.

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
import hmac
import json
import os
import re
import secrets
import threading
import time
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from fastapi import FastAPI, Header, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

# Defaults mirror llm-router.sh (canonical source, calibrated there) —
# keep in sync so bare `python server.py` behaves identically to the launcher.
HOST = os.environ.get("ROUTELLM_HOST", "127.0.0.1")
PORT = int(os.environ.get("ROUTELLM_PORT", "5500"))
SERVER_KEY = os.environ.get("ROUTELLM_KEY", "sk-route-local")
THRESHOLD = float(os.environ.get("ROUTELLM_THRESHOLD", "0.2"))
ROUTER_NAME = os.environ.get("ROUTELLM_ROUTER", "supra")
# In supra mode the Supra-Router-51M complexity gate is the primary signal
# (the MF score has ~zero separation on this workload: AUC 0.52 vs 0.66 for
# Supra over 428 labeled prompts; the MF gate only adds waste on top of Supra).
# MF scoring is then optional observability only.
SCORE_WITH_MF = os.environ.get("ROUTELLM_SCORE_WITH_MF", "0").lower() in {"1", "true", "yes", "on"}
SUPRA_ENABLED = os.environ.get("ROUTELLM_USE_SUPRA", "1") != "0"
SUPRA_THRESHOLD = int(os.environ.get("ROUTELLM_SUPRA_THRESHOLD", "3"))
SUPRA_MIN_SCORE = float(os.environ.get("ROUTELLM_SUPRA_MIN_SCORE", "0"))
ROUTELLM_CONTEXT_WINDOW = os.environ.get("ROUTELLM_CONTEXT_WINDOW", "auto")
ROUTELLM_MAX_TOKENS = int(os.environ.get("ROUTELLM_MAX_TOKENS", "131072"))
MODEL_ID = "auto"

GATEWAY_BASE = os.environ.get(
    "ROUTELLM_GATEWAY_BASE", "https://unified-ai-gateway.siddsantham.workers.dev/v1",
)
GATEWAY_KEY = os.environ.get("AI_GATEWAY_API_KEY") or os.environ.get("MANTIS_GATEWAY_API_KEY", "")
GATEWAY_HOST = "unified-ai-gateway.siddsantham.workers.dev"


def _gateway_force_bases() -> bool:
    profile = os.environ.get("ROUTELLM_ENDPOINT_PROFILE", "").lower()
    if profile == "direct":
        return False
    if profile == "cloudflare":
        return True
    mode = os.environ.get("ROUTELLM_GATEWAY_MODE", "").lower()
    valid = {"1", "true", "yes", "on", "cloudflare", "0", "false", "no", "off"}
    if mode and mode not in valid:
        raise ValueError("invalid ROUTELLM_GATEWAY_MODE")
    if mode in {"1", "true", "yes", "on", "cloudflare"}:
        return True
    if mode in {"0", "false", "no", "off"}:
        return False
    configured = (os.environ.get("EXPENSIVE_BASE"), os.environ.get("CHEAP_BASE"),
                  os.environ.get("MIDDLE_BASE"))
    # If one configured tier already uses the gateway, normalize all tiers to
    # it rather than accidentally sending a gateway credential to a direct URL.
    return any(GATEWAY_HOST in (base or "").lower() for base in configured) or not any(configured)


def _gateway_requested() -> bool:
    return _gateway_force_bases()


def _base(name: str, direct_default: str, gateway: bool) -> str:
    if gateway and _gateway_force_bases():
        return GATEWAY_BASE
    value = os.environ.get(name)
    if value is not None:
        return value
    return GATEWAY_BASE if gateway else direct_default


def _key(name: str, base: str) -> str:
    value = os.environ.get(name, "")
    profile = os.environ.get("ROUTELLM_ENDPOINT_PROFILE", "").lower()
    if profile != "direct" and (_gateway_requested() and base == GATEWAY_BASE or GATEWAY_HOST in base.lower()):
        return GATEWAY_KEY or value
    return value


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _optional_int(name: str) -> int | None:
    value = os.environ.get(name)
    return int(value) if value else None


def _backend(name: str, *, base: str, key: str, model: str, effort: str,
             max_tokens: int | None = None, usage_include: bool | None = None) -> dict:
    # OpenRouter accepts usage.include. OpenCode Go and Modal may reject
    # unknown request fields, so gateway usage is enabled only for namespaced
    # OpenRouter catalog IDs unless explicitly overridden.
    openrouter_model = model.split("/", 1)[0] in {"openai", "google", "anthropic", "openrouter"}
    default_usage = "openrouter.ai" in base or (GATEWAY_HOST in base and openrouter_model)
    usage_name = f"ROUTELLM_{name.upper()}_USAGE_INCLUDE"
    return {
        "tier": name, "base": base, "key": key, "model": model,
        "effort": effort, "max_tokens": max_tokens,
        "usage_include": (_env_bool(usage_name, default_usage)
                           if usage_include is None else usage_include),
    }


_GATEWAY_SELECTED = _gateway_requested()
EXPENSIVE_BASE = _base("EXPENSIVE_BASE", "https://openrouter.ai/api/v1", _GATEWAY_SELECTED)
CHEAP_BASE = _base("CHEAP_BASE", "https://opencode.ai/zen/go/v1", _GATEWAY_SELECTED)
MIDDLE_BASE = _base("MIDDLE_BASE", "", _GATEWAY_SELECTED)
EXPENSIVE = _backend(
    "expensive", base=EXPENSIVE_BASE, key=_key("EXPENSIVE_KEY", EXPENSIVE_BASE),
    model=os.environ.get("EXPENSIVE_MODEL", "openai/gpt-5.6-sol"),
    effort=os.environ.get("EXPENSIVE_REASONING_EFFORT", "medium"),
    max_tokens=_optional_int("EXPENSIVE_MAX_TOKENS"),
)
CHEAP = _backend(
    "cheap", base=CHEAP_BASE, key=_key("CHEAP_KEY", CHEAP_BASE),
    model=os.environ.get("CHEAP_MODEL", "deepseek-v4-flash"),
    effort=os.environ.get("CHEAP_REASONING_EFFORT", "none"),
    max_tokens=int(os.environ.get("CHEAP_MAX_TOKENS", str(ROUTELLM_MAX_TOKENS))),
)
MIDDLE = _backend(
    "middle", base=MIDDLE_BASE, key=_key("MIDDLE_KEY", MIDDLE_BASE),
    model=os.environ.get("MIDDLE_MODEL", "kimi-k3"),
    effort=os.environ.get("MIDDLE_REASONING_EFFORT", "medium"),
    max_tokens=int(os.environ.get("MIDDLE_MAX_TOKENS", str(ROUTELLM_MAX_TOKENS))),
)
BACKENDS = {"cheap": CHEAP, "middle": MIDDLE, "expensive": EXPENSIVE}
TIER_ORDER = {"cheap": 0, "middle": 1, "expensive": 2}
MIDDLE_MIN_COMPLEXITY = int(os.environ.get("ROUTELLM_MIDDLE_MIN_COMPLEXITY", "3"))
MIDDLE_CONFIGURED = bool(MIDDLE["base"])
EXPENSIVE_MIN_COMPLEXITY = int(os.environ.get(
    "ROUTELLM_EXPENSIVE_MIN_COMPLEXITY",
    str(SUPRA_THRESHOLD + (1 if MIDDLE_CONFIGURED else 0)),
))


# One async pool is created and closed by the ASGI lifespan.
TIMEOUT_S = float(os.environ.get("ROUTELLM_TIMEOUT_S", "600"))
_client: httpx.AsyncClient | None = None
RETRY_STATUSES = {429, 500, 502, 503, 504}

# Reject oversized bodies before they are buffered into memory (413).
MAX_BODY_BYTES = int(os.environ.get("ROUTELLM_MAX_BODY_BYTES", str(50 * 1024 * 1024)))

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
    backends = [EXPENSIVE, CHEAP] + ([MIDDLE] if MIDDLE_CONFIGURED else [])
    values = await asyncio.gather(*(
        _fetch_model_context_length(b["base"], b["key"], b["model"])
        for b in backends
    ))
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


def _extract_responses_prompt(body: dict) -> str:
    value = body.get("input")
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        return ""

    def item_text(item: dict) -> list[str]:
        content = item.get("content")
        if isinstance(content, str):
            return [content]
        if not isinstance(content, list):
            return []
        return [
            part["text"] for part in content
            if isinstance(part, dict) and part.get("type") in {"input_text", "text"}
            and isinstance(part.get("text"), str)
        ]

    for item in reversed(value):
        if isinstance(item, dict) and item.get("role") == "user":
            user_texts = item_text(item)
            if user_texts:
                return " ".join(user_texts)
    return " ".join(text for item in value if isinstance(item, dict) for text in item_text(item))


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
    from transformers import StoppingCriteria

    class _ComplexitySeen(StoppingCriteria):
        """Stop generation as soon as the 'Complexity:' field is emitted.

        Supra emits 'Domain: ... | Complexity: N | ...' and the complexity
        digit appears within the first ~10 generated tokens. Greedy decode is
        deterministic, so stopping early yields the exact same parsed value
        while cutting median inference from ~480ms to ~160ms (3x)."""

        def __call__(self, input_ids, scores, **kwargs) -> bool:
            text = tokenizer.decode(input_ids[0][-24:], skip_special_tokens=True)
            # Stop only once the complexity digit itself has been emitted;
            # stopping at the bare "Complexity:" prefix would parse as 0.
            return re.search(r"Complexity:\s*\d", text) is not None

    fmt = f"Task: {prompt}\nAnalysis: "
    inputs = tokenizer(fmt, return_tensors="pt", truncation=True,
                       max_length=tokenizer.model_max_length)
    import torch
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=128, do_sample=False,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            stopping_criteria=[_ComplexitySeen()],
        )
    supra_ms = int((time.time() - t0) * 1000)
    gen = tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True,
    ).strip()
    return _parse_supra_complexity(gen), supra_ms


def _decide_mf(trimmed_prompt: str) -> tuple[str, float, int | None, int | None]:
    """Legacy MF+Supra gate: MF score >= threshold, then Supra on the band below."""
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


def _decide_uncached(trimmed_prompt: str) -> tuple[str, float, int | None, int | None]:
    try:
        if ROUTER_NAME == "supra":
            # Supra-first: complexity >= threshold is the only gate. The MF
            # score is computed only for observability when SCORE_WITH_MF is
            # set; it never influences the decision. If Supra is unavailable
            # (missing torch/transformers), fall back to the legacy MF gate.
            supra_complexity = supra_ms = None
            score = None
            try:
                supra_complexity, supra_ms = _supra_complexity(trimmed_prompt)
            except Exception as err:
                print(f"Supra scoring failed ({err}); falling back to MF gate", flush=True)
            if SCORE_WITH_MF:
                try:
                    r = _load_router()
                    score = float(r.calculate_strong_win_rate(trimmed_prompt))
                except Exception as err:
                    print(f"MF scoring failed ({err})", flush=True)
            if supra_complexity is not None:
                expensive_cutoff = EXPENSIVE_MIN_COMPLEXITY
                if supra_complexity >= expensive_cutoff:
                    tier = "expensive"
                elif MIDDLE_CONFIGURED and supra_complexity >= MIDDLE_MIN_COMPLEXITY:
                    tier = "middle"
                else:
                    tier = "cheap"
                return tier, score, supra_complexity, supra_ms
            return _decide_mf(trimmed_prompt)
        return _decide_mf(trimmed_prompt)
    except Exception as err:
        print(f"Router decision failed ({err}); defaulting to expensive", flush=True)
        return tuple(["expensive", 1.0, None, None])  # type: ignore


@lru_cache(maxsize=256)
def _decide_cached(trimmed_prompt: str) -> tuple[str, float, int | None, int | None]:
    return _decide_uncached(trimmed_prompt)


def _decide(prompt: str, session_id: str | None = None) -> tuple[str, float, int | None, int | None]:
    # Session traffic does not consult the global prompt store: short turns
    # like "Proceed" are not transferable across coding sessions. Once a live
    # session exists, continuation turns also skip Supra/MF scoring entirely.
    if session_id is not None and _is_continuation(prompt):
        state = _session_get(session_id)
        if state is not None:
            return state["tier"], None, state.get("last_complexity"), None
    if session_id is None:
        pinned = _store_pinned(_prompt_hash(prompt))
        if pinned is not None:
            return pinned.get("decision", "cheap"), pinned.get("score"), None, None
    trimmed_prompt = prompt[-15000:] if len(prompt) > 15000 else prompt
    return _decide_cached(trimmed_prompt)


def _backend_for(decision: str) -> dict:
    if decision == "middle" and not MIDDLE_CONFIGURED:
        return EXPENSIVE if EXPENSIVE["base"] else CHEAP
    return BACKENDS.get(decision, EXPENSIVE)


def _supports_responses(backend: dict) -> bool:
    """Responses is supported only by OpenAI models via OR or our gateway."""
    base = str(backend.get("base", ""))
    if not base or not str(backend.get("model", "")).lower().startswith("openai/"):
        return False
    try:
        host = (urlsplit(base).hostname or "").lower()
    except ValueError:
        return False
    return host in {"openrouter.ai", GATEWAY_HOST}


def _responses_routes(decision: str) -> tuple[str, ...]:
    """Return only configured compatible tiers, never below the selected tier."""
    rank = TIER_ORDER.get(decision, TIER_ORDER["expensive"])
    routes = []
    seen = set()
    for tier in sorted(BACKENDS, key=TIER_ORDER.get):
        if TIER_ORDER[tier] < rank or (tier == "middle" and not MIDDLE_CONFIGURED):
            continue
        backend = _backend_for(tier)
        identity = (backend.get("base"), backend.get("model"))
        if identity in seen or not _supports_responses(backend):
            continue
        routes.append(tier)
        seen.add(identity)
    return tuple(routes)


def _responses_tier(decision: str) -> str | None:
    routes = _responses_routes(decision)
    return routes[0] if routes else None


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


def _build_responses_body(body: dict, backend: dict) -> dict:
    out_body = dict(body)
    out_body["model"] = backend["model"]
    if isinstance(out_body.get("max_output_tokens"), int) and backend.get("max_tokens"):
        out_body["max_output_tokens"] = min(out_body["max_output_tokens"], backend["max_tokens"])
    if backend.get("effort") and "reasoning" not in out_body:
        out_body["reasoning"] = {"effort": backend["effort"]}
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
    occurrence_id: str | None = None, api_format: str = "chat", **detail,
):
    row = {
        "ts": time.time(), "request_id": request_id,
        "occurrence_id": occurrence_id or uuid.uuid4().hex,
        "router": ROUTER_NAME, "threshold": THRESHOLD, "api_format": api_format,
        "score": round(score, 4) if isinstance(score, (int, float)) else None,
        "supra_complexity": supra_complexity, "supra_ms": supra_ms,
        "decision": decision, "model": backend_model, "ttfb_ms": ttfb_ms,
        "prompt_hash": _prompt_hash(prompt), **detail,
    }
    if cost_usd is not None:
        row["cost_usd"] = cost_usd
    if isinstance(usage, dict) and usage:
        row["usage"] = usage
        prompt_tokens = usage.get("prompt_tokens", usage.get("input_tokens"))
        details = usage.get("prompt_tokens_details") or usage.get("input_tokens_details") or {}
        cached_tokens = details.get("cached_tokens", usage.get("prompt_cache_hit_tokens"))
        if isinstance(prompt_tokens, (int, float)):
            row["prompt_tokens"] = prompt_tokens
        if isinstance(cached_tokens, (int, float)):
            row["cached_tokens"] = cached_tokens
            row["fresh_tokens"] = max(0, prompt_tokens - cached_tokens) if isinstance(prompt_tokens, (int, float)) else None
            row["cache_hit_ratio"] = (cached_tokens / prompt_tokens if prompt_tokens else None)
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

# --- Session affinity -------------------------------------------------------
# Session state is intentionally in-memory. A restart must not carry a model
# choice into a new context, while the learned prompt store remains persistent.
_CONTINUATION_RE = re.compile(
    r"^(?:ok(?:ay)?[,. ]*)?(?:proceed|continue|go ahead|do (?:it|that)|yes|yep|sure|"
    r"run (?:it|them|the tests)(?: again)?|try again|fix (?:it|that)|next)(?:[.! ]*)$", re.I,
)
_NEW_TASK_RE = re.compile(
    r"^(?:new task|different task|unrelated|switching topics?|on another topic)\b", re.I,
)
SESSION_TTL_S = float(os.environ.get("ROUTELLM_SESSION_TTL_S", "3600"))
SESSION_STATE_MAX = int(os.environ.get("ROUTELLM_SESSION_STATE_MAX", "4096"))
_session_state: OrderedDict[str, dict] = OrderedDict()
_session_lock = threading.Lock()
_STICKY_REASONS = frozenset({
    "continuation_sticky", "same_tier", "upgrade_hysteresis", "downgrade_hysteresis",
})


def _is_continuation(prompt: str) -> bool:
    value = prompt.strip()
    return len(value) <= 200 and "```" not in value and bool(_CONTINUATION_RE.fullmatch(value))


def _session_id(body: dict, request: Request) -> tuple[str | None, str | None]:
    """Extract an opaque, source-namespaced session identity."""
    raw, source = request.headers.get("x-route-session"), "header"
    if not raw and isinstance(body.get("metadata"), dict):
        raw, source = body["metadata"].get("session_id"), "metadata"
    if (not raw and os.environ.get("ROUTELLM_SESSION_FROM_USER", "0").lower()
            in {"1", "true", "yes", "on"} and isinstance(body.get("user"), str)):
        raw, source = body["user"], "user"
    if not isinstance(raw, str) or not raw or len(raw.encode()) > 256:
        return None, None
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in raw):
        return None, None
    secret = SERVER_KEY.encode("utf-8", errors="replace")
    digest = hmac.new(secret, (source + "\0" + raw).encode(), hashlib.sha256).hexdigest()[:24]
    return digest, source


def _session_get(session_id: str | None) -> dict | None:
    if not session_id:
        return None
    with _session_lock:
        state = _session_state.get(session_id)
        if state is None:
            return None
        if time.time() - state["last_seen"] > SESSION_TTL_S:
            _session_state.pop(session_id, None)
            return None
        _session_state.move_to_end(session_id)
        return dict(state)


def _session_route(session_id: str | None, prompt: str, proposed: str,
                   complexity: int | None, score: float | None = None,
                   *, new_task: bool = False) -> tuple[str, str]:
    state = _session_get(session_id)
    if not state:
        return proposed, "new_session"
    if new_task or _NEW_TASK_RE.match(prompt.strip()):
        return proposed, "new_task"
    current = state["tier"]
    if _is_continuation(prompt):
        return current, "continuation_sticky"
    if proposed == current:
        return current, "same_tier"
    current_rank, proposed_rank = TIER_ORDER[current], TIER_ORDER.get(proposed, 2)
    if proposed_rank < current_rank:
        return current, "downgrade_hysteresis"
    expensive_cutoff = EXPENSIVE_MIN_COMPLEXITY
    if ((complexity is not None and complexity >= expensive_cutoff)
            or (score is not None and score >= THRESHOLD)):
        return proposed, "strong_upgrade"
    return current, "upgrade_hysteresis"


def _session_note(session_id: str | None, tier: str, complexity: int | None,
                  usage: dict | None = None) -> None:
    if not session_id:
        return
    now = time.time()
    with _session_lock:
        prior = _session_state.get(session_id, {})
        state = {
            "tier": tier, "last_seen": now, "last_complexity": complexity,
            "turns": int(prior.get("turns", 0)) + 1,
        }
        if isinstance(usage, dict) and usage:
            details = usage.get("prompt_tokens_details") or usage.get("input_tokens_details") or {}
            state["prompt_tokens"] = usage.get("prompt_tokens", usage.get("input_tokens"))
            state["cached_tokens"] = details.get("cached_tokens", usage.get("prompt_cache_hit_tokens"))
        _session_state[session_id] = state
        _session_state.move_to_end(session_id)
        while len(_session_state) > SESSION_STATE_MAX:
            _session_state.popitem(last=False)


# --- Persistent per-prompt decision store -----------------------------------
# This workload is dominated by repeated prompts (top 25 prompts = ~38% of
# calls). Pins let the router learn per-prompt routing: prompts that succeed
# cheaply N times stop paying embedding + Supra scoring cost; prompts that
# explicitly refuse on the cheap backend K times skip the doomed cheap attempt
# and go straight to the expensive backend. The store is append-only
# JSONL (like the other telemetry journals) and reloaded on startup, so pins
# survive restarts. Pins expire after PIN_TTL_S and stats reset after 24h
# without a new note, so a changed prompt behavior re-learns.
DECISION_STORE_PATH = Path(os.environ.get("DECISION_STORE_FILE", str(DATA_DIR / "router/decision-state.jsonl")))
PIN_CHEAP_AFTER = int(os.environ.get("ROUTELLM_PIN_CHEAP_AFTER", "5"))
PIN_EXPENSIVE_AFTER = int(os.environ.get("ROUTELLM_PIN_EXPENSIVE_AFTER", "2"))
PIN_TTL_S = float(os.environ.get("ROUTELLM_PIN_TTL_S", str(7 * 86400)))
DECISION_STORE_MAX = int(os.environ.get("ROUTELLM_DECISION_STORE_MAX", "4096"))
_decision_store: dict[str, dict] = {}
_decision_store_lock = threading.Lock()


def _store_load() -> None:
    """Load the last entry per prompt hash from the decision journal."""
    if not DECISION_STORE_PATH.exists():
        return
    try:
        with open(DECISION_STORE_PATH, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                prompt_hash = entry.get("prompt_hash")
                if isinstance(prompt_hash, str) and prompt_hash:
                    _decision_store[prompt_hash] = entry
    except OSError:
        return
    if len(_decision_store) > DECISION_STORE_MAX:
        for key in sorted(_decision_store, key=lambda k: _decision_store[k].get("ts", 0))[
                : len(_decision_store) - DECISION_STORE_MAX]:
            del _decision_store[key]


def _store_pinned(prompt_hash: str) -> dict | None:
    """Return the stored entry when it is currently pinned, else None."""
    if not prompt_hash:
        return None
    entry = _decision_store.get(prompt_hash)
    if not entry:
        return None
    pin_until = entry.get("pin_until")
    if not isinstance(pin_until, (int, float)) or pin_until < time.time():
        return None
    return entry


def _store_note(prompt_hash: str, decision: str, *, ok: bool = False, score=None) -> None:
    """Record one routed outcome for a prompt and update pins (write-through).

    ok=True marks a clean completion (cheap successes build the cheap pin);
    ok=False marks an explicit refusal (repeated cheap refusals flip the pin
    to expensive so the next request skips the doomed cheap attempt).
    """
    if not prompt_hash:
        return
    now = time.time()
    with _decision_store_lock:
        entry = dict(_decision_store.get(prompt_hash)
                     or {"prompt_hash": prompt_hash, "decision": decision, "ok": 0, "fail": 0, "ts": 0})
        if now - float(entry.get("ts", 0)) > 86400:
            entry["ok"], entry["fail"] = 0, 0  # new streak after idle day
        entry["decision"] = decision
        entry["ts"] = now
        if score is not None:
            entry["score"] = score
        entry["ok"] = int(entry.get("ok", 0)) + (1 if ok else 0)
        entry["fail"] = int(entry.get("fail", 0)) + (0 if ok else 1)
        if decision == "cheap" and entry["fail"] >= PIN_EXPENSIVE_AFTER:
            entry["decision"] = "expensive"
            entry["pin_until"] = now + PIN_TTL_S
        elif decision == "cheap" and entry["ok"] >= PIN_CHEAP_AFTER and entry["fail"] == 0:
            entry["pin_until"] = now + PIN_TTL_S
        elif entry.get("pin_until") and float(entry.get("pin_until", 0)) < now:
            entry.pop("pin_until", None)
        _decision_store[prompt_hash] = entry
        if len(_decision_store) > DECISION_STORE_MAX:
            for key in sorted(_decision_store, key=lambda k: _decision_store[k].get("ts", 0))[
                    : len(_decision_store) - DECISION_STORE_MAX]:
                del _decision_store[key]
        _secure_append(DECISION_STORE_PATH, entry)


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
        # Repeated identical bodies are useful outcome telemetry, but are not
        # a safe escalation signal: agent loops legitimately repeat prompts
        # such as "Proceed" and polling instructions. Only an explicit cheap
        # refusal (recorded from the upstream response) may pin expensive.
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
        if ROUTER_NAME != "supra":
            await asyncio.to_thread(_load_router)
        if SUPRA_ENABLED or ROUTER_NAME == "supra":
            await asyncio.to_thread(_load_supra)
        _store_load()
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


def _validate_responses_request(body: dict) -> tuple[str, str | None] | None:
    if body.get("model") != MODEL_ID:
        return "Only model 'auto' is supported", "model"
    if "input" not in body or body.get("input") is None:
        return "'input' is required", "input"
    input_value = body.get("input")
    if not isinstance(input_value, (str, list)):
        return "'input' must be text or an array", "input"
    if not input_value:
        return "'input' must not be empty", "input"
    if "stream" in body and not isinstance(body["stream"], bool):
        return "'stream' has an invalid type", "stream"
    if "max_output_tokens" in body:
        value = body["max_output_tokens"]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            return "'max_output_tokens' must be a positive integer", "max_output_tokens"
    stable_types = {
        "instructions": (str, type(None)), "tools": list, "metadata": dict,
        "reasoning": dict, "tool_choice": (str, dict),
    }
    for name, expected in stable_types.items():
        if name in body and not isinstance(body[name], expected):
            return f"'{name}' has an invalid type", name
    for name in ("temperature", "top_p"):
        if name in body and (isinstance(body[name], bool) or not isinstance(body[name], (int, float))):
            return f"'{name}' has an invalid type", name
    # Responses extensions are preserved rather than rejected.
    return None


def _route_headers(decision, score, backend, request_id, supra_complexity, supra_ms,
                   *, pinned=False, api_format="chat"):
    upstream_path = "/responses" if api_format == "responses" else "/chat/completions"
    headers = {
        "x-request-id": request_id, "x-route-decision": decision,
        "x-route-score": f"{score:.4f}" if isinstance(score, (int, float)) else "n/a",
        "x-route-model": backend["model"],
        "x-route-router": ROUTER_NAME, "x-route-fallback": "false",
        "x-route-attempts": "1", "x-route-api": api_format,
        "x-route-upstream-path": upstream_path,
    }
    if pinned:
        headers["x-route-pinned"] = "true"
    if supra_complexity is not None:
        headers["x-route-supra-complexity"] = str(supra_complexity)
    if supra_ms is not None:
        headers["x-route-supra-ms"] = str(supra_ms)
    return headers


def _upstream_request(backend: dict, body: dict, *, api_format: str = "chat") -> httpx.Request:
    assert _client is not None
    if api_format == "responses":
        path, outgoing = "/responses", _build_responses_body(body, backend)
    else:
        path, outgoing = "/chat/completions", _build_outgoing_body(body, backend)
    return _client.build_request(
        "POST", backend["base"].rstrip("/") + path,
        json=outgoing,
        headers={"Authorization": f"Bearer {backend['key']}", "Content-Type": "application/json"},
    )


async def _send(backend: dict, body: dict, *, stream: bool, api_format: str = "chat") -> httpx.Response:
    if _client is None:
        raise RuntimeError("router is not ready")
    return await _client.send(_upstream_request(backend, body, api_format=api_format), stream=stream)


def _retryable(status: int) -> bool:
    return status in RETRY_STATUSES


def _failover_routes(decision: str) -> tuple[str, ...]:
    if not MIDDLE_CONFIGURED:
        return (decision, "cheap" if decision == "expensive" else "expensive")
    if decision == "cheap":
        return ("cheap", "middle")
    if decision == "middle":
        return ("middle", "expensive")
    return ("expensive", "middle")


async def _open_with_failover(body: dict, decision: str, deadline: float, *, stream: bool,
                              api_format: str = "chat"):
    attempts = []
    routes = _responses_routes(decision) if api_format == "responses" else _failover_routes(decision)
    for index, current in enumerate(routes):
        backend = _backend_for(current)
        try:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError
            async with asyncio.timeout(remaining):
                response = await _send(backend, body, stream=stream, api_format=api_format)
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


def _log_attempts(attempts, prompt: str, score: float, request_id: str, occurrence_id: str,
                  *, api_format: str = "chat") -> None:
    for index, (decision, backend, status, error) in enumerate(attempts, 1):
        _log(decision, score, backend["model"], prompt, None, request_id=request_id,
             occurrence_id=occurrence_id, record_type="attempt", attempt=index,
             tier=backend.get("tier"), status=status, error=error, api_format=api_format)


@app.get("/healthz")
async def healthz():
    return {"ok": _READY, "router": ROUTER_NAME, "threshold": THRESHOLD,
            "backend": "cloudflare" if _GATEWAY_SELECTED else "direct",
            "gateway": _GATEWAY_SELECTED, "tiers": list(BACKENDS),
            "ready": _READY, "cache": {**_cache_metrics, "entries": len(_resp_cache), "bytes": _cache_bytes}}


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


def _responses_event_state(event: bytes) -> tuple[bool, bool, dict | None]:
    """Return (completed, failed, usage) without assuming Chat choices."""
    event_name = None
    for line in event.splitlines():
        if line.startswith(b"event:"):
            event_name = line[6:].strip().decode("utf-8", errors="replace")
            break
    data_text = _sse_data(event)
    if data_text is None:
        return False, event_name in {"error", "response.failed", "response.incomplete"}, None
    if data_text.strip() == "[DONE]":
        return True, False, None
    try:
        payload = json.loads(data_text)
    except json.JSONDecodeError:
        return False, False, None
    event_type = payload.get("type") or event_name
    response = payload.get("response") if isinstance(payload.get("response"), dict) else {}
    usage = response.get("usage") or payload.get("usage")
    if event_type == "response.completed":
        return True, False, usage if isinstance(usage, dict) else None
    failed = event_type in {"error", "response.failed", "response.incomplete"} or "error" in payload
    return False, failed, usage if isinstance(usage, dict) else None


def _responses_stream_error(message: str, code: str, request_id: str) -> bytes:
    payload = {"type": "error", "error": {"message": message, "type": "upstream_error", "code": code},
               "request_id": request_id}
    return ("event: error\ndata: " + json.dumps(payload) + "\n\n").encode()


def _responses_usage(data: dict) -> dict | None:
    usage = data.get("usage")
    if isinstance(usage, dict):
        return usage
    response = data.get("response")
    if isinstance(response, dict) and isinstance(response.get("usage"), dict):
        return response["usage"]
    return None


@app.post("/v1/responses")
async def responses(request: Request, authorization: str | None = Header(default=None)):
    request_id = request.headers.get("x-request-id") or f"req_{uuid.uuid4().hex}"
    if not _authorize(authorization):
        return _openai_error("Invalid API key", 401, error_type="authentication_error", code="invalid_api_key")
    try:
        body = await _read_json_body(request)
    except OverflowError:
        return _openai_error("Request body is too large", 413, code="request_too_large")
    except ValueError as exc:
        return _openai_error(str(exc), 400, code="invalid_json")
    invalid = _validate_responses_request(body)
    if invalid:
        return _openai_error(invalid[0], 400, param=invalid[1], code="invalid_request")

    prompt = _extract_responses_prompt(body)
    prompt_hash = _prompt_hash(prompt)
    session_id, session_source = _session_id(body, request)
    if prompt.strip():
        if session_id is None:
            proposed, score, supra_complexity, supra_ms = await asyncio.to_thread(_decide, prompt)
        else:
            proposed, score, supra_complexity, supra_ms = await asyncio.to_thread(_decide, prompt, session_id)
    else:
        proposed, score, supra_complexity, supra_ms = "cheap", 0.0, None, None
    decision, route_reason = _session_route(
        session_id, prompt, proposed, supra_complexity, score,
        new_task=request.headers.get("x-route-new-task", "").lower() in {"1", "true", "yes"},
    )
    compatible = _responses_tier(decision)
    protocol_upgraded = compatible is not None and compatible != decision
    if compatible is None:
        return _openai_error(
            "Responses API requires an OpenAI model routed through OpenRouter or the Cloudflare gateway",
            503, error_type="configuration_error", code="responses_backend_unavailable",
        )
    if protocol_upgraded:
        decision, route_reason = compatible, "responses_protocol_upgrade"
    backend = _backend_for(decision)
    pinned = session_id is None and _store_pinned(prompt_hash) is not None
    occurrence_id = uuid.uuid4().hex
    _record_and_detect_retry(_request_hash({"messages": body.get("input")}), decision,
                             backend["model"], prompt_hash, request_id, occurrence_id)
    headers = _route_headers(decision, score, backend, request_id, supra_complexity, supra_ms,
                             pinned=pinned, api_format="responses")
    headers["x-route-reason"] = route_reason
    headers["x-route-sticky"] = str(route_reason in _STICKY_REASONS).lower()
    if session_id:
        headers["x-route-session"] = session_id

    deadline = time.monotonic() + TIMEOUT_S
    upstream = None
    try:
        selected, backend, upstream, attempts = await _open_with_failover(
            body, decision, deadline, stream=bool(body.get("stream")), api_format="responses",
        )
    except BaseException:
        if upstream is not None:
            await asyncio.shield(upstream.aclose())
        raise
    headers.update({
        "x-route-decision": selected, "x-route-model": backend["model"],
        "x-route-fallback": str(selected != decision).lower(), "x-route-attempts": str(len(attempts)),
        "x-route-switch": str(protocol_upgraded or selected != decision).lower(),
        "x-route-affinity": "warm" if session_id and _session_get(session_id) else "unknown",
    })
    _log_attempts(attempts, prompt, score, request_id, occurrence_id, api_format="responses")
    if upstream is None:
        _log(selected, score, backend["model"], prompt, None, request_id=request_id,
             occurrence_id=occurrence_id, record_type="decision", status=None,
             error="upstream_unavailable", attempts=len(attempts), pinned=pinned,
             tier=backend.get("tier"), route_reason=route_reason, session_id=session_id,
             session_source=session_source, api_format="responses")
        return _openai_error("Responses upstream was unavailable", 502,
                             error_type="upstream_error", code="upstream_unavailable")

    if not body.get("stream"):
        try:
            async with asyncio.timeout(max(0, deadline - time.monotonic())):
                content = await upstream.aread()
        except (httpx.TransportError, TimeoutError, asyncio.TimeoutError):
            await upstream.aclose()
            return _openai_error("Upstream response timed out", 504,
                                 error_type="upstream_error", code="upstream_timeout")
        finally:
            await upstream.aclose()
        data = _safe_json(content)
        usage = _responses_usage(data)
        clean_success = upstream.status_code == 200 and data.get("status") == "completed"
        if upstream.status_code != 200:
            _log_outcome(prompt_hash, "upstream_error", request_id=request_id,
                         decision_occurrence_id=occurrence_id, model=backend["model"],
                         status=upstream.status_code)
        elif not clean_success:
            _log_outcome(prompt_hash, "truncated" if data.get("status") == "incomplete" else "upstream_error",
                         request_id=request_id, decision_occurrence_id=occurrence_id,
                         model=backend["model"], responses_status=data.get("status"))
        if clean_success:
            if session_id is None:
                _store_note(prompt_hash, selected, ok=True, score=score)
            else:
                _session_note(session_id, selected, supra_complexity, usage)
        _log(selected, score, backend["model"], prompt, None, supra_complexity, supra_ms,
             cost_usd=_extract_cost(data), usage=usage, request_id=request_id,
             occurrence_id=occurrence_id, record_type="decision", attempts=len(attempts),
             status=upstream.status_code, pinned=pinned, tier=backend.get("tier"),
             route_reason=route_reason, session_id=session_id, session_source=session_source,
             api_format="responses")
        response_headers = {**headers, "content-type": upstream.headers.get("content-type", "application/json")}
        return Response(content=content, status_code=upstream.status_code,
                        media_type=None, headers=response_headers)

    if upstream.status_code != 200:
        content = await upstream.aread()
        response_headers = {**headers, "content-type": upstream.headers.get("content-type", "application/json")}
        await upstream.aclose()
        return Response(content=content, status_code=upstream.status_code,
                        media_type=None, headers=response_headers)

    async def response_events():
        completed = failed = False
        usage: dict = {}
        started = time.monotonic()
        try:
            async with asyncio.timeout(max(0, deadline - time.monotonic())):
                async for event in _iter_sse_events(upstream):
                    if await request.is_disconnected():
                        raise asyncio.CancelledError
                    event_completed, event_failed, event_usage = _responses_event_state(event)
                    if event_usage:
                        usage.update(event_usage)
                    yield event
                    completed = completed or event_completed
                    failed = failed or event_failed
                    if completed or failed:
                        break
            if failed:
                _log_outcome(prompt_hash, "upstream_error", request_id=request_id,
                             decision_occurrence_id=occurrence_id, model=backend["model"],
                             responses_terminal_error=True)
            elif not completed:
                _log_outcome(prompt_hash, "truncated", request_id=request_id,
                             decision_occurrence_id=occurrence_id, model=backend["model"], abrupt_eof=True)
                yield _responses_stream_error("Upstream Responses stream ended before completion",
                                              "upstream_truncated", request_id)
        except asyncio.CancelledError:
            _log_outcome(prompt_hash, "disconnected", request_id=request_id,
                         decision_occurrence_id=occurrence_id, model=backend["model"])
            raise
        except (httpx.TransportError, TimeoutError, asyncio.TimeoutError, ValueError):
            _log_outcome(prompt_hash, "upstream_error", request_id=request_id,
                         decision_occurrence_id=occurrence_id, model=backend["model"])
            yield _responses_stream_error("Upstream Responses stream failed",
                                          "upstream_transport_error", request_id)
        finally:
            await asyncio.shield(upstream.aclose())
            _log(selected, score, backend["model"], prompt,
                 int((time.monotonic() - started) * 1000), supra_complexity, supra_ms,
                 usage=usage or None, request_id=request_id, occurrence_id=occurrence_id,
                 record_type="decision", attempts=len(attempts), completed=completed,
                 pinned=pinned, tier=backend.get("tier"), route_reason=route_reason,
                 session_id=session_id, session_source=session_source, api_format="responses")
            if completed:
                if session_id is None:
                    _store_note(prompt_hash, selected, ok=True, score=score)
                else:
                    _session_note(session_id, selected, supra_complexity, usage or None)

    return StreamingResponse(response_events(), media_type="text/event-stream", headers=headers)


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
        result = (cached, 200, media, {"x-route-cache": "hit", "x-route-api": "chat",
                                      "x-route-upstream-path": "/chat/completions"})
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
        prompt_hash = _prompt_hash(prompt)
        session_id, session_source = _session_id(body, request)
        if prompt.strip():
            if session_id is None:
                proposed, score, supra_complexity, supra_ms = await asyncio.to_thread(_decide, prompt)
            else:
                proposed, score, supra_complexity, supra_ms = await asyncio.to_thread(
                    _decide, prompt, session_id)
        else:
            proposed, score, supra_complexity, supra_ms = "cheap", 0.0, None, None
        decision, route_reason = _session_route(
            session_id, prompt, proposed, supra_complexity, score,
            new_task=request.headers.get("x-route-new-task", "").lower() in {"1", "true", "yes"},
        )
        pinned = session_id is None and _store_pinned(prompt_hash) is not None
        backend = _backend_for(decision)
        _record_and_detect_retry(_request_hash(body), decision, backend["model"], prompt_hash,
                                 request_id, occurrence_id)
        headers = _route_headers(decision, score, backend, request_id, supra_complexity, supra_ms,
                                 pinned=pinned, api_format="chat")
        headers["x-route-reason"] = route_reason
        headers["x-route-sticky"] = str(route_reason in _STICKY_REASONS).lower()
        if session_id:
            headers["x-route-session"] = session_id
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
                    tried = [item[0] for item in attempts]
                    selected = next((tier for tier in _failover_routes(decision) if tier not in tried), None)
                    if selected is None:
                        raise ValueError("no streaming fallback route available")
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

        # A refusal on the cheap backend is a per-prompt failure signal. Keep
        # legacy global learning for sessionless traffic; session traffic gets
        # a local promotion so every following turn avoids the bad tier.
        for a_decision, _a_backend, _a_status, a_error in attempts:
            if a_error in ("refusal", "refusal_fallback") and a_decision == "cheap":
                if session_id is None:
                    _store_note(prompt_hash, "cheap", ok=False)
                else:
                    _session_note(session_id, "middle" if MIDDLE_CONFIGURED else "expensive", supra_complexity)

    except BaseException:
        _finish_inflight_nowait(cache_key, inflight, None)
        if upstream is not None:
            await asyncio.shield(upstream.aclose())
        raise

    headers.update({"x-route-decision": selected, "x-route-model": backend["model"],
                    "x-route-fallback": str(selected != decision).lower(), "x-route-attempts": str(len(attempts)),
                    "x-route-switch": str(selected != decision).lower(),
                    "x-route-affinity": "warm" if session_id and _session_get(session_id) else "unknown"})
    _log_attempts(attempts, prompt, score, request_id, occurrence_id, api_format="chat")
    if upstream is None:
        _log(selected, score, backend["model"], prompt, None, request_id=request_id,
             occurrence_id=occurrence_id, record_type="decision", status=None,
             error="upstream_unavailable", attempts=len(attempts), pinned=pinned,
             tier=backend.get("tier"), route_reason=route_reason,
             session_id=session_id, session_source=session_source)
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
                 error="upstream_timeout", attempts=len(attempts), pinned=pinned,
                 tier=backend.get("tier"), route_reason=route_reason,
                 session_id=session_id, session_source=session_source)
            await _finish_inflight(cache_key, inflight, None)
            return _openai_error("Upstream response timed out", 504, error_type="upstream_error", code="upstream_timeout")
        finally:
            await upstream.aclose()
        data = _safe_json(content)
        if upstream.status_code != 200:
            _log_outcome(_prompt_hash(prompt), "upstream_error", request_id=request_id,
                         decision_occurrence_id=occurrence_id, model=backend["model"], status=upstream.status_code)
        truncated = _length_truncated(data)
        if truncated:
            _log_outcome(_prompt_hash(prompt), "truncated", request_id=request_id,
                         decision_occurrence_id=occurrence_id, model=backend["model"])
        if upstream.status_code == 200:
            _cache_put(cache_key, body, content)
            if not truncated:
                if session_id is None:
                    _store_note(prompt_hash, selected, ok=True, score=score)
                else:
                    _session_note(session_id, selected, supra_complexity, data.get("usage"))
        _log(selected, score, backend["model"], prompt, None, supra_complexity, supra_ms,
             cost_usd=_extract_cost(data), usage=data.get("usage"), request_id=request_id,
             occurrence_id=occurrence_id, record_type="decision", attempts=len(attempts),
             status=upstream.status_code, pinned=pinned, tier=backend.get("tier"),
             route_reason=route_reason, session_id=session_id, session_source=session_source)
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
             occurrence_id=occurrence_id, record_type="decision", status=upstream.status_code,
             pinned=pinned, tier=backend.get("tier"), route_reason=route_reason,
             session_id=session_id, session_source=session_source)
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
                 attempts=len(attempts), completed=saw_done and saw_finish, pinned=pinned,
                 tier=backend.get("tier"), route_reason=route_reason,
                 session_id=session_id, session_source=session_source)
            if saw_done and saw_finish and not saw_length:
                if session_id is None:
                    _store_note(prompt_hash, selected, ok=True, score=score)
                else:
                    _session_note(session_id, selected, supra_complexity, usage or None)
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