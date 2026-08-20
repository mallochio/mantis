"""Minimal Mantis gateway: Supra routing + Bifrost transport."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import os
import re
import secrets
import sqlite3
import threading
import time
import tomllib
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from cachetools import LRUCache, TTLCache
from fastapi import FastAPI, Header, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

HOST = os.environ.get("MANTIS_ROUTER_HOST", "127.0.0.1")
PORT = int(os.environ.get("MANTIS_ROUTER_PORT", "5501"))
SERVER_KEY = os.environ.get("MANTIS_ROUTER_KEY", "sk-route-local")
MODEL_ID = "auto"
MANTIS_ROUTER_MAX_TOKENS = int(os.environ.get("MANTIS_ROUTER_MAX_TOKENS", "384000"))
TIMEOUT_S = float(os.environ.get("MANTIS_ROUTER_TIMEOUT_S", "600"))
MAX_BODY_BYTES = int(os.environ.get("MANTIS_ROUTER_MAX_BODY_BYTES", str(64 * 1024 * 1024)))
PIN_CHEAP_AFTER = int(os.environ.get("MANTIS_ROUTER_PIN_CHEAP_AFTER", "5"))
PIN_EXPENSIVE_AFTER = int(os.environ.get("MANTIS_ROUTER_PIN_EXPENSIVE_AFTER", "2"))
PIN_TTL_S = float(os.environ.get("MANTIS_ROUTER_PIN_TTL_S", str(7 * 86400)))
SESSION_TTL_S = float(os.environ.get("MANTIS_ROUTER_SESSION_TTL_S", "3600"))
SESSION_STATE_MAX = int(os.environ.get("MANTIS_ROUTER_SESSION_STATE_MAX", "4096"))
RESCORE_EVERY_N = int(os.environ.get("MANTIS_ROUTER_RESCORE_EVERY_N", "0"))
DOWNGRADE_IDLE_S = float(os.environ.get("MANTIS_ROUTER_DOWNGRADE_IDLE_S", "0"))
RESP_CACHE_TTL_S = float(os.environ.get("MANTIS_ROUTER_RESP_CACHE_TTL_S", "120"))
RESP_CACHE_MAX_ENTRIES = int(os.environ.get("MANTIS_ROUTER_RESP_CACHE_MAX_ENTRIES", "128"))
RESP_CACHE_MAX_BYTES = int(os.environ.get("MANTIS_ROUTER_RESP_CACHE_MAX_BYTES", str(8 * 1024 * 1024)))
AFFINITY_TTL_S = float(os.environ.get("MANTIS_ROUTER_RESPONSES_AFFINITY_TTL_S", "3600"))
AFFINITY_MAX = int(os.environ.get("MANTIS_ROUTER_RESPONSES_AFFINITY_MAX", "4096"))
DATA_DIR = Path(os.environ.get("MANTIS_DATA_DIR", str(Path.home() / ".local/share/mantis")))
DECISION_STORE_PATH = Path(os.environ.get("DECISION_STORE_FILE", str(DATA_DIR / "router/lean-decisions.db")))
BIFROST_BASE_URL = os.environ.get("BIFROST_BASE_URL", "http://127.0.0.1:8080/v1")
BIFROST_API_KEY = os.environ.get("BIFROST_API_KEY", "")

TARGET_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_VALID_EFFORTS = frozenset({"low", "medium", "high", "xhigh", "max"})
RETRY_STATUSES = {429, 500, 502, 503, 504}
_REFUSAL_RE = re.compile(
    r"cannot (assist|help|comply)|i('| a)?m sorry|not (able|allowed) to (assist|help)|can('?t) (assist|help)", re.I
)
_CONTINUATION_RE = re.compile(
    r"^(?:ok(?:ay)?[,. ]*)?(?:proceed|continue|go ahead|do (?:it|that)|yes|yep|sure|run (?:it|them|the tests)(?: again)?|try again|fix (?:it|that)|next)(?:[.! ]*)$",
    re.I,
)
_NEW_TASK_RE = re.compile(r"^(?:new task|different task|unrelated|switching topics?|on another topic)\b", re.I)


def _cfg_err(msg):
    return ValueError(f"invalid lean gateway config: {msg}")


def _valid_id(v, field="target"):
    if not isinstance(v, str) or not TARGET_ID_RE.fullmatch(v):
        raise _cfg_err(f"{field} must be an identifier")
    return v


def _valid_env(v, field="credential_env"):
    if not isinstance(v, str) or not ENV_NAME_RE.fullmatch(v):
        raise _cfg_err(f"{field} must name an env variable")
    return v


def _valid_url(v, field="base_url"):
    if not isinstance(v, str) or not v:
        raise _cfg_err(f"{field} must be a URL")
    try:
        p = urlsplit(v)
        _ = p.port
    except ValueError:
        raise _cfg_err(f"{field} is not a valid URL")
    if p.scheme.lower() not in {"http", "https"} or not p.hostname:
        raise _cfg_err(f"{field} must be http(s)")
    return v.rstrip("/")


def _reject_dup(pairs):
    r = {}
    for k, v in pairs:
        if k in r:
            raise ValueError(f"duplicate key {k!r}")
        r[k] = v
    return r


def _read_json():
    raw = os.environ.get("MANTIS_ROUTER_TARGETS_JSON")
    if not raw:
        return None
    try:
        data = json.loads(raw, object_pairs_hook=_reject_dup)
    except (TypeError, ValueError):
        raise _cfg_err("MANTIS_ROUTER_TARGETS_JSON must be valid JSON") from None
    if not isinstance(data, dict) or data.get("version") != 1:
        raise _cfg_err("JSON version must be 1")
    return data


def _read_catalog():
    path = Path(os.environ.get("AI_ROUTING_CONFIG", str(Path.home() / ".config/ai-routing/catalog.toml"))).expanduser()
    if not path.exists():
        raise _cfg_err(f"catalog not found: {path}")
    try:
        with path.open("rb") as f:
            data = tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError):
        raise _cfg_err("catalog could not be read") from None
    if not isinstance(data, dict) or data.get("version") != 1:
        raise _cfg_err("catalog version must be 1")
    gw = data.get("gateway", {})
    return {
        "targets": gw.get("targets", {}),
        "providers": data.get("providers", {}),
        "active_policy": gw.get("active_policy"),
        "policies": gw.get("policies", {}),
        "revision": gw.get("revision"),
        "invalid_complexity_target": gw.get("invalid_complexity_target"),
    }


def _load_spec():
    spec = _read_json()
    if spec is None:
        spec = _read_catalog()
    providers = {}
    for pid, p in (spec.get("providers") or {}).items():
        _valid_id(pid, "provider id")
        if not isinstance(p, dict):
            raise _cfg_err(f"provider {pid} must be a table")
        base = _valid_url(
            os.environ.get("BIFROST_BASE_URL") or p.get("base_url") or BIFROST_BASE_URL, f"provider.{pid}.base_url"
        )
        cred = p.get("credential_env")
        if cred:
            _valid_env(cred, f"provider.{pid}.credential_env")
        providers[pid] = {
            "base_url": base,
            "credential_env": cred,
            "developer_role": p.get("developer_role", "system"),
            "protocols": tuple(p.get("protocols") or ("chat_completions", "responses")),
        }
    targets = {}
    for tid, t in (spec.get("targets") or {}).items():
        _valid_id(tid, f"target.{tid}")
        if not isinstance(t, dict):
            raise _cfg_err(f"target {tid} must be a table")
        prov = t.get("provider")
        if prov:
            if prov not in providers:
                raise _cfg_err(f"target {tid} references unknown provider {prov}")
            bind = providers[prov]
        else:
            bind = {
                "base_url": _valid_url(
                    os.environ.get("BIFROST_BASE_URL") or t.get("base_url") or BIFROST_BASE_URL,
                    f"target.{tid}.base_url",
                ),
                "credential_env": t.get("credential_env"),
                "developer_role": t.get("developer_role", "system"),
                "protocols": tuple(t.get("protocols") or ("chat_completions", "responses")),
            }
        cred = t.get("credential_env") or bind.get("credential_env") or "BIFROST_API_KEY"
        _valid_env(cred, f"target.{tid}.credential_env")
        key = os.environ.get(cred) or BIFROST_API_KEY
        if not key:
            raise _cfg_err("BIFROST_API_KEY is required")
        model = t.get("upstream_model") or t.get("model")
        if not model:
            raise _cfg_err(f"target {tid} needs upstream_model")
        effort = (t.get("reasoning_effort") or "").lower()
        if effort and effort not in _VALID_EFFORTS:
            raise _cfg_err(f"target {tid} has invalid reasoning_effort")
        rank = t.get("rank")
        rank = int(rank) if isinstance(rank, int) and not isinstance(rank, bool) else list(spec["targets"]).index(tid)
        maxt = t.get("max_tokens")
        if maxt is not None and (isinstance(maxt, bool) or not isinstance(maxt, int) or maxt <= 0):
            raise _cfg_err(f"target {tid} has invalid max_tokens")
        protocols = tuple(t.get("protocols") or bind["protocols"])
        fallbacks = tuple(_valid_id(f, f"target.{tid}.fallbacks") for f in (t.get("fallbacks") or ()))
        targets[tid] = {
            "target": tid,
            "base_url": bind["base_url"],
            "key": key,
            "model": model,
            "rank": rank,
            "protocols": protocols,
            "reasoning_effort": effort,
            "max_tokens": maxt,
            "force_reasoning_effort": bool(t.get("force_reasoning_effort", False)),
            "fallbacks": fallbacks,
            "developer_role": bind.get("developer_role", "system"),
        }
    if not targets:
        raise _cfg_err("no targets configured")
    policies = spec.get("policies") or {}
    active = spec.get("active_policy") or next(iter(policies), None)
    if active is None:
        raise _cfg_err("no active policy")
    if active not in policies:
        raise _cfg_err(f"active_policy {active} not found")
    policy = policies[active]
    ctargets = policy.get("complexity_targets")
    if not isinstance(ctargets, (list, tuple)) or len(ctargets) != 5:
        raise _cfg_err("complexity_targets must be 5 ids")
    for c in ctargets:
        if c not in targets:
            raise _cfg_err(f"complexity_targets references unknown target {c}")
    cefforts = policy.get("complexity_efforts")
    if cefforts is None:
        cefforts = (None,) * 5
    elif not isinstance(cefforts, (list, tuple)) or len(cefforts) != 5:
        raise _cfg_err("complexity_efforts must be 5 values")
    cefforts = tuple((e.lower() if isinstance(e, str) and e.lower() in _VALID_EFFORTS else None) for e in cefforts)
    inv = policy.get("invalid_complexity_target") or spec.get("invalid_complexity_target")
    if inv is not None:
        _valid_id(inv, "invalid_complexity_target")
        if inv not in targets:
            raise _cfg_err("invalid_complexity_target is unknown")
    else:
        inv = max(targets, key=lambda x: (targets[x]["rank"], -list(targets).index(x)))
    return targets, tuple(ctargets), cefforts, inv


BACKENDS, SUPRA_TARGETS, SUPRA_EFFORTS, SUPRA_INVALID_TARGET = _load_spec()


def _safe_target():
    return max(BACKENDS, key=lambda t: (BACKENDS[t]["rank"], -list(BACKENDS).index(t)))


def _target_rank(t):
    return BACKENDS[t]["rank"] if t in BACKENDS else -1


def _target_order(t):
    return list(BACKENDS).index(t) if t in BACKENDS else len(BACKENDS)


def _target_for_complexity(c):
    return SUPRA_TARGETS[c - 1] if isinstance(c, int) and 1 <= c <= 5 else SUPRA_INVALID_TARGET


def _effort_for_complexity(c):
    return SUPRA_EFFORTS[c - 1] if isinstance(c, int) and 1 <= c <= 5 else None


def _apply_effort_override(backend, effort):
    if effort is None or effort == backend.get("reasoning_effort"):
        return backend
    return {**backend, "reasoning_effort": effort, "force_reasoning_effort": True}


def _supports_api(backend, api_format):
    return ("responses" if api_format == "responses" else "chat_completions") in (backend.get("protocols") or ())


def _backend_for(t):
    return BACKENDS.get(t, BACKENDS[_safe_target()])


def _ranked_compatible(decision, api_format):
    if decision not in BACKENDS:
        decision = _safe_target()
    floor = _target_rank(decision)
    return tuple(
        t
        for t in sorted(BACKENDS, key=lambda x: (_target_rank(x), _target_order(x)))
        if _target_rank(t) >= floor and _supports_api(_backend_for(t), api_format)
    )


def _api_routes(decision, api_format):
    if decision not in BACKENDS:
        decision = _safe_target()
    ranked = _ranked_compatible(decision, api_format)
    primary = (
        decision if _supports_api(_backend_for(decision), api_format) else (ranked[0] if ranked else _safe_target())
    )
    routes = [primary]
    for fb in _backend_for(primary).get("fallbacks", ()):
        if fb in BACKENDS and fb not in routes and _supports_api(_backend_for(fb), api_format):
            routes.append(fb)
            return tuple(routes)
    if primary != decision:
        for fb in _backend_for(decision).get("fallbacks", ()):
            if fb in BACKENDS and fb not in routes and _supports_api(_backend_for(fb), api_format):
                routes.append(fb)
                return tuple(routes)
    for t in ranked:
        if t not in routes:
            routes.append(t)
            break
    return tuple(routes[:2])


def _chat_tier(d):
    return (_api_routes(d, "chat") or (None,))[0]


def _responses_tier(d):
    return (_api_routes(d, "responses") or (None,))[0]


def _refusal_target(decision, api_format="chat"):
    for t in _api_routes(decision, api_format):
        if t != decision:
            return t
    return _safe_target()


def _lowest_rank(api_format):
    ranks = [_target_rank(t) for t in BACKENDS if _supports_api(_backend_for(t), api_format)]
    return min(ranks) if ranks else None


def _target_revision():
    safe = {tid: {k: v for k, v in b.items() if k != "key"} for tid, b in sorted(BACKENDS.items())}
    return hashlib.sha256(
        json.dumps(
            {
                "target_order": list(BACKENDS),
                "targets": safe,
                "complexity_targets": list(SUPRA_TARGETS),
                "complexity_efforts": list(SUPRA_EFFORTS),
                "invalid_target": SUPRA_INVALID_TARGET,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()[:24]


TARGET_CONFIG_REVISION = _target_revision()

# --- Supra ----------------------------------------------------------------
_supra_model = None
_supra_tokenizer = None
SUPRA_FALLBACK_COUNT = 0


def _load_supra():
    global _supra_model, _supra_tokenizer
    if _supra_model is not None:
        return
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained("SupraLabs/Supra-Router-51M")
    model = AutoModelForCausalLM.from_pretrained("SupraLabs/Supra-Router-51M", torch_dtype=torch.float32)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model.eval()
    _supra_tokenizer, _supra_model = tok, model


@lru_cache(maxsize=512)
def _supra_complexity(prompt: str) -> tuple[int, int]:
    from transformers import StoppingCriteria

    model, tokenizer = _supra_model, _supra_tokenizer

    class _Seen(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            return (
                re.search(r"Complexity:\s*\d", tokenizer.decode(input_ids[0][-24:], skip_special_tokens=True))
                is not None
            )

    import torch

    inputs = tokenizer(
        f"Task: {prompt}\nAnalysis: ", return_tensors="pt", truncation=True, max_length=tokenizer.model_max_length
    )
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            stopping_criteria=[_Seen()],
        )
    ms = int((time.time() - t0) * 1000)
    gen = tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()
    c = 0
    for part in gen.split("|"):
        m = re.search(r"Complexity:\s*(\d)", part, re.I)
        if m:
            c = int(m.group(1))
            break
    return c, ms


# --- SQLite ---------------------------------------------------------------
_db_conn = None
_db_lock = threading.RLock()
_decision_cache: dict = {}


def _init_db():
    global _db_conn
    DECISION_STORE_PATH.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    _db_conn = sqlite3.connect(str(DECISION_STORE_PATH), check_same_thread=False)
    _db_conn.row_factory = sqlite3.Row
    _db_conn.execute("""CREATE TABLE IF NOT EXISTS decisions (key TEXT PRIMARY KEY, api_format TEXT, decision TEXT, ok INTEGER DEFAULT 0,
                       fail INTEGER DEFAULT 0, score REAL, pin_until REAL, ts REAL, target_revision TEXT)""")
    _db_conn.commit()


def _store_load():
    _decision_cache.clear()
    if _db_conn is None:
        return
    for row in _db_conn.execute("SELECT * FROM decisions WHERE target_revision=?", (TARGET_CONFIG_REVISION,)):
        _decision_cache[row["key"]] = dict(row)


def _store_key(ph, api):
    return f"{api}:{ph}"


def _store_pinned(ph, api="chat"):
    e = _decision_cache.get(_store_key(ph, api))
    if not e or e.get("target_revision") != TARGET_CONFIG_REVISION:
        return None
    pt = e.get("pin_until")
    if pt and time.time() < pt and e["decision"] in BACKENDS and _supports_api(_backend_for(e["decision"]), api):
        return e
    return None


def _store_note(ph, decision, *, ok=False, score=None, api="chat"):
    if not ph or decision not in BACKENDS or not _supports_api(_backend_for(decision), api):
        return
    key = _store_key(ph, api)
    now = time.time()
    with _db_lock:
        row = _db_conn.execute("SELECT * FROM decisions WHERE key=?", (key,)).fetchone() if _db_conn else None
        e = (
            dict(row)
            if row
            else {
                "key": key,
                "api_format": api,
                "decision": decision,
                "ok": 0,
                "fail": 0,
                "ts": 0,
                "target_revision": TARGET_CONFIG_REVISION,
            }
        )
        if now - float(e.get("ts", 0)) > 86400:
            e["ok"], e["fail"] = 0, 0
        e["ts"] = now
        e["target_revision"] = TARGET_CONFIG_REVISION
        e["decision"] = decision
        if score is not None:
            e["score"] = score
        e["ok"] = int(e.get("ok", 0)) + (1 if ok else 0)
        e["fail"] = int(e.get("fail", 0)) + (0 if ok else 1)
        low = _lowest_rank(api)
        if not ok and low is not None and _target_rank(decision) == low and e["fail"] >= PIN_EXPENSIVE_AFTER:
            e["decision"] = _refusal_target(decision, api)
            e["pin_until"] = now + PIN_TTL_S
        elif ok and low is not None and _target_rank(decision) == low and e["ok"] >= PIN_CHEAP_AFTER and e["fail"] == 0:
            e["pin_until"] = now + PIN_TTL_S
        elif e.get("pin_until") and e["pin_until"] < now:
            e.pop("pin_until", None)
        if _db_conn:
            _db_conn.execute(
                "INSERT OR REPLACE INTO decisions (key,api_format,decision,ok,fail,score,pin_until,ts,target_revision) VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    e["key"],
                    e["api_format"],
                    e["decision"],
                    e["ok"],
                    e["fail"],
                    e.get("score"),
                    e.get("pin_until"),
                    e["ts"],
                    e["target_revision"],
                ),
            )
            _db_conn.commit()
        _decision_cache[key] = e


def _record_refusal_learning(ph, attempts, session_id, complexity, *, api="chat"):
    low = _lowest_rank(api)
    if low is None:
        return
    for attempted, _b, _s, error in attempts:
        if error != "refusal" or attempted not in BACKENDS or _target_rank(attempted) != low:
            continue
        if session_id is None:
            _store_note(ph, attempted, ok=False, api=api)
        else:
            _session_note(session_id, _refusal_target(attempted, api=api), complexity, None, api=api)


# --- session --------------------------------------------------------------
_session_cache = TTLCache(maxsize=SESSION_STATE_MAX, ttl=SESSION_TTL_S)
_session_lock = threading.RLock()


def _is_continuation(prompt):
    return len(prompt.strip()) <= 200 and "```" not in prompt and bool(_CONTINUATION_RE.fullmatch(prompt.strip()))


def _session_id(body, request):
    raw, src = request.headers.get("x-route-session"), "header"
    if not raw and isinstance(body.get("metadata"), dict):
        raw, src = body["metadata"].get("session_id"), "metadata"
    if (
        not raw
        and os.environ.get("MANTIS_ROUTER_SESSION_FROM_USER", "").lower() in {"1", "true", "yes", "on"}
        and isinstance(body.get("user"), str)
    ):
        raw, src = body["user"], "user"
    if not isinstance(raw, str) or not raw or len(raw.encode()) > 256 or any(ord(c) < 32 or ord(c) == 127 for c in raw):
        return None, None
    return hmac.new(
        SERVER_KEY.encode("utf-8", errors="replace"), (src + "\0" + raw).encode(), hashlib.sha256
    ).hexdigest()[:24], src


def _session_get(sid):
    if not sid:
        return None
    with _session_lock:
        st = _session_cache.get(sid)
        if st is None or st.get("target_revision") != TARGET_CONFIG_REVISION:
            return None
        _session_cache[sid] = st
        return dict(st)


def _session_target(st, api):
    if not st:
        return None
    t = (st.get("routes") or {}).get(api) or st.get("tier")
    return t if t in BACKENDS and _supports_api(_backend_for(t), api) else None


def _session_route(sid, prompt, proposed, complexity, score=None, *, new_task=False, api="chat"):
    proposed = proposed if proposed in BACKENDS else _safe_target()
    st = _session_get(sid)
    if not st:
        return proposed, "new_session"
    if new_task or _NEW_TASK_RE.match(prompt.strip()):
        return proposed, "new_task"
    cur = _session_target(st, api)
    if cur is None:
        return proposed, "protocol_unpinned"
    if _is_continuation(prompt):
        return cur, "continuation_sticky"
    if proposed == cur:
        return cur, "same_tier"
    cr, pr = _target_rank(cur), _target_rank(proposed)
    if pr < cr:
        if DOWNGRADE_IDLE_S > 0 and time.time() - st.get("last_seen", 0) > DOWNGRADE_IDLE_S:
            return proposed, "downgrade_idle"
        if RESCORE_EVERY_N and st.get("turns", 0) > 0 and st["turns"] % RESCORE_EVERY_N == 0:
            return proposed, "rescore_downgrade"
        if st.get("consecutive_low_turns", 0) >= 2:
            return proposed, "downgrade_consecutive_low"
        return cur, "downgrade_hysteresis"
    if pr > cr:
        return proposed, "strong_upgrade"
    return cur, "upgrade_hysteresis"


def _session_note(sid, tier, complexity, usage=None, *, api="chat"):
    if not sid or tier not in BACKENDS or not _supports_api(_backend_for(tier), api):
        return
    with _session_lock:
        p = _session_cache.get(sid) or {}
        routes = dict(p.get("routes") or {})
        routes[api] = tier
        low = p.get("consecutive_low_turns", 0) + 1 if complexity is not None and complexity <= 2 else 0
        st = {
            "tier": tier,
            "routes": routes,
            "target_revision": TARGET_CONFIG_REVISION,
            "last_seen": time.time(),
            "last_complexity": complexity,
            "consecutive_low_turns": low,
            "turns": int(p.get("turns", 0)) + 1,
        }
        if isinstance(usage, dict):
            st["prompt_tokens"] = usage.get("prompt_tokens", usage.get("input_tokens"))
            details = usage.get("prompt_tokens_details") or usage.get("input_tokens_details") or {}
            st["cached_tokens"] = details.get("cached_tokens", usage.get("prompt_cache_hit_tokens"))
        _session_cache[sid] = st


# --- decision -------------------------------------------------------------
def _decide(prompt, sid=None, api="chat"):
    if sid and _is_continuation(prompt):
        st = _session_get(sid)
        t = _session_target(st, api) if st else None
        if t:
            return t, None, st.get("last_complexity"), None
    if sid is None:
        p = _store_pinned(_prompt_hash(prompt), api)
        if p:
            return p["decision"], p.get("score"), None, None
    trimmed = prompt[-15000:] if len(prompt) > 15000 else prompt
    try:
        c, ms = _supra_complexity(trimmed)
        return _target_for_complexity(c), None, c, ms
    except Exception as err:
        global SUPRA_FALLBACK_COUNT
        SUPRA_FALLBACK_COUNT += 1
        print(f"Supra failed ({err}); defaulting", flush=True)
        return _safe_target(), None, None, None


# --- cache / in-flight ----------------------------------------------------
_resp_cache = TTLCache(maxsize=RESP_CACHE_MAX_ENTRIES, ttl=RESP_CACHE_TTL_S)
_cache_lock = threading.RLock()
_cache_bytes = 0
_cache_metrics = {"hits": 0, "misses": 0, "stores": 0, "evictions": 0}
_inflight: dict[str, asyncio.Future] = {}
_inflight_lock = asyncio.Lock()


def _prompt_hash(prompt):
    return hashlib.sha256(prompt.encode("utf-8", errors="replace")).hexdigest()[:24]


def _request_body_hash(body):
    return hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


def _cache_key(body, idem, *, sid=None):
    if not idem or len(idem) > 200:
        return None
    if body.get("tools") or body.get("functions") or body.get("function_call"):
        return None
    if any(
        isinstance(m, dict) and (m.get("role") in {"tool", "function"} or m.get("tool_calls") or m.get("function_call"))
        for m in body.get("messages", [])
    ):
        return None
    return hashlib.sha256(
        (TARGET_CONFIG_REVISION + ":" + (sid or "sessionless") + ":" + idem + ":" + _request_body_hash(body)).encode()
    ).hexdigest()


def _cache_get(key):
    if key is None:
        return None
    with _cache_lock:
        if key in _resp_cache:
            _cache_metrics["hits"] += 1
            return _resp_cache[key]
        _cache_metrics["misses"] += 1
        return None


def _cache_put(key, body, content, headers, media, status=200):
    global _cache_bytes
    if key is None or len(content) > RESP_CACHE_MAX_BYTES or not _response_replay_safe(body, content):
        return None
    with _cache_lock:
        while _resp_cache and (
            len(_resp_cache) >= RESP_CACHE_MAX_ENTRIES or _cache_bytes + len(content) > RESP_CACHE_MAX_BYTES
        ):
            k, v = _resp_cache.popitem(last=False)
            _cache_bytes -= len(v[0])
            _cache_metrics["evictions"] += 1
        _resp_cache[key] = (content, status, media, headers)
        _cache_bytes += len(content)
        _cache_metrics["stores"] += 1
        return _resp_cache[key]


async def _claim_inflight(key):
    if key is None:
        return True, None
    async with _inflight_lock:
        fut = _inflight.get(key)
        if fut is None:
            fut = asyncio.get_running_loop().create_future()
            _inflight[key] = fut
            return True, fut
        return False, fut


async def _finish_inflight(key, fut, result):
    if key is None or fut is None:
        return
    async with _inflight_lock:
        if _inflight.get(key) is fut:
            _inflight.pop(key, None)
        if not fut.done():
            fut.set_result(result)


def _finish_inflight_nowait(key, fut, result):
    if key is None or fut is None:
        return
    if _inflight.get(key) is fut:
        _inflight.pop(key, None)
    if fut and not fut.done():
        fut.set_result(result)


def _replayed_response(result, request_id):
    content, status, media, headers = result
    headers = {**headers, "x-request-id": request_id, "x-route-coalesced": "true"}
    if media == "text/event-stream":
        return StreamingResponse(iter([content]), status_code=status, media_type=media, headers=headers)
    return Response(content=content, status_code=status, media_type=media, headers=headers)


# --- affinity -------------------------------------------------------------
_aff = LRUCache(maxsize=AFFINITY_MAX)
_aff_lock = threading.RLock()
_AFF_HDR = "x-route-responses-affinity"


def _aff_key(kind, value):
    return (
        kind
        + ":"
        + hmac.new(
            SERVER_KEY.encode("utf-8", errors="replace"),
            (kind + "\0" + value).encode("utf-8", errors="replace"),
            hashlib.sha256,
        ).hexdigest()[:32]
    )


def _aff_record(backend):
    return {
        "target": backend["target"],
        "base_url": backend["base_url"].rstrip("/"),
        "model": backend["model"],
        "target_revision": TARGET_CONFIG_REVISION,
        "last_seen": time.time(),
    }


def _aff_put(key, rec):
    with _aff_lock:
        _aff[key] = dict(rec)


def _aff_get(key):
    with _aff_lock:
        r = _aff.get(key)
        if (
            r is None
            or time.time() - r.get("last_seen", 0) > AFFINITY_TTL_S
            or r.get("target_revision") != TARGET_CONFIG_REVISION
        ):
            _aff.pop(key, None)
            return None
        r["last_seen"] = time.time()
        _aff[key] = r
        return dict(r)


def _aff_token():
    n = secrets.token_urlsafe(24)
    s = hmac.new(SERVER_KEY.encode("utf-8", errors="replace"), n.encode(), hashlib.sha256).hexdigest()[:24]
    return f"{n}.{s}"


def _aff_token_rec(token):
    if not isinstance(token, str) or token.count(".") != 1:
        return None
    n, s = token.rsplit(".", 1)
    e = hmac.new(SERVER_KEY.encode("utf-8", errors="replace"), n.encode(), hashlib.sha256).hexdigest()[:24]
    if not secrets.compare_digest(s, e):
        return None
    return _aff_get(_aff_key("token", token))


def _aff_opaque_values(v):
    out = []

    def walk(x):
        if isinstance(x, list):
            for c in x:
                walk(c)
        elif isinstance(x, dict):
            t = x.get("type")
            if t in {"reasoning", "compaction"}:
                i = x.get("id")
                e = x.get("encrypted_content")
                if isinstance(i, str) and i:
                    out.append(("opaque-id", f"{t}\0{i}"))
                if isinstance(e, str) and e:
                    out.append(("opaque-content", f"{t}\0{e}"))
            for f in ("content", "output", "item", "response"):
                walk(x.get(f))

    walk(v)
    return out


def _aff_conv_value(v):
    if isinstance(v, str) and v:
        return v
    if isinstance(v, dict) and isinstance(v.get("id"), str) and v["id"]:
        return v["id"]
    return None


def _aff_continuation_values(body):
    out = []
    p = body.get("previous_response_id")
    if isinstance(p, str) and p:
        out.append(("response", p))
    c = _aff_conv_value(body.get("conversation"))
    if c:
        out.append(("conversation", c))
    out.extend(_aff_opaque_values(body.get("input")))
    return out


def _aff_needs(body):
    return bool(body.get("previous_response_id") or body.get("conversation") or _aff_opaque_values(body.get("input")))


def _aff_backend(body, token):
    recs = []
    if token:
        r = _aff_token_rec(token)
        if r is None:
            return None
        recs.append(r)
    for kind, value in _aff_continuation_values(body):
        r = _aff_get(_aff_key(kind, value))
        if r is None:
            return None
        recs.append(r)
    if not recs:
        return None
    b = (recs[0]["target"], recs[0]["base_url"], recs[0]["model"], recs[0]["target_revision"])
    if any((r["target"], r["base_url"], r["model"], r["target_revision"]) != b for r in recs[1:]):
        return None
    t = recs[0]["target"]
    if t not in BACKENDS:
        return None
    be = _backend_for(t)
    if (be["target"], be["base_url"].rstrip("/"), be["model"]) != (
        recs[0]["target"],
        recs[0]["base_url"],
        recs[0]["model"],
    ):
        return None
    return be if _supports_api(be, "responses") else None


def _aff_record_payload(backend, payload):
    if not isinstance(payload, dict):
        return
    r = _aff_record(backend)
    rid = payload.get("id") if isinstance(payload.get("id"), str) else None
    if not rid:
        rid = (payload.get("response") or {}).get("id") if isinstance(payload.get("response"), dict) else None
    if rid:
        _aff_put(_aff_key("response", rid), r)
    c = _aff_conv_value(payload.get("conversation")) or _aff_conv_value(
        (payload.get("response") or {}).get("conversation")
    )
    if c:
        _aff_put(_aff_key("conversation", c), r)
    for kind, value in _aff_opaque_values(payload):
        _aff_put(_aff_key(kind, value), r)


def _aff_issue(backend, payload=None):
    token = _aff_token()
    _aff_put(_aff_key("token", token), _aff_record(backend))
    if payload:
        _aff_record_payload(backend, payload)
    return token


# --- helpers --------------------------------------------------------------
def _safe_json(content):
    try:
        data = json.loads(content)
        return data if isinstance(data, dict) else {"error": {"message": content.decode(errors="replace")[:500]}}
    except Exception:
        return {"error": {"message": content.decode(errors="replace")[:500]}}


def _extract_prompt(body):
    for m in reversed(body.get("messages") or []):
        if isinstance(m, dict) and m.get("role") == "user":
            c = m.get("content")
            if isinstance(c, str):
                return c
            if isinstance(c, list):
                return " ".join(p.get("text", "") for p in c if isinstance(p, dict) and p.get("type") == "text")
            return str(c)
    return ""


def _extract_responses_prompt(body):
    v = body.get("input")
    if isinstance(v, str):
        return v
    if not isinstance(v, list):
        return ""

    def texts(item):
        c = item.get("content") if isinstance(item, dict) else None
        if isinstance(c, str):
            return [c]
        if not isinstance(c, list):
            return []
        return [
            p["text"]
            for p in c
            if isinstance(p, dict) and p.get("type") in {"input_text", "text"} and isinstance(p.get("text"), str)
        ]

    for item in reversed(v):
        if isinstance(item, dict) and item.get("role") == "user":
            ts = texts(item)
            if ts:
                return " ".join(ts)
    return " ".join(t for item in v if isinstance(item, dict) for t in texts(item))


def _is_refusal(status, data):
    if not isinstance(data, dict):
        return False
    # Bifrost and other providers may inject an error event inside an otherwise 200 SSE stream.
    code = data.get("status_code")
    if isinstance(code, int) and code >= 400:
        return True
    if data.get("is_bifrost_error"):
        return True
    parts = []
    err = data.get("error")
    if isinstance(err, dict):
        parts.extend(str(err.get(n, "")) for n in ("message", "code", "type"))
    elif err:
        parts.append(str(err))

    def nat(x):
        if not isinstance(x, dict):
            return False
        k = x.get("type")
        if k == "refusal" or (isinstance(k, str) and ".refusal" in k):
            return True
        for f in ("refusal", "text"):
            t = x.get(f)
            if isinstance(t, str):
                parts.append(t)
        for f in ("output", "content"):
            c = x.get(f)
            if isinstance(c, list) and any(nat(i) for i in c):
                return True
        for f in ("response", "item"):
            if nat(x.get(f)):
                return True
        return False

    if nat(data):
        return True
    ch = data.get("choices")
    if isinstance(ch, list):
        for c in ch:
            if not isinstance(c, dict):
                continue
            if c.get("finish_reason") == "content_filter":
                return True
            for m in (c.get("message"), c.get("delta")):
                if not isinstance(m, dict):
                    continue
                if isinstance(m.get("refusal"), str):
                    return True
                for f in ("content", "reasoning_content"):
                    v = m.get(f)
                    if isinstance(v, str):
                        parts.append(v)
    text = " ".join(parts)
    if status != 200 and re.search(r"content[\s_-]?filter", text, re.I):
        return True
    return bool(_REFUSAL_RE.search(text))


def _choice_has_payload(c):
    if not isinstance(c, dict):
        return False
    for o in (c.get("delta"), c.get("message")):
        if not isinstance(o, dict):
            continue
        v = o.get("content")
        if isinstance(v, str) and v.strip():
            return True
        if isinstance(v, list):
            for p in v:
                if isinstance(p, str) and p.strip():
                    return True
                if isinstance(p, dict) and str(p.get("text") or "").strip():
                    return True
        if o.get("tool_calls") or o.get("function_call"):
            return True
    return False


def _incomplete_tail(text):
    t = text.strip()
    if not t or t.endswith(("(", "[", "{")):
        return True
    return any(t.count(a) > t.count(b) for a, b in (("(", ")"), ("[", "]"), ("{", "}")))


def _is_empty_completion(data):
    if not isinstance(data, dict):
        return False
    ch = data.get("choices")
    if not isinstance(ch, list) or not ch:
        return True
    for c in ch:
        if not isinstance(c, dict):
            continue
        m = c.get("message") or {}
        if m.get("tool_calls") or m.get("function_call"):
            return False
        v = m.get("content")
        if isinstance(v, str) and v.strip() and not _incomplete_tail(v):
            return False
        if isinstance(v, list):
            text = "".join(
                p if isinstance(p, str) else str(p.get("text") or "") for p in v if isinstance(p, (str, dict))
            ).strip()
            if text and not _incomplete_tail(text):
                return False
    return True


def _length_truncated(data):
    ch = (data or {}).get("choices")
    return isinstance(ch, list) and any(isinstance(c, dict) and c.get("finish_reason") == "length" for c in ch)


def _possible_refusal_prefix(text):
    v = text.strip().lower()
    return any(
        prefix.startswith(v) or v.startswith(prefix)
        for prefix in ("i cannot", "i can't", "i’m sorry", "i'm sorry", "sorry", "unfortunately", "as an ai", "cannot")
        if len(v) >= 3
    )


# --- body builders --------------------------------------------------------
def _norm_msgs(msgs, dev_role="system"):
    out, seen, changed = [], set(), False
    for m in msgs:
        if not isinstance(m, dict):
            out.append(m)
            continue
        r = m.get("role")
        if r == "assistant":
            calls = m.get("tool_calls")
            if isinstance(calls, list):
                for c in calls:
                    if isinstance(c, dict) and isinstance(c.get("id"), str):
                        seen.add(c["id"])
            if m.get("function_call"):
                seen.add(None)
        elif r == "tool" and m.get("tool_call_id") not in seen:
            changed = True
            continue
        elif r == "function" and None not in seen:
            changed = True
            continue
        elif r == "developer" and dev_role != "native":
            m = {**m, "role": "system"}
            changed = True
        out.append(m)
    return out if changed else msgs


def _cap(backend):
    return min(backend.get("max_tokens") or MANTIS_ROUTER_MAX_TOKENS, MANTIS_ROUTER_MAX_TOKENS)


def _build_chat(body, backend):
    out = dict(body)
    if isinstance(out.get("messages"), list):
        out["messages"] = _norm_msgs(out["messages"], backend.get("developer_role", "system"))
    if isinstance(out.get("tools"), list):
        out["tools"] = sorted(
            out["tools"], key=lambda t: ((t.get("function") or {}).get("name") or t.get("name") or "")
        )
    if isinstance(out.get("functions"), list):
        out["functions"] = sorted(
            out["functions"], key=lambda t: ((t.get("function") or {}).get("name") or t.get("name") or "")
        )
    out["model"] = backend["model"]
    cap = _cap(backend)
    if isinstance(out.get("max_tokens"), int):
        out["max_tokens"] = min(out["max_tokens"], cap)
        out["max_completion_tokens"] = out["max_tokens"]
    elif isinstance(out.get("max_completion_tokens"), int):
        out["max_completion_tokens"] = min(out["max_completion_tokens"], cap)
        out["max_tokens"] = out["max_completion_tokens"]
    out.pop("stop", None)
    effort = backend.get("reasoning_effort") or out.get("reasoning_effort")
    if effort:
        out["reasoning_effort"] = effort
    return out


def _build_responses(body, backend):
    out = dict(body)
    if isinstance(out.get("tools"), list):
        out["tools"] = sorted(
            out["tools"], key=lambda t: ((t.get("function") or {}).get("name") or t.get("name") or "")
        )
    out["model"] = backend["model"]
    if isinstance(out.get("max_output_tokens"), int):
        out["max_output_tokens"] = min(out["max_output_tokens"], _cap(backend))
    reasoning = out.get("reasoning")
    if backend.get("force_reasoning_effort") and backend.get("reasoning_effort"):
        reasoning = dict(reasoning) if isinstance(reasoning, dict) else {}
        reasoning["effort"] = backend["reasoning_effort"]
        out["reasoning"] = reasoning
    elif backend.get("reasoning_effort") and "reasoning" not in out:
        out["reasoning"] = {"effort": backend["reasoning_effort"]}
    return out


# --- SSE ------------------------------------------------------------------
MAX_SSE = 1024 * 1024


def _sse_boundary(buf):
    ls = i = 0
    while i < len(buf):
        if buf[i] not in (10, 13):
            i += 1
            continue
        if buf[i] == 13 and i + 1 == len(buf):
            return None
        end = i + (2 if buf[i : i + 2] == b"\r\n" else 1)
        if i == ls:
            return end
        ls = end
        i = end
    return None


async def _iter_sse(resp):
    buf = bytearray()
    async for chunk in resp.aiter_bytes():
        buf.extend(chunk)
        while (end := _sse_boundary(buf)) is not None:
            yield bytes(buf[:end])
            del buf[:end]
        if len(buf) > MAX_SSE:
            raise ValueError("SSE event too large")
    if buf:
        if len(buf) > MAX_SSE:
            raise ValueError("SSE event too large")
        yield bytes(buf)


def _sse_data(event):
    vals = []
    for raw in event.splitlines():
        if raw.startswith(b"data:"):
            v = raw[5:]
            if v.startswith(b" "):
                v = v[1:]
            vals.append(v.decode("utf-8", errors="replace"))
    return "\n".join(vals) if vals else None


def _sse_refusal(content):
    buf = bytearray(content)
    while (end := _sse_boundary(buf)) is not None:
        ev = bytes(buf[:end])
        del buf[:end]
        d = _sse_data(ev)
        if not d or d.strip() == "[DONE]":
            continue
        try:
            p = json.loads(d)
            if isinstance(p, dict) and _is_refusal(200, p):
                return True
        except json.JSONDecodeError:
            continue
    d = _sse_data(bytes(buf))
    if d and d.strip() != "[DONE]":
        try:
            p = json.loads(d)
        except json.JSONDecodeError:
            return False
        return isinstance(p, dict) and _is_refusal(200, p)
    return False


def _response_replay_safe(body, content):
    if body.get("n", 1) != 1:
        return False
    if body.get("stream"):
        text = content.decode("utf-8", errors="replace")
        return (
            content.rstrip().endswith(b"data: [DONE]")
            and '"tool_calls"' not in text
            and '"function_call"' not in text
            and not re.search(r'"finish_reason"\s*:\s*"(?:content_filter|length)"', text)
            and not _REFUSAL_RE.search(text)
            and not _sse_refusal(content)
        )
    data = _safe_json(content)
    ch = data.get("choices") or []
    return (
        bool(ch)
        and not _is_refusal(200, data)
        and all(
            (c.get("message") or {}).get("finish_reason") not in {"content_filter", "length"}
            and not (c.get("message") or {}).get("tool_calls")
            and not (c.get("message") or {}).get("function_call")
            for c in ch
        )
    )


async def _prefetch_chat(events, deadline):
    prefix, text = [], []
    saw, tool = False, False
    pb = 0
    while True:
        rem = deadline - time.monotonic()
        if rem <= 0:
            raise TimeoutError
        try:
            ev = await asyncio.wait_for(anext(events), timeout=rem)
        except StopAsyncIteration:
            if prefix and saw and not tool and _incomplete_tail("".join(text)):
                return prefix, "empty_completion"
            return prefix, ("empty_completion" if prefix and not saw else None)
        prefix.append(ev)
        pb += len(ev)
        if pb >= MAX_SSE:
            return prefix, None
        d = _sse_data(ev)
        if d is None:
            continue
        if d.strip() == "[DONE]":
            if saw and not tool and _incomplete_tail("".join(text)):
                return prefix, "empty_completion"
            return prefix, (None if saw else "empty_completion")
        try:
            p = json.loads(d)
        except json.JSONDecodeError:
            return prefix, None
        if _is_refusal(200, p):
            return prefix, "refusal"
        for c in p.get("choices") or []:
            if not isinstance(c, dict):
                continue
            if c.get("finish_reason") == "content_filter":
                return prefix, "refusal"
            if _choice_has_payload(c):
                saw = True
            for o in (c.get("delta"), c.get("message")):
                if not isinstance(o, dict):
                    continue
                if o.get("tool_calls") or o.get("function_call"):
                    tool = True
                if isinstance(o.get("refusal"), str):
                    return prefix, "refusal"
                v = o.get("content")
                if isinstance(v, str) and v:
                    text.append(v)
        comb = "".join(text)
        if _REFUSAL_RE.search(comb):
            return prefix, "refusal"
        fr = next(
            (
                c.get("finish_reason")
                for c in (p.get("choices") or [])
                if isinstance(c, dict) and c.get("finish_reason") is not None
            ),
            None,
        )
        if fr is not None:
            if saw and not tool and _incomplete_tail(comb):
                return prefix, "empty_completion"
            return prefix, (None if saw else "empty_completion")
        if text and not _possible_refusal_prefix(comb):
            return prefix, None


def _resp_payload(event):
    d = _sse_data(event)
    if not d or d.strip() == "[DONE]":
        return None
    try:
        p = json.loads(d)
    except json.JSONDecodeError:
        return None
    return p if isinstance(p, dict) else None


def _resp_event_state(event):
    en = None
    for line in event.splitlines():
        if line.startswith(b"event:"):
            en = line[6:].strip().decode("utf-8", errors="replace")
            break
    d = _sse_data(event)
    if d is None:
        return (
            False,
            "upstream_error" if en in {"error", "response.failed", "response.incomplete"} else None,
            None,
            None,
        )
    if d.strip() == "[DONE]":
        return False, None, None, None
    try:
        p = json.loads(d)
    except json.JSONDecodeError:
        return False, None, None, None
    et = p.get("type") or en
    resp = p.get("response") if isinstance(p.get("response"), dict) else {}
    usage = resp.get("usage") or p.get("usage")
    rid = resp.get("id") if isinstance(resp.get("id"), str) else None
    if _is_refusal(200, p):
        return False, "refusal", usage, None
    if et == "response.completed":
        return (True, None, usage, rid) if resp.get("status") == "completed" else (False, "upstream_error", usage, rid)
    if et in {"error", "response.failed", "response.incomplete"} or "error" in p:
        return False, "upstream_error", usage, rid
    return False, None, usage, rid


async def _prefetch_resp(events, deadline):
    prefix, pb, n = [], 0, 0
    while True:
        rem = deadline - time.monotonic()
        if rem <= 0:
            raise TimeoutError
        try:
            ev = await asyncio.wait_for(anext(events), timeout=rem)
        except StopAsyncIteration:
            return prefix, False
        prefix.append(ev)
        pb += len(ev)
        n += 1
        if pb >= MAX_SSE or n >= 16:
            return prefix, False
        completed, failure, usage, rid = _resp_event_state(ev)
        if failure == "refusal":
            return prefix, True
        return prefix, False


def _stream_err(msg, code, rid):
    return (
        "data: "
        + json.dumps({"error": {"message": msg, "type": "upstream_error", "code": code}, "request_id": rid})
        + "\n\n"
    ).encode()


def _resp_stream_err(msg, code, rid):
    return (
        "event: error\ndata: "
        + json.dumps(
            {"type": "error", "error": {"message": msg, "type": "upstream_error", "code": code}, "request_id": rid}
        )
        + "\n\n"
    ).encode()


# --- upstream / failover --------------------------------------------------
_client = None


def _url_body(backend, body, api):
    if api == "responses":
        return backend["base_url"].rstrip("/") + "/responses", _build_responses(body, backend)
    return backend["base_url"].rstrip("/") + "/chat/completions", _build_chat(body, backend)


async def _send_upstream(backend, body, *, stream, api):
    if _client is None:
        raise RuntimeError("client not ready")
    url, out = _url_body(backend, body, api)
    req = _client.build_request(
        "POST",
        url,
        json=out,
        headers={"Authorization": f"Bearer {backend['key']}", "Content-Type": "application/json"},
        timeout=TIMEOUT_S,
    )
    return await _client.send(req, stream=stream)


def _retryable(status):
    return status in RETRY_STATUSES


async def _open_with_failover(body, decision, deadline, *, stream, api, effort=None, allow_failover=True):
    attempts = []
    routes = _api_routes(decision, api) if allow_failover else (decision,)
    for idx, route in enumerate(routes):
        backend = _backend_for(route)
        if idx == 0 and effort:
            backend = _apply_effort_override(backend, effort)
        resp = None
        try:
            rem = deadline - time.monotonic()
            if rem <= 0:
                raise TimeoutError
            resp = await asyncio.wait_for(_send_upstream(backend, body, stream=stream, api=api), timeout=rem)
        except (httpx.TransportError, TimeoutError, asyncio.TimeoutError) as e:
            attempts.append((route, backend, None, type(e).__name__))
            continue
        if resp.status_code != 200:
            content = await resp.aread()
            data = _safe_json(content)
            refusal = _is_refusal(resp.status_code, data)
            attempts.append((route, backend, resp.status_code, "refusal" if refusal else None))
            if (refusal or _retryable(resp.status_code)) and idx != len(routes) - 1:
                await resp.aclose()
                continue
            return route, backend, resp, attempts, None
        if not stream:
            content = await resp.aread()
            data = _safe_json(content)
            refusal = _is_refusal(200, data)
            empty = _is_empty_completion(data) if api == "chat" else False
            failure = "refusal" if refusal else ("empty_completion" if empty else None)
            attempts.append((route, backend, 200, failure))
            if failure and idx != len(routes) - 1:
                await resp.aclose()
                continue
            return route, backend, resp, attempts, None
        events = _iter_sse(resp)
        prefix, failure = (
            await _prefetch_resp(events, deadline) if api == "responses" else await _prefetch_chat(events, deadline)
        )
        attempts.append((route, backend, 200, failure))
        if failure and idx != len(routes) - 1:
            await resp.aclose()
            continue
        return route, backend, resp, attempts, (prefix, events)
    if not attempts:
        backend = _backend_for(decision)
        if effort:
            backend = _apply_effort_override(backend, effort)
        return decision, backend, None, [], None
    return attempts[-1][0], attempts[-1][1], None, attempts, None


# --- validation -----------------------------------------------------------
def _validate_chat(body):
    if body.get("model") != MODEL_ID:
        return "Only model 'auto' is supported", "model"
    msgs = body.get("messages")
    if not isinstance(msgs, list) or not msgs:
        return "'messages' must be non-empty", "messages"
    for i, m in enumerate(msgs):
        if not isinstance(m, dict) or m.get("role") not in {
            "system",
            "developer",
            "user",
            "assistant",
            "tool",
            "function",
        }:
            return "Invalid message role", f"messages.{i}.role"
        c = m.get("content")
        if c is not None and not isinstance(c, (str, list)):
            return "Invalid content", f"messages.{i}.content"
    for n in ("n", "max_tokens", "max_completion_tokens"):
        if n in body and body[n] <= 0:
            return f"'{n}' must be > 0", n
    return None


def _validate_resp(body):
    if body.get("model") != MODEL_ID:
        return "Only model 'auto' is supported", "model"
    if "input" not in body or body.get("input") is None:
        return "'input' required", "input"
    v = body["input"]
    if not isinstance(v, (str, list)) or not v:
        return "'input' must be non-empty text/array", "input"
    if "stream" in body and not isinstance(body["stream"], bool):
        return "'stream' invalid type", "stream"
    if "max_output_tokens" in body:
        m = body["max_output_tokens"]
        if isinstance(m, bool) or not isinstance(m, int) or m <= 0:
            return "'max_output_tokens' must be > 0", "max_output_tokens"
    return None


def _route_headers(decision, score, backend, rid, comp, sms, *, pinned=False, api="chat", reason=None):
    h = {
        "x-request-id": rid,
        "x-route-decision": decision,
        "x-route-target": backend["target"],
        "x-route-target-revision": TARGET_CONFIG_REVISION,
        "x-route-score": f"{score:.4f}" if isinstance(score, (int, float)) else "n/a",
        "x-route-model": backend["model"],
        "x-route-router": "supra",
        "x-route-fallback": "false",
        "x-route-attempts": "1",
        "x-route-api": api,
        "x-route-upstream-path": "/responses" if api == "responses" else "/chat/completions",
    }
    if pinned:
        h["x-route-pinned"] = "true"
    if comp is not None:
        h["x-route-supra-complexity"] = str(comp)
    if sms is not None:
        h["x-route-supra-ms"] = str(sms)
    if reason:
        h["x-route-supra-reason"] = reason
    return h


def _openai_err(msg, status, *, err="invalid_request_error", param=None, code=None):
    return JSONResponse({"error": {"message": msg, "type": err, "param": param, "code": code}}, status_code=status)


# --- middleware -----------------------------------------------------------
async def _body_limit(request, call_next):
    length = request.headers.get("content-length")
    if length and length.isdigit() and int(length) > MAX_BODY_BYTES:
        return _openai_err("Request body is too large", 413, code="request_too_large")
    return await call_next(request)


async def _read_json_body(request):
    chunks, size = [], 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > MAX_BODY_BYTES:
            raise OverflowError
        chunks.append(chunk)
    try:
        data = json.loads(b"".join(chunks))
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise ValueError("Invalid JSON body")
    if not isinstance(data, dict):
        raise ValueError("Request body must be a JSON object")
    return data


def _authorize(a):
    return bool(a and a.startswith("Bearer ") and secrets.compare_digest(a[7:].encode(), SERVER_KEY.encode()))


# --- app ------------------------------------------------------------------
_READY = False


@asynccontextmanager
async def lifespan(app):
    global _READY, _client
    _init_db()
    _client = httpx.AsyncClient(timeout=httpx.Timeout(TIMEOUT_S, connect=min(10.0, TIMEOUT_S)))
    try:
        await asyncio.to_thread(_load_supra)
        await asyncio.to_thread(_store_load)
        _READY = True
        yield
    finally:
        _READY = False
        if _client:
            await _client.aclose()
        _client = None


app = FastAPI(title="Mantis lean gateway", lifespan=lifespan)


@app.middleware("http")
async def mw(request, call_next):
    return await _body_limit(request, call_next)


@app.get("/healthz")
async def healthz():
    return {
        "ready": _READY,
        "router": "supra",
        "targets": list(BACKENDS),
        "supra_targets": list(SUPRA_TARGETS),
        "supra_invalid_target": SUPRA_INVALID_TARGET,
        "target_config_revision": TARGET_CONFIG_REVISION,
        "supra_fallback_count": SUPRA_FALLBACK_COUNT,
        "cache": {**_cache_metrics, "entries": len(_resp_cache), "bytes": _cache_bytes},
    }


@app.get("/v1/models")
async def models():
    ctx = int(os.environ.get("MANTIS_CONTEXT_LENGTH", "262144"))
    return {
        "object": "list",
        "data": [
            {
                "id": m,
                "object": "model",
                "owned_by": "mantis",
                "context_window": ctx,
                "max_tokens": MANTIS_ROUTER_MAX_TOKENS,
            }
            for m in [MODEL_ID] + list(BACKENDS)
        ],
    }


# --- chat endpoint --------------------------------------------------------
@app.post("/v1/chat/completions")
async def chat_completions(
    request: Request,
    authorization: str | None = Header(default=None),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
):
    rid = request.headers.get("x-request-id") or f"req_{secrets.token_hex(12)}"
    if not _authorize(authorization):
        return _openai_err("Invalid API key", 401, err="authentication_error", code="invalid_api_key")
    try:
        body = await _read_json_body(request)
    except OverflowError:
        return _openai_err("Request body is too large", 413, code="request_too_large")
    except ValueError as e:
        return _openai_err(str(e), 400, code="invalid_json")
    inv = _validate_chat(body)
    if inv:
        return _openai_err(inv[0], 400, param=inv[1], code="invalid_request")

    sid, _ = _session_id(body, request)
    new_task = request.headers.get("x-route-new-task", "").lower() in {"1", "true", "yes"}
    ckey = _cache_key(body, idempotency_key, sid=sid)
    if new_task:
        ckey = None
    cached = _cache_get(ckey)
    if cached:
        return _replayed_response(cached, rid)

    leader, inflight = await _claim_inflight(ckey)
    if not leader:
        res = await asyncio.shield(inflight)
        if res:
            return _replayed_response(res, rid)
        leader, inflight = await _claim_inflight(ckey)

    upstream = None
    try:
        prompt = _extract_prompt(body)
        ph = _prompt_hash(prompt)
        if prompt.strip():
            if sid is None:
                proposed, score, comp, sms = await asyncio.to_thread(_decide, prompt)
            else:
                proposed, score, comp, sms = await asyncio.to_thread(_decide, prompt, sid)
        else:
            proposed, score, comp, sms = _safe_target(), None, None, None
        decision, reason = _session_route(sid, prompt, proposed, comp, score, new_task=new_task, api="chat")
        compat = _chat_tier(decision)
        if compat is None:
            _finish_inflight_nowait(ckey, inflight, None)
            return _openai_err(
                "No chat-compatible target", 503, err="configuration_error", code="chat_backend_unavailable"
            )
        upgraded = compat != decision
        if upgraded:
            decision, reason = compat, "chat_protocol_upgrade"
        pinned = not upgraded and sid is None and _store_pinned(ph, "chat") is not None
        backend = _backend_for(decision)
        headers = _route_headers(decision, score, backend, rid, comp, sms, pinned=pinned, api="chat")
        headers["x-route-reason"] = reason
        if sid:
            headers["x-route-session"] = sid
        dl = time.monotonic() + TIMEOUT_S
        selected, backend, upstream, attempts, stream_extra = await _open_with_failover(
            body, decision, dl, stream=bool(body.get("stream")), api="chat", effort=_effort_for_complexity(comp)
        )
        _record_refusal_learning(ph, attempts, sid, comp, api="chat")
    except BaseException:
        _finish_inflight_nowait(ckey, inflight, None)
        if upstream:
            await asyncio.shield(upstream.aclose())
        raise

    headers.update(
        {
            "x-route-decision": selected,
            "x-route-target": backend["target"],
            "x-route-model": backend["model"],
            "x-route-fallback": str(selected != decision).lower(),
            "x-route-attempts": str(len(attempts)),
        }
    )

    if upstream is None:
        _finish_inflight_nowait(ckey, inflight, None)
        _store_note(ph, selected, ok=False, score=score, api="chat")
        return _openai_err("Upstream unavailable", 502, err="upstream_error", code="upstream_unavailable")

    if not body.get("stream"):
        try:
            content = await asyncio.wait_for(upstream.aread(), timeout=max(0, dl - time.monotonic()))
        except (httpx.TransportError, asyncio.TimeoutError, TimeoutError):
            await upstream.aclose()
            _finish_inflight_nowait(ckey, inflight, None)
            _store_note(ph, selected, ok=False, score=score, api="chat")
            return _openai_err("Upstream timed out", 504, err="upstream_error", code="upstream_timeout")
        finally:
            await upstream.aclose()
        data = _safe_json(content)
        refused = _is_refusal(upstream.status_code, data)
        empty = _is_empty_completion(data)
        trunc = _length_truncated(data)
        n_expected = int(body.get("n", 1) or 1)
        ch = data.get("choices") or []
        complete = (
            len([c for c in ch if isinstance(c, dict) and c.get("finish_reason")]) == n_expected
            and len(set(c.get("index", 0) for c in ch if isinstance(c, dict))) == n_expected
        )
        ok = upstream.status_code == 200 and not refused and not empty and not trunc and complete
        if ok:
            if sid is None:
                _store_note(ph, selected, ok=True, score=score, api="chat")
            else:
                _session_note(sid, selected, comp, data.get("usage"), api="chat")
        else:
            _store_note(ph, selected, ok=False, score=score, api="chat")
        result = (
            _cache_put(ckey, body, content, headers, "application/json", upstream.status_code)
            if ckey and ok and not body.get("stream")
            else None
        )
        await _finish_inflight(ckey, inflight, result)
        return Response(
            content=content, status_code=upstream.status_code, media_type="application/json", headers=headers
        )

    prefix, events = stream_extra or ([], _iter_sse(upstream))

    async def event_stream():
        parts, size, saw_done, saw_finish, saw_length, saw_refusal, saw_payload = (
            [],
            0,
            False,
            False,
            False,
            False,
            False,
        )
        usage = {}

        def remember(ev):
            nonlocal size, parts
            if parts is None:
                return
            size += len(ev)
            if size > RESP_CACHE_MAX_BYTES:
                parts = None
                return
            parts.append(ev)

        def track(ev):
            nonlocal saw_done, saw_finish, saw_length, saw_refusal, saw_payload
            d = _sse_data(ev)
            if d is None:
                return False
            if d.strip() == "[DONE]":
                saw_done = True
                return True
            try:
                p = json.loads(d)
            except json.JSONDecodeError:
                return False
            if _is_refusal(200, p):
                saw_refusal = True
            for c in p.get("choices") or []:
                if not isinstance(c, dict):
                    continue
                if _choice_has_payload(c):
                    saw_payload = True
                if c.get("finish_reason") is not None:
                    saw_finish = True
                    if c.get("finish_reason") == "length":
                        saw_length = True
            if isinstance(p.get("usage"), dict):
                usage.update(p["usage"])
            return False

        try:
            async with asyncio.timeout(max(0, dl - time.monotonic())):

                async def all_events():
                    for e in prefix:
                        yield e
                    async for e in events:
                        yield e

                async for ev in all_events():
                    if await request.is_disconnected():
                        raise asyncio.CancelledError
                    track(ev)
                    if (saw_finish or saw_done) and not saw_payload and not saw_refusal:
                        yield _stream_err("Upstream completed without content or tool calls", "empty_completion", rid)
                        break
                    remember(ev)
                    yield ev
                    if saw_done or saw_finish:
                        break
            if not saw_refusal and saw_done and saw_finish and saw_payload and not saw_length:
                if sid is None:
                    _store_note(ph, selected, ok=True, score=score, api="chat")
                else:
                    _session_note(sid, selected, comp, usage or None, api="chat")
            elif not saw_refusal:
                if not saw_done and not saw_finish:
                    yield _stream_err("Stream ended before completion", "upstream_truncated", rid)
                elif not saw_payload and not saw_refusal:
                    yield _stream_err("Upstream completed without content", "empty_completion", rid)
            if not saw_done:
                done = b"data: [DONE]\n\n"
                remember(done)
                yield done
                saw_done = True
        except asyncio.CancelledError:
            raise
        except (httpx.TransportError, TimeoutError, asyncio.TimeoutError, ValueError):
            yield _stream_err("Upstream stream failed", "upstream_transport_error", rid)
        finally:
            await upstream.aclose()
            result = None
            if parts is not None and saw_done and saw_finish and saw_payload and not saw_refusal and not saw_length:
                content = b"".join(parts)
                if _response_replay_safe(body, content):
                    result = _cache_put(ckey, body, content, headers, "text/event-stream", 200)
            await _finish_inflight(ckey, inflight, result)

    return StreamingResponse(event_stream(), media_type="text/event-stream", headers=headers)


# --- responses endpoint ---------------------------------------------------
@app.post("/v1/responses")
async def responses(request: Request, authorization: str | None = Header(default=None)):
    rid = request.headers.get("x-request-id") or f"req_{secrets.token_hex(12)}"
    if not _authorize(authorization):
        return _openai_err("Invalid API key", 401, err="authentication_error", code="invalid_api_key")
    try:
        body = await _read_json_body(request)
    except OverflowError:
        return _openai_err("Request body is too large", 413, code="request_too_large")
    except ValueError as e:
        return _openai_err(str(e), 400, code="invalid_json")
    inv = _validate_resp(body)
    if inv:
        return _openai_err(inv[0], 400, param=inv[1], code="invalid_request")

    prompt = _extract_responses_prompt(body)
    ph = _prompt_hash(prompt)
    sid, _ = _session_id(body, request)
    aff_token = request.headers.get(_AFF_HDR.lower())
    bound = _aff_backend(body, aff_token) if _aff_needs(body) else None
    if _aff_needs(body) and bound is None:
        return _openai_err(
            "Responses continuation requires a valid route affinity",
            409,
            err="configuration_error",
            code="responses_continuation_affinity_required",
        )

    if bound:
        decision = bound["target"]
        backend = bound
        reason = "responses_affinity"
        upgraded = False
        pinned = False
        score = comp = sms = None
    else:
        if prompt.strip():
            proposed, score, comp, sms = await asyncio.to_thread(_decide, prompt, sid, "responses")
        else:
            proposed, score, comp, sms = _safe_target(), None, None, None
        new_task = request.headers.get("x-route-new-task", "").lower() in {"1", "true", "yes"}
        decision, reason = _session_route(sid, prompt, proposed, comp, score, new_task=new_task, api="responses")
        compat = _responses_tier(decision)
        if compat is None:
            return _openai_err(
                "No responses-compatible target", 503, err="configuration_error", code="responses_backend_unavailable"
            )
        upgraded = compat != decision
        if upgraded:
            decision, reason = compat, "responses_protocol_upgrade"
        backend = _backend_for(decision)
        pinned = not upgraded and sid is None and _store_pinned(ph, "responses") is not None

    headers = _route_headers(decision, score, backend, rid, comp, sms, pinned=pinned, api="responses")
    headers["x-route-reason"] = reason
    if sid:
        headers["x-route-session"] = sid
    dl = time.monotonic() + TIMEOUT_S
    upstream = None
    try:
        selected, backend, upstream, attempts, stream_extra = await _open_with_failover(
            body,
            decision,
            dl,
            stream=bool(body.get("stream")),
            api="responses",
            effort=_effort_for_complexity(comp),
            allow_failover=bound is None,
        )
        if bound is None:
            _record_refusal_learning(ph, attempts, sid, comp, api="responses")
    except BaseException:
        if upstream:
            await asyncio.shield(upstream.aclose())
        raise

    headers.update(
        {
            "x-route-decision": selected,
            "x-route-target": backend["target"],
            "x-route-model": backend["model"],
            "x-route-fallback": str(selected != decision).lower(),
            "x-route-attempts": str(len(attempts)),
        }
    )

    if upstream is None:
        if bound is None:
            _store_note(ph, selected, ok=False, score=score, api="responses")
        return _openai_err("Responses upstream unavailable", 502, err="upstream_error", code="upstream_unavailable")

    if not body.get("stream"):
        try:
            content = await asyncio.wait_for(upstream.aread(), timeout=max(0, dl - time.monotonic()))
        except (httpx.TransportError, asyncio.TimeoutError, TimeoutError):
            await upstream.aclose()
            if bound is None:
                _store_note(ph, selected, ok=False, score=score, api="responses")
            return _openai_err("Upstream timed out", 504, err="upstream_error", code="upstream_timeout")
        finally:
            await upstream.aclose()
        data = _safe_json(content)
        usage = data.get("usage") or (data.get("response") or {}).get("usage")
        refused = _is_refusal(upstream.status_code, data)
        ok = upstream.status_code == 200 and data.get("status") == "completed" and not refused
        if ok:
            token = _aff_issue(backend, data)
            if token:
                headers[_AFF_HDR] = token
            if bound is None:
                if sid is None:
                    _store_note(ph, selected, ok=True, score=score, api="responses")
                else:
                    _session_note(sid, selected, comp, usage, api="responses")
        else:
            if bound is None:
                _store_note(ph, selected, ok=False, score=score, api="responses")
        return Response(
            content=content,
            status_code=upstream.status_code,
            media_type=upstream.headers.get("content-type", "application/json"),
            headers=headers,
        )

    prefix, events = stream_extra or ([], _iter_sse(upstream))
    token = _aff_issue(backend)
    if token:
        headers[_AFF_HDR] = token

    async def resp_stream():
        completed = failed = False
        terminal = None
        usage = {}
        try:
            async with asyncio.timeout(max(0, dl - time.monotonic())):

                async def all_events():
                    for e in prefix:
                        yield e
                    async for e in events:
                        yield e

                async for ev in all_events():
                    if await request.is_disconnected():
                        raise asyncio.CancelledError
                    ev_completed, fail, us, _ = _resp_event_state(ev)
                    if us:
                        usage.update(us)
                    p = _resp_payload(ev)
                    if p:
                        _aff_record_payload(backend, p)
                    yield ev
                    completed = completed or ev_completed
                    if fail:
                        terminal = fail
                        failed = True
                    if completed or (failed and terminal != "refusal"):
                        break
            if completed:
                if bound is None:
                    if sid is None:
                        _store_note(ph, selected, ok=True, score=score, api="responses")
                    else:
                        _session_note(sid, selected, comp, usage or None, api="responses")
            elif failed and terminal == "refusal":
                _record_refusal_learning(
                    ph, [(selected, backend, upstream.status_code, "refusal")], sid, comp, api="responses"
                )
            else:
                yield _resp_stream_err("Responses stream ended before completion", "upstream_truncated", rid)
        except asyncio.CancelledError:
            raise
        except (httpx.TransportError, TimeoutError, asyncio.TimeoutError, ValueError):
            yield _resp_stream_err("Responses stream failed", "upstream_transport_error", rid)
        finally:
            await asyncio.shield(upstream.aclose())

    return StreamingResponse(resp_stream(), media_type="text/event-stream", headers=headers)


def _bind_loopback(h):
    return h in {"127.0.0.1", "::1", "localhost"}


def _validate_bind():
    if not _bind_loopback(HOST) and (not os.environ.get("MANTIS_ROUTER_KEY") or SERVER_KEY == "sk-route-local"):
        raise RuntimeError("non-loopback binding requires an externally supplied, non-default MANTIS_ROUTER_KEY")


if __name__ == "__main__":
    import uvicorn

    _validate_bind()
    print(f"lean gateway: targets={','.join(BACKENDS)} port={PORT} revision={TARGET_CONFIG_REVISION}", flush=True)
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
