"""Environment, catalog loading, BACKENDS, and target helpers."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from urllib.parse import urlsplit

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
        raise _cfg_err(f"{field} is not a valid URL") from None
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
        import tomllib

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
