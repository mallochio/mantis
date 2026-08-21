"""Responses API origin binding."""

from __future__ import annotations

import hashlib
import hmac
import secrets
import threading
import time

from cachetools import LRUCache

from lean.config import (
    AFFINITY_MAX,
    AFFINITY_TTL_S,
    BACKENDS,
    SERVER_KEY,
    TARGET_CONFIG_REVISION,
    _backend_for,
    _supports_api,
)

_aff = LRUCache(maxsize=AFFINITY_MAX)
_aff_lock = threading.RLock()
AFF_HDR = "x-route-responses-affinity"


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
