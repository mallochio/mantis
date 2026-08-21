"""HMAC session IDs, session state, and ratchet logic."""

from __future__ import annotations

import hashlib
import hmac
import os
import re
import threading
import time

from cachetools import TTLCache

from lean.config import (
    BACKENDS,
    DOWNGRADE_IDLE_S,
    RESCORE_EVERY_N,
    SERVER_KEY,
    SESSION_STATE_MAX,
    SESSION_TTL_S,
    TARGET_CONFIG_REVISION,
    _backend_for,
    _safe_target,
    _supports_api,
    _target_rank,
)

_CONTINUATION_RE = re.compile(
    r"^(?:ok(?:ay)?[,. ]*)?(?:proceed|continue|go ahead|do (?:it|that)|yes|yep|sure|run (?:it|them|the tests)(?: again)?|try again|fix (?:it|that)|next)(?:[.! ]*)$",
    re.I,
)
_NEW_TASK_RE = re.compile(r"^(?:new task|different task|unrelated|switching topics?|on another topic)\b", re.I)

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
