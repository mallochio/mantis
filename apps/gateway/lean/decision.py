"""SQLite decision store and Supra-based routing decisions."""

from __future__ import annotations

import sqlite3
import threading
import time

import lean.supra
from lean.cache import _prompt_hash
from lean.config import (
    BACKENDS,
    DECISION_STORE_PATH,
    PIN_CHEAP_AFTER,
    PIN_EXPENSIVE_AFTER,
    PIN_TTL_S,
    TARGET_CONFIG_REVISION,
    _backend_for,
    _lowest_rank,
    _refusal_target,
    _safe_target,
    _supports_api,
    _target_for_complexity,
    _target_rank,
)
from lean.session import _is_continuation, _session_get, _session_note, _session_target
from lean.supra import _supra_complexity

_db_conn = None
_db_lock = threading.RLock()
_decision_cache: dict = {}


def _init_db():
    global _db_conn
    DECISION_STORE_PATH.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    _db_conn = sqlite3.connect(str(DECISION_STORE_PATH), check_same_thread=False)
    _db_conn.row_factory = sqlite3.Row
    _db_conn.execute(
        """CREATE TABLE IF NOT EXISTS decisions (
            key TEXT PRIMARY KEY,
            api_format TEXT,
            decision TEXT,
            ok INTEGER DEFAULT 0,
            fail INTEGER DEFAULT 0,
            score REAL,
            pin_until REAL,
            ts REAL,
            target_revision TEXT
        )"""
    )
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
                "INSERT OR REPLACE INTO decisions "
                "(key, api_format, decision, ok, fail, score, pin_until, ts, target_revision) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
        lean.supra.SUPRA_FALLBACK_COUNT += 1
        print(f"Supra failed ({err}); defaulting", flush=True)
        return _safe_target(), None, None, None
