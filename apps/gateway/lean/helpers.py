"""Shared helper functions used across the gateway."""

from __future__ import annotations

import json
import re

_REFUSAL_RE = re.compile(
    r"cannot (assist|help|comply)|i('| a)?m sorry|not (able|allowed) to (assist|help)|can('?t) (assist|help)", re.I
)


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
