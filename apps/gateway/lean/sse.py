"""SSE parsing, event helpers, and prefetch logic."""

from __future__ import annotations

import asyncio
import json
import re
import time

from lean.helpers import (
    _REFUSAL_RE,
    _choice_has_payload,
    _incomplete_tail,
    _is_refusal,
    _possible_refusal_prefix,
    _safe_json,
)

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
        _, failure, _, _ = _resp_event_state(ev)
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
