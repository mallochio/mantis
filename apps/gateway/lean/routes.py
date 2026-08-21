"""FastAPI endpoint handlers for the lean gateway."""

from __future__ import annotations

import asyncio
import json
import os
import secrets
import time

import httpx
from fastapi import Header, Request
from fastapi.responses import Response, StreamingResponse

import lean.state as _state
import lean.supra as _supra
from lean.affinity import AFF_HDR, _aff_backend, _aff_issue, _aff_needs, _aff_record_payload
from lean.cache import (
    _cache_bytes,
    _cache_get,
    _cache_key,
    _cache_metrics,
    _cache_put,
    _claim_inflight,
    _finish_inflight,
    _finish_inflight_nowait,
    _prompt_hash,
    _replayed_response,
    _resp_cache,
)
from lean.common import _authorize, _openai_err, _read_json_body
from lean.config import (
    BACKENDS,
    MANTIS_ROUTER_MAX_TOKENS,
    MODEL_ID,
    RESP_CACHE_MAX_BYTES,
    SUPRA_INVALID_TARGET,
    SUPRA_TARGETS,
    TARGET_CONFIG_REVISION,
    TIMEOUT_S,
    _backend_for,
    _chat_tier,
    _effort_for_complexity,
    _responses_tier,
    _safe_target,
)
from lean.decision import _decide, _record_refusal_learning, _store_note, _store_pinned
from lean.helpers import (
    _choice_has_payload,
    _extract_prompt,
    _extract_responses_prompt,
    _is_empty_completion,
    _is_refusal,
    _length_truncated,
    _safe_json,
)
from lean.proxy import _open_with_failover
from lean.session import _session_id, _session_note, _session_route
from lean.sse import (
    _iter_sse,
    _resp_event_state,
    _resp_payload,
    _resp_stream_err,
    _response_replay_safe,
    _sse_data,
    _stream_err,
)


def healthz():
    return {
        "ready": _state._READY,
        "router": "supra",
        "targets": list(BACKENDS),
        "supra_targets": list(SUPRA_TARGETS),
        "supra_invalid_target": SUPRA_INVALID_TARGET,
        "target_config_revision": TARGET_CONFIG_REVISION,
        "supra_fallback_count": _supra.SUPRA_FALLBACK_COUNT,
        "cache": {**_cache_metrics, "entries": len(_resp_cache), "bytes": _cache_bytes},
    }


def models():
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

    _leader, inflight = await _claim_inflight(ckey)
    if not _leader:
        res = await asyncio.shield(inflight)
        if res:
            return _replayed_response(res, rid)
        _leader, inflight = await _claim_inflight(ckey)

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
    aff_token = request.headers.get(AFF_HDR.lower())
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
                headers[AFF_HDR] = token
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
        headers[AFF_HDR] = token

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
