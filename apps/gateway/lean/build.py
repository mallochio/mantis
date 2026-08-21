"""Request body builders and message normalization."""

from __future__ import annotations

from lean.config import MANTIS_ROUTER_MAX_TOKENS


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
