#!/usr/bin/env python3
# OpenFugu — Apache-2.0. Part of an independent, open reimplementation of
# the Fugu orchestrator. NOT affiliated with Sakana AI. See NOTICE.
# Reference: OpenAI-compatible serving layer for the OpenFugu TRINITY + Conductor coordinators.
"""
serve.py — Fugu as a single OpenAI-compatible model endpoint.

A client POSTs to /v1/chat/completions as if calling one model; internally the
requested coordinator ("trinity" or "conductor") runs the full loop. The
model field in the request selects the coordinator.

stdlib http.server only — no FastAPI/uvicorn.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
import select
import socket
import sys
import threading
import time
import urllib.parse
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
# When running from the repo's openfugu-patch overlay, also find upstream OpenFugu.
if not (_HERE / "mini.py").exists():
    _OPENFUGU = _HERE.parent / "OpenFugu" / "openfugu"
    if _OPENFUGU.exists():
        sys.path.insert(0, str(_OPENFUGU))

from mini import (
    DEFAULT_SLOT_LABELS,
    HEAD_ROWS,
    HIDDEN,
    ROUTER_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    THINKER_PROMPT,
    VERIFICATION_PROMPT,
    Coordinator,
    FuguRouter,
)
from mini import LiteLLMWorker as _TrinityLiteLLMWorker
from ultra import ConductorExecutor, conductor_prompt, parse_workflow, visible_indices
from ultra import LiteLLMWorker as _ConductorLiteLLMWorker

ROUTER: FuguRouter | None = None
_router_lock = threading.Lock()
MODEL_NAME = os.environ.get("MANTIS_MODEL_NAME", "mantis")
MAX_TURNS = 5
# Reject bodies larger than this many bytes.
MAX_BODY_BYTES = int(os.environ.get("MANTIS_MAX_BODY_BYTES", str(5 * 1024 * 1024)))
WORKER_TIMEOUT = float(os.environ.get("MANTIS_WORKER_TIMEOUT", "240"))

# Aliases that carry a LiteLLM reasoning_effort parameter. OpenRouter/LiteLLM
# reject temperature != 1 for these models.
REASONING_ALIASES = ("claude-", "gpt-5.6-", "expensive", "cheap")

_args: argparse.Namespace | None = None
_coordinators: dict[str, object] = {}
_coordinator_lock = threading.Lock()
_history_context = threading.local()


class ClientDisconnectedError(Exception):
    """Raised when client disconnects during streaming or step execution."""


class RequestBodyTooLargeError(Exception):
    """Raised before an oversized request body is allocated."""


def _check_client_connected() -> None:
    """Check if current request client connection is broken or aborted."""
    if getattr(_history_context, "aborted", False):
        raise ClientDisconnectedError("Client disconnected")
    is_connected = getattr(_history_context, "is_client_connected", None)
    if is_connected is not None and not is_connected():
        _history_context.aborted = True
        raise ClientDisconnectedError("Client disconnected")


def _is_reasoning_model(model: str) -> bool:
    name = model.rsplit("/", 1)[-1]
    return any(name.startswith(p) for p in REASONING_ALIASES)


def _litellm_api_key() -> str | None:
    """Return the upstream proxy key, independent from Mantis ingress auth."""
    return os.environ.get("MANTIS_LITELLM_API_KEY") or os.environ.get("LITELLM_KEY")


def _litellm_base_url() -> str:
    return os.environ.get("MANTIS_BASE_URL", "http://127.0.0.1:3001/v1")


def _build_litellm_kwargs(
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float,
    timeout: float | None = None,
) -> dict[str, Any]:
    """Construct kwargs for a LiteLLM completion routed through the OpenRouter proxy."""
    if timeout is None:
        timeout = WORKER_TIMEOUT
    kw: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "custom_llm_provider": "openai",
        "timeout": timeout,
    }
    # LiteLLM's openai provider rejects temperature != 1 when reasoning_effort
    # is enabled (claude-*, gpt-5.6-*). Drop it for those models while keeping
    # it for the low-cost non-reasoning workers (deepseek/glm/opencode).
    if not _is_reasoning_model(model):
        kw["temperature"] = temperature
    return kw


class RejectAwareRouter:
    """Force a revision after a verifier rejects instead of re-verifying unchanged text."""

    def __init__(self, router: Any) -> None:
        self._router = router

    def route(self, *args: Any, **kwargs: Any) -> Any:
        result = self._router.route(*args, **kwargs)
        if getattr(_history_context, "force_worker", False):
            _history_context.force_worker = False
            result = {**result, "role_id": 0, "role_name": "Worker"}
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self._router, name)


class HistoryWorker:
    """Wrap a worker so every LLM call sees the conversation history.

    The coordinator classes only know the current query; this wrapper prepends
    the prior user/assistant turns (carried in a per-request thread-local) to
    the messages list handed to the underlying worker. This makes multi-turn
    coding sessions work without modifying the upstream Coordinator code."""

    def __init__(self, worker: Any) -> None:
        self._worker = worker

    def _combine(self, messages: Any) -> Any:
        history = getattr(_history_context, "history", None) or []
        if not history or not isinstance(messages, list):
            return messages
        system_context = "\n\n".join(
            str(m.get("content", ""))
            for m in history
            if isinstance(m, dict) and m.get("role") == "system" and m.get("content")
        )
        prior = [m for m in history if isinstance(m, dict) and m.get("role") != "system"]
        if messages and isinstance(messages[0], dict) and messages[0].get("role") == "system":
            first = messages[0]
            if system_context:
                first = {**first, "content": f"{first.get('content', '')}\n\n{system_context}"}
            return [first] + prior + list(messages[1:])
        if system_context:
            return [{"role": "system", "content": system_context}] + prior + list(messages)
        return prior + list(messages)

    def _model_name(self, agent_id: int) -> str:
        labels = (
            getattr(self._worker, "slot_models", None) or getattr(self._worker, "names", None) or []
        )
        if not labels:
            return f"slot-{agent_id}"
        return str(labels[agent_id % len(labels)])

    def _last_user_prompt(self, messages: Any) -> str:
        if not isinstance(messages, list):
            return ""
        for m in reversed(messages):
            if isinstance(m, dict) and m.get("role") == "user":
                return str(m.get("content", ""))
            if getattr(m, "role", None) == "user":
                return str(getattr(m, "content", ""))
        return ""

    def __call__(self, *args: Any) -> Any:
        _check_client_connected()
        if len(args) == 3:
            role_or_subtask, messages, agent_id = args
            combined = self._combine(messages)
            is_conductor = getattr(_history_context, "conductor_mode", False)
            known_roles = {"Worker", "Thinker", "Verifier"}
            role = role_or_subtask if role_or_subtask in known_roles else "Worker"

            if is_conductor:
                original_prompt = self._last_user_prompt(messages)
                call: dict[str, Any] = {
                    "role": role,
                    "agent_id": agent_id,
                    "model_name": self._model_name(agent_id),
                    "messages": combined,
                    "prompt": original_prompt,
                }
                calls = getattr(_history_context, "calls", None)
                if calls is not None:
                    calls.append(call)
                write_line = getattr(_history_context, "write_line", None)
                turn_index = len(calls) - 1 if calls else 0
                if write_line:
                    write_line(
                        {
                            "type": "step-start",
                            "turn": turn_index,
                            "role": role,
                            "agent_id": agent_id,
                            "model_name": call["model_name"],
                            "prompt": call["prompt"],
                        }
                    )
                reply = self._worker(role_or_subtask, combined, agent_id)
                call["reply"] = reply
                if write_line:
                    write_line(
                        {
                            "type": "step-end",
                            "turn": turn_index,
                            "role": role,
                            "agent_id": agent_id,
                            "model_name": call["model_name"],
                            "prompt": call["prompt"],
                            "reply": reply,
                        }
                    )

                if reply and reply.strip():
                    return reply

                # Bounded retry once for an empty Conductor node
                retry_instruction = (
                    "\n\nPrevious attempt produced an empty response; produce a complete answer."
                )
                last_msg = (
                    combined[-1]
                    if combined and isinstance(combined[-1], dict)
                    else {"role": "user", "content": ""}
                )
                retry_content = str(last_msg.get("content", "")) + retry_instruction
                retry_messages = combined[:-1] + [{**last_msg, "content": retry_content}]
                retry_prompt = self._last_user_prompt(retry_messages)
                retry_call: dict[str, Any] = {
                    "role": role,
                    "agent_id": agent_id,
                    "model_name": self._model_name(agent_id),
                    "messages": retry_messages,
                    "prompt": retry_prompt,
                }
                if calls is not None:
                    calls.append(retry_call)
                turn_index_retry = len(calls) - 1 if calls else 0
                if write_line:
                    write_line(
                        {
                            "type": "step-start",
                            "turn": turn_index_retry,
                            "role": role,
                            "agent_id": agent_id,
                            "model_name": retry_call["model_name"],
                            "prompt": retry_call["prompt"],
                        }
                    )
                reply_retry = self._worker(role_or_subtask, retry_messages, agent_id)
                retry_call["reply"] = reply_retry
                if write_line:
                    write_line(
                        {
                            "type": "step-end",
                            "turn": turn_index_retry,
                            "role": role,
                            "agent_id": agent_id,
                            "model_name": retry_call["model_name"],
                            "prompt": retry_call["prompt"],
                            "reply": reply_retry,
                        }
                    )

                if reply_retry and reply_retry.strip():
                    return reply_retry

                raise ValueError(
                    f"Conductor subtask node {agent_id} returned empty response after retry."
                )

            if role == "Worker":
                feedback = getattr(_history_context, "revision_feedback", None)
                if feedback and combined and isinstance(combined[-1], dict):
                    prev_content = combined[-1].get("content", "")
                    combined[-1] = {
                        **combined[-1],
                        "content": (
                            f"{prev_content}\n\n"
                            f"Revise the answer to address this verifier feedback:\n{feedback}"
                        ),
                    }
                    _history_context.revision_feedback = None
            original_prompt = self._last_user_prompt(messages)
            call = {
                "role": role,
                "agent_id": agent_id,
                "model_name": self._model_name(agent_id),
                "messages": combined,
                "prompt": original_prompt,
            }
            calls = getattr(_history_context, "calls", None)
            if calls is not None:
                calls.append(call)
            write_line = getattr(_history_context, "write_line", None)
            turn_index = len(calls) - 1 if calls else 0
            if write_line:
                write_line(
                    {
                        "type": "step-start",
                        "turn": turn_index,
                        "role": role,
                        "agent_id": agent_id,
                        "model_name": call["model_name"],
                        "prompt": call["prompt"],
                    }
                )
            reply = self._worker(role_or_subtask, combined, agent_id)
            call["reply"] = reply
            if role == "Worker" and not reply.strip():
                _history_context.force_worker = True
                _history_context.revision_feedback = (
                    "Previous worker returned no response; produce a complete answer."
                )
            elif role == "Verifier" and reply.strip().upper().startswith("REJECT"):
                _history_context.force_worker = True
                _history_context.revision_feedback = reply
            if write_line:
                write_line(
                    {
                        "type": "step-end",
                        "turn": turn_index,
                        "role": role,
                        "agent_id": agent_id,
                        "model_name": call["model_name"],
                        "prompt": call["prompt"],
                        "reply": reply,
                    }
                )
            return reply
        return self._worker(*args)

    def conduct(self, *args: Any) -> Any:
        _check_client_connected()
        if not (getattr(_history_context, "history", None) or []):
            return self._worker.conduct(*args)
        if len(args) == 2:
            model, messages = args
            return self._worker.conduct(model, self._combine(messages))
        if len(args) == 1:
            (messages,) = args
            return self._worker.conduct(self._combine(messages))
        return self._worker.conduct(*args)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._worker, name)


class OpenRouterTrinityWorker(_TrinityLiteLLMWorker):
    """TRINITY worker that dispatches each turn through the LiteLLM/OpenRouter proxy."""

    def __init__(
        self,
        slot_models: list[str] | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        api_key: str | None = None,
        api_base: str | None = None,
        timeout: float | None = None,
    ) -> None:
        super().__init__(
            slot_models=slot_models,
            max_tokens=max_tokens,
            temperature=temperature,
            api_key=api_key,
            api_base=api_base,
        )
        self.timeout = timeout if timeout is not None else WORKER_TIMEOUT

    def __call__(self, role_name: str, messages: list, agent_id: int) -> str:
        import litellm

        model = self.slot_models[agent_id % len(self.slot_models)]
        msgs = [{"role": m["role"], "content": m["content"]} for m in messages]
        kw = _build_litellm_kwargs(
            model, msgs, self.max_tokens, self.temperature, timeout=self.timeout
        )
        if self.api_key:
            kw["api_key"] = self.api_key
        if self.api_base:
            kw["api_base"] = self.api_base
        return str(litellm.completion(**kw).choices[0].message.content or "")


class OpenRouterConductorWorker(_ConductorLiteLLMWorker):
    """Conductor worker that dispatches plan and step calls through LiteLLM/OpenRouter."""

    def __init__(
        self,
        slot_models: list[str] | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        api_key: str | None = None,
        api_base: str | None = None,
        timeout: float | None = None,
    ) -> None:
        super().__init__(
            slot_models=slot_models,
            max_tokens=max_tokens,
            temperature=temperature,
            api_key=api_key,
            api_base=api_base,
        )
        self.timeout = timeout if timeout is not None else WORKER_TIMEOUT

    def _call(self, model: str, messages: list) -> str:
        kw = _build_litellm_kwargs(
            model, messages, self.max_tokens, self.temperature, timeout=self.timeout
        )
        if self.api_key:
            kw["api_key"] = self.api_key
        if self.api_base:
            kw["api_base"] = self.api_base
        return str(self.litellm.completion(**kw).choices[0].message.content or "")


class LocalPoolWorker:
    """Serving-time local worker pool — the same protocol the per-step trainer
    used. The Coordinator calls (role_name, messages, agent_id) -> reply; we
    dispatch to model[agent_id % n], each model resident on its own GPU. Replies
    are decoded greedily so serving is deterministic. No external API."""

    def __init__(self, specs: list[tuple[str, str, str]], max_new: int = 384) -> None:
        import torch as _torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = _torch
        self.max_new = max_new
        self.names: list[str] = []
        self.toks: list[Any] = []
        self.models: list[Any] = []
        self.devs: list[str] = []
        for name, path, dev in specs:
            tk = AutoTokenizer.from_pretrained(path)
            if tk.pad_token is None:
                tk.pad_token = tk.eos_token
            dtype = _torch.bfloat16 if dev.startswith("cuda") else _torch.float32
            m: Any = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype)
            m = m.to(dev).eval()
            self.names.append(name)
            self.toks.append(tk)
            self.models.append(m)
            self.devs.append(dev)

    def __call__(self, role_name: str, messages: list, agent_id: int) -> str:
        torch = self.torch
        wid = agent_id % len(self.models)
        tk, model, dev = self.toks[wid], self.models[wid], self.devs[wid]
        try:
            text = tk.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except (ValueError, TypeError, AttributeError):
            text = "\n".join(m["content"] for m in messages)
        ids = tk(text, return_tensors="pt", truncation=True, max_length=2048).to(dev)
        with torch.no_grad():
            out = model.generate(
                **ids,
                max_new_tokens=self.max_new,
                do_sample=False,
                pad_token_id=tk.pad_token_id,
            )
        return str(tk.decode(out[0, ids["input_ids"].shape[1] :], skip_special_tokens=True))


def _build_fugu_trace(result: Any) -> str:
    """Render a compact orchestration trace from a coordinator result."""
    turns = getattr(result, "turns", [])
    if not turns:
        return "steps:0:conductor"
    if hasattr(turns[0], "role_name"):
        parts = [f"{t.role_name}({t.agent_id})" for t in turns]
        tb = getattr(result, "terminated_by", "")
        return "→".join(parts) + (f":{tb}" if tb else "")
    return f"steps:{len(turns)}:conductor"


def _chat_response(result: Any, model: str) -> dict:
    text = getattr(result, "final", "")
    turns = getattr(result, "turns", [])
    trace = _build_fugu_trace(result)
    step_details = [
        {
            "turn": getattr(turn, "t", getattr(turn, "step", getattr(turn, "idx", 0))),
            "agent_id": getattr(turn, "agent_id", 0),
            "role": getattr(turn, "role", getattr(turn, "role_name", "Worker")),
            "reply": getattr(turn, "reply", getattr(turn, "text", "")),
            "prompt": getattr(turn, "prompt", ""),
            "model_name": getattr(turn, "model_name", ""),
        }
        for turn in turns
    ]
    return {
        "id": "chatcmpl-" + uuid.uuid4().hex[:24],
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": text,
                    "mantis_steps": step_details,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "mantis_turns": len(turns),
            "mantis_trace": trace,
            "mantis_steps": step_details,
            "fugu_turns": len(turns),
            "fugu_trace": trace,
        },
        "mantis_steps": step_details,
    }


def _resolve_conductor_model(worker) -> str:
    """Pick the model used for the Conductor planning call."""
    conductor_model = os.environ.get("MANTIS_CONDUCTOR_MODEL")
    if conductor_model is None and getattr(worker, "slot_models", None):
        conductor_model = worker.slot_models[0]
    if conductor_model is None:
        conductor_model = "openai/gpt-4o-mini"
    return conductor_model


def _run_conductor_workflow(
    worker: Any, query: str, slot_labels: list[str], completion: str, verbose: bool = False
):
    """Parse a Conductor completion and execute the resulting DAG."""
    mids, subs, acc = parse_workflow(completion)
    if not subs:
        raise ValueError(f"Conductor did not emit a parseable workflow. Raw: {completion[:200]}")
    res = ConductorExecutor(worker, slot_labels=slot_labels).execute(
        mids, subs, acc, verbose=verbose
    )
    # expose a turns attribute for _chat_response
    res.turns = res.steps
    return res


class ConductorCoordinator:
    """Per-request Conductor wrapper: one Conductor LM call produces a workflow
    DAG, then ConductorExecutor runs it. Exposes the same .run(query) interface
    as the TRINITY Coordinator."""

    def __init__(self, worker: Any, slot_labels: list[str] | None = None) -> None:
        self.worker = worker
        self.slot_labels = (
            slot_labels or getattr(worker, "slot_models", None) or DEFAULT_SLOT_LABELS
        )

    def _prepare_planning(self, query: str) -> tuple[str, list[dict[str, Any]], Any]:
        conductor_model = _resolve_conductor_model(self.worker)
        prompt_msgs = conductor_prompt(query, self.slot_labels)

        def _get_completion() -> str:
            return str(self.worker.conduct(conductor_model, prompt_msgs))

        return conductor_model, prompt_msgs, _get_completion

    def run(self, query: str, verbose: bool = False):
        _history_context.conductor_mode = True
        try:
            conductor_model, prompt_msgs, get_completion = self._prepare_planning(query)

            calls = getattr(_history_context, "calls", None)
            if calls is None:
                calls = []
                _history_context.calls = calls

            write_line = getattr(_history_context, "write_line", None)
            planner_turn = len(calls)
            planner_call: dict[str, Any] = {
                "role": "Planner",
                "agent_id": 0,
                "model_name": conductor_model,
                "messages": prompt_msgs,
                "prompt": query,
            }
            calls.append(planner_call)

            if write_line:
                write_line(
                    {
                        "type": "step-start",
                        "turn": planner_turn,
                        "role": "Planner",
                        "agent_id": 0,
                        "model_name": conductor_model,
                        "prompt": query,
                    }
                )

            completion = get_completion()
            planner_call["reply"] = completion

            if write_line:
                write_line(
                    {
                        "type": "step-end",
                        "turn": planner_turn,
                        "role": "Planner",
                        "agent_id": 0,
                        "model_name": conductor_model,
                        "prompt": query,
                        "reply": completion,
                    }
                )

            if not completion or not str(completion).strip():
                raise ValueError("Conductor planning returned an empty completion.")

            res = _run_conductor_workflow(self.worker, query, self.slot_labels, completion, verbose)

            if len(calls) > 1:
                turns = []
                for idx, call in enumerate(calls):
                    t = SimpleNamespace(
                        idx=idx,
                        turn=idx,
                        t=idx,
                        agent_id=call.get("agent_id", 0),
                        role=call.get("role", "Worker"),
                        role_name=call.get("role", "Worker"),
                        subtask=call.get("prompt", ""),
                        prompt=call.get("prompt", ""),
                        reply=call.get("reply", ""),
                        text=call.get("reply", ""),
                        model_name=call.get("model_name", ""),
                        sees=[],
                    )
                    turns.append(t)
                res.turns = turns
            else:
                planner_turn_obj = SimpleNamespace(
                    idx=0,
                    turn=0,
                    t=0,
                    agent_id=0,
                    role="Planner",
                    role_name="Planner",
                    subtask=query,
                    prompt=query,
                    reply=completion,
                    text=completion,
                    model_name=conductor_model,
                    sees=[],
                )
                dag_steps = getattr(res, "steps", [])
                for i, s in enumerate(dag_steps, start=1):
                    if hasattr(s, "idx"):
                        s.idx = i
                    if hasattr(s, "t"):
                        s.t = i
                    if hasattr(s, "turn"):
                        s.turn = i
                res.turns = [planner_turn_obj] + dag_steps

            return res
        finally:
            _history_context.conductor_mode = False


def choose_conductor_device(torch_module: Any, env_device: str | None = None) -> str:
    """Pick mps > cuda:0 > cpu, allowing an explicit env override."""
    if env_device and env_device != "auto":
        return env_device
    if torch_module.backends.mps.is_available():
        return "mps"
    if torch_module.cuda.is_available():
        return "cuda:0"
    return "cpu"


def choose_conductor_dtype(device: str, torch_module: Any, env_dtype: str | None = None) -> Any:
    """Return torch dtype for a Conductor device, honoring an explicit override."""
    if env_dtype:
        return getattr(torch_module, env_dtype)
    return torch_module.bfloat16 if device in ("mps", "cuda", "cuda:0") else torch_module.float32


class EnvLocalConductor:
    """Load a GRPO-trained Conductor checkpoint locally with transformers.

    Env overrides: MANTIS_CONDUCTOR_DEVICE (cpu/cuda:0/mps/auto),
                    MANTIS_CONDUCTOR_DTYPE (float32/bfloat16/float16),
                    MANTIS_CONDUCTOR_MAX_NEW.
    Defaults to bfloat16 on mps/cuda and float32 on cpu."""

    def __init__(self, ckpt: str, device: str | None = None, max_new: int | None = None) -> None:
        import torch as _torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = _torch
        self.ckpt = ckpt
        env_device = device if device is not None else os.environ.get("MANTIS_CONDUCTOR_DEVICE")
        self.device = choose_conductor_device(_torch, env_device)
        self.max_new = max_new or int(os.environ.get("MANTIS_CONDUCTOR_MAX_NEW", "512"))
        dtype_env = os.environ.get("MANTIS_CONDUCTOR_DTYPE")
        self.dtype = choose_conductor_dtype(self.device, _torch, dtype_env)
        self.temperature = float(os.environ.get("MANTIS_CONDUCTOR_TEMPERATURE", "0.7"))
        self.top_p = float(os.environ.get("MANTIS_CONDUCTOR_TOP_P", "0.9"))
        do_sample_env = os.environ.get("MANTIS_CONDUCTOR_DO_SAMPLE")
        if do_sample_env:
            self.do_sample = do_sample_env.lower() not in ("0", "false", "no", "")
        else:
            self.do_sample = True
        print(
            f"[serve] loading local Conductor ({ckpt}) on {self.device} dtype={self.dtype} "
            f"do_sample={self.do_sample} temp={self.temperature} top_p={self.top_p} ...",
            flush=True,
        )
        self.tok = AutoTokenizer.from_pretrained(ckpt)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model: Any = AutoModelForCausalLM.from_pretrained(ckpt, torch_dtype=self.dtype)
        if self.device == "auto" and _torch.cuda.device_count() > 1:
            pass  # leave device_map behavior to from_pretrained
        else:
            self.model = self.model.to(self.device)  # type: ignore[arg-type]
        self.model.eval()
        print("[serve] Conductor ready", flush=True)

    def _build_messages(self, messages: list) -> list[dict[str, str]]:
        """Add an assistant prefill that nudges the Conductor into the 3-list format.

        One-shot examples are intentionally avoided here: the 3B Conductor
        checkpoints tend to collapse into repeating a fixed example rather
        than following the actual user query.
        """
        return list(messages) + [{"role": "assistant", "content": "Plan:\n"}]

    def conduct(self, messages: list) -> str:
        torch = self.torch
        messages = self._build_messages(messages)
        try:
            text = self.tok.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
                continue_final_message=True,
                truncation=True,
                max_length=2048,
            )
        except (ValueError, TypeError, AttributeError):
            # Fallback for tokenizers without chat_template or old transformers.
            parts = [f"{m['role'].capitalize()}: {m['content']}" for m in messages]
            text = "\n\n".join(parts)
        ids = self.tok(text, return_tensors="pt", truncation=True, max_length=2048).to(self.device)
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new,
            "pad_token_id": self.tok.pad_token_id,
        }
        if self.do_sample:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = self.temperature
            gen_kwargs["top_p"] = self.top_p
        else:
            gen_kwargs["do_sample"] = False
        with torch.no_grad():
            out = self.model.generate(**ids, **gen_kwargs)
        completion = str(
            self.tok.decode(out[0, ids["input_ids"].shape[1] :], skip_special_tokens=True)
        )
        print(f"[serve] raw conductor completion: {completion[:1000]!r}", flush=True)
        return completion


class EnvConductorCoordinator(ConductorCoordinator):
    """ConductorCoordinator that can use a local transformers checkpoint
    (Llama-3.2-3B Conductor) or LiteLLM for the planning call."""

    def __init__(
        self,
        worker: Any,
        conductor: EnvLocalConductor | None = None,
        slot_labels: list[str] | None = None,
    ) -> None:
        super().__init__(worker, slot_labels=slot_labels)
        self.local_conductor = conductor

    def _prepare_planning(self, query: str) -> tuple[str, list[dict[str, Any]], Any]:
        prompt_msgs = conductor_prompt(query, self.slot_labels)
        lc = self.local_conductor
        if lc is not None:
            conductor_model = getattr(lc, "ckpt", "local-conductor")

            def _get_completion_local() -> str:
                return str(lc.conduct(prompt_msgs))

            return conductor_model, prompt_msgs, _get_completion_local

        conductor_model = _resolve_conductor_model(self.worker)

        def _get_completion_worker() -> str:
            return str(self.worker.conduct(conductor_model, prompt_msgs))

        return conductor_model, prompt_msgs, _get_completion_worker


def _split_messages(messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    """Return (query, history) where query is the last user message."""
    last_user_idx = -1
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            last_user_idx = i
            break
    if last_user_idx == -1:
        return "", messages
    return messages[last_user_idx].get("content", ""), messages[:last_user_idx]


def _json_object(raw: bytes) -> dict[str, Any]:
    value = json.loads(raw or b"{}")
    if not isinstance(value, dict):
        raise TypeError("request body must be a JSON object")
    return value


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send(self, code: int, body: dict) -> None:
        self.close_connection = True
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(data)

    def _read_request_body(self, max_bytes: int | None = None) -> bytes:
        """Read a bounded POST body, supporting Content-Length and chunked encoding."""
        max_bytes = MAX_BODY_BYTES if max_bytes is None else max_bytes
        te = self.headers.get("Transfer-Encoding", "")
        if te.lower() == "chunked":
            return self._read_chunked_body(max_bytes)
        n = int(self.headers.get("Content-Length", 0))
        if n > max_bytes:
            raise RequestBodyTooLargeError
        return self.rfile.read(n) if n > 0 else b""

    def _read_chunked_body(self, max_bytes: int | None = None) -> bytes:
        """Decode a bounded chunked transfer-coded request body."""
        max_bytes = MAX_BODY_BYTES if max_bytes is None else max_bytes
        body = bytearray()
        while True:
            line = self.rfile.readline()
            if not line:
                break
            size_str = line.split(b";", 1)[0].strip()
            try:
                chunk_size = int(size_str, 16)
            except ValueError:
                break
            if chunk_size < 0:
                raise ValueError("invalid negative chunk size")
            if chunk_size == 0:
                # consume optional trailers until final CRLF
                while True:
                    line = self.rfile.readline()
                    if not line or line == b"\r\n":
                        break
                break
            if len(body) + chunk_size > max_bytes:
                raise RequestBodyTooLargeError
            chunk = self.rfile.read(chunk_size)
            if len(chunk) != chunk_size:
                raise ValueError("truncated chunk data")
            body.extend(chunk)
            if self.rfile.read(2) != b"\r\n":
                raise ValueError("invalid chunk terminator")
        return bytes(body)

    def _auth_token(self) -> str | None:
        return os.environ.get("MANTIS_API_KEY") or os.environ.get("LITELLM_KEY")

    def _check_auth(self) -> bool:
        expected = self._auth_token()
        if not expected:
            self._send(
                500,
                {"error": "MANTIS_API_KEY or LITELLM_KEY is not configured"},
            )
            return False
        auth = self.headers.get("Authorization", "")
        if not auth.startswith("Bearer ") or auth[7:] != expected:
            self._send(401, {"error": "unauthorized"})
            return False
        return True

    def is_connected(self) -> bool:
        if getattr(self, "_disconnected", False):
            return False
        try:
            sock = getattr(self, "connection", None)
            if sock is None:
                return True
            r, _, _ = select.select([sock], [], [], 0)
            if r:
                buf = sock.recv(1, socket.MSG_PEEK)
                if not buf:
                    self._disconnected = True
                    return False
        except Exception:  # noqa: BLE001
            self._disconnected = True
            return False
        return True

    def _write_ndjson_line(self, obj: Any) -> None:
        try:
            self.wfile.write((json.dumps(obj) + "\n").encode())
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError) as err:
            self._disconnected = True
            _history_context.aborted = True
            raise ClientDisconnectedError("Client disconnected") from err

    def _run_stream(self, coordinator_mode: str, query: str, model: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        _history_context.write_line = self._write_ndjson_line
        _history_context.is_client_connected = self.is_connected
        _history_context.aborted = False
        _history_context.calls = []
        _history_context.force_worker = False
        _history_context.revision_feedback = None
        try:
            coord = get_coordinator(coordinator_mode)
            res = coord.run(query, verbose=False)
        except ClientDisconnectedError:
            print("[serve] client disconnected during stream, aborting orchestration", flush=True)
            return
        except Exception as e:  # noqa: BLE001
            with contextlib.suppress(
                BrokenPipeError, ConnectionResetError, OSError, ClientDisconnectedError
            ):
                self._write_ndjson_line({"type": "error", "error": str(e)})
            return
        finally:
            _history_context.write_line = None
            _history_context.is_client_connected = None
            _history_context.aborted = False
            _history_context.history = []
            _history_context.force_worker = False
            _history_context.revision_feedback = None
            calls = getattr(_history_context, "calls", [])
            _history_context.calls = []

        for turn, call in zip(getattr(res, "turns", []), calls, strict=False):
            turn.prompt = call.get("prompt", "")
            turn.model_name = call.get("model_name", "")
        body = _chat_response(res, model)
        try:
            self._write_ndjson_line(
                {
                    "type": "result",
                    "text": body["choices"][0]["message"]["content"],
                    "trace": body["usage"]["mantis_trace"],
                    "coordinator": coordinator_mode,
                    "mantis_steps": body.get("mantis_steps", []),
                    **body,
                }
            )
        except ClientDisconnectedError:
            print("[serve] client disconnected before writing final result", flush=True)

    def _handle_stream(self, coordinator_mode: str, query: str, model: str) -> None:
        self._run_stream(coordinator_mode, query, model)

    def _handle_warm(
        self, parsed: urllib.parse.ParseResult, mode_from_body: str | None = None
    ) -> None:
        if not self._check_auth():
            return
        qs = urllib.parse.parse_qs(parsed.query)
        mode = mode_from_body or (qs.get("mode", ["trinity"])[0] if qs.get("mode") else "trinity")
        if mode not in ("trinity", "conductor"):
            self._send(400, {"error": f"unknown mode: {mode}"})
            return
        try:
            get_coordinator(mode)
            self._send(
                200,
                {
                    "status": "ready",
                    "mode": mode,
                    "native_tool_runs": NATIVE_TOOL_RUNS,
                },
            )
        except Exception as e:  # noqa: BLE001
            self._send(500, {"error": str(e)})

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path in ("/warm", "/v1/warm"):
            self._handle_warm(parsed)
        elif parsed.path == "/v1/models":
            if not self._check_auth():
                return
            self._send(
                200,
                {
                    "object": "list",
                    "data": [{"id": MODEL_NAME, "object": "model", "owned_by": "mantis"}],
                },
            )
        elif parsed.path in ("/health", "/"):
            self._send(200, {"status": "ok", "model": MODEL_NAME})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path in ("/warm", "/v1/warm"):
            mode = None
            try:
                raw = self._read_request_body()
                if raw:
                    req = json.loads(raw)
                    if isinstance(req, dict):
                        mode = req.get("mode")
            except RequestBodyTooLargeError:
                self._send(413, {"error": "request body exceeds limit"})
                return
            except (json.JSONDecodeError, ValueError, KeyError, RuntimeError, TypeError):
                pass
            self._handle_warm(parsed, mode_from_body=mode)
            return

        if parsed.path == "/v1/runs":
            self._handle_create_run()
            return
        if parsed.path.startswith("/v1/runs/"):
            rest = parsed.path[len("/v1/runs/") :]
            run_id, action = (rest.split("/", 1) + [""])[:2]
            if run_id and action == "continue":
                self._handle_continue_run(run_id)
            elif run_id and not action:
                self._handle_fetch_run(run_id)  # advance without tool results
            else:
                self._send(404, {"error": "not found"})
            return

        if parsed.path.rstrip("/") != "/v1/chat/completions":
            self._send(404, {"error": "not found"})
            return
        if not self._check_auth():
            return
        try:
            raw = self._read_request_body()
            req = _json_object(raw)
            messages = req.get("messages", [])
            if not messages:
                self._send(400, {"error": "messages required"})
                return

            requested = (req.get("model") or "trinity").lower()
            coordinator_mode = "conductor" if requested in ("conductor", "ultra") else "trinity"
            model_name = req.get("model", MODEL_NAME)

            # the user query = last user message; history = everything before it
            query, history = _split_messages(messages)
            print(
                f"[serve] route request model={requested} -> coordinator={coordinator_mode}",
                flush=True,
            )
            if req.get("stream"):
                _history_context.history = history
                _history_context.calls = []
                return self._handle_stream(coordinator_mode, query, model_name)

            _history_context.history = history
            _history_context.calls = []
            _history_context.force_worker = False
            _history_context.revision_feedback = None
            _history_context.is_client_connected = self.is_connected
            _history_context.aborted = False
            try:
                coord = get_coordinator(coordinator_mode)
                res = coord.run(query, verbose=False)
            except ClientDisconnectedError:
                print(
                    "[serve] client disconnected during request, aborting orchestration",
                    flush=True,
                )
                return
            finally:
                _history_context.history = []
                _history_context.is_client_connected = None
                _history_context.aborted = False
                _history_context.force_worker = False
                _history_context.revision_feedback = None
                calls = getattr(_history_context, "calls", [])
                _history_context.calls = []
            for turn, call in zip(getattr(res, "turns", []), calls, strict=False):
                turn.prompt = call.get("prompt", "")
                turn.model_name = call.get("model_name", "")
            self._send(200, _chat_response(res, model_name))
        except RequestBodyTooLargeError:
            self._send(413, {"error": "request body exceeds limit"})
        except (json.JSONDecodeError, ValueError, KeyError, RuntimeError, TypeError) as e:
            self._send(500, {"error": str(e)})

    def _handle_create_run(self) -> None:
        if not self._check_auth():
            return
        try:
            raw = self._read_request_body()
            req = _json_object(raw)
            model = req.get("model") or "trinity"
            if not isinstance(model, str):
                raise TypeError("model must be a string")  # noqa: TRY301
            mode = model.lower()
            mode = "conductor" if mode in ("conductor", "ultra") else "trinity"
            if not req.get("messages"):
                self._send(400, {"error": "messages required"})
                return
            run = create_run(mode, req)
            print(f"[serve] created {mode} run {run.run_id}", flush=True)
            self._send(200, {"run_id": run.run_id, "mode": mode})
        except RequestBodyTooLargeError:
            self._send(413, {"error": "request body exceeds limit"})
        except (json.JSONDecodeError, ValueError, KeyError, RuntimeError, TypeError) as e:
            self._send(400, {"error": str(e)})

    def _handle_continue_run(self, run_id: str) -> None:
        if not self._check_auth():
            return
        try:
            raw = self._read_request_body()
            tool_results = None
            request_id = None
            if raw:
                req = _json_object(raw)
                tool_results = req.get("tool_results")
                request_id = req.get("request_id")
            event = advance_run(run_id, tool_results, request_id)
            self._send(200, event)
        except KeyError as e:
            self._send(404, {"error": str(e)})
        except RequestBodyTooLargeError:
            self._send(413, {"error": "request body exceeds limit"})
        except (json.JSONDecodeError, ValueError, RuntimeError, TypeError) as e:
            self._send(400, {"error": str(e)})

    def _handle_fetch_run(self, run_id: str) -> None:
        if not self._check_auth():
            return
        try:
            event = advance_run(run_id, None)
            self._send(200, event)
        except KeyError as e:
            self._send(404, {"error": str(e)})
        except (json.JSONDecodeError, ValueError, RuntimeError, TypeError) as e:
            self._send(400, {"error": str(e)})

    def do_DELETE(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        if not parsed.path.startswith("/v1/runs/"):
            self._send(404, {"error": "not found"})
            return
        if not self._check_auth():
            return
        run_id = parsed.path[len("/v1/runs/") :].rstrip("/")
        if not run_id or "/" in run_id:
            self._send(404, {"error": "not found"})
            return
        deleted = delete_run(run_id)
        self._send(200, {"deleted": deleted})

    def log_message(self, *a) -> None:  # quiet
        pass


def _parse_args() -> argparse.Namespace:
    global _args
    if _args is not None:
        return _args
    ap = argparse.ArgumentParser(description="Serve Mantis as one OpenAI-compatible model.")
    ap.add_argument(
        "--model",
        default=os.environ.get("MANTIS_MODEL", "Qwen/Qwen3-0.6B"),
        help="Qwen3-0.6B dir or HF id",
    )
    ap.add_argument(
        "--vector",
        default=os.environ.get("MANTIS_VECTOR", "model_iter_60.npy"),
        help="base vector (19456) — SVF + head",
    )
    ap.add_argument(
        "--head",
        default=os.environ.get("MANTIS_HEAD"),
        help="optional trained head-only vector/safetensors; overrides the "
        "head from --vector after SVF is applied",
    )
    default_workers = os.environ.get("MANTIS_WORKER_MODELS") or os.environ.get(
        "MANTIS_WORKER_MODEL"
    )
    ap.add_argument(
        "--slot-models",
        metavar="CSV",
        default=default_workers,
        help="litellm worker ids (CSV); also MANTIS_WORKER_MODELS",
    )
    ap.add_argument(
        "--local-models",
        metavar="CSV",
        default=os.environ.get("MANTIS_LOCAL_MODELS"),
        help="local HF worker model paths (CSV). "
        "Optional 'path@device' per entry; also MANTIS_LOCAL_MODELS",
    )
    ap.add_argument(
        "--host",
        default=os.environ.get("MANTIS_HOST", "0.0.0.0"),  # noqa: S104
    )
    ap.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("MANTIS_PORT", "8088")),
    )
    ap.add_argument(
        "--max-turns",
        type=int,
        default=int(os.environ.get("MANTIS_MAX_TURNS", "5")),
    )
    _args = ap.parse_args()
    return _args


def get_router() -> FuguRouter:
    global ROUTER
    if ROUTER is None:
        with _router_lock:
            if ROUTER is None:
                args = _parse_args()
                device = os.environ.get("MANTIS_DEVICE")
                print(f"[serve] loading TRINITY router ({args.model}) ...", flush=True)
                router = FuguRouter(args.model, args.vector, device=device, seed=0)
                if args.head:  # layer a trained head over base SVF
                    head = _load_head(args.head)
                    router.head = (
                        router.torch.from_numpy(head.copy())
                        .float()
                        .reshape(HEAD_ROWS, HIDDEN)
                        .to(router.device)
                    )
                    print(f"[serve] applied trained head from {args.head}", flush=True)
                ROUTER = router
    return ROUTER


def _load_head(path: str) -> np.ndarray:
    """Load a 10240-float head from .npy or from a safetensors file."""
    if path.endswith(".safetensors"):
        from safetensors import safe_open

        with safe_open(path, framework="pt") as f:
            head = f.get_tensor("trinity_router_head")
        head = head.to(torch.float32).numpy().reshape(-1)
    else:
        head = np.load(path).astype(np.float64)
    if head.shape != (HEAD_ROWS * HIDDEN,):
        raise ValueError(f"head must be {HEAD_ROWS * HIDDEN} floats, got {head.shape}")
    return np.asarray(head, dtype=np.float64)


def _worker_from_args(args: argparse.Namespace, mode: str) -> Any:
    if args.local_models:
        specs = []
        n_gpu = 0
        try:
            import torch as _torch

            n_gpu = _torch.cuda.device_count() if _torch.cuda.is_available() else 0
        except (ImportError, ModuleNotFoundError):
            pass
        for i, entry in enumerate(args.local_models.split(",")):
            if "@" in entry:
                path, dev = entry.rsplit("@", 1)
            else:
                path = entry
                dev = f"cuda:{(i % max(n_gpu - 1, 1)) + 1}" if n_gpu > 1 else "cpu"
            specs.append((os.path.basename(path.rstrip("/")) or f"w{i}", path, dev))
        return LocalPoolWorker(specs)

    # 4096 tokens to leave room for high reasoning effort (max/xhigh) while still
    # capping cost on long code outputs.
    slot_models = args.slot_models.split(",") if args.slot_models else None
    base_url = _litellm_base_url()
    api_key = _litellm_api_key()
    if mode == "conductor":
        worker = OpenRouterConductorWorker(slot_models=slot_models, max_tokens=4096)
    elif mode == "trinity":
        worker = OpenRouterTrinityWorker(slot_models=slot_models, max_tokens=4096)
    else:
        raise ValueError(f"unknown coordinator mode: {mode}")
    worker.api_key = api_key
    worker.api_base = base_url
    return worker


def load_coordinator(mode: str):
    global MAX_TURNS
    if mode not in ("trinity", "conductor"):
        raise ValueError(f"unknown coordinator mode: {mode}")
    args = _parse_args()
    MAX_TURNS = args.max_turns
    worker = HistoryWorker(_worker_from_args(args, mode))
    if mode == "trinity":
        return Coordinator(
            RejectAwareRouter(get_router()), worker, max_turns=args.max_turns, sample=True
        )
    local_ckpt = os.environ.get("MANTIS_LOCAL_CONDUCTOR")
    conductor = EnvLocalConductor(local_ckpt) if local_ckpt else None
    return EnvConductorCoordinator(
        worker, conductor=conductor, slot_labels=getattr(worker, "slot_models", None)
    )


def get_coordinator(mode: str):
    if mode not in _coordinators:
        with _coordinator_lock:
            if mode not in _coordinators:
                _coordinators[mode] = load_coordinator(mode)
    return _coordinators[mode]


def main() -> None:
    args = _parse_args()
    token = os.environ.get("MANTIS_API_KEY") or os.environ.get("LITELLM_KEY")
    if not token:
        print(
            "[serve] FATAL: set MANTIS_API_KEY or LITELLM_KEY before starting the server",
            flush=True,
        )
        raise SystemExit(1)
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(
        f"[serve] Mantis listening on {args.host}:{args.port} — POST /v1/chat/completions",
        flush=True,
    )
    srv.serve_forever()


# ---------------------------------------------------------------------------
# Resumable native-tool runs
# ---------------------------------------------------------------------------
# Pi forwards its active tool schemas; the backend selects a role/model and
# drives a tool loop, and Pi executes each tool natively and returns results.
# Each HTTP call advances the run by exactly one model invocation; events tell
# Pi whether to execute tools, acknowledge a completed role step, or return a
# final answer. State lives only in the bounded in-memory registry below.
RUN_TTL = float(os.environ.get("MANTIS_RUN_TTL", "600"))
MAX_TOOL_ROUNDS = int(os.environ.get("MANTIS_MAX_TOOL_ROUNDS_PER_STEP", "8"))
MAX_RUNS = int(os.environ.get("MANTIS_MAX_CONCURRENT_RUNS", "32"))
RUN_MAX_MSG_BYTES = 400_000

_runs: dict[str, NativeRun] = {}
_runs_lock = threading.Lock()
_runs_sweeper_started = False
NATIVE_TOOL_RUNS = True


def _sweep_runs() -> None:
    now = time.time()
    stale = [
        rid for rid, run in _runs.items() if run.in_flight == 0 and now - run.last_active > RUN_TTL
    ]
    for rid in stale:
        run = _runs.pop(rid, None)
        if run is not None:
            run.close()


def _ensure_runs_sweeper() -> None:
    global _runs_sweeper_started
    if _runs_sweeper_started:
        return
    _runs_sweeper_started = True

    def _loop() -> None:
        while True:
            time.sleep(RUN_TTL / 2 if RUN_TTL > 0 else 60)
            with _runs_lock:
                _sweep_runs()

    threading.Thread(target=_loop, daemon=True).start()


def _register_run(run: NativeRun) -> str:
    with _runs_lock:
        _ensure_runs_sweeper()
        if run.run_id in _runs:
            raise ValueError("run id already exists")
        while len(_runs) >= MAX_RUNS:
            _sweep_runs()
            if len(_runs) >= MAX_RUNS:  # still full: drop oldest
                oldest = min(_runs, key=lambda rid: _runs[rid].created)
                dropped = _runs.pop(oldest, None)
                if dropped is not None:
                    dropped.close()
        _runs[run.run_id] = run
    return cast(str, run.run_id)


def _convert_tools(tools: Any) -> list[dict[str, Any]]:
    """Convert Pi active-tool definitions to OpenAI function-tool format."""
    out: list[dict[str, Any]] = []
    if not isinstance(tools, list):
        return out
    for t in tools:
        if not isinstance(t, dict):
            continue
        name = t.get("name")
        if not name or name == "mantis_step":
            continue
        fn: dict[str, Any] = {"name": name}
        if t.get("description"):
            fn["description"] = t["description"]
        params = t.get("parameters")
        if isinstance(params, dict):
            fn["parameters"] = params
        elif not params:
            fn["parameters"] = {"type": "object", "properties": {}}
        else:
            continue  # non-dict parameters: cannot serialize a stable schema
        out.append({"type": "function", "function": fn})
    return out


def _model_completion(
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
) -> tuple[str, list[dict[str, Any]]]:
    """Call LiteLLM/OpenRouter; return (text, tool_calls)."""
    import litellm

    kw = _build_litellm_kwargs(model, messages, 4096, 0.7)
    api_key = _litellm_api_key()
    if not api_key:
        raise RuntimeError("set LITELLM_KEY or MANTIS_LITELLM_API_KEY for worker calls")
    kw["api_key"] = api_key
    kw["api_base"] = _litellm_base_url()
    if tools:
        kw["tools"] = tools
    resp = litellm.completion(**kw)
    msg = resp.choices[0].message
    text = str(getattr(msg, "content", None) or "")
    tcs = getattr(msg, "tool_calls", None) or []
    calls: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for tc in tcs:
        fn = tc.function
        try:
            args = json.loads(fn.arguments) if fn.arguments else {}
        except (json.JSONDecodeError, ValueError, TypeError):
            args = {}
        if not isinstance(args, dict):
            args = {}
        call_id = str(tc.id or f"tc_{uuid.uuid4().hex}")
        if call_id in seen_ids:
            call_id = f"tc_{uuid.uuid4().hex}"
        seen_ids.add(call_id)
        calls.append(
            {
                "id": call_id,
                "name": str(fn.name),
                "arguments": args,
            }
        )
    return text, calls


def _openai_tool_call(name: str, _id: str, arguments: dict) -> dict[str, Any]:
    return {
        "id": _id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def _validate_tool_results(tool_results: Any, expected_ids: set[str]) -> list[dict[str, Any]]:
    if not isinstance(tool_results, list):
        raise TypeError("tool_results must be a list")
    if not all(isinstance(result, dict) for result in tool_results):
        raise ValueError("each tool result must be an object")
    results = cast(list[dict[str, Any]], tool_results)
    ids = [result.get("tool_call_id") for result in results]
    if not all(isinstance(tool_id, str) and tool_id for tool_id in ids):
        raise ValueError("each tool result requires a tool_call_id")
    string_ids = cast(list[str], ids)
    if len(string_ids) != len(set(string_ids)) or set(string_ids) != expected_ids:
        raise ValueError(
            f"tool result id mismatch: expected {sorted(expected_ids)} got {sorted(string_ids)}"
        )
    return results


def _configured_slot_models(override: Any = None) -> list[str]:
    value = override
    if value is None:
        configured = getattr(_args, "slot_models", None) if _args is not None else None
        configured = (
            configured
            or os.environ.get("MANTIS_WORKER_MODELS")
            or os.environ.get("MANTIS_WORKER_MODEL")
        )
        value = configured.split(",") if configured else list(DEFAULT_SLOT_LABELS)
    if not isinstance(value, list):
        raise TypeError("slot_models must be a non-empty list of model names")
    models = [model.strip() for model in value if isinstance(model, str) and model.strip()]
    if len(models) != len(value) or not models:
        raise ValueError("slot_models must be a non-empty list of model names")
    return models


_learning_lock = threading.Lock()
_TEST_COMMAND = re.compile(
    r"(?:^|[;&|]\s*|\s)(?:python\d*\s+-m\s+(?:pytest|unittest)|pytest|npm\s+(?:run\s+)?test|"
    r"pnpm\s+(?:run\s+)?test|yarn\s+test|bun\s+test|cargo\s+test|go\s+test|dotnet\s+test|"
    r"mvn\s+test|gradle\s+test)(?:\s|$)",
    re.IGNORECASE,
)
_SECRET_PATTERNS = (
    re.compile(r"\b(?:sk|hf)_[A-Za-z0-9_-]{12,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b"),
    re.compile(r"\b(?:AKIA[A-Z0-9]{16}|github_pat_[A-Za-z0-9_]{20,})\b"),
    re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|xox[baprs]-[A-Za-z0-9-]{20,})\b"),
    re.compile(r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b"),
    re.compile(r"(?i)\b(api[_ -]?key|token|password|secret|authorization)\b\s*[:=]\s*\S+"),
    re.compile(r"(?i)\bBearer\s+\S+"),
)


def _learning_enabled() -> bool:
    return os.environ.get("MANTIS_LEARNING", "").lower() in {"1", "true", "yes", "on"}


def _redact_learning_task(task: str) -> str:
    redacted = task[: int(os.environ.get("MANTIS_LEARNING_MAX_TASK_CHARS", "12000"))]
    for pattern in _SECRET_PATTERNS:
        redacted = pattern.sub("[REDACTED]", redacted)
    return redacted


def _learning_path() -> Path:
    root = Path(
        os.path.expanduser(os.environ.get("MANTIS_LEARNING_DIR", "~/.local/share/mantis/learning"))
    )
    instance = os.environ.get("MANTIS_LEARNING_INSTANCE") or socket.gethostname()
    safe_instance = re.sub(r"[^A-Za-z0-9_.-]", "_", instance)[:80] or "local"
    return root / f"runs-{safe_instance}.jsonl"


def _tool_call_by_id(pending: dict[str, Any], tool_call_id: str) -> dict[str, Any] | None:
    for call in pending.get("asst", {}).get("tool_calls", []):
        if isinstance(call, dict) and call.get("id") == tool_call_id:
            return cast(dict[str, Any], call)
    return None


def _learning_record(run: NativeRun, event: dict[str, Any]) -> dict[str, Any]:
    turns = cast(list[dict[str, Any]], getattr(run, "turns", getattr(run, "steps", [])))
    final_worker = next((turn for turn in reversed(turns) if turn.get("role") == "Worker"), None)
    tests = [item for item in run.tool_observations if item["is_test"]]
    last_test_passed = bool(tests) and not tests[-1]["is_error"]
    accepted = event.get("terminated_by") == "verifier_accept"
    trainable = bool(run.kind == "trinity" and accepted and last_test_passed and final_worker)
    task = _redact_learning_task(str(getattr(run, "query", "")))
    return {
        "schema_version": 1,
        "timestamp": int(time.time()),
        "run_id": run.run_id,
        "mode": run.kind,
        "task": task,
        "task_hash": hashlib.sha256(task.encode()).hexdigest(),
        "pool": list(getattr(run, "slot_models", [])),
        "terminated_by": event.get("terminated_by", event.get("type", "")),
        "duration_seconds": round(time.time() - run.created, 3),
        "turn_count": len(turns),
        "test_seen": bool(tests),
        "last_test_passed": last_test_passed,
        "tool_error_count": sum(item["is_error"] for item in run.tool_observations),
        "verifier_accepted": accepted,
        "trainable": trainable,
        "label_worker": (
            final_worker.get("agent_id") if trainable and final_worker is not None else None
        ),
        "label_role": 0 if trainable else None,
    }


def _write_learning_record(run: NativeRun, event: dict[str, Any]) -> None:
    if not _learning_enabled():
        return
    with _learning_lock:
        if run.learning_logged:
            return
        path = _learning_path()
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
        os.chmod(path, 0o600)
        with os.fdopen(descriptor, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(_learning_record(run, event), ensure_ascii=False) + "\n")
        run.learning_logged = True


class NativeRun:
    """Base for a resumable orchestration run."""

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self.created = time.time()
        self.last_active = time.time()
        self.cancelled = False
        self.finished = False
        self.final_text = ""
        self.terminated_by: str | None = None
        self.kind = "run"
        self.lock = threading.Lock()
        self.request_lock = threading.Lock()
        self.request_events: dict[str, dict[str, Any]] = {}
        self.tool_observations: list[dict[str, Any]] = []
        self.learning_logged = False
        self.in_flight = 0

    def touch(self) -> None:
        self.last_active = time.time()

    def advance(self, tool_results: Any) -> dict[str, Any]:
        raise NotImplementedError

    def advance_idempotent(self, tool_results: Any, request_id: Any = None) -> dict[str, Any]:
        if request_id is None:
            return self.advance(tool_results)
        if not isinstance(request_id, str) or not request_id or len(request_id) > 128:
            raise ValueError("request_id must be a string of at most 128 characters")
        with self.request_lock:
            cached = self.request_events.get(request_id)
            if cached is not None:
                return cached
            event = self.advance(tool_results)
            self.request_events[request_id] = event
            while len(self.request_events) > 64:
                self.request_events.pop(next(iter(self.request_events)))
            return event

    def record_tool_results(
        self, pending: dict[str, Any], tool_results: list[dict[str, Any]]
    ) -> None:
        for result in tool_results:
            call = _tool_call_by_id(pending, str(result.get("tool_call_id", ""))) or {}
            function = call.get("function", {})
            try:
                arguments = json.loads(function.get("arguments", "{}"))
            except (json.JSONDecodeError, TypeError, ValueError):
                arguments = {}
            command = arguments.get("command", "") if isinstance(arguments, dict) else ""
            self.tool_observations.append(
                {
                    "name": str(function.get("name", "")),
                    "is_error": bool(result.get("is_error", False)),
                    "is_test": bool(
                        function.get("name") == "bash"
                        and isinstance(command, str)
                        and _TEST_COMMAND.search(command)
                    ),
                }
            )

    def close(self) -> None:
        self.cancelled = True


class TrinityRun(NativeRun):
    """Resumable TRINITY loop with native tool support.

    Replicates Coordinator semantics (role sampling, Thinker suggestion,
    Verifier accept/reject, cold-verifier -> Worker, empty-response recovery,
    multi-turn history) but lets each role's model call Pi tools before its text
    reply finalizes."""

    def __init__(
        self,
        run_id: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        slot_models: list[str] | None = None,
        max_turns: int = MAX_TURNS,
    ) -> None:
        super().__init__(run_id)
        self.kind = "trinity"
        self.tools = tools
        self.slot_models = slot_models or list(DEFAULT_SLOT_LABELS)
        self.max_turns = max_turns
        query, history = _split_messages(messages)
        self.query = query or ""
        self.history = history
        self.obs = self.query
        self.ref_id = 0
        self.last_response: str | None = None
        self.suggestion: str | None = None
        self.suggested_role: str | None = None
        self.force_worker = False
        self.revision_feedback: str | None = None
        self.turns: list[dict[str, Any]] = []
        self._pending: dict[str, Any] | None = None
        self._expected_ids: set[str] = set()
        self._tool_rounds = 0

    def _model_name(self, agent_id: int) -> str:
        return str(self.slot_models[agent_id % len(self.slot_models)])

    def _route(self) -> tuple[str, int]:
        msgs = [
            {
                "role": "system",
                "content": ROUTER_SYSTEM_PROMPT.format(num_agents=len(self.slot_models)),
            },
            {"role": "user", "content": self.obs},
        ]
        r = get_router().route(msgs, sample=True)
        role = r["role_name"]
        if self.suggested_role:
            role, self.suggested_role = self.suggested_role, None
        if self.force_worker:
            self.force_worker = False
            role = "Worker"
        if role == "Verifier" and self.last_response is None:
            role = "Worker"  # nothing to verify yet [FC]
        if role == "Thinker" and self.last_response is None:
            role = "Worker"  # a Thinker with no response to reason about is noise
        return role, int(r["agent_id"])

    def _role_prompt(self, role: str) -> str:
        if role == "Thinker":
            info = self.query
            if self.last_response:
                info += f"\n\nCurrent response:\n{self.last_response}"
            return cast(str, THINKER_PROMPT.format(info=info))
        if role == "Verifier":
            vp = VERIFICATION_PROMPT.format(query=self.query, response=self.last_response or "")
            if self.suggestion:
                vp += (
                    f"These are useful suggestions when drafting your response:\n"
                    f"<suggestion>{self.suggestion}</suggestion>"
                )
            return cast(str, vp)
        content = self.query
        if self.suggestion:
            content += (
                f"when drafting your response, thinking of following:\n"
                f"<suggestion>{self.suggestion}</suggestion>"
            )
        return cast(str, content)

    def _build_messages(self, role: str) -> list[dict[str, Any]]:
        prior_sys = "\n\n".join(
            str(m.get("content", ""))
            for m in self.history
            if isinstance(m, dict) and m.get("role") == "system" and m.get("content")
        )
        sys_content = SYSTEM_PROMPT
        if prior_sys:
            sys_content = f"{sys_content}\n\n{prior_sys}"
        prior = [m for m in self.history if isinstance(m, dict) and m.get("role") != "system"]
        msgs: list[dict[str, Any]] = [{"role": "system", "content": sys_content}]
        msgs.extend(prior)
        user_content = self._role_prompt(role)
        if role == "Worker" and self.revision_feedback:
            user_content = (
                f"{user_content}\n\n"
                f"Revise the answer to address this verifier feedback:\n{self.revision_feedback}"
            )
            self.revision_feedback = None
        msgs.append({"role": "user", "content": user_content})
        return msgs

    def _role_complete(self, role: str, agent_id: int, turn: int, messages: list, reply: str):
        if role == "Worker":
            self.last_response = reply
            self.suggestion = None
            thought = self._extract_thought(reply)
            if thought:
                self.obs += (
                    f"\n<reference_thought_{self.ref_id}>{thought}"
                    f"</reference_thought_{self.ref_id}>"
                )
                self.ref_id += 1
        elif role == "Thinker":
            self.suggested_role, self.suggestion = self._parse_thinker(reply)
        elif role == "Verifier":
            self.suggestion = None
            if self._parse_verification(reply):
                self.terminated_by = "verifier_accept"
                self.final_text = self.last_response or reply
        if role == "Worker" and not reply.strip():
            self.force_worker = True
            nope = "produce a complete answer."
            self.revision_feedback = f"Previous worker returned no response; {nope}"
        elif role == "Verifier" and reply.strip().upper().startswith("REJECT"):
            self.force_worker = True
            self.revision_feedback = reply
        step = {
            "turn": turn,
            "role": role,
            "agent_id": agent_id,
            "model_name": self._model_name(agent_id),
            "prompt": messages[-1]["content"] if messages else "",
            "reply": reply,
        }
        self.turns.append(step)
        self._pending = None
        self._expected_ids = set()
        self._tool_rounds = 0
        return {"type": "step_complete", **step}

    def _run_model(self, role: str, agent_id: int, turn: int, messages: list):
        model = self._model_name(agent_id)
        text, calls = _model_completion(model, messages, self.tools)
        if calls:
            asst: dict[str, Any] = {
                "role": "assistant",
                "content": text,
                "tool_calls": [
                    _openai_tool_call(c["name"], c["id"], c["arguments"]) for c in calls
                ],
            }
            self._pending = {
                "role": role,
                "agent_id": agent_id,
                "turn": turn,
                "messages": messages,
                "asst": asst,
            }
            self._expected_ids = {c["id"] for c in calls}
            self._tool_rounds = 1
            return {
                "type": "tool_calls",
                "role": role,
                "agent_id": agent_id,
                "model_name": model,
                "turn": turn,
                "tool_calls": calls,
            }
        return self._role_complete(role, agent_id, turn, messages, text)

    def _apply_tool_results(self, tool_results: list[dict[str, Any]]) -> None:
        pending = self._pending
        if pending is None:
            raise RuntimeError("no pending tool call")  # noqa: TRY301
        tool_results = _validate_tool_results(tool_results, self._expected_ids)
        self.record_tool_results(pending, tool_results)
        self._tool_rounds += 1
        if self._tool_rounds > MAX_TOOL_ROUNDS:
            raise ValueError(f"exceeded max tool rounds per step ({MAX_TOOL_ROUNDS})")
        pending["messages"].append(pending["asst"])
        for r in tool_results:
            pending["messages"].append(
                {
                    "role": "tool",
                    "tool_call_id": r.get("tool_call_id"),
                    "content": str(r.get("content", ""))[:RUN_MAX_MSG_BYTES],
                }
            )
        self._expected_ids = set()
        self._pending = pending

    def advance(self, tool_results: Any) -> dict[str, Any]:
        with self.lock:
            self.touch()
            if self.cancelled:
                return {"type": "error", "error": "run cancelled"}
            if self.finished:
                return {"type": "error", "error": "run already finished"}
            try:
                if self._pending is None and tool_results is not None:
                    return {"type": "error", "error": "unexpected tool results"}
                if self._pending is not None:
                    if tool_results is None:
                        return {"type": "error", "error": "missing tool results"}
                    self._apply_tool_results(tool_results)
                    p = self._pending
                    if p is None:
                        raise RuntimeError("no pending tool call")  # noqa: TRY301
                    return cast(
                        dict[str, Any],
                        self._run_model(p["role"], p["agent_id"], p["turn"], p["messages"]),
                    )

                # Finish conditions before starting a new coordinator turn.
                if self.terminated_by is not None:
                    self.finished = True
                    return self._final()
                if len(self.turns) >= self.max_turns:
                    self.terminated_by = "max_turns"
                    self.finished = True
                    return self._final()

                turn = len(self.turns)
                role, agent_id = self._route()
                messages = self._build_messages(role)
                return cast(dict[str, Any], self._run_model(role, agent_id, turn, messages))
            except ValueError as e:
                return {"type": "error", "error": str(e)}
            except Exception as e:  # noqa: BLE001 - model/network failures abort the run
                self.close()
                return {"type": "error", "error": str(e)}

    def _final(self) -> dict[str, Any]:
        text = self.final_text or (self.turns[-1]["reply"] if self.turns else "")
        return {
            "type": "final",
            "text": text,
            "terminated_by": self.terminated_by or "",
            "steps": self.turns,
        }

    @staticmethod
    def _extract_thought(reply: str) -> str:
        return reply.strip()

    @staticmethod
    def _parse_thinker(text: str):
        import re

        role = None
        m = re.search(
            r"<suggested_role>\s*(solver|thinker|verifier)\s*</suggested_role>",
            text,
            re.IGNORECASE,
        )
        if m:
            role = {"solver": "Worker", "thinker": "Thinker", "verifier": "Verifier"}[
                m.group(1).lower()
            ]
        sug = None
        s = re.search(r"<suggestion>\s*([\s\S]*?)\s*</suggestion>", text, re.IGNORECASE)
        if s:
            sug = s.group(1).strip() or None
        return role, sug

    @staticmethod
    def _parse_verification(text: str) -> bool:
        return text.strip().upper().startswith("ACCEPT")


class ConductorRun(NativeRun):
    """Resumable Conductor run: planning step then DAG nodes, all tool-capable."""

    def __init__(
        self,
        run_id: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        slot_models: list[str] | None = None,
        max_steps: int = 5,
    ) -> None:
        super().__init__(run_id)
        self.kind = "conductor"
        self.tools = tools
        self.slot_models = slot_models or list(DEFAULT_SLOT_LABELS)
        self.max_steps = max_steps
        query, history = _split_messages(messages)
        self.query = query or ""
        self.history = history
        self.conductor_model = _resolve_conductor_model(
            SimpleNamespace(slot_models=self.slot_models)
        )
        self.steps: list[dict[str, Any]] = []
        self._workflow: tuple[list, list, list] | None = None
        self._outputs: list[str] = []
        self._next_node = 0
        self._pending: dict[str, Any] | None = None
        self._expected_ids: set[str] = set()
        self._tool_rounds = 0

    def _planner_messages(self) -> list[dict[str, Any]]:
        prior = [m for m in self.history if isinstance(m, dict) and m.get("role") != "system"]
        return cast(
            list[dict[str, Any]],
            conductor_prompt(self.query, self.slot_models) + prior[-4:],
        )

    def _node_messages(self, node_index: int, mid: int, sub: str) -> list[dict[str, Any]]:
        if self._workflow is None:
            raise ValueError("workflow required")
        sees = visible_indices(self._workflow[2], node_index)
        mids = self._workflow[0]
        subs = self._workflow[1]
        ctx = ""
        for j in sees:
            prev_mid = mids[j]
            ctx += (
                f"\n<Subtask assigned to Agent {prev_mid}>{subs[j]}"
                f"</Subtask assigned to Agent {prev_mid}>"
                f"\n<Agent {prev_mid} response>{self._outputs[j].strip()}"
                f"</Agent {prev_mid} response>"
            )
        user = (
            f"USER QUESTION context:\n{ctx}\n\nYour subtask: {sub}"
            if ctx
            else f"Your subtask: {sub}"
        )
        return [{"role": "user", "content": user}]

    def _run_model(self, role: str, model: str, messages: list) -> dict[str, Any]:
        seq = len(self.steps)
        text, calls = _model_completion(model, messages, self.tools)
        if calls:
            asst = {
                "role": "assistant",
                "content": text,
                "tool_calls": [
                    _openai_tool_call(c["name"], c["id"], c["arguments"]) for c in calls
                ],
            }
            self._pending = {"role": role, "model": model, "messages": messages, "asst": asst}
            self._expected_ids = {c["id"] for c in calls}
            self._tool_rounds = 1
            return {
                "type": "tool_calls",
                "role": role,
                "model_name": model,
                "turn": seq,
                "tool_calls": calls,
            }
        return self._finalize_text(role, text, seq)

    def _finalize_text(self, role: str, text: str, seq: int) -> dict[str, Any]:
        self._pending = None
        self._expected_ids = set()
        self._tool_rounds = 0
        if role == "Planner":
            try:
                self._workflow = parse_workflow(text)
            except Exception as e:  # noqa: BLE001
                return {
                    "type": "error",
                    "error": f"Conductor did not emit a parseable workflow: {e}",
                }
            mids, subs, access = self._workflow
            if not subs or not (len(mids) == len(subs) == len(access)):
                return {
                    "type": "error",
                    "error": "Conductor emitted an empty or malformed workflow",
                }
            self.steps.append(
                {
                    "turn": seq,
                    "role": "Planner",
                    "agent_id": 0,
                    "model_name": self.conductor_model,
                    "prompt": self.query,
                    "reply": text,
                }
            )
            return {
                "type": "step_complete",
                "turn": seq,
                "role": "Planner",
                "agent_id": 0,
                "model_name": self.conductor_model,
                "prompt": self.query,
                "reply": text,
            }

        node_index = self._next_node - 1
        if self._workflow is None:
            raise ValueError("workflow required")
        mids = self._workflow[0]
        subs = self._workflow[1]
        mid = int(mids[node_index]) % len(self.slot_models)
        self._outputs.append(text)
        self.steps.append(
            {
                "turn": seq,
                "role": "Worker",
                "agent_id": mid,
                "model_name": self.slot_models[mid],
                "prompt": subs[node_index],
                "reply": text,
            }
        )
        return {
            "type": "step_complete",
            "turn": seq,
            "role": "Worker",
            "agent_id": mid,
            "model_name": self.slot_models[mid],
            "prompt": subs[node_index],
            "reply": text,
        }

    def _apply_tool_results(self, tool_results: list[dict[str, Any]]) -> None:
        pending = self._pending
        if pending is None:
            raise RuntimeError("no pending tool call")  # noqa: TRY301
        tool_results = _validate_tool_results(tool_results, self._expected_ids)
        self.record_tool_results(pending, tool_results)
        self._tool_rounds += 1
        if self._tool_rounds > MAX_TOOL_ROUNDS:
            raise ValueError(f"exceeded max tool rounds per step ({MAX_TOOL_ROUNDS})")
        pending["messages"].append(pending["asst"])
        for r in tool_results:
            pending["messages"].append(
                {
                    "role": "tool",
                    "tool_call_id": r.get("tool_call_id"),
                    "content": str(r.get("content", ""))[:RUN_MAX_MSG_BYTES],
                }
            )
        self._expected_ids = set()
        self._pending = pending

    def advance(self, tool_results: Any) -> dict[str, Any]:
        with self.lock:
            self.touch()
            if self.cancelled:
                return {"type": "error", "error": "run cancelled"}
            if self.finished:
                return {"type": "error", "error": "run already finished"}
            try:
                if self._pending is None and tool_results is not None:
                    return {"type": "error", "error": "unexpected tool results"}
                if self._pending is not None:
                    if tool_results is None:
                        return {"type": "error", "error": "missing tool results"}
                    self._apply_tool_results(tool_results)
                    p = self._pending
                    if p is None:
                        raise RuntimeError("no pending tool call")  # noqa: TRY301
                    return cast(
                        dict[str, Any], self._run_model(p["role"], p["model"], p["messages"])
                    )

                if self._workflow is None:
                    return self._run_model(
                        "Planner", self.conductor_model, self._planner_messages()
                    )
                mids, subs, access = self._workflow
                if self._next_node >= len(subs):
                    self.finished = True
                    self.final_text = self._outputs[-1] if self._outputs else ""
                    self.terminated_by = "conductor_done"
                    return {
                        "type": "final",
                        "text": self.final_text,
                        "terminated_by": "conductor_done",
                        "steps": self.steps,
                    }
                if self._next_node >= self.max_steps:
                    self.finished = True
                    self.final_text = self._outputs[-1] if self._outputs else ""
                    self.terminated_by = "max_steps"
                    return {
                        "type": "final",
                        "text": self.final_text,
                        "terminated_by": "max_steps",
                        "steps": self.steps,
                    }
                node_index = self._next_node
                self._next_node += 1
                mid = int(mids[node_index]) % len(self.slot_models)
                model = self.slot_models[mid]
                return self._run_model(
                    "Worker", model, self._node_messages(node_index, mid, subs[node_index])
                )
            except ValueError as e:
                return {"type": "error", "error": str(e)}
            except Exception as e:  # noqa: BLE001
                self.close()
                return {"type": "error", "error": str(e)}

    def close(self) -> None:
        self.cancelled = True


def create_run(mode: str, body: dict[str, Any]) -> NativeRun:
    messages = body.get("messages") or []
    if (
        not isinstance(messages, list)
        or not messages
        or not all(isinstance(message, dict) for message in messages)
    ):
        raise ValueError("messages must be a non-empty list of objects")
    tools = _convert_tools(body.get("tools"))
    slot_models = _configured_slot_models(body.get("slot_models"))
    requested_id = body.get("run_id")
    if requested_id is not None and (
        not isinstance(requested_id, str)
        or len(requested_id) != 32
        or any(char not in "0123456789abcdef" for char in requested_id.lower())
    ):
        raise ValueError("run_id must be a 32-character hexadecimal string")
    run_id = requested_id or uuid.uuid4().hex
    run: NativeRun
    if mode == "conductor":
        run = ConductorRun(run_id, messages, tools, slot_models=slot_models)
    else:
        run = TrinityRun(run_id, messages, tools, slot_models=slot_models)
    _register_run(run)
    return run


def get_run(run_id: str) -> NativeRun:
    with _runs_lock:
        run = _runs.get(run_id)
    if run is None:
        raise KeyError(f"unknown or expired run: {run_id}")
    return run


def advance_run(run_id: str, tool_results: Any, request_id: Any = None) -> dict[str, Any]:
    with _runs_lock:
        run = _runs.get(run_id)
        if run is None:
            raise KeyError(f"unknown or expired run: {run_id}")
        run.in_flight += 1
    try:
        event = run.advance_idempotent(tool_results, request_id)
        if event.get("type") in ("final", "error"):
            _write_learning_record(run, event)
        return event
    finally:
        with _runs_lock:
            run.in_flight -= 1
            run.touch()


def delete_run(run_id: str) -> bool:
    with _runs_lock:
        run = _runs.pop(run_id, None)
    if run is not None:
        _write_learning_record(run, {"type": "error", "terminated_by": "deleted"})
        run.close()
    return run is not None


if __name__ == "__main__":
    main()
