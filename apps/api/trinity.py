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
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
# When running from the repo's orchestration runtime, also find upstream OpenFugu.
if not (_HERE / "mini.py").exists():
    _OPENFUGU = _HERE.parent / "OpenFugu" / "openfugu"
    if _OPENFUGU.exists():
        sys.path.insert(0, str(_OPENFUGU))

import providers
import serve_config
import utils
from mini import (
    HEAD_ROWS,
    HIDDEN,
    FuguRouter,
)
from serve_config import (
    ROUTER,
    WORKER_TIMEOUT,
    _router_lock,
)


class RejectAwareRouter:
    """Force a revision after a verifier rejects instead of re-verifying unchanged text."""

    def __init__(self, router: Any) -> None:
        self._router = router

    def route(self, *args: Any, **kwargs: Any) -> Any:
        result = self._router.route(*args, **kwargs)
        if getattr(serve_config._history_context, "force_worker", False):
            serve_config._history_context.force_worker = False
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

    conductor_model: str | None

    def __init__(self, worker: Any) -> None:
        self._worker = worker
        self.conductor_model = None

    def _combine(self, messages: Any) -> Any:
        history = getattr(serve_config._history_context, "history", None) or []
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
        providers._check_client_connected()
        if len(args) == 3:
            role_or_subtask, messages, agent_id = args
            combined = self._combine(messages)
            is_conductor = getattr(serve_config._history_context, "conductor_mode", False)
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
                calls = getattr(serve_config._history_context, "calls", None)
                if calls is not None:
                    calls.append(call)
                write_line = getattr(serve_config._history_context, "write_line", None)
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
                feedback = getattr(serve_config._history_context, "revision_feedback", None)
                if feedback and combined and isinstance(combined[-1], dict):
                    prev_content = combined[-1].get("content", "")
                    combined[-1] = {
                        **combined[-1],
                        "content": (
                            f"{prev_content}\n\n"
                            f"Revise the answer to address this verifier feedback:\n{feedback}"
                        ),
                    }
                    serve_config._history_context.revision_feedback = None
            original_prompt = self._last_user_prompt(messages)
            call = {
                "role": role,
                "agent_id": agent_id,
                "model_name": self._model_name(agent_id),
                "messages": combined,
                "prompt": original_prompt,
            }
            calls = getattr(serve_config._history_context, "calls", None)
            if calls is not None:
                calls.append(call)
            write_line = getattr(serve_config._history_context, "write_line", None)
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
                serve_config._history_context.force_worker = True
                serve_config._history_context.revision_feedback = (
                    "Previous worker returned no response; produce a complete answer."
                )
            elif role == "Verifier" and reply.strip().upper().startswith("REJECT"):
                serve_config._history_context.force_worker = True
                serve_config._history_context.revision_feedback = reply
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
        providers._check_client_connected()
        if not (getattr(serve_config._history_context, "history", None) or []):
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


class DirectTrinityWorker:
    def __init__(
        self,
        slot_models: list[str],
        max_tokens: int = 4096,
        temperature: float = 0.7,
        timeout: float | None = None,
    ) -> None:
        self.slot_models = slot_models
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout if timeout is not None else WORKER_TIMEOUT

    def __call__(self, role_name: str, messages: list, agent_id: int) -> str:
        model = self.slot_models[agent_id % len(self.slot_models)]
        msgs = [{"role": m["role"], "content": m["content"]} for m in messages]
        return providers._direct_completion(
            model, msgs, self.max_tokens, self.temperature, self.timeout
        )


class DirectConductorWorker(DirectTrinityWorker):
    def _call(self, model: str, messages: list) -> str:
        return providers._direct_completion(
            model, messages, self.max_tokens, self.temperature, self.timeout
        )

    def conduct(self, model: str, messages: list) -> str:
        return self._call(model, messages)


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


def get_router() -> FuguRouter:
    global ROUTER
    if ROUTER is None:
        with _router_lock:
            if ROUTER is None:
                args = utils._parse_args()
                device = os.environ.get("MANTIS_DEVICE")
                print(f"[serve] loading TRINITY router ({args.model}) ...", flush=True)
                router = FuguRouter(
                    args.model,
                    args.vector,
                    dtype=os.environ.get("MANTIS_ROUTER_DTYPE", "float32"),
                    device=device,
                    seed=0,
                )
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
    slot_models = args.slot_models.split(",") if args.slot_models else []
    if not slot_models:
        raise ValueError("MANTIS_WORKER_MODELS is required")
    if mode == "conductor":
        return DirectConductorWorker(slot_models=slot_models, max_tokens=4096)
    if mode == "trinity":
        return DirectTrinityWorker(slot_models=slot_models, max_tokens=4096)
    raise ValueError(f"unknown coordinator mode: {mode}")


__all__ = [k for k in globals() if not k.startswith("__")]
