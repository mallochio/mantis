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

import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
# When running from the repo's orchestration runtime, also find upstream OpenFugu.
if not (_HERE / "mini.py").exists():
    _OPENFUGU = _HERE.parent / "OpenFugu" / "openfugu"
    if _OPENFUGU.exists():
        sys.path.insert(0, str(_OPENFUGU))

import model_catalog
import serve_config
import trinity
import utils
from mini import (
    DEFAULT_SLOT_LABELS,
    Coordinator,
)
from ultra import ConductorExecutor, conductor_prompt, parse_workflow


def _load_mantis_catalog() -> model_catalog.MantisCatalog | None:
    try:
        return model_catalog.load_mantis_catalog(require_contract=False)
    except (model_catalog.CatalogError, OSError, ValueError):
        return None


def _resolve_conductor_model(worker: Any) -> str:
    """Pick the model used for the Conductor planning call.

    Prefer an explicit conductor_model attached to the worker (set from the
    shared catalog), then the first configured slot, then a safe default.
    """
    conductor_model = getattr(worker, "conductor_model", None)
    if isinstance(conductor_model, str) and conductor_model:
        return conductor_model
    slot_models = getattr(worker, "slot_models", None)
    if isinstance(slot_models, list) and slot_models and isinstance(slot_models[0], str):
        return str(slot_models[0])
    return "openai/gpt-4o-mini"


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
        serve_config._history_context.conductor_mode = True
        try:
            conductor_model, prompt_msgs, get_completion = self._prepare_planning(query)

            calls = getattr(serve_config._history_context, "calls", None)
            if calls is None:
                calls = []
                serve_config._history_context.calls = calls

            write_line = getattr(serve_config._history_context, "write_line", None)
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
            serve_config._history_context.conductor_mode = False


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
            self.model = self.model.to(self.device)
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
    (Llama-3.2-3B Conductor) or Bifrost for the planning call."""

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


def load_coordinator(mode: str):
    if mode not in ("trinity", "conductor"):
        raise ValueError(f"unknown coordinator mode: {mode}")
    args = utils._parse_args()
    serve_config.MAX_TURNS = args.max_turns
    worker = trinity.HistoryWorker(trinity._worker_from_args(args, mode))
    if mode == "trinity":
        return Coordinator(
            trinity.RejectAwareRouter(trinity.get_router()),
            worker,
            max_turns=args.max_turns,
            sample=True,
        )
    local_ckpt = os.environ.get("MANTIS_LOCAL_CONDUCTOR")
    conductor = EnvLocalConductor(local_ckpt) if local_ckpt else None
    catalog = _load_mantis_catalog()
    worker.conductor_model = (
        catalog.conductor_model if catalog is not None else None
    )
    return EnvConductorCoordinator(
        worker, conductor=conductor, slot_labels=getattr(worker, "slot_models", None)
    )


def get_coordinator(mode: str):
    if mode not in serve_config._coordinators:
        with serve_config._coordinator_lock:
            if mode not in serve_config._coordinators:
                serve_config._coordinators[mode] = load_coordinator(mode)
    return serve_config._coordinators[mode]


__all__ = [k for k in globals() if not k.startswith("__")]
