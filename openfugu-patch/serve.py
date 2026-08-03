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
import json
import os
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
# When running from the repo's openfugu-patch overlay, also find upstream OpenFugu.
if not (_HERE / "mini.py").exists():
    _OPENFUGU = _HERE.parent / "OpenFugu" / "openfugu"
    if _OPENFUGU.exists():
        sys.path.insert(0, str(_OPENFUGU))

from mini import DEFAULT_SLOT_LABELS, HEAD_ROWS, HIDDEN, Coordinator, FuguRouter
from mini import LiteLLMWorker as _TrinityLiteLLMWorker
from ultra import ConductorExecutor, conductor_prompt, parse_workflow
from ultra import LiteLLMWorker as _ConductorLiteLLMWorker

ROUTER: FuguRouter | None = None
MODEL_NAME = os.environ.get("MANTIS_MODEL_NAME", os.environ.get("FUGU_MODEL_NAME", "mantis"))
MAX_TURNS = 5
# Reject bodies larger than this many bytes.
MAX_BODY_BYTES = int(
    os.environ.get(
        "MANTIS_MAX_BODY_BYTES",
        os.environ.get("FUGU_MAX_BODY_BYTES", str(5 * 1024 * 1024)),
    )
)

# Aliases that carry a LiteLLM reasoning_effort parameter. OpenRouter/LiteLLM
# reject temperature != 1 for these models.
REASONING_ALIASES = ("claude-", "gpt-5.6-", "expensive", "cheap")

_args: argparse.Namespace | None = None
_coordinators: dict[str, object] = {}
_history_context = threading.local()


def _is_reasoning_model(model: str) -> bool:
    return any(model.startswith(p) for p in REASONING_ALIASES)


def _build_litellm_kwargs(
    model: str, messages: list[dict[str, str]], max_tokens: int, temperature: float
) -> dict[str, Any]:
    """Construct kwargs for a LiteLLM completion routed through the OpenRouter proxy."""
    kw: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "custom_llm_provider": "openai",
    }
    # LiteLLM's openai provider rejects temperature != 1 when reasoning_effort
    # is enabled (claude-*, gpt-5.6-*). Drop it for those models while keeping
    # it for the low-cost non-reasoning workers (deepseek/glm/opencode).
    if not _is_reasoning_model(model):
        kw["temperature"] = temperature
    return kw


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
        # Strip any system messages from history; the coordinator adds its own.
        prior = [m for m in history if isinstance(m, dict) and m.get("role") != "system"]
        if not prior:
            return messages
        if messages and isinstance(messages[0], dict) and messages[0].get("role") == "system":
            return [messages[0]] + prior + list(messages[1:])
        return prior + list(messages)

    def __call__(self, *args: Any) -> Any:
        if len(args) == 3:
            role_or_subtask, messages, agent_id = args
            return self._worker(role_or_subtask, self._combine(messages), agent_id)
        return self._worker(*args)

    def conduct(self, *args: Any) -> Any:
        if not (getattr(_history_context, "history", None) or []):
            return self._worker.conduct(*args)
        if len(args) == 2:
            model, messages = args
            return self._worker.conduct(model, self._combine(messages))
        if len(args) == 1:
            messages, = args
            return self._worker.conduct(self._combine(messages))
        return self._worker.conduct(*args)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._worker, name)


class OpenRouterTrinityWorker(_TrinityLiteLLMWorker):
    """TRINITY worker that dispatches each turn through the LiteLLM/OpenRouter proxy."""

    def __call__(self, role_name: str, messages: list, agent_id: int) -> str:
        import litellm

        model = self.slot_models[agent_id % len(self.slot_models)]
        msgs = [{"role": m["role"], "content": m["content"]} for m in messages]
        kw = _build_litellm_kwargs(model, msgs, self.max_tokens, self.temperature)
        if self.api_key:
            kw["api_key"] = self.api_key
        if self.api_base:
            kw["api_base"] = self.api_base
        return str(litellm.completion(**kw).choices[0].message.content or "")


class OpenRouterConductorWorker(_ConductorLiteLLMWorker):
    """Conductor worker that dispatches plan and step calls through LiteLLM/OpenRouter."""

    def _call(self, model: str, messages: list) -> str:
        kw = _build_litellm_kwargs(model, messages, self.max_tokens, self.temperature)
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
            text = tk.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
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
    step_details = [{
        "turn": getattr(turn, "t", getattr(turn, "step", 0)),
        "agent_id": getattr(turn, "agent_id", 0),
        "role": getattr(turn, "role", getattr(turn, "role_name", "Worker")),
        "reply": getattr(turn, "reply", getattr(turn, "text", "")),
    } for turn in turns]
    return {
        "id": "chatcmpl-" + uuid.uuid4().hex[:24],
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": text,
                "mantis_steps": step_details,
            },
            "finish_reason": "stop",
        }],
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
    conductor_model = os.environ.get(
        "MANTIS_CONDUCTOR_MODEL", os.environ.get("FUGU_CONDUCTOR_MODEL")
    )
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

    def run(self, query: str, verbose: bool = False):
        conductor_model = _resolve_conductor_model(self.worker)
        completion = self.worker.conduct(conductor_model, conductor_prompt(query, self.slot_labels))
        return _run_conductor_workflow(self.worker, query, self.slot_labels, completion, verbose)


def choose_conductor_device(torch_module: Any, env_device: str | None = None) -> str:
    """Pick mps > cuda:0 > cpu, allowing an explicit env override."""
    if env_device and env_device != "auto":
        return env_device
    if torch_module.backends.mps.is_available():
        return "mps"
    if torch_module.cuda.is_available():
        return "cuda:0"
    return "cpu"


def choose_conductor_dtype(
    device: str, torch_module: Any, env_dtype: str | None = None
) -> Any:
    """Return torch dtype for a Conductor device, honoring an explicit override."""
    if env_dtype:
        return getattr(torch_module, env_dtype)
    return torch_module.bfloat16 if device in ("mps", "cuda", "cuda:0") else torch_module.float32


class EnvLocalConductor:
    """Load a GRPO-trained Conductor checkpoint locally with transformers.

    Env overrides: FUGU_CONDUCTOR_DEVICE (cpu/cuda:0/mps/auto),
                    FUGU_CONDUCTOR_DTYPE (float32/bfloat16/float16),
                    FUGU_CONDUCTOR_MAX_NEW.
    Defaults to bfloat16 on mps/cuda and float32 on cpu."""

    def __init__(self, ckpt: str, device: str | None = None, max_new: int | None = None) -> None:
        import torch as _torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = _torch
        self.ckpt = ckpt
        env_device = device if device is not None else os.environ.get("FUGU_CONDUCTOR_DEVICE")
        self.device = choose_conductor_device(_torch, env_device)
        self.max_new = max_new or int(os.environ.get("FUGU_CONDUCTOR_MAX_NEW", "512"))
        dtype_env = os.environ.get("FUGU_CONDUCTOR_DTYPE")
        self.dtype = choose_conductor_dtype(self.device, _torch, dtype_env)
        self.temperature = float(os.environ.get("FUGU_CONDUCTOR_TEMPERATURE", "0.7"))
        self.top_p = float(os.environ.get("FUGU_CONDUCTOR_TOP_P", "0.9"))
        do_sample_env = os.environ.get("FUGU_CONDUCTOR_DO_SAMPLE")
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

    def __init__(self, worker: Any, conductor: EnvLocalConductor | None = None,
                 slot_labels: list[str] | None = None) -> None:
        super().__init__(worker, slot_labels=slot_labels)
        self.local_conductor = conductor

    def run(self, query: str, verbose: bool = False):
        if self.local_conductor is not None:
            completion = self.local_conductor.conduct(conductor_prompt(query, self.slot_labels))
        else:
            completion = self.worker.conduct(
                _resolve_conductor_model(self.worker),
                conductor_prompt(query, self.slot_labels),
            )
        return _run_conductor_workflow(self.worker, query, self.slot_labels, completion, verbose)


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


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send(self, code: int, body: dict) -> None:
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _auth_token(self) -> str | None:
        return (
            os.environ.get("MANTIS_API_KEY")
            or os.environ.get("FUGU_API_KEY")
            or os.environ.get("LITELLM_KEY")
        )

    def _check_auth(self) -> bool:
        expected = self._auth_token()
        if not expected:
            self._send(
                500,
                {"error": "MANTIS_API_KEY (or FUGU_API_KEY / LITELLM_KEY) is not configured"},
            )
            return False
        auth = self.headers.get("Authorization", "")
        if not auth.startswith("Bearer ") or auth[7:] != expected:
            self._send(401, {"error": "unauthorized"})
            return False
        return True

    def do_GET(self) -> None:
        if self.path == "/v1/models":
            if not self._check_auth():
                return
            self._send(200, {"object": "list", "data": [
                {"id": MODEL_NAME, "object": "model", "owned_by": "mantis"}]})
        elif self.path in ("/health", "/"):
            self._send(200, {"status": "ok", "model": MODEL_NAME})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path.rstrip("/") != "/v1/chat/completions":
            self._send(404, {"error": "not found"})
            return
        if not self._check_auth():
            return
        try:
            n = int(self.headers.get("Content-Length", 0))
            if n > MAX_BODY_BYTES:
                self._send(413, {"error": f"request body exceeds {MAX_BODY_BYTES} bytes"})
                return
            req = json.loads(self.rfile.read(n) or b"{}")
            messages = req.get("messages", [])
            if not messages:
                self._send(400, {"error": "messages required"})
                return

            requested = (req.get("model") or "trinity").lower()
            coordinator_mode = "conductor" if requested in ("conductor", "ultra") else "trinity"

            # the user query = last user message; history = everything before it
            query, history = _split_messages(messages)
            print(
                f"[serve] route request model={requested} -> coordinator={coordinator_mode}",
                flush=True,
            )
            _history_context.history = history
            try:
                coord = get_coordinator(coordinator_mode)
                res = coord.run(query, verbose=False)
            finally:
                _history_context.history = []
            self._send(
                200, _chat_response(res, req.get("model", MODEL_NAME))
            )
        except (json.JSONDecodeError, ValueError, KeyError, RuntimeError) as e:
            self._send(500, {"error": str(e)})

    def log_message(self, *a) -> None:  # quiet
        pass


def _parse_args() -> argparse.Namespace:
    global _args
    if _args is not None:
        return _args
    ap = argparse.ArgumentParser(description="Serve Mantis as one OpenAI-compatible model.")
    ap.add_argument(
        "--model",
        default=os.environ.get("MANTIS_MODEL", os.environ.get("FUGU_MODEL", "Qwen/Qwen3-0.6B")),
        help="Qwen3-0.6B dir or HF id",
    )
    ap.add_argument(
        "--vector",
        default=os.environ.get("MANTIS_VECTOR", os.environ.get("FUGU_VECTOR", "model_iter_60.npy")),
        help="base vector (19456) — SVF + head",
    )
    ap.add_argument(
        "--head",
        default=os.environ.get("MANTIS_HEAD", os.environ.get("FUGU_HEAD")),
        help="optional trained head-only vector/safetensors; overrides the "
        "head from --vector after SVF is applied",
    )
    default_workers = os.environ.get(
        "MANTIS_WORKER_MODELS",
        os.environ.get(
            "FUGU_WORKER_MODELS",
            os.environ.get("MANTIS_WORKER_MODEL", os.environ.get("FUGU_WORKER_MODEL")),
        ),
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
        default=os.environ.get("MANTIS_LOCAL_MODELS", os.environ.get("FUGU_LOCAL_MODELS")),
        help="local HF worker model paths (CSV). "
        "Optional 'path@device' per entry; also MANTIS_LOCAL_MODELS",
    )
    ap.add_argument(
        "--host",
        default=os.environ.get("MANTIS_HOST", os.environ.get("FUGU_HOST", "0.0.0.0")),  # noqa: S104
    )
    ap.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("MANTIS_PORT", os.environ.get("FUGU_PORT", "8088"))),
    )
    ap.add_argument(
        "--max-turns",
        type=int,
        default=int(os.environ.get("MANTIS_MAX_TURNS", os.environ.get("FUGU_MAX_TURNS", "5"))),
    )
    _args = ap.parse_args()
    return _args


def get_router() -> FuguRouter:
    global ROUTER
    if ROUTER is None:
        args = _parse_args()
        device = os.environ.get("MANTIS_DEVICE", os.environ.get("FUGU_DEVICE"))
        print(f"[serve] loading TRINITY router ({args.model}) ...", flush=True)
        ROUTER = FuguRouter(args.model, args.vector, device=device, seed=0)
        if args.head:  # layer a trained head over base SVF
            head = _load_head(args.head)
            ROUTER.head = ROUTER.torch.from_numpy(head.copy()).float().reshape(
                HEAD_ROWS, HIDDEN).to(ROUTER.device)
            print(f"[serve] applied trained head from {args.head}", flush=True)
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
    if mode == "conductor":
        return OpenRouterConductorWorker(slot_models=slot_models, max_tokens=4096)
    if mode == "trinity":
        return OpenRouterTrinityWorker(slot_models=slot_models, max_tokens=4096)
    raise ValueError(f"unknown coordinator mode: {mode}")


def load_coordinator(mode: str):
    global MAX_TURNS
    if mode not in ("trinity", "conductor"):
        raise ValueError(f"unknown coordinator mode: {mode}")
    args = _parse_args()
    MAX_TURNS = args.max_turns
    worker = HistoryWorker(_worker_from_args(args, mode))
    if mode == "trinity":
        return Coordinator(get_router(), worker, max_turns=args.max_turns, sample=True)
    local_ckpt = os.environ.get("MANTIS_LOCAL_CONDUCTOR", os.environ.get("FUGU_LOCAL_CONDUCTOR"))
    conductor = EnvLocalConductor(local_ckpt) if local_ckpt else None
    return EnvConductorCoordinator(
        worker, conductor=conductor, slot_labels=getattr(worker, "slot_models", None)
    )


def get_coordinator(mode: str):
    if mode not in _coordinators:
        _coordinators[mode] = load_coordinator(mode)
    return _coordinators[mode]


def main() -> None:
    args = _parse_args()
    token = (
        os.environ.get("MANTIS_API_KEY")
        or os.environ.get("FUGU_API_KEY")
        or os.environ.get("LITELLM_KEY")
    )
    if not token:
        print(
            "[serve] FATAL: set MANTIS_API_KEY (or FUGU_API_KEY / LITELLM_KEY)"
            " before starting the server",
            flush=True,
        )
        raise SystemExit(1)
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(
        f"[serve] Mantis listening on {args.host}:{args.port} — POST /v1/chat/completions",
        flush=True,
    )
    srv.serve_forever()


if __name__ == "__main__":
    main()
