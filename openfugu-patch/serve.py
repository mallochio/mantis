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
import argparse, glob, json, os, sys, time, uuid
import numpy as np
import torch
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# reuse the faithful implementation
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mini import (FuguRouter, Coordinator, LiteLLMWorker as TrinityLiteLLMWorker,
                  MockWorker, DEFAULT_SLOT_LABELS, HEAD_ROWS, HIDDEN)
from ultra import (LiteLLMWorker as ConductorLiteLLMWorker, ConductorExecutor,
                   conductor_prompt, parse_workflow,
                   DEFAULT_SLOT_LABELS as CONDUCTOR_DEFAULT_SLOT_LABELS)

ROUTER: FuguRouter | None = None
MODEL_NAME = "fugu"
MAX_TURNS = 5

# Aliases that carry a LiteLLM reasoning_effort parameter. OpenRouter/LiteLLM
# reject temperature != 1 for these models.
REASONING_ALIASES = ("claude-", "gpt-5.6-", "expensive", "cheap")


def _is_reasoning_model(model: str) -> bool:
    return any(model.startswith(p) for p in REASONING_ALIASES)


# LiteLLM needs an explicit OpenAI-compatible provider when the model name is a
# LiteLLM proxy alias (not a provider-prefixed id). This keeps fugu-local routing
# through the internal LiteLLM proxy and, ultimately, OpenRouter.
class TrinityLiteLLMWorker(TrinityLiteLLMWorker):
    def __call__(self, role_name: str, messages: list, agent_id: int) -> str:
        import litellm
        model = self.slot_models[agent_id % len(self.slot_models)]
        msgs = [{"role": m["role"], "content": m["content"]} for m in messages]
        # LiteLLM's openai provider rejects temperature != 1 when reasoning_effort
        # is enabled (claude-*, gpt-5.6-*). Drop it for those models while keeping
        # it for the low-cost non-reasoning workers (deepseek/glm/opencode).
        kw = dict(model=model, messages=msgs,
                  max_tokens=self.max_tokens,
                  custom_llm_provider="openai")
        if not _is_reasoning_model(model):
            kw["temperature"] = self.temperature
        if self.api_key:
            kw["api_key"] = self.api_key
        if self.api_base:
            kw["api_base"] = self.api_base
        return litellm.completion(**kw).choices[0].message.content or ""


class ConductorLiteLLMWorker(ConductorLiteLLMWorker):
    def _call(self, model, messages):
        kw = dict(model=model, messages=messages,
                  max_tokens=self.max_tokens,
                  custom_llm_provider="openai")
        if not _is_reasoning_model(model):
            kw["temperature"] = self.temperature
        if self.api_key:
            kw["api_key"] = self.api_key
        if self.api_base:
            kw["api_base"] = self.api_base
        return self.litellm.completion(**kw).choices[0].message.content or ""

_args = None
_coordinators: dict[str, object] = {}


def _chat_response(text: str, model: str, usage_turns: int) -> dict:
    return {
        "id": "chatcmpl-" + uuid.uuid4().hex[:24],
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop",
        }],
        # surface the orchestration depth without exposing which workers ran
        "usage": {"fugu_turns": usage_turns},
    }


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send(self, code: int, body: dict):
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/v1/models":
            self._send(200, {"object": "list", "data": [
                {"id": MODEL_NAME, "object": "model", "owned_by": "openfugu"}]})
        elif self.path in ("/health", "/"):
            self._send(200, {"status": "ok", "model": MODEL_NAME})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path.rstrip("/") != "/v1/chat/completions":
            self._send(404, {"error": "not found"}); return
        try:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or b"{}")
            messages = req.get("messages", [])
            if not messages:
                self._send(400, {"error": "messages required"}); return

            requested = (req.get("model") or "trinity").lower()
            if requested in ("conductor", "ultra"):
                coordinator_mode = "conductor"
            else:
                coordinator_mode = "trinity"

            # the user query = last user message; coordinator runs the full loop
            query = next((m["content"] for m in reversed(messages)
                          if m.get("role") == "user"), "")
            print(f"[serve] route request model={requested} -> coordinator={coordinator_mode}", flush=True)
            coord = get_coordinator(coordinator_mode)
            res = coord.run(query, verbose=False)
            self._send(200, _chat_response(res.final, req.get("model", MODEL_NAME),
                                           len(res.turns)))
        except Exception as e:
            self._send(500, {"error": str(e)})

    def log_message(self, *a):       # quiet
        pass


class LocalPoolWorker:
    """Serving-time local worker pool — the same protocol the per-step trainer
    used. The Coordinator calls (role_name, messages, agent_id) -> reply; we
    dispatch to model[agent_id % n], each model resident on its own GPU. Replies
    are decoded greedily so serving is deterministic. No external API."""
    def __init__(self, specs, max_new=384):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch, self.max_new = torch, max_new
        self.names, self.toks, self.models, self.devs = [], [], [], []
        for name, path, dev in specs:
            tk = AutoTokenizer.from_pretrained(path)
            if tk.pad_token is None:
                tk.pad_token = tk.eos_token
            try:
                m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).to(dev).eval()
            except TypeError:
                m = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16).to(dev).eval()
            self.names.append(name); self.toks.append(tk); self.models.append(m); self.devs.append(dev)

    def __call__(self, role_name, messages, agent_id):
        torch = self.torch
        wid = agent_id % len(self.models)
        tk, model, dev = self.toks[wid], self.models[wid], self.devs[wid]
        try:
            text = tk.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            text = "\n".join(m["content"] for m in messages)
        ids = tk(text, return_tensors="pt", truncation=True, max_length=2048).to(dev)
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=self.max_new, do_sample=False,
                                 pad_token_id=tk.pad_token_id)
        return tk.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)


class ConductorCoordinator:
    """Per-request Conductor wrapper: one Conductor LM call produces a workflow
    DAG, then ConductorExecutor runs it. Exposes the same .run(query) interface
    as the TRINITY Coordinator."""
    def __init__(self, worker, slot_labels=None):
        self.worker = worker
        self.slot_labels = slot_labels or getattr(worker, "slot_models", None) or DEFAULT_SLOT_LABELS

    def run(self, query: str, verbose: bool = False):
        conductor_model = os.environ.get("FUGU_CONDUCTOR_MODEL")
        if conductor_model is None:
            if hasattr(self.worker, "slot_models") and self.worker.slot_models:
                conductor_model = self.worker.slot_models[0]
            else:
                conductor_model = "openai/gpt-4o-mini"
        completion = self.worker.conduct(conductor_model, conductor_prompt(query, self.slot_labels))
        mids, subs, acc = parse_workflow(completion)
        if not subs:
            raise ValueError(f"Conductor did not emit a parseable workflow. Raw: {completion[:200]}")
        res = ConductorExecutor(self.worker, slot_labels=self.slot_labels).execute(mids, subs, acc, verbose=verbose)
        # expose a turns attribute for _chat_response
        res.turns = res.steps
        return res


def _parse_args():
    global _args
    if _args is not None:
        return _args
    ap = argparse.ArgumentParser(description="Serve Fugu as one OpenAI-compatible model.")
    ap.add_argument("--model", default=os.environ.get("FUGU_MODEL", "Qwen/Qwen3-0.6B"),
                    help="Qwen3-0.6B dir or HF id")
    ap.add_argument("--vector", default=os.environ.get("FUGU_VECTOR", "model_iter_60.npy"),
                    help="base vector (19456) — SVF + head")
    ap.add_argument("--head", default=os.environ.get("FUGU_HEAD"),
                    help="optional trained head-only vector/safetensors; overrides the "
                         "head from --vector after SVF is applied")
    ap.add_argument("--slot-models", metavar="CSV",
                    default=os.environ.get("FUGU_WORKER_MODELS", os.environ.get("FUGU_WORKER_MODEL")),
                    help="litellm worker ids (CSV); also FUGU_WORKER_MODELS")
    ap.add_argument("--local-models", metavar="CSV", default=os.environ.get("FUGU_LOCAL_MODELS"),
                    help="local HF worker model paths (CSV). "
                         "Optional 'path@device' per entry; also FUGU_LOCAL_MODELS")
    ap.add_argument("--host", default=os.environ.get("FUGU_HOST", "0.0.0.0"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("FUGU_PORT", "8088")))
    ap.add_argument("--max-turns", type=int, default=int(os.environ.get("FUGU_MAX_TURNS", "5")))
    _args = ap.parse_args()
    return _args


def get_router():
    global ROUTER
    if ROUTER is None:
        args = _parse_args()
        device = os.environ.get("FUGU_DEVICE")
        print(f"[serve] loading TRINITY router ({args.model}) ...", flush=True)
        ROUTER = FuguRouter(args.model, args.vector, device=device, seed=0)
        if args.head:                                  # layer a trained head over base SVF
            head = _load_head(args.head)
            ROUTER.head = ROUTER.torch.from_numpy(head.copy()).float().reshape(
                HEAD_ROWS, HIDDEN).to(ROUTER.device)
            print(f"[serve] applied trained head from {args.head}", flush=True)
    return ROUTER


def _load_head(path: str):
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
    return head


class EnvLocalConductor:
    """Load a GRPO-trained Conductor checkpoint locally with transformers.

    Env overrides: FUGU_CONDUCTOR_DEVICE (cpu/cuda:0/mps/auto),
                    FUGU_CONDUCTOR_DTYPE (float32/bfloat16/float16),
                    FUGU_CONDUCTOR_MAX_NEW.
    Falls back to float32 on CPU/MPS; bfloat16 only on CUDA."""
    def __init__(self, ckpt: str, device: str | None = None, max_new: int | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.ckpt = ckpt
        if device is None:
            device = os.environ.get("FUGU_CONDUCTOR_DEVICE")
        if device is None or device == "auto":
            if torch.cuda.is_available(): device = "cuda:0"
            elif torch.backends.mps.is_available(): device = "mps"
            else: device = "cpu"
        self.device = device
        self.max_new = max_new or int(os.environ.get("FUGU_CONDUCTOR_MAX_NEW", "512"))
        dtype_env = os.environ.get("FUGU_CONDUCTOR_DTYPE")
        if dtype_env:
            self.dtype = getattr(torch, dtype_env)
        else:
            self.dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32
        print(f"[serve] loading local Conductor ({ckpt}) on {self.device} dtype={self.dtype} ...", flush=True)
        self.tok = AutoTokenizer.from_pretrained(ckpt)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        try:
            self.model = AutoModelForCausalLM.from_pretrained(ckpt, dtype=self.dtype)
        except TypeError:
            self.model = AutoModelForCausalLM.from_pretrained(ckpt, torch_dtype=self.dtype)
        if self.device == "auto" and torch.cuda.device_count() > 1:
            pass  # leave device_map behavior to from_pretrained
        else:
            self.model = self.model.to(self.device)
        self.model.eval()
        print(f"[serve] Conductor ready", flush=True)

    def conduct(self, messages):
        torch = self.torch
        try:
            text = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            text = "\n".join(m["content"] for m in messages)
        ids = self.tok(text, return_tensors="pt", truncation=True, max_length=2048).to(self.device)
        with torch.no_grad():
            out = self.model.generate(**ids, max_new_tokens=self.max_new, do_sample=False,
                                      pad_token_id=self.tok.pad_token_id)
        return self.tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)


class EnvConductorCoordinator(ConductorCoordinator):
    """ConductorCoordinator that can use a local transformers checkpoint
    (Llama-3.2-3B Conductor) or LiteLLM for the planning call."""
    def __init__(self, worker, conductor=None, slot_labels=None):
        self.worker = worker
        self.conductor = conductor
        self.slot_labels = slot_labels or getattr(worker, "slot_models", None) or CONDUCTOR_DEFAULT_SLOT_LABELS

    def run(self, query: str, verbose: bool = False):
        if self.conductor is not None:
            completion = self.conductor.conduct(conductor_prompt(query, self.slot_labels))
        else:
            conductor_model = os.environ.get("FUGU_CONDUCTOR_MODEL")
            if conductor_model is None and hasattr(self.worker, "slot_models") and self.worker.slot_models:
                conductor_model = self.worker.slot_models[0]
            if conductor_model is None:
                conductor_model = "openai/gpt-4o-mini"
            completion = self.worker.conduct(conductor_model, conductor_prompt(query, self.slot_labels))
        mids, subs, acc = parse_workflow(completion)
        if not subs:
            raise ValueError(f"Conductor did not emit a parseable workflow. Raw: {completion[:200]}")
        res = ConductorExecutor(self.worker, slot_labels=self.slot_labels).execute(mids, subs, acc, verbose=verbose)
        res.turns = res.steps
        return res


def _worker_from_args(args, mode: str):
    if args.local_models:
        specs = []
        n_gpu = 0
        try:
            import torch
            n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
        except Exception:
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
        return ConductorLiteLLMWorker(slot_models=slot_models, max_tokens=4096)
    return TrinityLiteLLMWorker(slot_models=slot_models, max_tokens=4096)


def load_coordinator(mode: str):
    global MAX_TURNS
    args = _parse_args()
    MAX_TURNS = args.max_turns
    worker = _worker_from_args(args, mode)
    if mode == "trinity":
        return Coordinator(get_router(), worker, max_turns=args.max_turns, sample=True)
    if mode == "conductor":
        local_ckpt = os.environ.get("FUGU_LOCAL_CONDUCTOR")
        conductor = EnvLocalConductor(local_ckpt) if local_ckpt else None
        return EnvConductorCoordinator(worker, conductor=conductor, slot_labels=getattr(worker, "slot_models", None))
    raise ValueError(f"unknown coordinator mode: {mode}")


def get_coordinator(mode: str):
    if mode not in _coordinators:
        _coordinators[mode] = load_coordinator(mode)
    return _coordinators[mode]


def main():
    args = _parse_args()
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[serve] Fugu listening on {args.host}:{args.port} — POST /v1/chat/completions", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
