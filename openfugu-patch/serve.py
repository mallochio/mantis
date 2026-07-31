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
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# reuse the faithful implementation
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mini import (FuguRouter, Coordinator, LiteLLMWorker as TrinityLiteLLMWorker,
                  MockWorker, DEFAULT_SLOT_LABELS, HEAD_ROWS, HIDDEN)
from ultra import (LiteLLMWorker as ConductorLiteLLMWorker, ConductorExecutor,
                   conductor_prompt, parse_workflow, N_AGENTS as CONDUCTOR_N_AGENTS)

ROUTER: FuguRouter | None = None
MODEL_NAME = "fugu"
MAX_TURNS = 5

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
    ap.add_argument("--head", default=None,
                    help="optional trained head-only vector (10240); overrides the "
                         "head from --vector after SVF is applied")
    ap.add_argument("--slot-models", metavar="CSV", help="litellm worker ids; omit for mock")
    ap.add_argument("--local-models", metavar="CSV",
                    help="local HF worker model paths (real per-step pool, no API). "
                         "Optional 'path@device' per entry; default round-robin GPUs.")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8088)
    ap.add_argument("--max-turns", type=int, default=5)
    _args = ap.parse_args()
    return _args


def get_router():
    global ROUTER
    if ROUTER is None:
        args = _parse_args()
        print(f"[serve] loading TRINITY router ({args.model}) ...", flush=True)
        ROUTER = FuguRouter(args.model, args.vector, seed=0)
        if args.head:                                  # layer a trained head over base SVF
            h = np.load(args.head).astype(np.float64)
            if h.shape != (HEAD_ROWS * HIDDEN,):
                raise ValueError(f"--head must be {HEAD_ROWS * HIDDEN} floats, got {h.shape}")
            ROUTER.head = ROUTER.torch.from_numpy(h.copy()).float().reshape(
                HEAD_ROWS, HIDDEN).to(ROUTER.device)
            print(f"[serve] applied trained head from {args.head}", flush=True)
    return ROUTER


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

    slot_models = args.slot_models.split(",") if args.slot_models else None
    if mode == "conductor":
        return ConductorLiteLLMWorker(slot_models=slot_models)
    return TrinityLiteLLMWorker(slot_models=slot_models)


def load_coordinator(mode: str):
    global MAX_TURNS
    args = _parse_args()
    MAX_TURNS = args.max_turns
    worker = _worker_from_args(args, mode)
    if mode == "trinity":
        return Coordinator(get_router(), worker, max_turns=args.max_turns, sample=True)
    if mode == "conductor":
        return ConductorCoordinator(worker)
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
