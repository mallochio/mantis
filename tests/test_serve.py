"""Unit tests for openfugu-patch/serve.py."""
from __future__ import annotations

import json
import sys
import threading
import time
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pytest
import serve
import torch


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------
def test_is_reasoning_model():
    assert serve._is_reasoning_model("claude-sonnet-5")
    assert serve._is_reasoning_model("gpt-5.6-terra")
    assert not serve._is_reasoning_model("deepseek-v4-flash")
    assert not serve._is_reasoning_model("glm-5.2")


def test_build_litellm_kwargs_reasoning():
    kw = serve._build_litellm_kwargs("claude-opus-5", [], 1024, 0.2)
    assert kw["model"] == "claude-opus-5"
    assert kw["max_tokens"] == 1024
    assert kw["custom_llm_provider"] == "openai"
    assert "temperature" not in kw


def test_build_litellm_kwargs_non_reasoning():
    kw = serve._build_litellm_kwargs("deepseek-v4-flash", [], 1024, 0.2)
    assert kw["temperature"] == 0.2


def test_resolve_conductor_model_env(monkeypatch):
    monkeypatch.setenv("FUGU_CONDUCTOR_MODEL", "openai/gpt-5.6-sol")
    assert serve._resolve_conductor_model(SimpleNamespace()) == "openai/gpt-5.6-sol"


def test_resolve_conductor_model_from_slot_models(monkeypatch):
    monkeypatch.delenv("FUGU_CONDUCTOR_MODEL", raising=False)
    worker = SimpleNamespace(slot_models=["slot-0", "slot-1"])
    assert serve._resolve_conductor_model(worker) == "slot-0"


def test_resolve_conductor_model_default(monkeypatch):
    monkeypatch.delenv("FUGU_CONDUCTOR_MODEL", raising=False)
    assert serve._resolve_conductor_model(SimpleNamespace()) == "openai/gpt-4o-mini"


def test_chat_response():
    resp = serve._chat_response("hello", "fugu", 3)
    assert resp["model"] == "fugu"
    assert resp["choices"][0]["message"]["content"] == "hello"
    assert resp["usage"]["fugu_turns"] == 3
    assert resp["id"].startswith("chatcmpl-")


# ---------------------------------------------------------------------------
# LiteLLM worker wrappers
# ---------------------------------------------------------------------------
def _fake_litellm_module(content: str = "ok") -> Any:
    def _completion(**kw):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )

    mod = SimpleNamespace(completion=_completion)
    return mod


def test_openrouter_trinity_worker_call(monkeypatch):
    monkeypatch.setitem(sys.modules, "litellm", _fake_litellm_module("trinity"))
    worker = serve.OpenRouterTrinityWorker(slot_models=["claude-sonnet-5"], api_key="k", api_base="http://x")
    result = worker("Worker", [{"role": "user", "content": "hi"}], 0)
    assert result == "trinity"


def test_openrouter_conductor_worker_call(monkeypatch):
    monkeypatch.setitem(sys.modules, "litellm", _fake_litellm_module("conductor"))
    worker = serve.OpenRouterConductorWorker(slot_models=["gpt-5.6-luna"], api_key="k")
    result = worker._call("gpt-5.6-luna", [{"role": "user", "content": "hi"}])
    assert result == "conductor"


# ---------------------------------------------------------------------------
# Local pool worker with mocked transformers
# ---------------------------------------------------------------------------
class _FakeBatch:
    def __init__(self, input_ids: torch.Tensor):
        self._data = {"input_ids": input_ids}

    def to(self, device):
        return self

    def keys(self):
        return self._data.keys()

    def __getitem__(self, key: str):
        return self._data[key]


def _make_tokenizer_mock(input_ids: torch.Tensor) -> MagicMock:
    tk = MagicMock()
    tk.pad_token = None
    tk.eos_token = 0
    tk.pad_token_id = 0
    tk.return_value = _FakeBatch(input_ids)
    tk.decode.return_value = "decoded"
    return tk


def _make_model_mock(out: torch.Tensor) -> MagicMock:
    model = MagicMock()
    model.to.return_value = model
    model.eval.return_value = model
    model.generate.return_value = out
    return model


def test_local_pool_worker(monkeypatch):
    fake_tok = _make_tokenizer_mock(torch.tensor([[1, 2, 3]]))
    fake_model = _make_model_mock(torch.tensor([[1, 2, 3, 4, 5]]))

    fake_transformers = MagicMock()
    fake_transformers.AutoTokenizer.from_pretrained.return_value = fake_tok
    fake_transformers.AutoModelForCausalLM.from_pretrained.return_value = fake_model
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    worker = serve.LocalPoolWorker([("m", "/path", "cpu")])
    result = worker("Worker", [{"role": "user", "content": "hi"}], 0)
    assert result == "decoded"
    fake_transformers.AutoModelForCausalLM.from_pretrained.assert_called_once()


# ---------------------------------------------------------------------------
# Env local conductor with mocked transformers
# ---------------------------------------------------------------------------
def test_env_local_conductor(monkeypatch):
    fake_tok = _make_tokenizer_mock(torch.tensor([[1, 2, 3]]))
    fake_model = _make_model_mock(torch.tensor([[1, 2, 3, 4, 5]]))

    fake_transformers = MagicMock()
    fake_transformers.AutoTokenizer.from_pretrained.return_value = fake_tok
    fake_transformers.AutoModelForCausalLM.from_pretrained.return_value = fake_model
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    conductor = serve.EnvLocalConductor("/ckpt", device="cpu", max_new=10)
    result = conductor.conduct([{"role": "user", "content": "hi"}])
    assert result == "decoded"


# ---------------------------------------------------------------------------
# Conductor coordinator end-to-end
# ---------------------------------------------------------------------------
def test_env_conductor_coordinator_with_local(monkeypatch):
    workflow = (
        "model_id: [0, 1]\n"
        "subtasks: ['plan', 'solve']\n"
        "access_list: ['all', [0]]"
    )
    local_conductor = MagicMock()
    local_conductor.conduct.return_value = workflow
    worker = MagicMock()
    worker.side_effect = ["planned", "solved"]

    coord = serve.EnvConductorCoordinator(worker, conductor=local_conductor, slot_labels=["a", "b"])
    res = coord.run("query")
    assert res.final == "solved"


def test_env_conductor_coordinator_litellm(monkeypatch):
    workflow = (
        "model_id: [0]\n"
        "subtasks: ['answer']\n"
        "access_list: ['all']"
    )

    class FakeWorker:
        def __init__(self, *args, **kwargs):
            self.slot_models = ["gpt-5.6-luna"]

        def conduct(self, model, prompt):
            return workflow

        def __call__(self, sub, messages, agent_id):
            return "done"

    monkeypatch.setenv("FUGU_CONDUCTOR_MODEL", "gpt-5.6-luna")
    coord = serve.EnvConductorCoordinator(FakeWorker())
    res = coord.run("query")
    assert res.final == "done"


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------
def _start_server(handler_cls, port: int = 0) -> tuple[Any, int]:
    srv = serve.ThreadingHTTPServer(("127.0.0.1", port), handler_cls)
    if port == 0:
        port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    return srv, port


def _fake_get_coordinator(mode: str):
    class FakeCoord:
        def run(self, query, verbose=False):
            return SimpleNamespace(final="42", turns=[1, 2, 3])

    return FakeCoord()


def test_handler_health():
    old = serve.get_coordinator
    try:
        serve.get_coordinator = _fake_get_coordinator
        srv, port = _start_server(serve.Handler)
        time.sleep(0.1)
        resp = urlopen(f"http://127.0.0.1:{port}/health")
        body = json.loads(resp.read().decode())
        assert body["status"] == "ok"
    finally:
        serve.get_coordinator = old
        srv.shutdown()


def test_handler_models():
    old = serve.get_coordinator
    try:
        serve.get_coordinator = _fake_get_coordinator
        srv, port = _start_server(serve.Handler)
        time.sleep(0.1)
        resp = urlopen(f"http://127.0.0.1:{port}/v1/models")
        body = json.loads(resp.read().decode())
        assert body["data"][0]["id"] == "fugu"
    finally:
        serve.get_coordinator = old
        srv.shutdown()


def test_handler_post_trinity():
    old = serve.get_coordinator
    try:
        serve.get_coordinator = _fake_get_coordinator
        srv, port = _start_server(serve.Handler)
        time.sleep(0.1)
        payload = json.dumps({"model": "trinity", "messages": [{"role": "user", "content": "2+2"}]}).encode()
        req = Request(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        resp = urlopen(req)
        body = json.loads(resp.read().decode())
        assert body["choices"][0]["message"]["content"] == "42"
        assert body["usage"]["fugu_turns"] == 3
    finally:
        serve.get_coordinator = old
        srv.shutdown()


def test_handler_post_conductor():
    old = serve.get_coordinator
    try:
        serve.get_coordinator = _fake_get_coordinator
        srv, port = _start_server(serve.Handler)
        time.sleep(0.1)
        payload = json.dumps({"model": "conductor", "messages": [{"role": "user", "content": "hi"}]}).encode()
        req = Request(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        resp = urlopen(req)
        body = json.loads(resp.read().decode())
        assert body["choices"][0]["message"]["content"] == "42"
    finally:
        serve.get_coordinator = old
        srv.shutdown()


def test_handler_post_bad_path():
    srv, port = _start_server(serve.Handler)
    try:
        time.sleep(0.1)
        with pytest.raises(HTTPError):
            urlopen(f"http://127.0.0.1:{port}/unknown")
    finally:
        srv.shutdown()


def test_handler_post_not_chat_completions():
    srv, port = _start_server(serve.Handler)
    try:
        time.sleep(0.1)
        req = Request(
            f"http://127.0.0.1:{port}/v1/chat/completions/extra",
            data=b"{}",
            headers={"Content-Type": "application/json"},
        )
        with pytest.raises(HTTPError):
            urlopen(req)
    finally:
        srv.shutdown()


def test_handler_get_unknown():
    srv, port = _start_server(serve.Handler)
    try:
        time.sleep(0.1)
        with pytest.raises(HTTPError):
            urlopen(f"http://127.0.0.1:{port}/not-a-path")
    finally:
        srv.shutdown()


def test_handler_post_empty_messages():
    old = serve.get_coordinator
    try:
        serve.get_coordinator = _fake_get_coordinator
        srv, port = _start_server(serve.Handler)
        time.sleep(0.1)
        payload = json.dumps({"model": "trinity", "messages": []}).encode()
        req = Request(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with pytest.raises(HTTPError):
            urlopen(req)
    finally:
        serve.get_coordinator = old
        srv.shutdown()


def test_handler_post_invalid_json():
    old = serve.get_coordinator
    try:
        serve.get_coordinator = _fake_get_coordinator
        srv, port = _start_server(serve.Handler)
        time.sleep(0.1)
        req = Request(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            data=b"not json",
            headers={"Content-Type": "application/json"},
        )
        with pytest.raises(HTTPError):
            urlopen(req)
    finally:
        serve.get_coordinator = old
        srv.shutdown()


def test_handler_post_coordinator_error():
    def _raising_coordinator(mode: str):
        class BadCoord:
            def run(self, query, verbose=False):
                raise ValueError("boom")
        return BadCoord()

    old = serve.get_coordinator
    try:
        serve.get_coordinator = _raising_coordinator
        srv, port = _start_server(serve.Handler)
        time.sleep(0.1)
        payload = json.dumps({"model": "trinity", "messages": [{"role": "user", "content": "hi"}]}).encode()
        req = Request(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with pytest.raises(HTTPError):
            urlopen(req)
    finally:
        serve.get_coordinator = old
        srv.shutdown()


# ---------------------------------------------------------------------------
# Worker wrappers
# ---------------------------------------------------------------------------
def test_openrouter_conductor_worker_with_api_base(monkeypatch):
    captured: dict[str, Any] = {}

    def _completion(**kw):
        captured.update(kw)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))]
        )

    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(completion=_completion))
    worker = serve.OpenRouterConductorWorker(slot_models=["gpt-5.6-luna"], api_base="http://x/")
    worker._call("gpt-5.6-luna", [{"role": "user", "content": "hi"}])
    assert captured["api_base"] == "http://x/"


def test_run_conductor_workflow_empty():
    worker = MagicMock()
    with pytest.raises(ValueError):
        serve._run_conductor_workflow(worker, "q", ["a"], "no workflow here")


def test_conductor_coordinator_run():
    class FakeWorker:
        slot_models = ["a", "b"]

        def conduct(self, model, prompt):
            return (
                "model_id: [0]\n"
                "subtasks: ['answer']\n"
                "access_list: ['all']"
            )

        def __call__(self, sub, messages, agent_id):
            return "final answer"

    coord = serve.ConductorCoordinator(FakeWorker())
    res = coord.run("what is 2+2?")
    assert res.final == "final answer"


# ---------------------------------------------------------------------------
# Local / env conductor corner cases
# ---------------------------------------------------------------------------
def _fake_transformers_module(input_ids: torch.Tensor, *, raise_chat: bool = False) -> Any:
    tk = MagicMock()
    tk.pad_token = None
    tk.eos_token = 0
    tk.pad_token_id = 0
    tk.return_value = _FakeBatch(input_ids)
    tk.decode.return_value = "decoded"
    if raise_chat:
        tk.apply_chat_template.side_effect = ValueError("no chat template")

    model = MagicMock()
    model.to.return_value = model
    model.eval.return_value = model
    model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])

    mod = MagicMock()
    mod.AutoTokenizer.from_pretrained.return_value = tk
    mod.AutoModelForCausalLM.from_pretrained.return_value = model
    return mod


def test_local_pool_worker_fallback_text(monkeypatch):
    fake = _fake_transformers_module(torch.tensor([[1, 2, 3]]), raise_chat=True)
    monkeypatch.setitem(sys.modules, "transformers", fake)
    worker = serve.LocalPoolWorker([("m", "/path", "cpu")])
    result = worker("Worker", [{"role": "user", "content": "hi"}], 0)
    assert result == "decoded"


def test_env_local_conductor_auto_cpu(monkeypatch):
    fake = _fake_transformers_module(torch.tensor([[1, 2, 3]]))
    monkeypatch.setitem(sys.modules, "transformers", fake)
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    monkeypatch.setattr("torch.backends.mps.is_available", lambda: False)
    monkeypatch.delenv("FUGU_CONDUCTOR_DEVICE", raising=False)
    conductor = serve.EnvLocalConductor("/ckpt", device=None)
    assert conductor.device == "cpu"
    assert conductor.dtype == torch.float32


def test_env_local_conductor_dtype_env(monkeypatch):
    fake = _fake_transformers_module(torch.tensor([[1, 2, 3]]))
    monkeypatch.setitem(sys.modules, "transformers", fake)
    monkeypatch.setenv("FUGU_CONDUCTOR_DTYPE", "float32")
    conductor = serve.EnvLocalConductor("/ckpt", device="cpu")
    assert conductor.dtype == torch.float32


def test_env_local_conductor_fallback_text(monkeypatch):
    fake = _fake_transformers_module(torch.tensor([[1, 2, 3]]), raise_chat=True)
    monkeypatch.setitem(sys.modules, "transformers", fake)
    conductor = serve.EnvLocalConductor("/ckpt", device="cpu", max_new=10)
    result = conductor.conduct([{"role": "user", "content": "hello"}])
    assert result == "decoded"


# ---------------------------------------------------------------------------
# Router / coordinator wiring
# ---------------------------------------------------------------------------
def test_parse_args_defaults(monkeypatch):
    old = serve._args
    try:
        serve._args = None
        monkeypatch.setattr(sys, "argv", ["serve", "--slot-models", "a,b", "--max-turns", "3"])
        args = serve._parse_args()
        assert args.slot_models == "a,b"
        assert args.max_turns == 3
        assert args.model == "Qwen/Qwen3-0.6B"
    finally:
        serve._args = old


def test_load_head_npy(tmp_path):
    head = np.zeros(serve.HEAD_ROWS * serve.HIDDEN)
    path = tmp_path / "head.npy"
    np.save(path, head)
    loaded = serve._load_head(str(path))
    assert loaded.shape == (serve.HEAD_ROWS * serve.HIDDEN,)


def test_load_head_safetensors(tmp_path):
    try:
        from safetensors.torch import save_file
    except ImportError:
        pytest.skip("safetensors not installed")
    path = tmp_path / "head.safetensors"
    save_file({"trinity_router_head": torch.zeros(serve.HEAD_ROWS * serve.HIDDEN)}, str(path))
    loaded = serve._load_head(str(path))
    assert loaded.shape == (serve.HEAD_ROWS * serve.HIDDEN,)


def test_load_head_bad_shape(tmp_path):
    path = tmp_path / "bad.npy"
    np.save(path, np.zeros(serve.HEAD_ROWS * serve.HIDDEN - 1))
    with pytest.raises(ValueError):
        serve._load_head(str(path))


def test_worker_from_args(monkeypatch):
    class FakeLocal:
        def __init__(self, specs):
            self.specs = specs

    class FakeTrinity:
        def __init__(self, slot_models=None, max_tokens=1024):
            self.slot_models = slot_models
            self.max_tokens = max_tokens

    class FakeConductor:
        def __init__(self, slot_models=None, max_tokens=1024):
            self.slot_models = slot_models
            self.max_tokens = max_tokens

    monkeypatch.setattr(serve, "LocalPoolWorker", FakeLocal)
    monkeypatch.setattr(serve, "OpenRouterTrinityWorker", FakeTrinity)
    monkeypatch.setattr(serve, "OpenRouterConductorWorker", FakeConductor)

    args = SimpleNamespace(local_models="/a@cpu,/b", slot_models="x,y")
    local = serve._worker_from_args(args, "trinity")
    assert isinstance(local, FakeLocal)

    args = SimpleNamespace(local_models=None, slot_models="x,y")
    t = serve._worker_from_args(args, "trinity")
    assert isinstance(t, FakeTrinity)
    assert t.max_tokens == 4096

    c = serve._worker_from_args(args, "conductor")
    assert isinstance(c, FakeConductor)


def test_get_router_and_head(monkeypatch, tmp_path):
    old_router = serve.ROUTER
    old_args = serve._args
    try:
        serve.ROUTER = None
        serve._args = SimpleNamespace(
            model="m", vector=str(tmp_path / "vec.npy"),
            head=str(tmp_path / "head.npy"), max_turns=5,
        )
        np.save(serve._args.vector, np.zeros(serve.HEAD_ROWS * serve.HIDDEN))
        np.save(serve._args.head, np.zeros(serve.HEAD_ROWS * serve.HIDDEN))

        class FakeRouter:
            device = "cpu"
            torch = torch
            head = torch.zeros(serve.HEAD_ROWS, serve.HIDDEN)

            def __init__(self, *args, **kwargs):
                pass

        monkeypatch.setattr(serve, "FuguRouter", FakeRouter)
        router = serve.get_router()
        assert router is not None
        assert router.head.shape == (serve.HEAD_ROWS, serve.HIDDEN)
    finally:
        serve.ROUTER = old_router
        serve._args = old_args


def test_load_coordinator_trinity(monkeypatch):
    monkeypatch.setattr(
        serve, "_args",
        SimpleNamespace(local_models=None, slot_models="a,b", max_turns=5)
    )
    class FakeRouter:
        pass

    class FakeCoord:
        def __init__(self, router, worker, max_turns, sample):
            self.router = router
            self.worker = worker
            self.max_turns = max_turns
            self.sample = sample

    class FakeWorker:
        slot_models = ["a", "b"]

    monkeypatch.setattr(serve, "get_router", lambda: FakeRouter())
    monkeypatch.setattr(serve, "Coordinator", FakeCoord)
    monkeypatch.setattr(
        serve, "_worker_from_args",
        lambda args, mode: FakeWorker(),
    )
    monkeypatch.delenv("FUGU_LOCAL_CONDUCTOR", raising=False)
    coord = serve.load_coordinator("trinity")
    assert coord.max_turns == serve.MAX_TURNS


def test_load_coordinator_conductor(monkeypatch):
    monkeypatch.setattr(
        serve, "_args",
        SimpleNamespace(local_models=None, slot_models="a,b", max_turns=5)
    )
    class FakeWorker:
        slot_models = ["a", "b"]

    class FakeLocalConductor:
        def __init__(self, ckpt):
            self.ckpt = ckpt

    class FakeCoord:
        def __init__(self, worker, conductor, slot_labels):
            self.worker = worker
            self.conductor = conductor
            self.slot_labels = slot_labels

    monkeypatch.setattr(
        serve, "_worker_from_args",
        lambda args, mode: FakeWorker(),
    )
    monkeypatch.setattr(serve, "EnvLocalConductor", FakeLocalConductor)
    monkeypatch.setattr(serve, "EnvConductorCoordinator", FakeCoord)
    monkeypatch.setenv("FUGU_LOCAL_CONDUCTOR", "di-zhang-fdu/openfugu-conductor-3b")
    coord = serve.load_coordinator("conductor")
    assert isinstance(coord.conductor, FakeLocalConductor)


def test_load_coordinator_unknown(monkeypatch):
    monkeypatch.setattr(
        serve, "_args",
        SimpleNamespace(local_models=None, slot_models="a,b", max_turns=5)
    )
    with pytest.raises(ValueError):
        serve.load_coordinator("other")


def test_get_coordinator_caches(monkeypatch):
    old = serve._coordinators.copy()
    try:
        serve._coordinators = {}
        fake = SimpleNamespace()
        monkeypatch.setattr(serve, "load_coordinator", MagicMock(return_value=fake))
        assert serve.get_coordinator("trinity") is fake
        assert serve.get_coordinator("trinity") is fake
        assert serve.load_coordinator.call_count == 1
    finally:
        serve._coordinators = old


def test_main_serve(monkeypatch):
    class FakeServer:
        calls = []

        def __init__(self, addr, handler):
            self.calls.append((addr, handler))

        def serve_forever(self):
            pass

    old_args = serve._args
    try:
        serve._args = None
        monkeypatch.setattr(sys, "argv", ["serve", "--port", "0"])
        monkeypatch.setattr(serve, "ThreadingHTTPServer", FakeServer)
        serve.main()
        assert len(FakeServer.calls) == 1
        assert FakeServer.calls[0][0] == ("0.0.0.0", 0)
    finally:
        serve._args = old_args


def test_worker_from_args_no_torch(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)

    class FakeLocal:
        def __init__(self, specs):
            self.specs = specs

    monkeypatch.setattr(serve, "LocalPoolWorker", FakeLocal)
    args = SimpleNamespace(local_models="/a@cpu", slot_models="x,y")
    worker = serve._worker_from_args(args, "trinity")
    assert isinstance(worker, FakeLocal)


def test_env_local_conductor_auto_mps(monkeypatch):
    fake = _fake_transformers_module(torch.tensor([[1, 2, 3]]))
    monkeypatch.setitem(sys.modules, "transformers", fake)
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    monkeypatch.setattr("torch.backends.mps.is_available", lambda: True)
    monkeypatch.delenv("FUGU_CONDUCTOR_DEVICE", raising=False)
    conductor = serve.EnvLocalConductor("/ckpt", device=None)
    assert conductor.device == "mps"
