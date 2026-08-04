"""Unit tests for openfugu-patch/serve.py."""

from __future__ import annotations

import http.client
import json
import os
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

os.environ["MANTIS_API_KEY"] = "test-key"
os.environ["FUGU_API_KEY"] = "test-key"


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
    monkeypatch.setenv("MANTIS_CONDUCTOR_MODEL", "openai/gpt-5.6-sol")
    assert serve._resolve_conductor_model(SimpleNamespace()) == "openai/gpt-5.6-sol"


def test_resolve_conductor_model_from_slot_models(monkeypatch):
    monkeypatch.delenv("MANTIS_CONDUCTOR_MODEL", raising=False)
    monkeypatch.delenv("FUGU_CONDUCTOR_MODEL", raising=False)
    worker = SimpleNamespace(slot_models=["slot-0", "slot-1"])
    assert serve._resolve_conductor_model(worker) == "slot-0"


def test_resolve_conductor_model_default(monkeypatch):
    monkeypatch.delenv("MANTIS_CONDUCTOR_MODEL", raising=False)
    monkeypatch.delenv("FUGU_CONDUCTOR_MODEL", raising=False)
    assert serve._resolve_conductor_model(SimpleNamespace()) == "openai/gpt-4o-mini"


def test_chat_response():
    result = SimpleNamespace(final="hello", turns=[1, 2, 3])
    resp = serve._chat_response(result, "fugu")
    assert resp["model"] == "fugu"
    assert resp["choices"][0]["message"]["content"] == "hello"
    assert resp["usage"]["fugu_turns"] == 3
    assert resp["usage"]["fugu_trace"] == "steps:3:conductor"
    assert resp["id"].startswith("chatcmpl-")


def test_build_fugu_trace_trinity():
    turns = [
        SimpleNamespace(role_name="Worker", agent_id=4),
        SimpleNamespace(role_name="Thinker", agent_id=1),
        SimpleNamespace(role_name="Verifier", agent_id=1),
    ]
    result = SimpleNamespace(final="ok", turns=turns, terminated_by="verifier_accept")
    assert serve._build_fugu_trace(result) == "Worker(4)→Thinker(1)→Verifier(1):verifier_accept"


def test_build_fugu_trace_empty():
    result = SimpleNamespace(final="ok", turns=[])
    assert serve._build_fugu_trace(result) == "steps:0:conductor"


# ---------------------------------------------------------------------------
# LiteLLM worker wrappers
# ---------------------------------------------------------------------------
def _fake_litellm_module(content: str = "ok") -> Any:
    def _completion(**kw):
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    mod = SimpleNamespace(completion=_completion)
    return mod


def test_openrouter_trinity_worker_call(monkeypatch):
    monkeypatch.setitem(sys.modules, "litellm", _fake_litellm_module("trinity"))
    worker = serve.OpenRouterTrinityWorker(
        slot_models=["claude-sonnet-5"], api_key="k", api_base="http://x"
    )
    result = worker("Worker", [{"role": "user", "content": "hi"}], 0)
    assert result == "trinity"


def test_openrouter_conductor_worker_call(monkeypatch):
    monkeypatch.setitem(sys.modules, "litellm", _fake_litellm_module("conductor"))
    worker = serve.OpenRouterConductorWorker(slot_models=["gpt-5.6-luna"], api_key="k")
    result = worker._call("gpt-5.6-luna", [{"role": "user", "content": "hi"}])
    assert result == "conductor"


def test_split_messages():
    assert serve._split_messages([{"role": "user", "content": "hi"}]) == ("hi", [])
    msgs = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
    ]
    query, history = serve._split_messages(msgs)
    assert query == "q2"
    assert history == [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"}]


def test_history_worker_with_history():
    class FakeWorker:
        def __init__(self):
            self.calls = []

        def __call__(self, role, messages, agent_id):
            self.calls.append(("call", role, messages, agent_id))
            return "ok"

    serve._history_context.history = [
        {"role": "system", "content": "repository context"},
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
    ]
    serve._history_context.calls = []
    try:
        worker = serve.HistoryWorker(FakeWorker())
        result = worker(
            "Worker", [{"role": "system", "content": "sys"}, {"role": "user", "content": "q2"}], 0
        )
        assert len(serve._history_context.calls) == 1
        call = serve._history_context.calls[0]
        assert call["role"] == "Worker"
        assert call["agent_id"] == 0
        assert call["prompt"] == "q2"
        assert call["model_name"] == "slot-0"
        assert result == "ok"
        assert worker._worker.calls[0][1] == "Worker"
        msgs = worker._worker.calls[0][2]
        assert msgs[0] == {"role": "system", "content": "sys\n\nrepository context"}
        assert msgs[1] == {"role": "user", "content": "q1"}
        assert msgs[2] == {"role": "assistant", "content": "a1"}
        assert msgs[3] == {"role": "user", "content": "q2"}
    finally:
        serve._history_context.history = []
        serve._history_context.calls = []


def test_history_worker_multi_turn_no_duplicate_assistant_text():
    """Assert each prior assistant response appears exactly once per worker call message list."""

    class FakeWorker:
        def __init__(self):
            self.calls = []

        def __call__(self, role, messages, agent_id):
            self.calls.append((role, messages, agent_id))
            return "ok"

    serve._history_context.history = [
        {"role": "user", "content": "Write hello world"},
        {"role": "assistant", "content": "UNIQUE_ASSISTANT_RESPONSE_1"},
        {"role": "user", "content": "Add a docstring"},
        {"role": "assistant", "content": "UNIQUE_ASSISTANT_RESPONSE_2"},
    ]
    serve._history_context.calls = []
    try:
        worker = serve.HistoryWorker(FakeWorker())
        result = worker("Worker", [{"role": "user", "content": "Make it a function"}], 0)
        assert result == "ok"
        assert len(worker._worker.calls) == 1
        _, msgs, _ = worker._worker.calls[0]
        assert msgs == [
            {"role": "user", "content": "Write hello world"},
            {"role": "assistant", "content": "UNIQUE_ASSISTANT_RESPONSE_1"},
            {"role": "user", "content": "Add a docstring"},
            {"role": "assistant", "content": "UNIQUE_ASSISTANT_RESPONSE_2"},
            {"role": "user", "content": "Make it a function"},
        ]
        # Count occurrences of prior assistant texts across all message content strings
        all_content = [m["content"] for m in msgs if isinstance(m, dict)]
        count_1 = sum(c.count("UNIQUE_ASSISTANT_RESPONSE_1") for c in all_content)
        count_2 = sum(c.count("UNIQUE_ASSISTANT_RESPONSE_2") for c in all_content)
        assert count_1 == 1
        assert count_2 == 1
        assert msgs[-1]["content"] == "Make it a function"
    finally:
        serve._history_context.history = []
        serve._history_context.calls = []


def test_verifier_accept_terminates_without_extra_turns():
    class FakeRouter:
        def __init__(self):
            self.calls = 0

        def route(self, *_args, **_kwargs):
            self.calls += 1
            if self.calls > 2:
                raise AssertionError("router called after verifier acceptance")
            role = "Worker" if self.calls == 1 else "Verifier"
            return {"agent_id": 0, "role_id": 0 if role == "Worker" else 2, "role_name": role}

    class FakeWorker:
        names = ["test-model"]

        def __init__(self):
            self.calls = 0

        def __call__(self, role, _messages, _agent_id):
            self.calls += 1
            return "draft" if role == "Worker" else "ACCEPT — complete"

    router = FakeRouter()
    worker = FakeWorker()
    coord = serve.Coordinator(
        serve.RejectAwareRouter(router),
        serve.HistoryWorker(worker),
        max_turns=5,
        sample=False,
    )

    result = coord.run("answer this")

    assert [turn.role_name for turn in result.turns] == ["Worker", "Verifier"]
    assert result.final == "draft"
    assert result.terminated_by == "verifier_accept"
    assert router.calls == 2
    assert worker.calls == 2


def test_verifier_rejection_forces_worker_revision():
    class FakeRouter:
        def __init__(self):
            self.calls = 0

        def route(self, *_args, **_kwargs):
            self.calls += 1
            role = "Worker" if self.calls == 1 else "Verifier"
            return {"agent_id": 0, "role_id": 0 if role == "Worker" else 2, "role_name": role}

    class FakeWorker:
        names = ["test-model"]

        def __call__(self, role, messages, _agent_id):
            prompt = messages[-1]["content"]
            if role == "Worker":
                if "returned no response" in prompt:
                    return "revised"
                return "" if "verifier feedback" in prompt else "draft"
            return "ACCEPT — fixed" if "<response>\nrevised" in prompt else "REJECT — fix the draft"

    serve._history_context.history = []
    serve._history_context.calls = []
    serve._history_context.force_worker = False
    serve._history_context.revision_feedback = None
    try:
        coord = serve.Coordinator(
            serve.RejectAwareRouter(FakeRouter()),
            serve.HistoryWorker(FakeWorker()),
            max_turns=5,
            sample=False,
        )
        result = coord.run("answer this")
        assert [turn.role_name for turn in result.turns] == [
            "Worker",
            "Verifier",
            "Worker",
            "Worker",
            "Verifier",
        ]
        assert result.final == "revised"
        assert result.terminated_by == "verifier_accept"
    finally:
        serve._history_context.history = []
        serve._history_context.calls = []
        serve._history_context.force_worker = False
        serve._history_context.revision_feedback = None


def test_history_worker_first_turn_no_context_prefix():
    """When there is no prior assistant message the worker query is unchanged."""

    class FakeWorker:
        def __init__(self):
            self.calls = []

        def __call__(self, role, messages, agent_id):
            self.calls.append((role, messages, agent_id))
            return "ok"

    serve._history_context.history = [{"role": "user", "content": "q1"}]
    serve._history_context.calls = []
    try:
        worker = serve.HistoryWorker(FakeWorker())
        result = worker(
            "Worker", [{"role": "system", "content": "sys"}, {"role": "user", "content": "q2"}], 0
        )
        assert result == "ok"
        msgs = worker._worker.calls[0][1]
        assert msgs[0] == {"role": "system", "content": "sys"}
        assert msgs[1] == {"role": "user", "content": "q1"}
        assert msgs[2] == {"role": "user", "content": "q2"}
    finally:
        serve._history_context.history = []
        serve._history_context.calls = []


def test_history_worker_conduct():
    class FakeWorker:
        def __init__(self):
            self.calls = []

        def conduct(self, model, messages):
            self.calls.append(("conduct", model, messages))
            return "plan"

    serve._history_context.history = [{"role": "assistant", "content": "a1"}]
    try:
        worker = serve.HistoryWorker(FakeWorker())
        result = worker.conduct("model", [{"role": "user", "content": "q"}])
        assert result == "plan"
        assert worker._worker.calls[0][2] == [
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q"},
        ]
    finally:
        serve._history_context.history = []
        serve._history_context.calls = []


def test_chat_response_includes_prompt_and_model_name():
    from types import SimpleNamespace

    turn = SimpleNamespace(
        t=1,
        agent_id=2,
        role_name="Thinker",
        reply="ok",
        prompt="solve it",
        model_name="openai/gpt-4o-mini",
    )
    res = SimpleNamespace(final="final", turns=[turn], terminated_by="verifier_accept")
    body = serve._chat_response(res, "mantis")
    step = body["mantis_steps"][0]
    assert step["prompt"] == "solve it"
    assert step["model_name"] == "openai/gpt-4o-mini"


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
    workflow = "model_id: [0, 1]\nsubtasks: ['plan', 'solve']\naccess_list: ['all', [0]]"
    local_conductor = MagicMock()
    local_conductor.conduct.return_value = workflow
    worker = MagicMock()
    worker.side_effect = ["planned", "solved"]

    coord = serve.EnvConductorCoordinator(worker, conductor=local_conductor, slot_labels=["a", "b"])
    res = coord.run("query")
    assert res.final == "solved"


def test_env_conductor_coordinator_litellm(monkeypatch):
    workflow = "model_id: [0]\nsubtasks: ['answer']\naccess_list: ['all']"

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


def _fake_get_coordinator(mode: str = "trinity"):
    class FakeCoord:
        def run(self, query, verbose=False):
            if mode == "conductor":
                return SimpleNamespace(final="42", turns=[1, 2, 3, 4, 5])
            return SimpleNamespace(
                final="42",
                turns=[
                    SimpleNamespace(role_name="Worker", agent_id=4),
                    SimpleNamespace(role_name="Thinker", agent_id=1),
                    SimpleNamespace(role_name="Verifier", agent_id=1),
                ],
                terminated_by="verifier_accept",
            )

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
        req = Request(
            f"http://127.0.0.1:{port}/v1/models",
            headers={"Authorization": "Bearer test-key"},
        )
        resp = urlopen(req)
        body = json.loads(resp.read().decode())
        assert body["data"][0]["id"] in ("mantis", "fugu")
    finally:
        serve.get_coordinator = old
        srv.shutdown()


def test_handler_warm_trinity_and_conductor():
    called_modes = []

    def _mock_get_coordinator(mode: str):
        called_modes.append(mode)
        return _fake_get_coordinator(mode)

    old = serve.get_coordinator
    try:
        serve.get_coordinator = _mock_get_coordinator
        srv, port = _start_server(serve.Handler)
        time.sleep(0.1)

        # GET /warm?mode=trinity
        req1 = Request(
            f"http://127.0.0.1:{port}/warm?mode=trinity",
            headers={"Authorization": "Bearer test-key"},
        )
        resp1 = urlopen(req1)
        body1 = json.loads(resp1.read().decode())
        assert resp1.status == 200
        assert body1["status"] == "ready"
        assert body1["mode"] == "trinity"
        assert body1["native_tool_runs"] is True

        # POST /warm with mode=conductor
        payload = json.dumps({"mode": "conductor"}).encode()
        req2 = Request(
            f"http://127.0.0.1:{port}/warm",
            data=payload,
            headers={"Content-Type": "application/json", "Authorization": "Bearer test-key"},
        )
        resp2 = urlopen(req2)
        body2 = json.loads(resp2.read().decode())
        assert resp2.status == 200
        assert body2["status"] == "ready"
        assert body2["mode"] == "conductor"
        assert body2["native_tool_runs"] is True

        assert called_modes == ["trinity", "conductor"]
    finally:
        serve.get_coordinator = old
        srv.shutdown()


def test_handler_warm_invalid_mode():
    old = serve.get_coordinator
    try:
        serve.get_coordinator = _fake_get_coordinator
        srv, port = _start_server(serve.Handler)
        time.sleep(0.1)

        req = Request(
            f"http://127.0.0.1:{port}/warm?mode=invalid",
            headers={"Authorization": "Bearer test-key"},
        )
        with pytest.raises(HTTPError) as exc:
            urlopen(req)
        assert exc.value.code == 400
        err_body = json.loads(exc.value.read().decode())
        assert "unknown mode" in err_body.get("error", "")
    finally:
        serve.get_coordinator = old
        srv.shutdown()


def test_handler_warm_missing_auth():
    srv, port = _start_server(serve.Handler)
    try:
        time.sleep(0.1)
        req = Request(f"http://127.0.0.1:{port}/warm?mode=trinity")
        with pytest.raises(HTTPError) as exc:
            urlopen(req)
        assert exc.value.code == 401
    finally:
        srv.shutdown()


def test_handler_missing_auth():
    srv, port = _start_server(serve.Handler)
    try:
        time.sleep(0.1)
        req = Request(
            f"http://127.0.0.1:{port}/v1/models",
        )
        with pytest.raises(HTTPError) as exc:
            urlopen(req)
        assert exc.value.code == 401
    finally:
        srv.shutdown()


def test_handler_post_too_large():
    old_max = serve.MAX_BODY_BYTES
    try:
        serve.MAX_BODY_BYTES = 16
        srv, port = _start_server(serve.Handler)
        time.sleep(0.1)
        payload = json.dumps(
            {"model": "trinity", "messages": [{"role": "user", "content": "hello world"}]}
        ).encode()
        req = Request(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json", "Authorization": "Bearer test-key"},
        )
        with pytest.raises(HTTPError) as exc:
            urlopen(req)
        assert exc.value.code == 413
    finally:
        serve.MAX_BODY_BYTES = old_max
        srv.shutdown()


def test_handler_post_trinity():
    old = serve.get_coordinator
    try:
        serve.get_coordinator = _fake_get_coordinator
        srv, port = _start_server(serve.Handler)
        time.sleep(0.1)
        payload = json.dumps(
            {"model": "trinity", "messages": [{"role": "user", "content": "2+2"}]}
        ).encode()
        req = Request(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json", "Authorization": "Bearer test-key"},
        )
        resp = urlopen(req)
        body = json.loads(resp.read().decode())
        assert body["choices"][0]["message"]["content"] == "42"
        assert body["usage"]["fugu_turns"] == 3
        assert body["usage"]["fugu_trace"] == "Worker(4)→Thinker(1)→Verifier(1):verifier_accept"
    finally:
        serve.get_coordinator = old
        srv.shutdown()


def test_handler_post_conductor():
    old = serve.get_coordinator
    try:
        serve.get_coordinator = _fake_get_coordinator
        srv, port = _start_server(serve.Handler)
        time.sleep(0.1)
        payload = json.dumps(
            {"model": "conductor", "messages": [{"role": "user", "content": "hi"}]}
        ).encode()
        req = Request(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json", "Authorization": "Bearer test-key"},
        )
        resp = urlopen(req)
        body = json.loads(resp.read().decode())
        assert body["choices"][0]["message"]["content"] == "42"
        assert body["usage"]["fugu_turns"] == 5
        assert body["usage"]["fugu_trace"] == "steps:5:conductor"
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
            headers={"Content-Type": "application/json", "Authorization": "Bearer test-key"},
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
            headers={"Content-Type": "application/json", "Authorization": "Bearer test-key"},
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
            headers={"Content-Type": "application/json", "Authorization": "Bearer test-key"},
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
        payload = json.dumps(
            {"model": "trinity", "messages": [{"role": "user", "content": "hi"}]}
        ).encode()
        req = Request(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json", "Authorization": "Bearer test-key"},
        )
        with pytest.raises(HTTPError):
            urlopen(req)
    finally:
        serve.get_coordinator = old
        srv.shutdown()


def test_handler_post_stream():
    class FakeWorker:
        slot_models = ["gpt-5.6-luna", "gpt-5.6-sol", "deepseek-v4"]
        names = slot_models

        def __call__(self, role, messages, agent_id):
            return "ok"

    class StreamCoord:
        def run(self, query, verbose=False):
            worker = serve.HistoryWorker(FakeWorker())
            r1 = worker("Worker", [{"role": "user", "content": query}], 4)
            r2 = worker("Thinker", [{"role": "user", "content": query}], 1)
            r3 = worker("Verifier", [{"role": "user", "content": query}], 1)
            return SimpleNamespace(
                final="final answer",
                turns=[
                    SimpleNamespace(t=1, agent_id=4, role_name="Worker", reply=r1),
                    SimpleNamespace(t=2, agent_id=1, role_name="Thinker", reply=r2),
                    SimpleNamespace(t=3, agent_id=1, role_name="Verifier", reply=r3),
                ],
                terminated_by="verifier_accept",
            )

    old = serve.get_coordinator
    try:
        serve.get_coordinator = lambda _mode: StreamCoord()
        srv, port = _start_server(serve.Handler)
        time.sleep(0.1)
        payload = json.dumps(
            {"model": "trinity", "messages": [{"role": "user", "content": "hi"}], "stream": True}
        ).encode()
        req = Request(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json", "Authorization": "Bearer test-key"},
        )
        resp = urlopen(req)
        lines = [line for line in resp.read().decode().split("\n") if line.strip()]
        events = [json.loads(line) for line in lines]
        starts = [e for e in events if e.get("type") == "step-start"]
        ends = [e for e in events if e.get("type") == "step-end"]
        results = [e for e in events if e.get("type") == "result"]
        assert len(starts) == 3
        assert len(ends) == 3
        assert len(results) == 1
        assert results[0]["text"] == "final answer"
        assert results[0]["mantis_steps"][0]["model_name"] == "gpt-5.6-sol"
        assert results[0]["mantis_steps"][0]["prompt"] == "hi"
    finally:
        serve.get_coordinator = old
        srv.shutdown()


def test_handler_post_chunked():
    """Pi/fetch can send chunked POST bodies; the handler must consume them."""

    class SimpleCoord:
        def run(self, query, verbose=False):
            return SimpleNamespace(
                final="chunked ok",
                turns=[
                    SimpleNamespace(
                        t=0,
                        agent_id=2,
                        role_name="Worker",
                        reply="ok",
                        prompt=query,
                        model_name="gpt-5.6-luna",
                    )
                ],
                terminated_by="verifier_accept",
            )

    old = serve.get_coordinator
    try:
        serve.get_coordinator = lambda _mode: SimpleCoord()
        srv, port = _start_server(serve.Handler)
        time.sleep(0.1)
        payload = json.dumps(
            {"model": "trinity", "messages": [{"role": "user", "content": "hi"}]}
        ).encode()
        chunks = [payload[i : i + 10] for i in range(0, len(payload), 10)]
        conn = http.client.HTTPConnection("127.0.0.1", port)
        conn.putrequest("POST", "/v1/chat/completions")
        conn.putheader("Content-Type", "application/json")
        conn.putheader("Authorization", "Bearer test-key")
        conn.putheader("Transfer-Encoding", "chunked")
        conn.endheaders()
        for chunk in chunks:
            conn.send(f"{len(chunk):X}\r\n".encode() + chunk + b"\r\n")
        conn.send(b"0\r\n\r\n")
        resp = conn.getresponse()
        body = resp.read().decode()
        assert resp.status == 200
        data = json.loads(body)
        assert data["choices"][0]["message"]["content"] == "chunked ok"
        assert data["mantis_steps"][0]["model_name"] == "gpt-5.6-luna"
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
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))])

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
            return "model_id: [0]\nsubtasks: ['answer']\naccess_list: ['all']"

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
            model="m",
            vector=str(tmp_path / "vec.npy"),
            head=str(tmp_path / "head.npy"),
            max_turns=5,
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
        serve, "_args", SimpleNamespace(local_models=None, slot_models="a,b", max_turns=5)
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
        serve,
        "_worker_from_args",
        lambda args, mode: FakeWorker(),
    )
    monkeypatch.delenv("FUGU_LOCAL_CONDUCTOR", raising=False)
    coord = serve.load_coordinator("trinity")
    assert coord.max_turns == serve.MAX_TURNS


def test_load_coordinator_conductor(monkeypatch):
    monkeypatch.setattr(
        serve, "_args", SimpleNamespace(local_models=None, slot_models="a,b", max_turns=5)
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
        serve,
        "_worker_from_args",
        lambda args, mode: FakeWorker(),
    )
    monkeypatch.setattr(serve, "EnvLocalConductor", FakeLocalConductor)
    monkeypatch.setattr(serve, "EnvConductorCoordinator", FakeCoord)
    monkeypatch.setenv("FUGU_LOCAL_CONDUCTOR", "di-zhang-fdu/openfugu-conductor-3b")
    coord = serve.load_coordinator("conductor")
    assert isinstance(coord.conductor, FakeLocalConductor)


def test_load_coordinator_unknown(monkeypatch):
    monkeypatch.setattr(
        serve, "_args", SimpleNamespace(local_models=None, slot_models="a,b", max_turns=5)
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
        assert FakeServer.calls[0][0] == ("0.0.0.0", 0)  # noqa: S104
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


# ---------------------------------------------------------------------------
# Conductor Recovery & Observability (Plan 002) Tests
# ---------------------------------------------------------------------------


def test_conductor_planning_planner_step_emitted():
    events = []

    class FakeWorker:
        slot_models = ["model-a", "model-b"]

        def conduct(self, model, prompt):
            return "model_id: [0]\nsubtasks: ['step 1']\naccess_list: ['all']"

        def __call__(self, sub, messages, agent_id):
            return "subtask output"

    serve._history_context.write_line = events.append
    serve._history_context.calls = []
    serve._history_context.history = []
    try:
        coord = serve.ConductorCoordinator(serve.HistoryWorker(FakeWorker()))
        res = coord.run("test query")
        assert res.final == "subtask output"

        # Check emitted events
        start_events = [e for e in events if e.get("type") == "step-start"]
        end_events = [e for e in events if e.get("type") == "step-end"]

        assert len(start_events) == 2
        assert len(end_events) == 2

        # Turn 0: Planner
        assert start_events[0]["role"] == "Planner"
        assert start_events[0]["turn"] == 0
        assert start_events[0]["prompt"] == "test query"

        assert end_events[0]["role"] == "Planner"
        assert end_events[0]["turn"] == 0
        assert "model_id: [0]" in end_events[0]["reply"]

        # Turn 1: Worker step
        assert start_events[1]["role"] == "Worker"
        assert start_events[1]["turn"] == 1

        # Check turn numbering on result
        assert len(res.turns) == 2
        assert res.turns[0].role == "Planner"
        assert res.turns[0].turn == 0
        assert res.turns[1].role == "Worker"
        assert res.turns[1].turn == 1
    finally:
        serve._history_context.write_line = None
        serve._history_context.calls = []
        serve._history_context.history = []
        serve._history_context.conductor_mode = False


def test_conductor_planning_empty_or_malformed_error():
    class FakeEmptyWorker:
        slot_models = ["model-a"]

        def conduct(self, model, prompt):
            return ""

        def __call__(self, sub, messages, agent_id):
            return "should not be called"

    coord = serve.ConductorCoordinator(serve.HistoryWorker(FakeEmptyWorker()))
    with pytest.raises(ValueError, match="empty completion"):
        coord.run("query")

    class FakeMalformedWorker:
        slot_models = ["model-a"]

        def conduct(self, model, prompt):
            return "This is not a valid workflow."

        def __call__(self, sub, messages, agent_id):
            return "should not be called"

    coord_malformed = serve.ConductorCoordinator(serve.HistoryWorker(FakeMalformedWorker()))
    with pytest.raises(ValueError, match="Conductor did not emit a parseable workflow"):
        coord_malformed.run("query")


def test_conductor_node_empty_retry_success():
    events = []

    class FakeFlakyWorker:
        slot_models = ["model-a"]

        def __init__(self):
            self.attempts = 0

        def conduct(self, model, prompt):
            return "model_id: [0]\nsubtasks: ['flaky step']\naccess_list: ['all']"

        def __call__(self, sub, messages, agent_id):
            self.attempts += 1
            if self.attempts == 1:
                return ""
            return "recovered output"

    serve._history_context.write_line = events.append
    serve._history_context.calls = []
    try:
        coord = serve.ConductorCoordinator(serve.HistoryWorker(FakeFlakyWorker()))
        res = coord.run("query")
        assert res.final == "recovered output"

        start_events = [e for e in events if e.get("type") == "step-start"]
        end_events = [e for e in events if e.get("type") == "step-end"]

        assert len(start_events) == 3
        assert len(end_events) == 3

        # Turn 0: Planner
        assert start_events[0]["role"] == "Planner"
        assert start_events[0]["turn"] == 0

        # Turn 1: Worker attempt 1 (failed)
        assert start_events[1]["role"] == "Worker"
        assert start_events[1]["turn"] == 1
        assert end_events[1]["reply"] == ""

        # Turn 2: Worker attempt 2 (retry succeeded)
        assert start_events[2]["role"] == "Worker"
        assert start_events[2]["turn"] == 2
        assert "empty response" in start_events[2]["prompt"]
        assert end_events[2]["reply"] == "recovered output"

        assert len(res.turns) == 3
    finally:
        serve._history_context.write_line = None
        serve._history_context.calls = []
        serve._history_context.conductor_mode = False


def test_conductor_node_empty_retry_exhaustion():
    events = []

    class FakeAlwaysEmptyWorker:
        slot_models = ["model-a"]

        def conduct(self, model, prompt):
            return "model_id: [0]\nsubtasks: ['always empty']\naccess_list: ['all']"

        def __call__(self, sub, messages, agent_id):
            return "   "

    serve._history_context.write_line = events.append
    serve._history_context.calls = []
    try:
        coord = serve.ConductorCoordinator(serve.HistoryWorker(FakeAlwaysEmptyWorker()))
        with pytest.raises(ValueError, match="returned empty response after retry"):
            coord.run("query")

        start_events = [e for e in events if e.get("type") == "step-start"]
        end_events = [e for e in events if e.get("type") == "step-end"]
        assert len(start_events) == 3  # Planner, attempt 1, attempt 2
        assert len(end_events) == 3
    finally:
        serve._history_context.write_line = None
        serve._history_context.calls = []
        serve._history_context.conductor_mode = False


def test_conductor_worker_preserves_multi_turn_history_without_feedback_leakage():
    class HistoryCaptureWorker:
        slot_models = ["model-a"]

        def __init__(self):
            self.messages = []

        def __call__(self, _subtask, messages, _agent_id):
            self.messages = messages
            return "done"

    serve._history_context.history = [
        {"role": "user", "content": "first request"},
        {"role": "assistant", "content": "first answer"},
    ]
    serve._history_context.conductor_mode = True
    serve._history_context.revision_feedback = "TRINITY FEEDBACK MUST NOT LEAK"
    try:
        raw_worker = HistoryCaptureWorker()
        worker = serve.HistoryWorker(raw_worker)
        assert (
            worker("different subtask", [{"role": "user", "content": "current node"}], 0) == "done"
        )
        assert raw_worker.messages == [
            {"role": "user", "content": "first request"},
            {"role": "assistant", "content": "first answer"},
            {"role": "user", "content": "current node"},
        ]
        assert all(
            "TRINITY FEEDBACK MUST NOT LEAK" not in m["content"] for m in raw_worker.messages
        )
    finally:
        serve._history_context.history = []
        serve._history_context.conductor_mode = False
        serve._history_context.revision_feedback = None
        serve._history_context.calls = []


def test_conductor_no_feedback_leakage():
    class LeakCheckWorker:
        slot_models = ["model-a"]

        def __init__(self):
            self.received_prompts = []

        def conduct(self, model, prompt):
            return "model_id: [0, 0]\nsubtasks: ['step 1', 'step 2']\naccess_list: [[], [0]]"

        def __call__(self, sub, messages, agent_id):
            prompt_content = messages[-1]["content"]
            self.received_prompts.append(prompt_content)
            return "done step"

    serve._history_context.revision_feedback = "LEAKED VERIFIER REJECTION FEEDBACK"
    serve._history_context.force_worker = True
    serve._history_context.calls = []
    try:
        worker = LeakCheckWorker()
        coord = serve.ConductorCoordinator(serve.HistoryWorker(worker))
        res = coord.run("query")
        assert res.final == "done step"

        # Conductor subtasks must not contain the Trinity revision_feedback
        for prompt in worker.received_prompts:
            assert "LEAKED VERIFIER REJECTION FEEDBACK" not in prompt

        # Trinity feedback state must remain untouched during Conductor execution
        assert serve._history_context.revision_feedback == "LEAKED VERIFIER REJECTION FEEDBACK"
        assert serve._history_context.force_worker is True
    finally:
        serve._history_context.revision_feedback = None
        serve._history_context.force_worker = False
        serve._history_context.calls = []
        serve._history_context.conductor_mode = False


def test_conductor_complete_turn_numbering():
    class MultiStepWorker:
        slot_models = ["model-a"]

        def conduct(self, model, prompt):
            return (
                "model_id: [0, 0, 0]\n"
                "subtasks: ['task A', 'task B', 'task C']\n"
                "access_list: [[], [0], [1]]"
            )

        def __call__(self, sub, messages, agent_id):
            return f"out for {sub}"

    serve._history_context.calls = []
    try:
        coord = serve.ConductorCoordinator(serve.HistoryWorker(MultiStepWorker()))
        res = coord.run("multi step query")
        turns = res.turns
        assert len(turns) == 4  # 1 Planner + 3 Subtasks
        for expected_turn, turn in enumerate(turns):
            assert turn.turn == expected_turn
            assert turn.idx == expected_turn
            assert turn.t == expected_turn

        assert turns[0].role == "Planner"
        assert turns[1].role == "Worker"
        assert turns[2].role == "Worker"
        assert turns[3].role == "Worker"
    finally:
        serve._history_context.calls = []
        serve._history_context.conductor_mode = False


def test_worker_timeout_passthrough(monkeypatch):
    captured_kw: dict[str, Any] = {}

    def _completion(**kw):
        captured_kw.update(kw)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))])

    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(completion=_completion))
    monkeypatch.setenv("MANTIS_WORKER_TIMEOUT", "120")

    worker_trinity = serve.OpenRouterTrinityWorker(slot_models=["gpt-5.6-luna"], timeout=120)
    res_trinity = worker_trinity("Worker", [{"role": "user", "content": "test"}], 0)
    assert res_trinity == "ok"
    assert captured_kw["timeout"] == 120.0

    captured_kw.clear()
    worker_conductor = serve.OpenRouterConductorWorker(slot_models=["gpt-5.6-luna"], timeout=180)
    res_conductor = worker_conductor._call("gpt-5.6-luna", [{"role": "user", "content": "test"}])
    assert res_conductor == "ok"
    assert captured_kw["timeout"] == 180.0


def test_disconnect_prevents_subsequent_steps(monkeypatch):
    worker_calls = 0

    class DisconnectingWorker:
        slot_models = ["model-0"]
        names = ["model-0"]

        def __call__(self, role, messages, agent_id):
            nonlocal worker_calls
            worker_calls += 1
            if worker_calls == 1:
                # Simulate client disconnect after step 0
                serve._history_context.aborted = True
            return f"reply-{worker_calls}"

    class MultiStepCoord:
        def run(self, query, verbose=False):
            worker = serve.HistoryWorker(DisconnectingWorker())
            worker("Worker", [{"role": "user", "content": query}], 0)
            # This second worker call should raise ClientDisconnectedError
            r2 = worker("Thinker", [{"role": "user", "content": query}], 0)
            return SimpleNamespace(final=r2, turns=[])

    old = serve.get_coordinator
    try:
        serve.get_coordinator = lambda _mode: MultiStepCoord()
        srv, port = _start_server(serve.Handler)
        time.sleep(0.1)

        payload = json.dumps(
            {"model": "trinity", "messages": [{"role": "user", "content": "hi"}], "stream": True}
        ).encode()
        req = Request(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json", "Authorization": "Bearer test-key"},
        )
        resp = urlopen(req)
        # Read the first event
        resp.readline()
        resp.close()

        time.sleep(0.2)
        # Ensure only 1 worker call occurred before aborting
        assert worker_calls == 1
        assert getattr(serve._history_context, "write_line", None) is None
        assert getattr(serve._history_context, "is_client_connected", None) is None
        assert getattr(serve._history_context, "aborted", False) is False
    finally:
        serve.get_coordinator = old
        srv.shutdown()


# ---------------------------------------------------------------------------
# Resumable native-tool runs
# ---------------------------------------------------------------------------


def _run_messages(text: str = "do the task") -> list[dict[str, str]]:
    return [{"role": "system", "content": "system"}, {"role": "user", "content": text}]


def test_litellm_upstream_config_is_separate_from_ingress(monkeypatch):
    monkeypatch.setenv("MANTIS_API_KEY", "ingress")
    monkeypatch.setenv("LITELLM_KEY", "proxy")
    monkeypatch.setenv("MANTIS_BASE_URL", "http://proxy/v1")
    assert serve._litellm_api_key() == "proxy"
    assert serve._litellm_base_url() == "http://proxy/v1"
    assert serve._is_reasoning_model("openai/gpt-5.6-sol")
    monkeypatch.delenv("LITELLM_KEY")
    monkeypatch.delenv("MANTIS_LITELLM_API_KEY", raising=False)
    monkeypatch.delenv("FUGU_LITELLM_API_KEY", raising=False)
    assert serve._litellm_api_key() is None


def test_configured_slot_models(monkeypatch):
    monkeypatch.setattr(serve, "_args", None)
    monkeypatch.setenv("MANTIS_WORKER_MODELS", "one, two")
    assert serve._configured_slot_models() == ["one", "two"]
    assert serve._configured_slot_models(["override"]) == ["override"]
    with pytest.raises(TypeError):
        serve._configured_slot_models("not-a-list")
    with pytest.raises(ValueError):
        serve._configured_slot_models([])


def test_validate_tool_results_exact_ids():
    expected = {"a", "b"}
    results = [{"tool_call_id": "b", "content": "2"}, {"tool_call_id": "a", "content": "1"}]
    assert serve._validate_tool_results(results, expected) == results
    for invalid in (
        None,
        [{"tool_call_id": "a"}, {"tool_call_id": "a"}],
        [{"tool_call_id": "a"}],
        [{"content": "missing"}, {"tool_call_id": "b"}],
    ):
        with pytest.raises((TypeError, ValueError)):
            serve._validate_tool_results(invalid, expected)


def test_trinity_native_tool_run_and_accept(monkeypatch):
    roles = iter([("Worker", 0), ("Verifier", 0)])
    monkeypatch.setattr(
        serve,
        "get_router",
        lambda: SimpleNamespace(
            route=lambda *_args, **_kwargs: dict(
                zip(("role_name", "agent_id"), next(roles), strict=True)
            )
        ),
    )
    calls = []

    def complete(model, messages, tools):
        calls.append((model, list(messages), tools))
        if len(calls) == 1:
            return "", [{"id": "read-1", "name": "read", "arguments": {"path": "README.md"}}]
        if len(calls) == 2:
            assert messages[-1] == {"role": "tool", "tool_call_id": "read-1", "content": "file"}
            return "answer", []
        return "ACCEPT", []

    monkeypatch.setattr(serve, "_model_completion", complete)
    run = serve.TrinityRun(
        "r1", _run_messages(), [{"type": "function"}], slot_models=["worker"], max_turns=4
    )
    assert run.advance(None)["type"] == "tool_calls"
    worker = run.advance([{"tool_call_id": "read-1", "content": "file"}])
    assert worker["type"] == "step_complete" and worker["reply"] == "answer"
    verifier = run.advance(None)
    assert verifier["role"] == "Verifier" and verifier["reply"] == "ACCEPT"
    final = run.advance(None)
    assert final["type"] == "final"
    assert final["text"] == "answer"
    assert final["terminated_by"] == "verifier_accept"
    assert len(calls) == 3
    assert run.advance(None)["error"] == "run already finished"


def test_trinity_rejects_wrong_tool_result(monkeypatch):
    monkeypatch.setattr(
        serve,
        "get_router",
        lambda: SimpleNamespace(route=lambda *_a, **_k: {"role_name": "Worker", "agent_id": 0}),
    )
    monkeypatch.setattr(
        serve,
        "_model_completion",
        lambda *_a: ("", [{"id": "expected", "name": "read", "arguments": {}}]),
    )
    run = serve.TrinityRun("r2", _run_messages(), [], slot_models=["worker"])
    assert (
        run.advance([{"tool_call_id": "early", "content": "x"}])["error"]
        == "unexpected tool results"
    )
    assert run.advance(None)["type"] == "tool_calls"
    event = run.advance([{"tool_call_id": "wrong", "content": "x"}])
    assert event["type"] == "error" and "mismatch" in event["error"]


def test_conductor_native_tool_run(monkeypatch):
    monkeypatch.setattr(serve, "parse_workflow", lambda _text: ([0], ["inspect"], [[]]))
    replies = iter(
        [
            ("plan", []),
            ("", [{"id": "bash-1", "name": "bash", "arguments": {"command": "pwd"}}]),
            ("done", []),
        ]
    )
    monkeypatch.setattr(serve, "_model_completion", lambda *_a: next(replies))
    run = serve.ConductorRun("c1", _run_messages(), [], slot_models=["worker"])
    assert run.advance(None)["role"] == "Planner"
    assert run.advance(None)["type"] == "tool_calls"
    step = run.advance([{"tool_call_id": "bash-1", "content": "working-dir"}])
    assert step["type"] == "step_complete" and step["reply"] == "done"
    final = run.advance(None)
    assert final["type"] == "final" and final["text"] == "done"


def test_create_run_uses_configured_pool_and_delete(monkeypatch):
    monkeypatch.setattr(serve, "_args", None)
    monkeypatch.setenv("MANTIS_WORKER_MODELS", "configured-a,configured-b")
    run = serve.create_run("trinity", {"messages": _run_messages(), "tools": []})
    try:
        assert run.slot_models == ["configured-a", "configured-b"]
        assert serve.get_run(run.run_id) is run
        assert serve.delete_run(run.run_id)
        with pytest.raises(KeyError):
            serve.get_run(run.run_id)
    finally:
        serve.delete_run(run.run_id)


def test_create_run_validates_messages_and_slots():
    requested = "a" * 32
    run = serve.create_run(
        "trinity", {"messages": _run_messages(), "slot_models": ["worker"], "run_id": requested}
    )
    try:
        assert run.run_id == requested
    finally:
        serve.delete_run(requested)
    with pytest.raises(ValueError):
        serve.create_run("trinity", {"messages": _run_messages(), "run_id": "invalid"})
    with pytest.raises(ValueError):
        serve.create_run("trinity", {"messages": []})
    with pytest.raises(TypeError):
        serve.create_run("trinity", {"messages": _run_messages(), "slot_models": "bad"})


def test_run_http_lifecycle_and_invalid_json(monkeypatch):
    monkeypatch.setattr(serve, "_args", None)
    monkeypatch.setenv("MANTIS_WORKER_MODELS", "worker")
    srv, port = _start_server(serve.Handler)
    try:
        time.sleep(0.05)
        headers = {"Authorization": "Bearer test-key", "Content-Type": "application/json"}
        conn = http.client.HTTPConnection("127.0.0.1", port)
        conn.request("POST", "/v1/runs", body="[]", headers=headers)
        assert conn.getresponse().status == 400
        conn.close()

        conn = http.client.HTTPConnection("127.0.0.1", port)
        conn.request(
            "POST",
            "/v1/runs",
            body=json.dumps({"model": "trinity", "messages": _run_messages()}),
            headers=headers,
        )
        response = conn.getresponse()
        assert response.status == 200
        run_id = json.loads(response.read())["run_id"]
        conn.close()

        conn = http.client.HTTPConnection("127.0.0.1", port)
        conn.request("POST", f"/v1/runs/{run_id}/continue", body="[]", headers=headers)
        assert conn.getresponse().status == 400
        conn.close()

        conn = http.client.HTTPConnection("127.0.0.1", port)
        conn.request("DELETE", f"/v1/runs/{run_id}", headers=headers)
        response = conn.getresponse()
        assert response.status == 200 and json.loads(response.read())["deleted"] is True
        conn.close()
    finally:
        srv.shutdown()


def test_bounded_request_body_helpers():
    handler = object.__new__(serve.Handler)
    handler.headers = {"Content-Length": "4"}
    handler.rfile = __import__("io").BytesIO(b"data")
    with pytest.raises(serve.RequestBodyTooLargeError):
        handler._read_request_body(max_bytes=3)

    handler.headers = {"Transfer-Encoding": "chunked"}
    handler.rfile = __import__("io").BytesIO(b"4\r\ndata\r\n0\r\n\r\n")
    with pytest.raises(serve.RequestBodyTooLargeError):
        handler._read_request_body(max_bytes=3)
    handler.rfile = __import__("io").BytesIO(b"-1\r\n")
    with pytest.raises(ValueError, match="negative"):
        handler._read_request_body()
    handler.rfile = __import__("io").BytesIO(b"4\r\nab")
    with pytest.raises(ValueError, match="truncated"):
        handler._read_request_body()
    handler.rfile = __import__("io").BytesIO(b"2\r\nabXX0\r\n\r\n")
    with pytest.raises(ValueError, match="terminator"):
        handler._read_request_body()


def test_convert_tools_and_model_completion(monkeypatch):
    converted = serve._convert_tools(
        [
            None,
            {},
            {"name": "mantis_step", "parameters": {}},
            {"name": "read", "description": "read files", "parameters": {"type": "object"}},
            {"name": "bash"},
            {"name": "bad", "parameters": "invalid"},
        ]
    )
    assert [tool["function"]["name"] for tool in converted] == ["read", "bash"]
    assert converted[1]["function"]["parameters"]["type"] == "object"
    assert serve._convert_tools("bad") == []

    tool_calls = [
        SimpleNamespace(id="same", function=SimpleNamespace(name="read", arguments='{"path":"a"}')),
        SimpleNamespace(id="same", function=SimpleNamespace(name="bash", arguments="not-json")),
        SimpleNamespace(id=None, function=SimpleNamespace(name="edit", arguments="[]")),
    ]
    completion = MagicMock(
        return_value=SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=None, tool_calls=tool_calls))]
        )
    )
    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(completion=completion))
    monkeypatch.setenv("LITELLM_KEY", "proxy")
    text, calls = serve._model_completion("openai/gpt-5.6-sol", [], converted)
    assert text == ""
    assert len({call["id"] for call in calls}) == 3
    assert calls[1]["arguments"] == {} and calls[2]["arguments"] == {}
    kwargs = completion.call_args.kwargs
    assert kwargs["api_key"] == "proxy" and kwargs["tools"] == converted
    assert "temperature" not in kwargs


def test_run_registry_sweep_and_capacity(monkeypatch):
    monkeypatch.setattr(serve, "_runs", {})
    monkeypatch.setattr(serve, "_runs_sweeper_started", True)
    monkeypatch.setattr(serve, "MAX_RUNS", 1)
    first = serve.NativeRun("first")
    second = serve.NativeRun("second")
    serve._register_run(first)
    serve._register_run(second)
    assert first.cancelled and serve.get_run("second") is second
    second.last_active = 0
    second.in_flight = 1
    serve._sweep_runs()
    assert serve.get_run("second") is second and not second.cancelled
    second.in_flight = 0
    serve._sweep_runs()
    assert second.cancelled
    with pytest.raises(KeyError):
        serve.get_run("second")
    assert not serve.delete_run("missing")


def test_trinity_recovery_thinker_and_limits(monkeypatch):
    run = serve.TrinityRun("recovery", _run_messages(), [], slot_models=["worker"], max_turns=1)
    thinker = run._role_complete(
        "Thinker",
        0,
        0,
        [{"content": "think"}],
        "<suggested_role>solver</suggested_role><suggestion>be precise</suggestion>",
    )
    assert thinker["role"] == "Thinker"
    assert run.suggested_role == "Worker" and run.suggestion == "be precise"
    assert "be precise" in run._role_prompt("Worker")

    run._role_complete("Verifier", 0, 1, [{"content": "verify"}], "REJECT: fix it")
    assert run.force_worker and "REJECT" in (run.revision_feedback or "")
    run._role_complete("Worker", 0, 2, [{"content": "work"}], "")
    assert run.force_worker and "no response" in (run.revision_feedback or "")

    limited = serve.TrinityRun("limit", _run_messages(), [], slot_models=["worker"], max_turns=0)
    event = limited.advance(None)
    assert event["type"] == "final" and event["terminated_by"] == "max_turns"
    limited.close()
    assert limited.advance(None)["error"] == "run cancelled"

    monkeypatch.setattr(
        serve,
        "get_router",
        lambda: SimpleNamespace(route=lambda *_a, **_k: {"role_name": "Worker", "agent_id": 0}),
    )
    monkeypatch.setattr(
        serve, "_model_completion", lambda *_a: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    failed = serve.TrinityRun("failed", _run_messages(), [], slot_models=["worker"])
    assert failed.advance(None) == {"type": "error", "error": "boom"}
    assert failed.cancelled


def test_trinity_cold_roles_are_workers(monkeypatch):
    routed = iter(["Verifier", "Thinker", "Worker"])
    monkeypatch.setattr(
        serve,
        "get_router",
        lambda: SimpleNamespace(route=lambda *_a, **_k: {"role_name": next(routed), "agent_id": 0}),
    )
    run = serve.TrinityRun("cold", _run_messages(), [], slot_models=["worker"])
    assert run._route()[0] == "Worker"
    assert run._route()[0] == "Worker"
    run.suggested_role = "Thinker"
    run.last_response = "answer"
    assert run._route()[0] == "Thinker"


def test_conductor_errors_visibility_and_state(monkeypatch):
    run = serve.ConductorRun("errors", _run_messages(), [], slot_models=["worker"])
    with pytest.raises(ValueError):
        run._node_messages(0, 0, "task")
    run._workflow = ([0, 0], ["first", "second"], [[], [0]])
    run._outputs = ["first output"]
    assert "first output" in run._node_messages(1, 0, "second")[0]["content"]

    monkeypatch.setattr(
        serve, "parse_workflow", lambda _text: (_ for _ in ()).throw(ValueError("bad plan"))
    )
    event = run._finalize_text("Planner", "bad", 0)
    assert event["type"] == "error" and "parseable" in event["error"]
    monkeypatch.setattr(serve, "parse_workflow", lambda _text: ([], [], []))
    assert "malformed" in run._finalize_text("Planner", "bad", 0)["error"]

    run.close()
    assert run.advance(None)["error"] == "run cancelled"
    finished = serve.ConductorRun("finished", _run_messages(), [], slot_models=["worker"])
    finished.finished = True
    assert finished.advance(None)["error"] == "run already finished"
    unexpected = serve.ConductorRun("unexpected", _run_messages(), [], slot_models=["worker"])
    assert unexpected.advance([{"tool_call_id": "early"}])["error"] == "unexpected tool results"

    capped = serve.ConductorRun("capped", _run_messages(), [], slot_models=["worker"], max_steps=0)
    capped._workflow = ([0], ["task"], [[]])
    event = capped.advance(None)
    assert event["type"] == "final" and event["terminated_by"] == "max_steps"


def test_advance_run_idempotency(monkeypatch):
    class CountingRun(serve.NativeRun):
        def __init__(self):
            super().__init__("idempotent")
            self.calls = 0

        def advance(self, tool_results):
            self.calls += 1
            return {"type": "step_complete", "reply": str(tool_results)}

    monkeypatch.setattr(serve, "_runs", {})
    monkeypatch.setattr(serve, "_runs_sweeper_started", True)
    run = CountingRun()
    serve._register_run(run)
    first = serve.advance_run(run.run_id, [{"tool_call_id": "one"}], "request-1")
    second = serve.advance_run(run.run_id, [{"tool_call_id": "different"}], "request-1")
    assert first == second and run.calls == 1
    with pytest.raises(ValueError, match="request_id"):
        serve.advance_run(run.run_id, None, "")


def test_learning_record_is_redacted_and_high_confidence(tmp_path, monkeypatch):
    monkeypatch.setenv("MANTIS_LEARNING", "1")
    monkeypatch.setenv("MANTIS_LEARNING_DIR", str(tmp_path))
    monkeypatch.setenv("MANTIS_LEARNING_INSTANCE", "test/host")
    run = serve.TrinityRun(
        "learn",
        [{"role": "user", "content": "fix tests token=secret-value"}],
        [],
        slot_models=[f"m{i}" for i in range(7)],
    )
    run.turns = [
        {"role": "Worker", "agent_id": 3, "reply": "fixed"},
        {"role": "Verifier", "agent_id": 1, "reply": "ACCEPT"},
    ]
    pending = {
        "asst": {
            "tool_calls": [
                {
                    "id": "test-1",
                    "function": {
                        "name": "bash",
                        "arguments": json.dumps({"command": "python3 -m unittest -v"}),
                    },
                }
            ]
        }
    }
    run.record_tool_results(pending, [{"tool_call_id": "test-1", "is_error": False}])
    serve._write_learning_record(run, {"type": "final", "terminated_by": "verifier_accept"})
    path = tmp_path / "runs-test_host.jsonl"
    record = json.loads(path.read_text())
    assert path.stat().st_mode & 0o777 == 0o600
    assert record["trainable"] is True
    assert record["label_worker"] == 3 and record["label_role"] == 0
    assert record["last_test_passed"] is True
    assert "secret-value" not in record["task"] and "[REDACTED]" in record["task"]
    serve._write_learning_record(run, {"type": "error"})
    assert len(path.read_text().splitlines()) == 1


def test_learning_record_skips_ambiguous_runs(monkeypatch):
    monkeypatch.delenv("MANTIS_LEARNING", raising=False)
    run = serve.TrinityRun("ambiguous", _run_messages(), [], slot_models=["worker"])
    run.turns = [{"role": "Worker", "agent_id": 0, "reply": "answer"}]
    record = serve._learning_record(run, {"type": "final", "terminated_by": "max_turns"})
    assert not record["trainable"] and not record["test_seen"]
