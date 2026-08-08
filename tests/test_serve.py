"""Unit tests for openfugu-patch/serve.py."""

from __future__ import annotations

import json
import os
import sys
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
import serve
import torch

os.environ["MANTIS_API_KEY"] = "test-key"


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------
def test_is_reasoning_model():
    assert serve._is_reasoning_model("claude-sonnet-5")
    assert serve._is_reasoning_model("gpt-5.6-terra")
    assert not serve._is_reasoning_model("deepseek-v4-flash")
    assert not serve._is_reasoning_model("glm-5.2")


def test_build_request_routes_providers(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    monkeypatch.setenv("OPENCODE_API_KEY", "oc-key")
    url, headers, body = serve._build_request("openrouter/openai/gpt-5.6-sol|medium", [], 1024, 0.2)
    assert url == "https://openrouter.ai/api/v1/chat/completions"
    assert headers["Authorization"] == "Bearer or-key"
    assert body["model"] == "openai/gpt-5.6-sol"
    assert body["reasoning_effort"] == "medium"
    assert "temperature" not in body

    url, headers, body = serve._build_request("opencode-go/deepseek-v4-flash", [], 1024, 0.2)
    assert url == "https://opencode.ai/zen/go/v1/chat/completions"
    assert headers["Authorization"] == "Bearer oc-key"
    assert body["model"] == "deepseek-v4-flash"
    assert body["temperature"] == 0.2


def test_resolve_conductor_model_env(monkeypatch):
    monkeypatch.setenv("MANTIS_CONDUCTOR_MODEL", "openai/gpt-5.6-sol")
    assert serve._resolve_conductor_model(SimpleNamespace()) == "openai/gpt-5.6-sol"


def test_resolve_conductor_model_from_slot_models(monkeypatch):
    monkeypatch.delenv("MANTIS_CONDUCTOR_MODEL", raising=False)
    worker = SimpleNamespace(slot_models=["slot-0", "slot-1"])
    assert serve._resolve_conductor_model(worker) == "slot-0"


def test_resolve_conductor_model_default(monkeypatch):
    monkeypatch.delenv("MANTIS_CONDUCTOR_MODEL", raising=False)
    assert serve._resolve_conductor_model(SimpleNamespace()) == "openai/gpt-4o-mini"


def test_direct_workers(monkeypatch):
    monkeypatch.setattr(serve, "_direct_completion", lambda *args: args[0])
    trinity = serve.DirectTrinityWorker(
        ["openrouter/anthropic/claude-sonnet-5|medium", "opencode-go/deepseek-v4-flash"]
    )
    assert (
        trinity("Worker", [{"role": "user", "content": "hi"}], 1) == "opencode-go/deepseek-v4-flash"
    )

    conductor = serve.DirectConductorWorker(["openrouter/openai/gpt-5.6-luna|max"])
    assert (
        conductor._call("openrouter/openai/gpt-5.6-luna|max", [])
        == "openrouter/openai/gpt-5.6-luna|max"
    )


def test_direct_provider_completions(monkeypatch):
    class Response:
        def __init__(self, body):
            self.body = body
            self.status_code = 200
            self.text = json.dumps(body)

        def raise_for_status(self):
            return None

        def json(self):
            return self.body

    responses = iter(
        [
            Response({"choices": [{"message": {"content": "answer"}}]}),
            Response(
                {
                    "choices": [
                        {
                            "message": {
                                "content": "",
                                "tool_calls": [
                                    {
                                        "id": "call-1",
                                        "function": {
                                            "name": "read",
                                            "arguments": '{"path":"README.md"}',
                                        },
                                    }
                                ],
                            }
                        }
                    ]
                }
            ),
        ]
    )
    client = SimpleNamespace(post=lambda *_a, **_k: next(responses))
    monkeypatch.setenv("OPENROUTER_API_KEY", "provider-key")
    monkeypatch.setattr(serve, "_upstream_streaming_enabled", lambda: False)
    monkeypatch.setattr(serve, "_provider_client", client)

    assert (
        serve._direct_completion(
            "openrouter/model", [{"role": "user", "content": "hi"}], 10, 0.7, 1
        )
        == "answer"
    )
    text, calls = serve._model_completion(
        "openrouter/model",
        [{"role": "user", "content": "read"}],
        [{"type": "function", "function": {"name": "read"}}],
    )
    assert text == ""
    assert calls == [{"id": "call-1", "name": "read", "arguments": {"path": "README.md"}}]


def test_provider_metadata_is_captured_for_public_response(monkeypatch):
    class Response:
        status_code = 200
        text = ""

        def raise_for_status(self):
            return None

        def json(self):
            return {
                "choices": [
                    {
                        "message": {
                            "content": "answer",
                            "reasoning_details": [{"type": "summary", "text": "checked"}],
                            "citations": [{"url": "https://example.test"}],
                        }
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
            }

    run = serve.NativeRun("metadata")
    run.capture_metadata = True
    serve._history_context.active_run = run
    monkeypatch.setenv("OPENROUTER_API_KEY", "provider-key")
    monkeypatch.setattr(serve, "_upstream_streaming_enabled", lambda: False)
    monkeypatch.setattr(serve, "_provider_client", SimpleNamespace(post=lambda *_a, **_k: Response()))
    try:
        serve._provider_response("openrouter/model", [], 10, 0.7)
    finally:
        serve._history_context.active_run = None
    body = serve._completion_response(
        "mantis", [], run, {"type": "final", "text": "answer"}
    )
    message = body["choices"][0]["message"]
    assert message["reasoning_details"][0]["text"] == "checked"
    assert message["citations"][0]["url"] == "https://example.test"


def test_split_messages():
    assert serve._split_messages([{"role": "user", "content": "hi"}]) == ("hi", [])
    content = [
        {"type": "text", "text": "inspect"},
        {"type": "image_url", "image_url": {"url": "https://example.test/image.png"}},
    ]
    assert serve._split_messages([{"role": "user", "content": content}]) == ("inspect", [])
    assert serve._with_images("worker prompt", content)[1]["type"] == "image_url"
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

    monkeypatch.setenv("MANTIS_CONDUCTOR_MODEL", "gpt-5.6-luna")
    coord = serve.EnvConductorCoordinator(FakeWorker())
    res = coord.run("query")
    assert res.final == "done"



def test_direct_conductor_worker(monkeypatch):
    captured: list[Any] = []
    monkeypatch.setattr(serve, "_direct_completion", lambda *args: captured.extend(args) or "ok")
    worker = serve.DirectConductorWorker(["openrouter/openai/gpt-5.6-luna"])
    assert (
        worker._call("openrouter/openai/gpt-5.6-luna", [{"role": "user", "content": "hi"}]) == "ok"
    )
    assert captured[0] == "openrouter/openai/gpt-5.6-luna"


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
    monkeypatch.delenv("MANTIS_CONDUCTOR_DEVICE", raising=False)
    conductor = serve.EnvLocalConductor("/ckpt", device=None)
    assert conductor.device == "cpu"
    assert conductor.dtype == torch.float32


def test_env_local_conductor_dtype_env(monkeypatch):
    fake = _fake_transformers_module(torch.tensor([[1, 2, 3]]))
    monkeypatch.setitem(sys.modules, "transformers", fake)
    monkeypatch.setenv("MANTIS_CONDUCTOR_DTYPE", "float32")
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
    monkeypatch.setattr(serve, "DirectTrinityWorker", FakeTrinity)
    monkeypatch.setattr(serve, "DirectConductorWorker", FakeConductor)

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
    monkeypatch.delenv("MANTIS_LOCAL_CONDUCTOR", raising=False)
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
    monkeypatch.setenv("MANTIS_LOCAL_CONDUCTOR", "di-zhang-fdu/openfugu-conductor-3b")
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
    monkeypatch.delenv("MANTIS_CONDUCTOR_DEVICE", raising=False)
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
    captured: list[Any] = []
    monkeypatch.setattr(serve, "_direct_completion", lambda *args: captured.extend(args) or "ok")
    worker = serve.DirectTrinityWorker(["opencode-go/deepseek-v4-flash"], timeout=120)
    assert worker("Worker", [{"role": "user", "content": "test"}], 0) == "ok"
    assert captured[-1] == 120


def _run_messages(text: str = "do the task") -> list[dict[str, str]]:
    return [{"role": "system", "content": "system"}, {"role": "user", "content": text}]


def test_provider_config_is_separate_from_ingress(monkeypatch):
    monkeypatch.setenv("MANTIS_API_KEY", "ingress")
    monkeypatch.setenv("OPENROUTER_API_KEY", "provider")
    _, headers, _ = serve._build_request("openrouter/openai/gpt-5.6-sol|medium", [], 10, 0.7)
    assert headers["Authorization"] == "Bearer provider"
    assert serve._is_reasoning_model("openai/gpt-5.6-sol")


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



def test_run_registry_sweep_and_capacity(monkeypatch):
    monkeypatch.setattr(serve, "_runs", {})
    monkeypatch.setattr(serve, "_runs_sweeper_started", True)
    monkeypatch.setattr(serve, "MAX_RUNS", 1)
    first = serve.NativeRun("first")
    second = serve.NativeRun("second")
    serve._register_run(first)
    with pytest.raises(serve.RunCapacityError):
        serve._register_run(second)
    assert not first.cancelled and serve.get_run("first") is first
    first.last_active = 0
    first.in_flight = 1
    serve._sweep_runs()
    assert serve.get_run("first") is first and not first.cancelled
    first.in_flight = 0
    serve._sweep_runs()
    assert first.cancelled
    with pytest.raises(KeyError):
        serve.get_run("first")
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



def test_standard_tool_validation_and_model_ids():
    tools = serve._convert_tools(
        [
            {
                "type": "function",
                "function": {
                    "name": "read",
                    "description": "read",
                    "parameters": {"type": "object"},
                },
            },
            {"name": "legacy"},
        ]
    )
    assert [tool["function"]["name"] for tool in tools] == ["read"]
    assert serve._mode_for_model("mantis") == "trinity"
    assert serve._mode_for_model("mantis-ultra") == "conductor"
    with pytest.raises(ValueError):
        serve._mode_for_model("unknown")
