"""Session-bound sidekick transcript reuse tests."""

from __future__ import annotations

import api
import fusion
import fusion_session
import pytest
import runs
import serve_config
from fastapi.testclient import TestClient

from tests.test_fusion_tool_mode import call, install, reply, use_tool_catalog


def _api_client(monkeypatch, tmp_path):
    use_tool_catalog(monkeypatch, tmp_path)
    monkeypatch.setenv("MANTIS_API_KEY", "test-key")
    monkeypatch.setenv("MANTIS_RUN_STORE", "memory")
    monkeypatch.setattr(fusion, "RUN_STORE", "memory")
    monkeypatch.setattr(runs, "RUN_STORE", "memory")
    serve_config._runs.clear()
    return TestClient(api.app)


def _post_chat(client, messages, session_id):
    headers = {"Authorization": "Bearer test-key"}
    if session_id:
        headers["X-Mantis-Session-Id"] = session_id
    return client.post(
        "/v1/chat/completions",
        headers=headers,
        json={
            "model": "mantis/fusion",
            "messages": messages,
            "tool_options": {"delegation": "tool"},
        },
    )


def _session_worker(monkeypatch):
    return install(
        monkeypatch,
        [
            reply("", call("c1", "sidekick", message="first handoff")),
            reply("first answer"),
            reply("", call("c2", "sidekick", message="second handoff")),
            reply("second answer"),
            reply("", call("c3", "sidekick", message="fresh handoff")),
            reply("third answer"),
        ],
        [reply("first report"), reply("second report"), reply("fresh report")],
    )


def _conversation_transcripts():
    first = [{"role": "user", "content": "first user turn"}]
    second = [
        *first,
        {"role": "assistant", "content": "first answer"},
        {"role": "user", "content": "second user turn"},
    ]
    third = [
        *second,
        {"role": "assistant", "content": "second answer"},
        {"role": "user", "content": "third user turn"},
    ]
    return first, second, third


@pytest.fixture(autouse=True)
def clear_session_contexts():
    """Isolate the process-local sidekick session store."""
    with fusion_session._SESSION_LOCK:
        fusion_session._SESSION_CONTEXTS.clear()
    yield
    with fusion_session._SESSION_LOCK:
        fusion_session._SESSION_CONTEXTS.clear()


def _run(run_id: str, messages: list[dict[str, str]] | None = None) -> fusion.FusionRun:
    run = fusion.FusionRun(run_id, "task")
    run.handoffs = object()
    run.status = "completed"
    run.sidekick_messages = [
        {"role": "system", "content": "current sidekick policy"},
        *(messages or []),
    ]
    return run


def test_session_context_restores_only_for_identified_sessions():
    saved = _run("session-saved", [{"role": "user", "content": "prior task"}])
    fusion_session.save_sidekick(saved, "session-key")
    resumed = _run("session-resumed")
    fresh = _run("session-fresh")

    assert fusion_session.restore_sidekick(resumed, "session-key")
    assert [message["content"] for message in resumed.sidekick_messages] == [
        "current sidekick policy",
        "prior task",
    ]
    assert not fusion_session.restore_sidekick(fresh, None)
    assert fresh.sidekick_messages == [{"role": "system", "content": "current sidekick policy"}]


def test_session_store_bounds_context_and_number_of_sessions(monkeypatch):
    monkeypatch.setattr(fusion_session, "_MAX_SESSION_CONTEXTS", 2)
    monkeypatch.setattr(fusion_session, "_MAX_CONTEXT_MESSAGES", 2)
    messages = [{"role": "user", "content": f"turn {index}"} for index in range(4)]

    for index in range(3):
        fusion_session.save_sidekick(_run(f"session-{index}", messages), f"key-{index}")

    assert len(fusion_session._SESSION_CONTEXTS) == 2
    assert not fusion_session.restore_sidekick(_run("old"), "key-0")
    newest = _run("new")
    assert fusion_session.restore_sidekick(newest, "key-2")
    assert len(newest.sidekick_messages) == 3
    assert [message["content"] for message in newest.sidekick_messages[1:]] == [
        "turn 2",
        "turn 3",
    ]


def test_no_session_id_does_not_save_context():
    run = _run("no-session", [{"role": "user", "content": "private context"}])

    fusion_session.save_sidekick(run, None)

    assert not fusion_session._SESSION_CONTEXTS


def test_api_starts_new_runs_and_reuses_context_only_with_session_id(monkeypatch, tmp_path):
    client = _api_client(monkeypatch, tmp_path)
    worker = _session_worker(monkeypatch)
    first_messages, second_messages, third_messages = _conversation_transcripts()
    responses = [
        _post_chat(client, first_messages, "conversation"),
        _post_chat(client, second_messages, "conversation"),
        _post_chat(client, third_messages, None),
    ]

    assert [response.status_code for response in responses] == [200, 200, 200]
    run_ids = [response.headers["x-mantis-run-id"] for response in responses]
    assert len(set(run_ids)) == 3
    resumed, _ = worker.seen["sidekick"][1]
    fresh, _ = worker.seen["sidekick"][2]
    assert [message["content"] for message in resumed[-3:]] == [
        "first handoff",
        "first report",
        "second handoff",
    ]
    assert [message["content"] for message in fresh[1:]] == ["fresh handoff"]
