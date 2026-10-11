"""Native client history rebasing at public Fusion boundaries."""

from copy import deepcopy

import fusion
import fusion_history
import fusion_lead
import pytest
import runs

from tests.test_fusion_tool_mode import BASH_TOOL, call, install, reply, tool_run, use_tool_catalog


def history(*calls):
    return [
        {"role": "system", "content": "new rules"},
        {"role": "developer", "content": "developer rules"},
        {"role": "user", "content": "compacted summary"},
        {"role": "assistant", "tool_calls": list(calls)},
        {"role": "tool", "tool_call_id": "public", "content": "result"},
    ]


def test_mixed_private_work_survives_compacted_public_history(monkeypatch, tmp_path):
    use_tool_catalog(monkeypatch, tmp_path)
    worker = install(monkeypatch, [
        reply("", call("d", "sidekick", message="work")),
        reply("", call("c", "bash", command="ls")), reply("done")
    ], [reply("private report")])
    run = tool_run(tools=[BASH_TOOL])
    run.advance()
    incoming = history(call("public", "bash", command="ls"))
    run.advance_idempotent(
        [{"tool_call_id": "lead:c", "content": "result"}], "resume",
        client_history=incoming, client_tool_ids={"public": "lead:c"}
    )
    seen = worker.seen["lead"][-1][0]
    assert "new rules" in seen[0]["content"]
    assert seen[1] == incoming[1]
    assert not any(m.get("content") == "build the feature" for m in seen)
    assert sum(m.get("tool_call_id") == "delegate:d" for m in seen) == 1
    assert sum(m.get("tool_call_id") == "lead:c" for m in seen) == 1
    before = deepcopy(run.main_messages)
    run.advance_idempotent([], "resume", client_history=incoming, client_tool_ids={})
    assert run.main_messages == before


def test_sidekick_only_pending_preserves_private_lead_batch(monkeypatch, tmp_path):
    use_tool_catalog(monkeypatch, tmp_path)
    worker = install(monkeypatch, [reply("", call("d", "sidekick", message="work")),
                                    reply("done")],
                     [reply("", call("c", "bash", command="ls")), reply("report")])
    run = tool_run(tools=[BASH_TOOL])
    event = run.advance()
    pending = event["pending_tool_calls"][0]["id"]
    run.advance_idempotent([{ "tool_call_id": pending, "content": "result"}],
                          client_history=history(call("public", "bash", command="ls")),
                          client_tool_ids={"public": pending})
    seen = worker.seen["lead"][-1][0]
    assert not any(m.get("tool_call_id") == pending for m in seen)
    assert any(m.get("tool_call_id") == "delegate:d" for m in seen)


def test_checkpoint_serializes_and_delta_only_does_not_rebase(monkeypatch, tmp_path):
    use_tool_catalog(monkeypatch, tmp_path)
    run = tool_run()
    fusion_lead.prepare_run(run)
    run.fusion_public_checkpoint = len(run.main_messages)
    restored = deepcopy(run)
    before = deepcopy(restored.main_messages)
    fusion_history.rebase(restored, [{"role": "tool", "content": "result"}], {})
    assert restored.main_messages == before
    assert restored.fusion_public_checkpoint == run.fusion_public_checkpoint


def test_full_history_requires_exact_pending_assistant(monkeypatch, tmp_path):
    use_tool_catalog(monkeypatch, tmp_path)
    run = tool_run()
    fusion_lead.prepare_run(run)
    run.status = "awaiting_tools"
    run.pending_tool_calls = [call("lead:c", "bash")]
    run.fusion_public_checkpoint = len(run.main_messages)
    with pytest.raises(ValueError, match="pending assistant"):
        fusion_history.rebase(run, history(call("wrong", "bash")), {"public": "lead:c"})


def test_next_public_boundary_drops_previous_private_span(monkeypatch, tmp_path):
    use_tool_catalog(monkeypatch, tmp_path)
    worker = install(monkeypatch, [
        reply("", call("d", "sidekick", message="work")),
        reply("", call("c", "bash", command="ls")),
        reply("", call("next", "bash", command="pwd")), reply("done")
    ], [reply("private report")])
    run = tool_run(tools=[BASH_TOOL])
    run.advance()
    run.advance_idempotent([{ "tool_call_id": "lead:c", "content": "result"}],
                          client_history=history(call("public", "bash", command="ls")),
                          client_tool_ids={"public": "lead:c"})
    run.advance_idempotent([{ "tool_call_id": "lead:next", "content": "result"}],
                          client_history=history(call("public", "bash", command="pwd")),
                          client_tool_ids={"public": "lead:next"})
    seen = worker.seen["lead"][-1][0]
    assert not any(m.get("tool_call_id") in {"delegate:d", "lead:c"} for m in seen)
    assert sum(m.get("tool_call_id") == "lead:next" for m in seen) == 1


def test_header_followup_replaces_history_without_duplicating_user(monkeypatch, tmp_path):
    use_tool_catalog(monkeypatch, tmp_path)
    worker = install(monkeypatch, [reply("first"), reply("second")], [])
    run = tool_run()
    run.advance()
    incoming = [{"role": "system", "content": "new rules"},
                {"role": "user", "content": "next request"}]
    run.advance_idempotent(message="next request", client_history=incoming)
    seen = worker.seen["lead"][-1][0]
    assert [m.get("content") for m in seen[1:]] == ["next request"]


def test_file_store_resume_and_cached_retry(monkeypatch, tmp_path):
    use_tool_catalog(monkeypatch, tmp_path)
    worker = install(monkeypatch, [reply("", call("c", "bash", command="ls")),
                                   reply("done")], [])
    monkeypatch.setattr(fusion, "RUN_STORE", "file")
    monkeypatch.setattr(runs, "RUN_STORE", "file")
    run = fusion.create_fusion_run("task", tools=[BASH_TOOL])
    fusion.advance_fusion_run(run.run_id)
    incoming = history(call("public", "bash", command="ls"))
    event = fusion.advance_fusion_run(
        run.run_id, request_id="resume", tool_results=[{"tool_call_id": "lead:c", "content": "ok"}],
        client_history=incoming, client_tool_ids={"public": "lead:c"}
    )
    repeated = fusion.advance_fusion_run(run.run_id, request_id="resume", client_history=incoming)
    assert repeated == event
    assert len(worker.seen["lead"]) == 2
    assert fusion.get_run(run.run_id).fusion_public_checkpoint is not None
