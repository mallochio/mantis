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

import hashlib
import json
import os
import pickle
import re
import socket
import sys
import threading
import time
import uuid
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import jsonschema

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
# When running from the repo's orchestration runtime, also find upstream OpenFugu.
if not (_HERE / "mini.py").exists():
    _OPENFUGU = _HERE.parent / "OpenFugu" / "openfugu"
    if _OPENFUGU.exists():
        sys.path.insert(0, str(_OPENFUGU))

import conductor
import providers
import serve_config
import trinity
import utils
from mini import (
    DEFAULT_SLOT_LABELS,
    ROUTER_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    THINKER_PROMPT,
    VERIFICATION_PROMPT,
)
from serve_config import (
    _REDIS_PREFIX,
    _SECRET_PATTERNS,
    _TEST_COMMAND,
    _TEST_TOOL_NAMES,
    MAX_RUNS,
    MAX_TOOL_ROUNDS,
    REDIS_LOCK_TIMEOUT,
    REDIS_URL,
    RUN_MAX_MSG_BYTES,
    RUN_STORE,
)
from ultra import conductor_prompt, parse_workflow, visible_indices


def _learning_enabled() -> bool:
    return os.environ.get("MANTIS_LEARNING", "").lower() in {"1", "true", "yes", "on"}


def _redact_learning_task(task: str) -> str:
    redacted = task[: int(os.environ.get("MANTIS_LEARNING_MAX_TASK_CHARS", "12000"))]
    for pattern in _SECRET_PATTERNS:
        redacted = pattern.sub("[REDACTED]", redacted)
    return redacted


def _learning_path() -> Path:
    root = Path(
        os.path.expanduser(os.environ.get("MANTIS_LEARNING_DIR", "~/.local/share/mantis/learning"))
    )
    instance = os.environ.get("MANTIS_LEARNING_INSTANCE") or socket.gethostname()
    safe_instance = re.sub(r"[^A-Za-z0-9_.-]", "_", instance)[:80] or "local"
    return root / f"runs-{safe_instance}.jsonl"


def _tool_call_by_id(pending: dict[str, Any], tool_call_id: str) -> dict[str, Any] | None:
    for call in pending.get("asst", {}).get("tool_calls", []):
        if isinstance(call, dict) and call.get("id") == tool_call_id:
            return cast(dict[str, Any], call)
    return None


def _learning_record(run: NativeRun, event: dict[str, Any]) -> dict[str, Any]:
    turns = cast(list[dict[str, Any]], getattr(run, "turns", getattr(run, "steps", [])))
    final_worker = next((turn for turn in reversed(turns) if turn.get("role") == "Worker"), None)
    tests = [item for item in run.tool_observations if item["is_test"]]
    last_test_passed = bool(tests) and not tests[-1]["is_error"]
    accepted = event.get("terminated_by") == "verifier_accept"
    trainable = bool(run.kind == "trinity" and accepted and last_test_passed and final_worker)
    task = _redact_learning_task(str(getattr(run, "query", "")))
    steps = [
        {
            "role": str(turn.get("role", "")),
            "model": turn.get("model_name"),
            "agent_id": turn.get("agent_id"),
        }
        for turn in turns[:50]
        if isinstance(turn, dict)
    ]
    return {
        "schema_version": 1,
        "timestamp": int(time.time()),
        "run_id": run.run_id,
        "mode": run.kind,
        "task": task,
        "task_hash": hashlib.sha256(task.encode()).hexdigest(),
        "pool": list(getattr(run, "slot_models", [])),
        "terminated_by": event.get("terminated_by", event.get("type", "")),
        "duration_seconds": round(time.time() - run.created, 3),
        "turn_count": len(turns),
        "test_seen": bool(tests),
        "last_test_passed": last_test_passed,
        "tool_error_count": sum(item["is_error"] for item in run.tool_observations),
        "verifier_accepted": accepted,
        "trainable": trainable,
        "steps": steps,
        "error": event.get("error") if isinstance(event, dict) else None,
        "label_worker": (
            final_worker.get("agent_id") if trainable and final_worker is not None else None
        ),
        "label_role": 0 if trainable else None,
    }


def _write_learning_record(run: NativeRun, event: dict[str, Any]) -> None:
    if not _learning_enabled():
        return
    with serve_config._learning_lock:
        if run.learning_logged:
            return
        path = _learning_path()
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
        os.chmod(path, 0o600)
        with os.fdopen(descriptor, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(_learning_record(run, event), ensure_ascii=False) + "\n")
        run.learning_logged = True


def _redis() -> Any:
    if serve_config._redis_client is None:
        if not REDIS_URL:
            raise RuntimeError("MANTIS_REDIS_URL is required when MANTIS_RUN_STORE=redis")
        try:
            import redis
        except ImportError as error:
            raise RuntimeError("install the redis extra to use MANTIS_RUN_STORE=redis") from error
        serve_config._redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=False)
    return serve_config._redis_client


def _redis_key(run_id: str) -> str:
    return f"{_REDIS_PREFIX}{run_id}"


def _redis_index_key() -> str:
    return f"{_REDIS_PREFIX}index"


def _redis_get(run_id: str) -> NativeRun | None:
    raw = _redis().get(_redis_key(run_id))
    return pickle.loads(raw) if raw else None  # noqa: S301 - Redis is a trusted deployment dependency


def _redis_put(run: NativeRun) -> None:
    client = _redis()
    payload = pickle.dumps(run, protocol=pickle.HIGHEST_PROTOCOL)
    ttl = max(serve_config.RUN_TTL, REDIS_LOCK_TIMEOUT) if run.in_flight else serve_config.RUN_TTL
    client.setex(_redis_key(run.run_id), max(1, int(ttl)), payload)
    client.sadd(_redis_index_key(), run.run_id)
    client.expire(_redis_index_key(), max(1, int(serve_config.RUN_TTL)))


@contextmanager
def _redis_run_lock(run_id: str):
    lock = _redis().lock(f"{_REDIS_PREFIX}lock:{run_id}", timeout=REDIS_LOCK_TIMEOUT)
    acquired = lock.acquire(blocking=True, blocking_timeout=REDIS_LOCK_TIMEOUT)
    if not acquired:
        raise providers.RunCapacityError("Mantis run is busy")
    try:
        yield
    finally:
        lock.release()


@contextmanager
def _redis_registry_lock():
    lock = _redis().lock(f"{_REDIS_PREFIX}registry-lock", timeout=60)
    acquired = lock.acquire(blocking=True, blocking_timeout=60)
    if not acquired:
        raise providers.RunCapacityError("Mantis tool-run registry is busy")
    try:
        yield
    finally:
        lock.release()


def _record_abandoned(run: NativeRun | None) -> None:
    """Write a learning record for a run that expired before completing."""
    if run is None or not _learning_enabled():
        return
    if not (getattr(run, "turns", None) or getattr(run, "steps", None)):
        return  # never progressed; nothing to learn from
    _write_learning_record(run, {"type": "error", "terminated_by": "abandoned"})


def _sweep_runs() -> None:
    if RUN_STORE == "redis":
        client = _redis()
        now = time.time()
        for raw_id in client.smembers(_redis_index_key()):
            run_id = raw_id.decode() if isinstance(raw_id, bytes) else str(raw_id)
            run = _redis_get(run_id)
            if run is None:
                client.srem(_redis_index_key(), run_id)
                continue
            if run.in_flight == 0 and now - run.last_active > serve_config.RUN_TTL:
                _record_abandoned(run)
                client.delete(_redis_key(run_id))
                client.srem(_redis_index_key(), run_id)
        return
    now = time.time()
    stale = [
        rid
        for rid, run in serve_config._runs.items()
        if run.in_flight == 0 and now - run.last_active > serve_config.RUN_TTL
    ]
    for rid in stale:
        run = serve_config._runs.pop(rid, None)
        if run is not None:
            _record_abandoned(run)
            run.close()


def _ensure_runs_sweeper() -> None:
    if serve_config._runs_sweeper_started:
        return
    serve_config._runs_sweeper_started = True

    def _loop() -> None:
        while True:
            time.sleep(serve_config.RUN_TTL / 2 if serve_config.RUN_TTL > 0 else 60)
            with serve_config._runs_lock:
                _sweep_runs()

    threading.Thread(target=_loop, daemon=True).start()


def _register_run(run: NativeRun) -> str:
    if RUN_STORE == "redis":
        with _redis_registry_lock():
            client = _redis()
            if _redis_get(run.run_id) is not None:
                raise ValueError("run id already exists")
            if client.scard(_redis_index_key()) >= MAX_RUNS:
                _sweep_runs()
            if client.scard(_redis_index_key()) >= MAX_RUNS:
                raise providers.RunCapacityError("Mantis tool-run capacity is full")
            _redis_put(run)
        return cast(str, run.run_id)
    with serve_config._runs_lock:
        _ensure_runs_sweeper()
        if run.run_id in serve_config._runs:
            raise ValueError("run id already exists")
        if len(serve_config._runs) >= MAX_RUNS:
            _sweep_runs()
        if len(serve_config._runs) >= MAX_RUNS:
            raise providers.RunCapacityError("Mantis tool-run capacity is full")
        serve_config._runs[run.run_id] = run
    return cast(str, run.run_id)


class NativeRun:
    """Base for a resumable orchestration run."""

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self.created = time.time()
        self.last_active = time.time()
        self.cancelled = False
        self.finished = False
        self.final_text = ""
        self.terminated_by: str | None = None
        self.kind = "run"
        self.lock = threading.Lock()
        self.request_lock = threading.Lock()
        self.request_events: dict[str, dict[str, Any]] = {}
        self.tool_observations: list[dict[str, Any]] = []
        self.learning_logged = False
        self.in_flight = 0
        self.tool_choice: Any = None
        self.active_tool_choice: Any = None
        self.response_format: dict[str, Any] | None = None
        self.active_response_format: dict[str, Any] | None = None
        self.controls: dict[str, Any] = {}
        self.active_controls: dict[str, Any] = {}
        self.capture_metadata = False
        self.response_metadata: dict[str, Any] = {}
        self.usage: dict[str, Any] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        self.usage_models: dict[str, dict[str, int]] = {}
        self._activity: list[dict[str, Any]] = []
        self._started_monotonic = time.monotonic()
        self._next_tool_call = 0
        self.cache_namespace = ""

    def own_tool_calls(self, calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        owned: list[dict[str, Any]] = []
        for call in calls:
            owned_id = f"c{self._next_tool_call:x}"
            updated = {**call, "id": owned_id}
            # Accept legacy call-attached metadata while keeping provider IDs on
            # each call in the canonical representation.
            legacy = updated.pop("_message_metadata", None)
            if isinstance(legacy, dict) and "_assistant_metadata" not in updated:
                updated["_assistant_metadata"] = {
                    key: value for key, value in legacy.items() if key != "_anthropic_tool_ids"
                }
                raw_ids = legacy.get("_anthropic_tool_ids")
                if isinstance(raw_ids, dict):
                    updated["_anthropic_tool_id"] = raw_ids.get(call.get("id"), call.get("id"))
            provider_id = updated.pop("_anthropic_tool_id", None)
            if isinstance(provider_id, str) and provider_id:
                updated["_anthropic_tool_id"] = provider_id
            owned.append(updated)
            self._next_tool_call += 1
        return owned

    def record_activity(
        self,
        activity_type: str,
        *,
        role: str | None = None,
        model: str | None = None,
        status: str = "completed",
        duration_ms: float | None = None,
        summary: str | None = None,
        attempt: int | None = None,
        detail: str | None = None,
    ) -> None:
        entry: dict[str, Any] = {
            "type": activity_type,
            "role": role,
            "model": model,
            "status": status,
            "summary": summary or providers._activity_summary(activity_type, role),
        }
        if detail is not None:
            entry["error"] = detail
        if duration_ms is not None:
            entry["duration_ms"] = round(duration_ms, 1)
        if attempt is not None:
            entry["attempt"] = attempt
        self._activity.append(entry)
        providers._emit_progress(
            {"run_id": self.run_id, **{k: v for k, v in entry.items() if k != "error"}}
        )

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state.pop("lock", None)
        state.pop("request_lock", None)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self.lock = threading.Lock()
        self.request_lock = threading.Lock()
        self.request_events = getattr(self, "request_events", {})

    def add_usage(self, usage: Any, model: str | None = None) -> None:
        if not isinstance(usage, dict):
            return
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = usage.get(key)
            if isinstance(value, int) and value >= 0:
                self.usage[key] += value
        details = usage.get("completion_tokens_details")
        if isinstance(details, dict):
            target = self.usage.setdefault("completion_tokens_details", {})
            for key, value in details.items():
                if isinstance(value, int) and value >= 0:
                    target[key] = target.get(key, 0) + value
        if model:
            model_usage = self.usage_models.setdefault(
                model, {"prompt_tokens": 0, "completion_tokens": 0}
            )
            for key in ("prompt_tokens", "completion_tokens"):
                value = usage.get(key)
                if isinstance(value, int) and value >= 0:
                    model_usage[key] += value
        prompt_details = usage.get("prompt_tokens_details")
        if model and isinstance(prompt_details, dict):
            cached = prompt_details.get("cached_tokens")
            if isinstance(cached, int) and cached > 0:
                model_usage = self.usage_models.setdefault(
                    model, {"prompt_tokens": 0, "completion_tokens": 0}
                )
                model_usage["cached_tokens"] = model_usage.get("cached_tokens", 0) + cached

    def validate_output(self, text: str) -> None:
        if not self.response_format:
            return
        try:
            value = json.loads(text)
        except json.JSONDecodeError as error:
            raise ValueError(f"structured output is not valid JSON: {error.msg}") from error
        if self.response_format.get("type") == "json_schema":
            schema = self.response_format.get("json_schema", {}).get("schema", {})
            try:
                jsonschema.validate(value, schema)
            except jsonschema.ValidationError as error:
                message = f"structured output does not match schema: {error.message}"
                raise ValueError(message) from error

    def touch(self) -> None:
        self.last_active = time.time()

    def advance(self, tool_results: Any) -> dict[str, Any]:
        raise NotImplementedError

    def advance_idempotent(self, tool_results: Any, request_id: Any = None) -> dict[str, Any]:
        if request_id is None:
            return self.advance(tool_results)
        if not isinstance(request_id, str) or not request_id or len(request_id) > 128:
            raise ValueError("request_id must be a string of at most 128 characters")
        with self.request_lock:
            cached = self.request_events.get(request_id)
            if cached is not None:
                return cached
            event = self.advance(tool_results)
            self.request_events[request_id] = event
            while len(self.request_events) > 64:
                self.request_events.pop(next(iter(self.request_events)))
            return event

    def record_tool_results(
        self, pending: dict[str, Any], tool_results: list[dict[str, Any]]
    ) -> None:
        for result in tool_results:
            call = _tool_call_by_id(pending, str(result.get("tool_call_id", ""))) or {}
            function = call.get("function", {})
            try:
                arguments = json.loads(function.get("arguments", "{}"))
            except (json.JSONDecodeError, TypeError, ValueError):
                arguments = {}
            command = (
                str(arguments.get("command") or arguments.get("code") or "")
                if isinstance(arguments, dict)
                else ""
            )
            self.tool_observations.append(
                {
                    "name": str(function.get("name", "")),
                    "is_error": bool(result.get("is_error", False)),
                    "is_test": bool(
                        str(function.get("name", "")).lower() in _TEST_TOOL_NAMES
                        and _TEST_COMMAND.search(command)
                    ),
                }
            )

    def close(self) -> None:
        self.cancelled = True


class TrinityRun(NativeRun):
    """Resumable TRINITY loop with native tool support.

    Replicates Coordinator semantics (role sampling, Thinker suggestion,
    Verifier accept/reject, cold-verifier -> Worker, empty-response recovery,
    multi-turn history) but lets each role's model call client-provided tools before its text
    reply finalizes."""

    def __init__(
        self,
        run_id: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        slot_models: list[str] | None = None,
        max_turns: int = serve_config.MAX_TURNS,
    ) -> None:
        super().__init__(run_id)
        self.kind = "trinity"
        self.tools = tools
        self.slot_models = slot_models or list(DEFAULT_SLOT_LABELS)
        self.max_turns = max_turns
        query, history = utils._split_messages(messages)
        self.query = query or ""
        self.query_content = utils._last_user_content(messages)
        self.history = history
        self.obs = self.query
        self.ref_id = 0
        self.last_response: str | None = None
        self.suggestion: str | None = None
        self.suggested_role: str | None = None
        self.force_worker = False
        self.revision_feedback: str | None = None
        self.turns: list[dict[str, Any]] = []
        self._pending: dict[str, Any] | None = None
        self._expected_ids: set[str] = set()
        self._tool_rounds = 0

    def _model_name(self, agent_id: int) -> str:
        return str(self.slot_models[agent_id % len(self.slot_models)])

    def _route(self) -> tuple[str, int]:
        msgs = [
            {
                "role": "system",
                "content": ROUTER_SYSTEM_PROMPT.format(num_agents=len(self.slot_models)),
            },
            {"role": "user", "content": self.obs},
        ]
        r = trinity.get_router().route(msgs, sample=True)
        role = r["role_name"]
        if self.suggested_role:
            role, self.suggested_role = self.suggested_role, None
        if self.force_worker:
            self.force_worker = False
            role = "Worker"
        if role == "Verifier" and self.last_response is None:
            role = "Worker"  # nothing to verify yet [FC]
        if role == "Thinker" and self.last_response is None:
            role = "Worker"  # a Thinker with no response to reason about is noise
        return role, int(r["agent_id"])

    def _role_prompt(self, role: str) -> str:
        if role == "Thinker":
            info = self.query
            if self.last_response:
                info += f"\n\nCurrent response:\n{self.last_response}"
            return cast(str, THINKER_PROMPT.format(info=info))
        if role == "Verifier":
            vp = VERIFICATION_PROMPT.format(query=self.query, response=self.last_response or "")
            if self.suggestion:
                vp += (
                    f"These are useful suggestions when drafting your response:\n"
                    f"<suggestion>{self.suggestion}</suggestion>"
                )
            return cast(str, vp)
        content = self.query
        if self.suggestion:
            content += (
                f"when drafting your response, thinking of following:\n"
                f"<suggestion>{self.suggestion}</suggestion>"
            )
        return cast(str, content)

    def _build_messages(self, role: str) -> list[dict[str, Any]]:
        prior_sys = "\n\n".join(
            str(m.get("content", ""))
            for m in self.history
            if isinstance(m, dict) and m.get("role") == "system" and m.get("content")
        )
        sys_content = SYSTEM_PROMPT
        if prior_sys:
            sys_content = f"{sys_content}\n\n{prior_sys}"
        prior = [m for m in self.history if isinstance(m, dict) and m.get("role") != "system"]
        msgs: list[dict[str, Any]] = [{"role": "system", "content": sys_content}]
        msgs.extend(prior)
        user_content = self._role_prompt(role)
        if role == "Worker" and self.revision_feedback:
            user_content = (
                f"{user_content}\n\n"
                f"Revise the answer to address this verifier feedback:\n{self.revision_feedback}"
            )
            self.revision_feedback = None
        msgs.append(
            {
                "role": "user",
                "content": utils._with_images(user_content, self.query_content)
                if role == "Worker"
                else user_content,
            }
        )
        return msgs

    def _role_complete(self, role: str, agent_id: int, turn: int, messages: list, reply: str):
        if role == "Worker":
            self.last_response = reply
            self.suggestion = None
            thought = self._extract_thought(reply)
            if thought:
                self.obs += (
                    f"\n<reference_thought_{self.ref_id}>{thought}"
                    f"</reference_thought_{self.ref_id}>"
                )
                self.ref_id += 1
        elif role == "Thinker":
            self.suggested_role, self.suggestion = self._parse_thinker(reply)
        elif role == "Verifier":
            self.suggestion = None
            if self._parse_verification(reply):
                self.terminated_by = "verifier_accept"
                self.final_text = self.last_response or reply
                self.record_activity(
                    "verify_accept",
                    role=role,
                    model=self._model_name(agent_id),
                    summary="Verifier accepted the draft",
                )
        if role == "Worker" and not reply.strip():
            self.force_worker = True
            nope = "produce a complete answer."
            self.revision_feedback = f"Previous worker returned no response; {nope}"
        elif role == "Verifier" and reply.strip().upper().startswith("REJECT"):
            self.force_worker = True
            self.revision_feedback = reply
            self.record_activity(
                "verify_reject",
                role=role,
                model=self._model_name(agent_id),
                summary="Verifier requested a revision",
            )
        step = {
            "turn": turn,
            "role": role,
            "agent_id": agent_id,
            "model_name": self._model_name(agent_id),
            "prompt": messages[-1]["content"] if messages else "",
            "reply": reply,
        }
        self.turns.append(step)
        self._pending = None
        self._expected_ids = set()
        self._tool_rounds = 0
        return {"type": "step_complete", **step}

    def _run_model(self, role: str, agent_id: int, turn: int, messages: list):
        model = self._model_name(agent_id)
        self.active_response_format = self.response_format if role == "Worker" else None
        self.active_tool_choice = (
            self.tool_choice if role == "Worker" and self._tool_rounds == 0 else None
        )
        self.active_controls = self.controls if role == "Worker" else {}
        self.capture_metadata = role == "Worker"
        started = time.monotonic()
        self.record_activity(
            "step",
            role=role,
            model=model,
            status="started",
            summary=providers._running_summary(role),
        )
        try:
            text, calls = utils._model_completion(model, messages, self.tools)
            calls = self.own_tool_calls(calls)
            duration_ms = (time.monotonic() - started) * 1000.0
        except Exception as error:
            self.record_activity(
                "step",
                role=role,
                model=model,
                status="failed",
                duration_ms=(time.monotonic() - started) * 1000.0,
                detail=str(error)[:300],
            )
            raise
        finally:
            self.active_response_format = None
            self.active_tool_choice = None
            self.active_controls = {}
            self.capture_metadata = False
        self.record_activity(
            "step",
            role=role,
            model=model,
            duration_ms=duration_ms,
        )
        if calls:
            self.record_activity("tool_call", role=role, model=model)
            message_metadata = calls[0].pop("_assistant_metadata", {})
            provider_ids = {
                c["id"]: c.pop("_anthropic_tool_id")
                for c in calls
                if isinstance(c.get("_anthropic_tool_id"), str)
            }
            asst: dict[str, Any] = {
                "role": "assistant",
                "content": text,
                "tool_calls": [
                    utils._openai_tool_call(c["name"], c["id"], c["arguments"]) for c in calls
                ],
            }
            asst.update(message_metadata)
            if provider_ids:
                asst["_anthropic_tool_ids"] = provider_ids
            self._pending = {
                "role": role,
                "agent_id": agent_id,
                "turn": turn,
                "messages": messages,
                "asst": asst,
            }
            self._expected_ids = {c["id"] for c in calls}
            self._tool_rounds = 1
            return {
                "type": "tool_calls",
                "role": role,
                "agent_id": agent_id,
                "model_name": model,
                "turn": turn,
                "tool_calls": calls,
            }
        return self._role_complete(role, agent_id, turn, messages, text)

    def _apply_tool_results(self, tool_results: list[dict[str, Any]]) -> None:
        pending = self._pending
        if pending is None:
            raise RuntimeError("no pending tool call")  # noqa: TRY301
        tool_results = utils._validate_tool_results(tool_results, self._expected_ids)
        self.record_tool_results(pending, tool_results)
        failed = any(bool(result.get("is_error")) for result in tool_results)
        self.record_activity(
            "tool_result",
            role=str(pending.get("role") or ""),
            model=str(pending.get("model") or pending.get("model_name") or ""),
            status="failed" if failed else "completed",
        )
        self._tool_rounds += 1
        if self._tool_rounds > MAX_TOOL_ROUNDS:
            raise ValueError(f"exceeded max tool rounds per step ({MAX_TOOL_ROUNDS})")
        pending["messages"].append(pending["asst"])
        for r in tool_results:
            pending["messages"].append(
                {
                    "role": "tool",
                    "tool_call_id": r.get("tool_call_id"),
                    "content": str(r.get("content", ""))[:RUN_MAX_MSG_BYTES],
                }
            )
        self._expected_ids = set()
        self._pending = pending

    def advance(self, tool_results: Any) -> dict[str, Any]:
        with self.lock:
            self.touch()
            if self.cancelled:
                return {"type": "error", "error": "run cancelled"}
            if self.finished:
                return {"type": "error", "error": "run already finished"}
            try:
                if self._pending is None and tool_results is not None:
                    return {"type": "error", "error": "unexpected tool results"}
                if self._pending is not None:
                    if tool_results is None:
                        return {"type": "error", "error": "missing tool results"}
                    self._apply_tool_results(tool_results)
                    p = self._pending
                    if p is None:
                        raise RuntimeError("no pending tool call")  # noqa: TRY301
                    return cast(
                        dict[str, Any],
                        self._run_model(p["role"], p["agent_id"], p["turn"], p["messages"]),
                    )

                # Finish conditions before starting a new coordinator turn.
                if self.terminated_by is not None:
                    self.finished = True
                    return self._final()
                if len(self.turns) >= self.max_turns:
                    self.terminated_by = "max_turns"
                    self.finished = True
                    return self._final()

                turn = len(self.turns)
                role, agent_id = self._route()
                messages = self._build_messages(role)
                return cast(dict[str, Any], self._run_model(role, agent_id, turn, messages))
            except ValueError as e:
                return {"type": "error", "error": str(e)}
            except Exception as e:  # noqa: BLE001 - model/network failures abort the run
                self.close()
                return {"type": "error", "error": str(e)}

    def _final(self) -> dict[str, Any]:
        text = self.final_text
        if not text:
            # Reasoning workers can return content:null when effort eats the
            # budget; fall back to the last non-empty reply so the client
            # never receives an empty final answer.
            for step in reversed(self.turns):
                if step.get("reply", "").strip():
                    text = step["reply"]
                    break
        return {
            "type": "final",
            "text": text,
            "terminated_by": self.terminated_by or "",
            "steps": self.turns,
        }

    @staticmethod
    def _extract_thought(reply: str) -> str:
        return reply.strip()

    @staticmethod
    def _parse_thinker(text: str):
        import re

        role = None
        m = re.search(
            r"<suggested_role>\s*(solver|thinker|verifier)\s*</suggested_role>",
            text,
            re.IGNORECASE,
        )
        if m:
            role = {"solver": "Worker", "thinker": "Thinker", "verifier": "Verifier"}[
                m.group(1).lower()
            ]
        sug = None
        s = re.search(r"<suggestion>\s*([\s\S]*?)\s*</suggestion>", text, re.IGNORECASE)
        if s:
            sug = s.group(1).strip() or None
        return role, sug

    @staticmethod
    def _parse_verification(text: str) -> bool:
        return text.strip().upper().startswith("ACCEPT")


class ConductorRun(NativeRun):
    """Resumable Conductor run: planning step then DAG nodes, all tool-capable."""

    def __init__(
        self,
        run_id: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        slot_models: list[str] | None = None,
        max_steps: int = 5,
    ) -> None:
        super().__init__(run_id)
        self.kind = "conductor"
        self.tools = tools
        self.slot_models = slot_models or list(DEFAULT_SLOT_LABELS)
        self.max_steps = max_steps
        query, history = utils._split_messages(messages)
        self.query = query or ""
        self.query_content = utils._last_user_content(messages)
        self.history = history
        self.conductor_model = conductor._resolve_conductor_model(
            SimpleNamespace(slot_models=self.slot_models)
        )
        self.steps: list[dict[str, Any]] = []
        self._workflow: tuple[list, list, list] | None = None
        self._outputs: list[str] = []
        self._next_node = 0
        self._pending: dict[str, Any] | None = None
        self._expected_ids: set[str] = set()
        self._tool_rounds = 0

    def _planner_messages(self) -> list[dict[str, Any]]:
        prior = [m for m in self.history if isinstance(m, dict) and m.get("role") != "system"]
        return cast(
            list[dict[str, Any]],
            conductor_prompt(self.query, self.slot_models) + prior[-4:],
        )

    def _node_messages(self, node_index: int, mid: int, sub: str) -> list[dict[str, Any]]:
        if self._workflow is None:
            raise ValueError("workflow required")
        sees = visible_indices(self._workflow[2], node_index)
        mids = self._workflow[0]
        subs = self._workflow[1]
        ctx = ""
        for j in sees:
            prev_mid = mids[j]
            ctx += (
                f"\n<Subtask assigned to Agent {prev_mid}>{subs[j]}"
                f"</Subtask assigned to Agent {prev_mid}>"
                f"\n<Agent {prev_mid} response>{self._outputs[j].strip()}"
                f"</Agent {prev_mid} response>"
            )
        user = (
            f"Relevant completed subtasks:\n{ctx}\n\nYour subtask: {sub}"
            if ctx
            else f"Your subtask: {sub}"
        )
        return [
            {
                "role": "system",
                "content": "Complete the assigned subtask in the context of the original request.",
            },
            {
                "role": "user",
                "content": utils._with_images(
                    f"Original request:\n{self.query}", self.query_content
                ),
            },
            {"role": "user", "content": user},
        ]

    def _run_model(self, role: str, model: str, messages: list) -> dict[str, Any]:
        seq = len(self.steps)
        is_final_worker = bool(
            role == "Worker"
            and self._workflow is not None
            and self._next_node >= len(self._workflow[1])
        )
        self.active_response_format = self.response_format if is_final_worker else None
        self.active_tool_choice = (
            self.tool_choice if role == "Worker" and self._tool_rounds == 0 else None
        )
        self.active_controls = self.controls if role == "Worker" else {}
        self.capture_metadata = is_final_worker
        started = time.monotonic()
        self.record_activity(
            "step",
            role=role,
            model=model,
            status="started",
            summary=providers._running_summary(role),
        )
        try:
            text, calls = utils._model_completion(model, messages, self.tools)
            calls = self.own_tool_calls(calls)
            duration_ms = (time.monotonic() - started) * 1000.0
        except Exception as error:
            self.record_activity(
                "step",
                role=role,
                model=model,
                status="failed",
                duration_ms=(time.monotonic() - started) * 1000.0,
                detail=str(error)[:300],
            )
            raise
        finally:
            self.active_response_format = None
            self.active_tool_choice = None
            self.active_controls = {}
            self.capture_metadata = False
        self.record_activity(
            "step",
            role=role,
            model=model,
            duration_ms=duration_ms,
        )
        if calls:
            self.record_activity("tool_call", role=role, model=model)
            message_metadata = calls[0].pop("_assistant_metadata", {})
            provider_ids = {
                c["id"]: c.pop("_anthropic_tool_id")
                for c in calls
                if isinstance(c.get("_anthropic_tool_id"), str)
            }
            asst = {
                "role": "assistant",
                "content": text,
                "tool_calls": [
                    utils._openai_tool_call(c["name"], c["id"], c["arguments"]) for c in calls
                ],
            }
            asst.update(message_metadata)
            if provider_ids:
                asst["_anthropic_tool_ids"] = provider_ids
            self._pending = {"role": role, "model": model, "messages": messages, "asst": asst}
            self._expected_ids = {c["id"] for c in calls}
            self._tool_rounds = 1
            return {
                "type": "tool_calls",
                "role": role,
                "model_name": model,
                "turn": seq,
                "tool_calls": calls,
            }
        return self._finalize_text(role, text, seq)

    def _finalize_text(self, role: str, text: str, seq: int) -> dict[str, Any]:
        self._pending = None
        self._expected_ids = set()
        self._tool_rounds = 0
        if role == "Planner":
            try:
                self._workflow = parse_workflow(text)
            except Exception as e:  # noqa: BLE001
                return {
                    "type": "error",
                    "error": f"Conductor did not emit a parseable workflow: {e}",
                }
            mids, subs, access = self._workflow
            if not subs or not (len(mids) == len(subs) == len(access)):
                return {
                    "type": "error",
                    "error": "Conductor emitted an empty or malformed workflow",
                }
            self.steps.append(
                {
                    "turn": seq,
                    "role": "Planner",
                    "agent_id": 0,
                    "model_name": self.conductor_model,
                    "prompt": self.query,
                    "reply": text,
                }
            )
            return {
                "type": "step_complete",
                "turn": seq,
                "role": "Planner",
                "agent_id": 0,
                "model_name": self.conductor_model,
                "prompt": self.query,
                "reply": text,
            }

        node_index = self._next_node - 1
        if self._workflow is None:
            raise ValueError("workflow required")
        mids = self._workflow[0]
        subs = self._workflow[1]
        mid = int(mids[node_index]) % len(self.slot_models)
        self._outputs.append(text)
        self.steps.append(
            {
                "turn": seq,
                "role": "Worker",
                "agent_id": mid,
                "model_name": self.slot_models[mid],
                "prompt": subs[node_index],
                "reply": text,
            }
        )
        return {
            "type": "step_complete",
            "turn": seq,
            "role": "Worker",
            "agent_id": mid,
            "model_name": self.slot_models[mid],
            "prompt": subs[node_index],
            "reply": text,
        }

    def _apply_tool_results(self, tool_results: list[dict[str, Any]]) -> None:
        pending = self._pending
        if pending is None:
            raise RuntimeError("no pending tool call")  # noqa: TRY301
        tool_results = utils._validate_tool_results(tool_results, self._expected_ids)
        self.record_tool_results(pending, tool_results)
        failed = any(bool(result.get("is_error")) for result in tool_results)
        self.record_activity(
            "tool_result",
            role=str(pending.get("role") or ""),
            model=str(pending.get("model") or pending.get("model_name") or ""),
            status="failed" if failed else "completed",
        )
        self._tool_rounds += 1
        if self._tool_rounds > MAX_TOOL_ROUNDS:
            raise ValueError(f"exceeded max tool rounds per step ({MAX_TOOL_ROUNDS})")
        pending["messages"].append(pending["asst"])
        for r in tool_results:
            pending["messages"].append(
                {
                    "role": "tool",
                    "tool_call_id": r.get("tool_call_id"),
                    "content": str(r.get("content", ""))[:RUN_MAX_MSG_BYTES],
                }
            )
        self._expected_ids = set()
        self._pending = pending

    def advance(self, tool_results: Any) -> dict[str, Any]:
        with self.lock:
            self.touch()
            if self.cancelled:
                return {"type": "error", "error": "run cancelled"}
            if self.finished:
                return {"type": "error", "error": "run already finished"}
            try:
                if self._pending is None and tool_results is not None:
                    return {"type": "error", "error": "unexpected tool results"}
                if self._pending is not None:
                    if tool_results is None:
                        return {"type": "error", "error": "missing tool results"}
                    self._apply_tool_results(tool_results)
                    p = self._pending
                    if p is None:
                        raise RuntimeError("no pending tool call")  # noqa: TRY301
                    return cast(
                        dict[str, Any], self._run_model(p["role"], p["model"], p["messages"])
                    )

                if self._workflow is None:
                    return self._run_model(
                        "Planner", self.conductor_model, self._planner_messages()
                    )
                mids, subs, access = self._workflow
                if self._next_node >= len(subs):
                    self.finished = True
                    self.terminated_by = "conductor_done"
                    return self._final_conductor("conductor_done")
                if self._next_node >= self.max_steps:
                    self.finished = True
                    self.terminated_by = "max_steps"
                    return self._final_conductor("max_steps")
                node_index = self._next_node
                self._next_node += 1
                mid = int(mids[node_index]) % len(self.slot_models)
                model = self.slot_models[mid]
                return self._run_model(
                    "Worker", model, self._node_messages(node_index, mid, subs[node_index])
                )
            except ValueError as e:
                return {"type": "error", "error": str(e)}
            except Exception as e:  # noqa: BLE001
                self.close()
                return {"type": "error", "error": str(e)}

    def close(self) -> None:
        self.cancelled = True

    def _final_conductor(self, terminated_by: str) -> dict[str, Any]:
        text = ""
        for out in reversed(self._outputs):
            if str(out or "").strip():
                text = str(out)
                break
        self.final_text = text
        return {
            "type": "final",
            "text": text,
            "terminated_by": terminated_by,
            "steps": self.steps,
        }


def create_run(mode: str, body: dict[str, Any]) -> NativeRun:
    messages = body.get("messages") or []
    if (
        not isinstance(messages, list)
        or not messages
        or not all(isinstance(message, dict) for message in messages)
    ):
        raise ValueError("messages must be a non-empty list of objects")
    tools = utils._convert_tools(body.get("tools"))
    slot_models = utils._configured_slot_models(body.get("slot_models"))
    requested_id = body.get("run_id")
    if requested_id is not None and (
        not isinstance(requested_id, str)
        or len(requested_id) != 32
        or any(char not in "0123456789abcdef" for char in requested_id.lower())
    ):
        raise ValueError("run_id must be a 32-character hexadecimal string")
    run_id = requested_id or uuid.uuid4().hex
    run: NativeRun
    if mode == "conductor":
        run = ConductorRun(run_id, messages, tools, slot_models=slot_models)
    else:
        run = TrinityRun(run_id, messages, tools, slot_models=slot_models)
    run.tool_choice = body.get("tool_choice")
    run.response_format = body.get("response_format")
    run.cache_namespace = providers._prompt_cache_namespace(messages, tools)
    output_limit = body.get("max_completion_tokens", body.get("max_tokens"))
    if output_limit is not None:
        run.controls["max_tokens"] = min(int(output_limit), providers.upstream_output_token_cap())
    if body.get("reasoning"):
        run.controls["reasoning"] = body["reasoning"]
    elif body.get("reasoning_effort") is not None:
        run.controls["reasoning_effort"] = body["reasoning_effort"]
    if body.get("web_search_options") is not None:
        run.controls["web_search_options"] = body["web_search_options"]
    run.record_activity("run", status="started", summary=f"Started Mantis {mode} orchestration")
    _register_run(run)
    return run


def get_run(run_id: str) -> NativeRun:
    run = _redis_get(run_id) if RUN_STORE == "redis" else serve_config._runs.get(run_id)
    if run is None:
        raise KeyError(f"unknown or expired run: {run_id}")
    return run


def advance_run(run_id: str, tool_results: Any, request_id: Any = None) -> dict[str, Any]:
    lock = _redis_run_lock(run_id) if RUN_STORE == "redis" else nullcontext()
    with lock:
        run = get_run(run_id)
        run.in_flight += 1
        if RUN_STORE == "redis":
            _redis_put(run)
        try:
            serve_config._history_context.active_run = run
            event = run.advance_idempotent(tool_results, request_id)
            if event.get("type") in ("final", "error"):
                _write_learning_record(run, event)
            return event
        finally:
            serve_config._history_context.active_run = None
            run.in_flight -= 1
            run.touch()
            if RUN_STORE == "redis":
                _redis_put(run)


def delete_run(run_id: str, error: str | None = None) -> bool:
    if RUN_STORE == "redis":
        with _redis_run_lock(run_id):
            run = _redis_get(run_id)
            if run is None:
                return False
            _redis().delete(_redis_key(run_id))
            _redis().srem(_redis_index_key(), run_id)
    else:
        with serve_config._runs_lock:
            run = serve_config._runs.pop(run_id, None)
        if run is None:
            return False
    _write_learning_record(run, {"type": "error", "terminated_by": "deleted", "error": error})
    run.close()
    return True


__all__ = [k for k in globals() if not k.startswith("__")]
