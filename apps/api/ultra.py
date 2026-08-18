#!/usr/bin/env python3
# OpenFugu — Apache-2.0. Part of an independent, open reimplementation of
# the Fugu orchestrator. NOT affiliated with Sakana AI. See NOTICE.
# Reference: Learning to Orchestrate Agents in NL with the Conductor
# (arXiv:2512.04388, Sakana AI). Independent reimplementation of the workflow
# DAG executor from the paper.
"""
fugu_ultra.py — a faithful, runnable reconstruction of Sakana Fugu-Ultra's
Conductor line: instead of routing one worker per turn (that's fugu_mini.py /
TRINITY), a Conductor LM emits an ENTIRE agentic workflow in one shot — three
equal-length lists (model_id / subtasks / access_list) forming a DAG over a
worker pool — which is then executed in topological order.

Provenance, stated honestly:
  [EXEC]  the execution engine — 3-list parse, DAG order, access-list visibility
          injection — is a faithful reimplementation of the TRINITY/Conductor
          authors' conductor_engine.py + conductor_utils.py.
  [DOC]   the GRPO-trained 7B Conductor weights are NOT public, so here the
          Conductor is a *prompted off-the-shelf model*. The Conductor paper's
          own claim is that prompting works (just below the RL-optimized model);
          this reproduces the mechanism, not the trained policy.
"""

from __future__ import annotations

import ast
import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

N_AGENTS = 7
MAX_STEPS = 5  # [DOC] Conductor workflows up to 5 steps
PLANNER_PREFILL = "Plan:\n"
_SMART = str.maketrans("“”‘’", "\"\"''")

DEFAULT_SLOT_LABELS = [  # [DATA] training metadata; remappable to any provider
    "gpt-5",
    "claude-sonnet-4",
    "gemini-2.5-pro",
    "deepseek-r1-distill-qwen-32b",
    "gemma-3-27b-it",
    "qwen3-32b-reasoning",
    "qwen3-32b-direct",
]


# ---- 3-list parsing (faithful to conductor_utils._extract_any) [EXEC] --------
def _balanced_list(after: str) -> str | None:
    """Extract the first balanced [...] list, respecting quotes/escapes."""
    depth = 0
    start = None
    q = None
    esc = False
    for i, ch in enumerate(after):
        if esc:
            esc = False
            continue
        if ch == "\\":
            esc = True
            continue
        if q:
            if ch == q:
                q = None
            continue
        if ch in "\"'":
            q = ch
            continue
        if ch == "[":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0 and start is not None:
                return after[start : i + 1]
    return None


def extract_list(text: str, labels: list[str]) -> list[Any]:
    """Find 'label: [ ... ]' then parse via ast -> json -> CSV fallback. [EXEC]"""
    tag = "|".join(re.escape(label) for label in labels)
    m = re.search(rf"({tag})\s*[:=]\s*", text, re.IGNORECASE)
    if not m:
        return []
    raw = _balanced_list(text[m.end() :])
    if not raw:
        return []
    raw = raw.translate(_SMART).strip()
    try:
        return list(ast.literal_eval(raw))
    except (SyntaxError, ValueError):
        pass
    try:
        return list(json.loads(re.sub(r"'", '"', raw)))
    except (TypeError, ValueError, json.JSONDecodeError):
        pass
    items = [x.strip(" \"'") for x in raw.strip("[]").split(",") if x.strip()]
    return [int(x) if x.isdigit() else x for x in items]


def _parse_json_workflow(text: str) -> tuple[list, list, list] | None:
    stripped = text.strip()
    fenced = re.match(r"^```(?:json)?\s*\n?(.*?)\n?```\s*$", stripped, re.DOTALL | re.IGNORECASE)
    if fenced:
        stripped = fenced.group(1).strip()
    if not stripped.startswith("{"):
        return None
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    model_ids = data.get("model_id", data.get("model_ids"))
    subtasks = data.get("subtasks", data.get("subtask"))
    access = data.get("access_list", data.get("access"))
    if not (isinstance(model_ids, list) and isinstance(subtasks, list) and isinstance(access, list)):
        return None
    if not (model_ids and subtasks and access):
        return None
    return [int(x) for x in model_ids], list(subtasks), list(access)


def parse_workflow(text: str) -> tuple[list, list, list]:
    parsed = _parse_json_workflow(text)
    if parsed is not None:
        return parsed
    model_ids = extract_list(text, ["model_id", "model id", "model_ids", "model ids"])
    subtasks = extract_list(text, ["subtasks", "subtask"])
    access = extract_list(text, ["access_list", "access list", "access"])
    return model_ids, subtasks, access


def planner_response_format() -> dict[str, Any]:
    """Structured planner output: three equal-length workflow lists."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "conductor_workflow",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "model_id": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "minItems": 1,
                        "maxItems": MAX_STEPS,
                    },
                    "subtasks": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                        "maxItems": MAX_STEPS,
                    },
                    "access_list": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": MAX_STEPS,
                        "items": {
                            "anyOf": [
                                {
                                    "type": "array",
                                    "items": {"type": "integer"},
                                },
                                {"type": "string", "enum": ["all"]},
                            ],
                        },
                    },
                },
                "required": ["model_id", "subtasks", "access_list"],
                "additionalProperties": False,
            },
        },
    }


def _is_all(x) -> bool:
    return isinstance(x, str) and x.strip().lower() in ("all", "[all]", "'all'")


# EXEC_MARKER


# ---- access-list visibility (choose_position: indices of earlier steps) [EXEC]
def visible_indices(access_list: list, step: int) -> list[int]:
    """Which earlier step outputs are visible to `step`. Forward references are
    rejected (topological order). 'all' => every earlier step. Faithful to
    _ascribe_history_positional_complex / _ascribe_history_binary."""
    if step == 0:
        return []
    a = access_list[step] if step < len(access_list) else []
    if _is_all(a):
        return list(range(step))
    if a in ([], "", None):
        return []
    out = []
    for pos in dict.fromkeys(a if isinstance(a, (list, tuple)) else [a]):
        if not isinstance(pos, int):
            continue
        if pos >= step:  # forward reference -> reject [EXEC]
            raise ValueError(f"step {step} references future/own step {pos} (not a DAG order)")
        if 0 <= pos < step:
            out.append(pos)
    return sorted(out)


# ---- the Conductor prompt (prompted stand-in for the RL-trained 7B) [DOC] ----
def conductor_prompt(query: str, slot_labels: list[str]) -> list[dict]:
    pool = "\n".join(f"  {i}: {name}" for i, name in enumerate(slot_labels))
    sys = (
        "You are a Conductor that orchestrates a pool of worker LLMs to solve a task. "
        "Design an agentic workflow as THREE equal-length Python lists:\n"
        "  model_id   = [int, ...]   # which worker (0-indexed) runs each step\n"
        "  subtasks   = [str, ...]   # the natural-language instruction for each step\n"
        "  access_list= [list, ...]  # for each step, the indices of EARLIER steps whose\n"
        '                            # outputs that step may see ([] = none, may use "all")\n'
        "Rules: lists must be equal length (<=5 steps); access_list may only reference "
        "strictly earlier steps (it is a DAG executed in order); the LAST step's output "
        "is the final answer. Pick workers to match each subtask's demands: prefer cheaper "
        "workers for search, inspection, or reading; reserve frontier reasoning models for "
        "synthesis, complex debugging, and final code generation.\n\n"
        f"AVAILABLE LANGUAGE MODELS:\n{pool}\n\n"
        "Return a JSON object with keys model_id, subtasks, and access_list, or output "
        "the three lists explicitly as 'model_id: [...]', 'subtasks: [...]', "
        "'access_list: [...]'. You may reason first, but the three lists must appear."
    )
    return [
        {"role": "system", "content": sys},
        {"role": "user", "content": f"USER QUESTION: {query}"},
    ]


def planner_repair_messages(
    query: str,
    slot_labels: list[str],
    bad_text: str,
    error: str,
) -> list[dict]:
    """One-shot repair turn after a planner parse/validation failure."""
    pool = "\n".join(f"  {i}: {name}" for i, name in enumerate(slot_labels))
    sys = (
        "You are a Conductor repairing an invalid workflow plan. "
        "Return a JSON object with exactly three keys: model_id (array of ints), "
        "subtasks (array of strings), and access_list (array of int arrays or \"all\"). "
        "Lists must be equal length (1-5 steps); access_list may only reference earlier steps.\n\n"
        f"AVAILABLE LANGUAGE MODELS:\n{pool}"
    )
    clipped = bad_text.strip()[:8000] or "(empty)"
    return [
        {"role": "system", "content": sys},
        {"role": "user", "content": f"USER QUESTION: {query}"},
        {"role": "assistant", "content": clipped},
        {
            "role": "user",
            "content": (
                f"The previous workflow was invalid: {error}. "
                "Emit a corrected workflow as JSON with model_id, subtasks, and access_list."
            ),
        },
    ]


WorkerFn = Callable[[str, list, int], str]  # (subtask, messages, agent_id) -> reply


@dataclass
class Step:
    idx: int
    agent_id: int
    subtask: str
    sees: list[int]
    reply: str


@dataclass
class UltraResult:
    final: str
    steps: list[Step] = field(default_factory=list)
    workflow: dict = field(default_factory=dict)
    turns: list[Any] = field(default_factory=list)


class ConductorExecutor:
    """Parse a Conductor completion into a workflow DAG and execute it. [EXEC]

    Each step prompts its worker with its subtask plus the outputs of the steps
    named in access_list, injected as <Agent N response> blocks (the engine's
    exact context-assembly format)."""

    def __init__(self, worker: WorkerFn, slot_labels=None, max_steps=MAX_STEPS):
        self.worker = worker
        self.slot_labels = slot_labels or DEFAULT_SLOT_LABELS
        self.max_steps = max_steps

    def validate(self, model_ids, subtasks, access):
        if not (subtasks and model_ids and access):
            raise ValueError("workflow missing one of model_id/subtasks/access_list")
        if not (len(model_ids) == len(subtasks) == len(access)):
            raise ValueError(
                f"lists unequal length: {len(model_ids)}/{len(subtasks)}/{len(access)}"
            )
        if len(subtasks) > self.max_steps:
            subtasks, model_ids, access = (
                subtasks[: self.max_steps],
                model_ids[: self.max_steps],
                access[: self.max_steps],
            )
        for step, model_id in enumerate(model_ids):
            if not isinstance(model_id, int) or isinstance(model_id, bool):
                raise TypeError(f"step {step} has a non-integer model_id: {model_id!r}")
            if not 0 <= model_id < len(self.slot_labels):
                raise ValueError(f"step {step} has an out-of-range model_id: {model_id}")
        return model_ids, subtasks, access

    def execute(self, model_ids, subtasks, access, verbose=False) -> UltraResult:
        model_ids, subtasks, access = self.validate(model_ids, subtasks, access)
        res = UltraResult(
            final="", workflow={"model_id": model_ids, "subtasks": subtasks, "access_list": access}
        )
        outputs: list[str] = []
        for t, (mid, sub) in enumerate(zip(model_ids, subtasks, strict=False)):
            sees = visible_indices(access, t)
            ctx = ""
            for j in sees:
                ctx += (
                    f"\n<Subtask assigned to Agent {model_ids[j]}>{subtasks[j]}"
                    f"</Subtask assigned to Agent {model_ids[j]}>"
                    f"\n<Agent {model_ids[j]} response>{outputs[j].strip()}"
                    f"</Agent {model_ids[j]} response>"
                )
            user = (
                f"USER QUESTION context:\n{ctx}\n\nYour subtask: {sub}"
                if ctx
                else f"Your subtask: {sub}"
            )
            reply = self.worker(sub, [{"role": "user", "content": user}], mid)
            outputs.append(reply)
            res.steps.append(Step(t, mid, sub, sees, reply))
            if verbose:
                print(f"  step {t}: agent={mid}({self.slot_labels[mid]}) sees={sees}")
                print(f"    subtask: {sub[:80]}")
                print(f"    -> {reply.strip()[:90]}")
        res.final = outputs[-1] if outputs else ""  # last step = answer [EXEC]
        return res


class MockWorker:
    """Offline: deterministic replies so parser+DAG can be tested with no keys."""

    def __call__(self, subtask, messages, agent_id):
        return f"[agent {agent_id}] result for: {subtask[:50]}"


CANNED = (  # a Conductor-style completion for the offline self-test
    "Plan: derive then implement then verify.\n"
    "model_id: [2, 0, 1]\n"
    'subtasks: ["Devise an algorithm for the task", '
    '"Implement it in Python using the devised algorithm", '
    '"Verify the implementation is correct"]\n'
    "access_list: [[], [0], [0, 1]]\n"
)


def self_test() -> int:
    """Offline: parse a canned workflow and execute the DAG with a mock pool.
    Checks parsing, equal-length validation, topological visibility."""
    mids, subs, acc = parse_workflow(CANNED)
    print("parsed workflow:")
    print(f"  model_id   = {mids}")
    print(f"  subtasks   = {[s[:30] + '...' for s in subs]}")
    print(f"  access_list= {acc}")
    assert mids == [2, 0, 1], mids
    assert acc == [[], [0], [0, 1]], acc
    assert len(mids) == len(subs) == len(acc) == 3
    # visibility
    assert visible_indices(acc, 0) == []
    assert visible_indices(acc, 1) == [0]
    assert visible_indices(acc, 2) == [0, 1]
    # forward-ref rejection
    try:
        visible_indices([[], [2], []], 1)
        raise AssertionError("should have rejected")
    except ValueError:
        pass
    res = ConductorExecutor(MockWorker()).execute(mids, subs, acc, verbose=True)
    assert len(res.steps) == 3 and res.final
    print("\nPASS — parser, equal-length, DAG order, forward-ref ban, execution all OK")
    return 0
