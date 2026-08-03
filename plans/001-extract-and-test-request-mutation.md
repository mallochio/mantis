# Plan 001: Extract request-mutation logic into a tested pure function

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving to the
> next step. If anything in the "STOP conditions" section occurs, stop and
> report — do not improvise. When done, update the status row for this plan
> in `plans/README.md` — unless a reviewer dispatched you and told you they
> maintain the index.
>
> **Drift check (run first)**: `git diff --stat aa6d8c4..HEAD -- server.py test_server_helpers.py`
> If either file changed since this plan was written, compare the "Current
> state" excerpts against the live code before proceeding; on a mismatch,
> treat it as a STOP condition.

## Status

- **Priority**: P1
- **Effort**: S
- **Risk**: LOW
- **Depends on**: none
- **Category**: tests
- **Planned at**: commit `aa6d8c4`, 2026-08-03

## Why this matters

Every request to this router is silently rewritten before being forwarded:
`max_tokens` becomes `max_completion_tokens` (and is clamped), `stop` is
dropped, `temperature` is dropped for gpt-5.6 models, `reasoning_effort` is
injected, `developer` roles become `system`, and the model is overwritten.
Clients (Pi, OpenCode) depend on this. None of it is tested — a regression
would break every client silently. Extracting the block into a pure function
makes it testable and is the verification baseline any later refactor stands
on. Also removes one line of dead code in the same handler.

## Current state

`server.py` — the only server file; the handler at lines 263-289:

```python
    backend = _backend_for(decision)

    out_body = dict(body)
    if isinstance(out_body.get("messages"), list):
        out_body["messages"] = _normalize_messages_for_backend(out_body["messages"])
    out_body["model"] = backend["model"]
    if isinstance(out_body.get("max_tokens"), int):
        out_body["max_completion_tokens"] = out_body.pop("max_tokens")
    if isinstance(out_body.get("max_completion_tokens"), int) and backend.get("max_tokens"):
        out_body["max_completion_tokens"] = min(out_body["max_completion_tokens"], backend["max_tokens"])
    out_body.pop("stop", None)
    if backend["model"].startswith("gpt-5.6-") and out_body.get("temperature") not in (None, 1):
        out_body.pop("temperature")
    if backend["effort"]:
        out_body["reasoning_effort"] = backend["effort"]
```

And at `server.py:303` (streaming branch), one dead line — a `httpx.stream`
request object is created and never used:

```python
        req = httpx.stream("POST", url, json=out_body, headers=headers, timeout=None)
        client = httpx.Client(timeout=None)
```

`test_server_helpers.py` — the existing test file, an assert-based demo script
with **no test framework** (this is the repo convention — match it):

```python
from server import _normalize_messages_for_backend, _parse_supra_complexity

def demo():
    assert _normalize_messages_for_backend([{"role": "developer", "content": "x"}]) == [
        {"role": "system", "content": "x"}
    ]
    ...

if __name__ == "__main__":
    demo()
```

`EXPENSIVE` / `CHEAP` dicts (with `"effort"`, `"model"`, `"max_tokens"` keys)
live at `server.py:47-59`. Tests construct fake backend dicts; they do NOT
need the real env config.

## Commands you will need

| Purpose | Command | Expected on success |
|---------|---------|---------------------|
| Compile check | `cd ~/.config/llm-router && .venv/bin/python -m py_compile server.py` | exit 0 |
| Tests | `.venv/bin/python test_server_helpers.py` | exit 0, no output |

(All python runs use `.venv/bin/python` — the venv holds fastapi/httpx/etc.;
bare `python` may not.)

## Scope

**In scope**:
- `server.py`
- `test_server_helpers.py`

**Out of scope**:
- `llm-router.sh`, `_test_scores.sh`, `_test_headless.sh`, `README.md`
- Any change to actual routing behavior — this plan only *moves* existing
  logic into a function and tests it; decisions must stay identical.
- The `_decide` / `_supra_complexity` / model-loading internals (see plans 003/004).

## Git workflow

- Branch: `advisor/001-request-mutation-tests`
- One commit at the end. Message style follows `git log` (e.g. `fix(router): ...`):
  `test(router): extract and cover outgoing-body mutation logic`
- Do NOT push or open a PR unless the operator instructed it.

## Steps

### Step 1: Add the pure function

In `server.py`, directly above `_normalize_messages_for_backend` (line ~212),
add:

```python
def _build_outgoing_body(body: dict, backend: dict) -> dict:
    out_body = dict(body)
    if isinstance(out_body.get("messages"), list):
        out_body["messages"] = _normalize_messages_for_backend(out_body["messages"])
    out_body["model"] = backend["model"]
    if isinstance(out_body.get("max_tokens"), int):
        out_body["max_completion_tokens"] = out_body.pop("max_tokens")
    if isinstance(out_body.get("max_completion_tokens"), int) and backend.get("max_tokens"):
        out_body["max_completion_tokens"] = min(out_body["max_completion_tokens"], backend["max_tokens"])
    out_body.pop("stop", None)
    if backend["model"].startswith("gpt-5.6-") and out_body.get("temperature") not in (None, 1):
        out_body.pop("temperature")
    if backend["effort"]:
        out_body["reasoning_effort"] = backend["effort"]
    return out_body
```

Move the code verbatim — no behavioral changes.

**Verify**: `.venv/bin/python -m py_compile server.py` → exit 0.

### Step 2: Replace the inline block in the handler

At `server.py:276-289`, replace the `out_body = dict(body)` … `reasoning_effort` block with:

```python
    out_body = _build_outgoing_body(body, backend)
```

**Verify**: `.venv/bin/python -m py_compile server.py` → exit 0; then
`grep -n "out_body\[.messages.\] = " server.py` → no match (the reassignment
now lives inside `_build_outgoing_body`).

### Step 3: Remove the dead `httpx.stream` line

At `server.py:303`, delete the line:

```python
        req = httpx.stream("POST", url, json=out_body, headers=headers, timeout=None)
```

The following line (`client = httpx.Client(timeout=None)`) stays — the
streaming `gen()` closure uses it.

**Verify**: `grep -n "req = httpx.stream" server.py` → no match; `.venv/bin/python -m py_compile server.py` → exit 0.

### Step 4: Add tests

Extend `test_server_helpers.py` (import `_build_outgoing_body` alongside the
existing imports) and add asserts to `demo()`:

- happy path: body `{"messages": [{"role": "user", "content": "hi"}], "max_tokens": 100, "temperature": 0.7, "stop": ["\n"]}` with backend `{"model": "deepseek-v4-pro", "effort": "xhigh", "max_tokens": 131072}` →
  - `model` == `"deepseek-v4-pro"`, `max_completion_tokens` == 100, no `max_tokens` key, no `stop` key, `reasoning_effort` == `"xhigh"`, `temperature` still 0.7 (non-gpt model keeps it)
- clamp: same body with backend `{"model": "deepseek-v4-pro", "effort": "", "max_tokens": 64}` → `max_completion_tokens` == 64 (min wins), no `reasoning_effort` (empty effort)
- gpt-5.6 temperature drop: backend `{"model": "gpt-5.6-luna", "effort": "xhigh", "max_tokens": None}`, body `temperature: 0.7` → `temperature` key gone; body with `temperature: 1` → temperature kept
- developer→system: body `{"messages": [{"role": "developer", "content": "x"}, {"role": "user", "content": "y"}], "max_tokens": 5}` → messages roles are `system`, `user`
- original body untouched: the input dict is not mutated by the call (compare before/after)

Use the existing `assert` style with no framework. Backend dicts in tests are
plain fakes — do not import `EXPENSIVE`/`CHEAP` (they depend on env).

**Verify**: `.venv/bin/python test_server_helpers.py` → exit 0, no output.
Confirm test count: `.venv/bin/python -c "import test_server_helpers"` runs only the import; run the file directly.

## Test plan

New asserts in `test_server_helpers.py::demo()` covering: remap + clamp,
stop-drop, temperature-drop (gpt-5.6 only, temp==1 kept), effort injection
(and skipped when empty), developer→system, and input immutability. Pattern:
the existing `demo()` function in that file.

## Done criteria

Machine-checkable. ALL must hold:

- [ ] `.venv/bin/python -m py_compile server.py` exits 0
- [ ] `.venv/bin/python test_server_helpers.py` exits 0
- [ ] `grep -n "req = httpx.stream" server.py` returns nothing
- [ ] `git diff --stat aa6d8c4..HEAD -- server.py test_server_helpers.py` shows only those two files (plus nothing else)
- [ ] `grep -c "assert " test_server_helpers.py` ≥ 12 (the 4 existing + ≥8 new)
- [ ] `plans/README.md` status row updated

## STOP conditions

Stop and report back (do not improvise) if:

- The code at the cited locations doesn't match the excerpts (drift).
- A verification fails twice after a reasonable fix attempt.
- Extracting the function requires changing behavior (e.g. the clamp, the
  temperature rule) — that is out of scope; the tests must pass against the
  *existing* behavior.

## Maintenance notes

- `_build_outgoing_body` is the contract between this router and LiteLLM —
  any future model-family special-casing (like the gpt-5.6 temperature rule)
  goes here, with a test.
- A reviewer should confirm the diff is a pure move: `git diff` on the handler
  should show deletion of the inline block and one call site.
- Deferred: documenting these mutations in README (plan 005).
