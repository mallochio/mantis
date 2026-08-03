# Plan 002: Make config defaults a single source of truth

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving to the
> next step. If anything in the "STOP conditions" section occurs, stop and
> report — do not improvise. When done, update the status row for this plan
> in `plans/README.md` — unless a reviewer dispatched you and told you they
> maintain the index.
>
> **Drift check (run first)**: `git diff --stat aa6d8c4..HEAD -- server.py llm-router.sh`
> If either file changed since this plan was written, compare the "Current
> state" excerpts against the live code before proceeding; on a mismatch,
> treat it as a STOP condition.

## Status

- **Priority**: P1
- **Effort**: S
- **Risk**: LOW
- **Depends on**: none
- **Category**: correctness
- **Planned at**: commit `aa6d8c4`, 2026-08-03

## Why this matters

The router has three sources of truth for defaults that disagree. The module
docstring (`server.py:13`) documents `ROUTELLM_THRESHOLD=0.156`; the code
default (`server.py:36`) is `"0.45"`; the launcher (`llm-router.sh:94-95`)
exports `0.156` with a calibration comment. When started via
`./llm-router.sh` the script's value wins — but anyone (or any agent) running
bare `python server.py` gets a router whose routing behavior is calibrated
completely differently. The same duplication pattern covers every
`EXPENSIVE_*`/`CHEAP_*`/`ROUTELLM_*` default. The fix: server.py defaults must
equal the launcher's, both files carry a mutual cross-reference comment, and
the server logs its effective config at startup so drift becomes visible in
`logs/server.out` instead of silent.

## Current state

`server.py:36`:

```python
THRESHOLD = float(os.environ.get("ROUTELLM_THRESHOLD", "0.45"))
```

`server.py:13` (docstring):

```
  ROUTELLM_THRESHOLD=0.156
```

`llm-router.sh:94-95`:

```bash
# 0.156 = calibrated for 30% strong-model calls via RouteLLM.
export ROUTELLM_THRESHOLD="${ROUTELLM_THRESHOLD:-0.156}"
```

The other server defaults (lines 33-58): `HOST 127.0.0.1`, `PORT 5500`,
`SERVER_KEY sk-route-local`, `ROUTER_NAME mf`, `SUPRA_ENABLED 1`,
`SUPRA_THRESHOLD 3`, `ROUTELLM_MAX_TOKENS 131072`, `LITELLM_BASE
http://127.0.0.1:3001/v1`, `LITELLM_KEY sk-mundial`,
`EXPENSIVE_MODEL gpt-5.6-luna`, `EXPENSIVE_REASONING_EFFORT xhigh`,
`CHEAP_MODEL deepseek-v4-pro`, `CHEAP_REASONING_EFFORT xhigh`,
`CHEAP_MAX_TOKENS 131072` — all duplicated in `llm-router.sh:82-99`.

`_get_context_window` (`server.py:89-111`) silently returns `1000000` when
both model-list fetches fail or are absent (`server.py:107`), which clients
read as "1M token context" — a lie told without any log.

## Commands you will need

| Purpose | Command | Expected on success |
|---------|---------|---------------------|
| Compile check | `cd ~/.config/llm-router && .venv/bin/python -m py_compile server.py` | exit 0 |
| Tests | `.venv/bin/python test_server_helpers.py` | exit 0 |
| Defaults parity | `grep -oE '"(0\.156|gpt-5\.6-luna|deepseek-v4-pro|xhigh|131072|sk-route-local|sk-mundial|auto|mf)"' server.py` | see steps |
| Bash syntax | `bash -n llm-router.sh` | exit 0 |

## Scope

**In scope**:
- `server.py` — default values, docstring, startup log
- `llm-router.sh` — one comment line only

**Out of scope**:
- Moving defaults into a config file (`.env`/yaml) — deliberately rejected:
  two-file sync with visible logging is enough for a personal tool.
- Changing any *behavior* — this plan changes the bare-`python server.py`
  threshold from 0.45 to 0.156, which is the intended correction, and
  nothing else.
- The launcher's `~/.zshrc` env import.

## Git workflow

- Branch: `advisor/002-config-defaults`
- Commit style per `git log` (e.g. `fix(router): ...`):
  `fix(router): align bare-run defaults with launcher and log effective config`
- Do NOT push or open a PR unless the operator instructed it.

## Steps

### Step 1: Sync server.py defaults to the launcher's

In `server.py`, change line 36 to:

```python
THRESHOLD = float(os.environ.get("ROUTELLM_THRESHOLD", "0.156"))
```

Then check every other default in lines 33-58 against `llm-router.sh:82-99`.
They are already identical (verify with grep below); if any differs, fix the
server.py value to match the script — the script is canonical (it carries the
calibration comment and is the documented entry point).

Add a comment above line 33:

```python
# Defaults mirror llm-router.sh (canonical source, calibrated there) —
# keep in sync so bare `python server.py` behaves identically to the launcher.
```

**Verify**:
- `grep -c '"0.156"' server.py` → ≥ 2 (docstring line 13 + line 36)
- `grep -c '"0.45"' server.py` → 0
- `.venv/bin/python -m py_compile server.py` → exit 0

### Step 2: Cross-reference from the launcher

In `llm-router.sh`, extend the existing comment at line 94:

```bash
# 0.156 = calibrated for 30% strong-model calls via RouteLLM.
# Canonical defaults live here; server.py mirrors them (bare-run parity).
export ROUTELLM_THRESHOLD="${ROUTELLM_THRESHOLD:-0.156}"
```

**Verify**: `bash -n llm-router.sh` → exit 0.

### Step 3: Log effective config at startup

In `server.py`, in the `if __name__ == "__main__":` block (currently just
`uvicorn.run(...)`), add before it:

```python
    print(
        "effective config: "
        f"router={ROUTER_NAME} threshold={THRESHOLD} supra={SUPRA_ENABLED} "
        f"supra_threshold={SUPRA_THRESHOLD} expensive={EXPENSIVE['model']} "
        f"cheap={CHEAP['model']} port={PORT}",
        flush=True,
    )
```

Also in `_get_context_window`, at `server.py:107`, replace the silent
fallback with a logged one:

```python
    _cached_context_window = 1000000
    print(
        "WARNING: could not determine context window from model lists; "
        "falling back to 1000000. Set ROUTELLM_CONTEXT_WINDOW to override.",
        flush=True,
    )
```

**Verify**: `.venv/bin/python -m py_compile server.py` → exit 0; the
`effective config:` line appears in `logs/server.out` after the next launch
via `./llm-router.sh`.

## Test plan

No new tests — the change is default values and log lines. Existing tests
(`.venv/bin/python test_server_helpers.py`) must stay green; `_build_outgoing_body`
tests from plan 001 are unaffected because they use fake backend dicts.

## Done criteria

Machine-checkable. ALL must hold:

- [ ] `grep -c '"0.45"' server.py` returns 0
- [ ] `grep -c '"0.156"' server.py` ≥ 2
- [ ] Every `os.environ.get("X", default)` in `server.py:33-58` has a matching
  `export X=` default in `llm-router.sh:82-99` (spot-check 5 of them)
- [ ] `.venv/bin/python -m py_compile server.py` and `bash -n llm-router.sh` exit 0
- [ ] `.venv/bin/python test_server_helpers.py` exits 0
- [ ] `grep -n "effective config:" server.py` matches; `grep -n "WARNING: could not determine" server.py` matches
- [ ] `plans/README.md` status row updated

## STOP conditions

Stop and report back (do not improvise) if:

- Any default in `server.py:33-58` does NOT match `llm-router.sh:82-99` in
  addition to the threshold (i.e. drift beyond what this plan lists) — report
  the mismatch instead of silently picking a winner.
- The `_get_context_window` code differs from the excerpt (drift).
- A verification fails twice after a reasonable fix attempt.

## Maintenance notes

- The parity rule is: **when you change a default in `llm-router.sh`, change
  it in `server.py` too** — the cross-reference comments exist to make that a
  two-step habit.
- The startup log line is the tripwire: a diff between `logs/server.out`'s
  effective config and the intended one means env from `~/.zshrc` is
  overriding something.
- Deferred: emitting the effective config as structured JSON (would need a
  logging setup; print is the repo's existing convention).
