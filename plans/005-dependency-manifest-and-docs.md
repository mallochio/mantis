# Plan 005: Dependency manifest and honest README

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving to the
> next step. If anything in the "STOP conditions" section occurs, stop and
> report — do not improvise. When done, update the status row for this plan
> in `plans/README.md` — unless a reviewer dispatched you and told you they
> maintain the index.
>
> **Drift check (run first)**: `git diff --stat aa6d8c4..HEAD -- README.md`
> plus `ls requirements.txt` (should not exist yet). If README changed or a
> manifest already appeared, compare against the excerpts below; on a
> mismatch, treat it as a STOP condition.

## Status

- **Priority**: P3
- **Effort**: S
- **Risk**: LOW
- **Depends on**: none
- **Category**: dx
- **Planned at**: commit `aa6d8c4`, 2026-08-03

## Why this matters

The README's setup says `pip install routellm fastapi uvicorn httpx
transformers torch` — unpinned, and with no `requirements.txt`/`pyproject.toml`
in the repo. If the `.venv` ever dies, rebuilding it is guesswork: which
versions did the router actually run against? The pinned manifest fixes that.
Separately, the router silently rewrites every client request (drops `stop`,
drops `temperature` for gpt-5.6, remaps/clamps `max_tokens`, injects
`reasoning_effort`, converts `developer`→`system` — see plan 001) and the
README documents none of it. Clients built against those behaviors (Pi,
OpenCode) deserve a written contract; a future maintainer debugging "my stop
sequences vanished" needs the doc.

## Current state

`README.md` (full file, 2.2KB): a one-line description, a `Files` list, a
`Setup` block with the unpinned install command, a `Run` block, an
`Architecture` diagram, and a `Response headers` list. The `Setup` block:

```bash
# Python deps
python -m venv .venv
. .venv/bin/activate
pip install routellm fastapi uvicorn httpx transformers torch
```

The working `.venv` exists at repo root and currently runs the router — its
installed versions are the ground truth for the pins.

## Commands you will need

| Purpose | Command | Expected on success |
|---------|---------|---------------------|
| Enumerate installed top-level deps | `cd ~/.config/llm-router && .venv/bin/pip freeze` | one `name==version` line per package |
| Write/read manifest | `cat requirements.txt` | pins for the 6 direct deps |

## Scope

**In scope**:
- `requirements.txt` (create)
- `README.md`

**Out of scope**:
- Upgrading or downgrading any dependency — this plan *records* versions, it
  does not change them.
- `pip-audit` / vulnerability scanning — no security review was requested for
  this run; the manifest is a reproducibility fix.
- A `pyproject.toml` with metadata/build config — overkill for a personal
  launcher repo; plain `requirements.txt` matches the "fewest files" rule.
- The test scripts and `llm-router.sh` — unchanged.

## Git workflow

- Branch: `advisor/005-deps-and-docs`
- Commit style per `git log` (e.g. `fix(router): ...`):
  `docs: pin dependencies and document request rewriting`
- Do NOT push or open a PR unless the operator instructed it.

## Steps

### Step 1: Write requirements.txt

Generate pins for the six direct imports only (fastapi, uvicorn, httpx,
routellm, transformers, torch) from the working venv's freeze:

```bash
cd ~/.config/llm-router && .venv/bin/pip freeze | grep -E '^(fastapi|uvicorn|httpx|routellm|transformers|torch)=='
```

Write the six matching lines to `requirements.txt`, preserving the versions
verbatim (do NOT add any other packages — transitive deps resolve
automatically).

**Verify**: `cat requirements.txt` → exactly six `name==version` lines; each
name is one of the six above.

### Step 2: Update the README setup block

Replace the install command with:

```bash
# Python deps (versions pinned in requirements.txt)
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

Also update the `Files` list to include:

```
- `requirements.txt` — pinned Python dependencies
```

**Verify**: `grep -n "requirements.txt" README.md` → two matches (Files list +
setup block).

### Step 3: Document request rewriting

Add a `## Request handling` section to README.md (after `Response headers`),
listing exactly what the router does to a client request before forwarding —
this is the contract from plan 001's `_build_outgoing_body`:

- `model` is always overwritten with the chosen backend model
  (`gpt-5.6-luna` / `deepseek-v4-pro`); the client's value is ignored.
- `max_tokens` is renamed to `max_completion_tokens` and clamped to the
  backend's `CHEAP_MAX_TOKENS` (default 131072).
- `stop` sequences are dropped.
- `temperature` is dropped for gpt-5.6 models unless it is `1`.
- `reasoning_effort` is injected from `EXPENSIVE_REASONING_EFFORT` /
  `CHEAP_REASONING_EFFORT`.
- `developer`-role messages are rewritten to `system`.

Note under the section header that these are deliberate compatibility
mutations for LiteLLM's Responses API bridge, not bugs.

**Verify**: `grep -n "Request handling" README.md` matches;
`grep -c "stop" README.md` ≥ 1.

## Test plan

None — documentation and a manifest. Existing suite
(`.venv/bin/python test_server_helpers.py`) must stay green (it imports
`server`, which is untouched).

## Done criteria

Machine-checkable. ALL must hold:

- [ ] `requirements.txt` exists with exactly 6 pinned direct deps
- [ ] Every line in `requirements.txt` matches a `name==version` present in `.venv/bin/pip freeze`
- [ ] `grep -n "pip install -r requirements.txt" README.md` matches
- [ ] `grep -n "Request handling" README.md` matches; the section lists all 6 mutations
- [ ] `.venv/bin/python test_server_helpers.py` exits 0
- [ ] `git status` shows only `requirements.txt`, `README.md` (and `plans/`) modified
- [ ] `plans/README.md` status row updated

## STOP conditions

Stop and report back (do not improvise) if:

- `pip freeze` shows none of the six packages (venv broken/absent) — report
  instead of guessing versions.
- README.md has drifted so far the excerpts don't locate the setup block.
- A verification fails twice after a reasonable fix attempt.

## Maintenance notes

- `pip install -r requirements.txt` recreates the environment; a future
  dependency bump means editing one file (and noting it in README if the
  mutation contract changes — e.g. if a new gpt-5.6-family rule appears in
  `_build_outgoing_body`, document it here in the same commit).
- A reviewer should verify no version in `requirements.txt` postdates what
  the venv actually has (that would be an upgrade in disguise).
