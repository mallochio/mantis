# Plan 004: Make routing latency observable and tunable

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving to the
> next step. If anything in the "STOP conditions" section occurs, stop and
> report — do not improvise. When done, update the status row for this plan
> in `plans/README.md` — unless a reviewer dispatched you and told you they
> maintain the index.
>
> **Drift check (run first)**: `git diff --stat aa6d8c4..HEAD -- server.py _test_scores.sh`
> If either file changed since this plan was written, compare the "Current
> state" excerpts against the live code before proceeding; on a mismatch,
> treat it as a STOP condition.

## Status

- **Priority**: P2
- **Effort**: M
- **Risk**: MED (routing *may* change, but only when the new gate is enabled)
- **Depends on**: none
- **Category**: perf
- **Planned at**: commit `aa6d8c4`, 2026-08-03

## Why this matters

Every request pays two scoring costs before anything is forwarded: an OpenAI
embeddings call (MF win-rate) and — whenever the score lands below the
threshold — a full CPU generation from Supra-Router-51M (128 new tokens,
greedy). Logged cheap-route TTFBs are ~1.3s. Two things make this worse than
necessary and both are currently invisible:

1. Identical or near-identical prompts (common in agentic loops) are
   re-scored from scratch every time — no cache.
2. Supra runs even for scores far below the threshold, where it essentially
   never flips the decision — but nothing in the log records how long Supra
   took or how often its answer changed the outcome, so there's no data to
   tune with.

This plan makes the cost visible (`supra_ms` in the decision log), avoids
re-paying it (small LRU cache, default off behavior preserved), and adds a
calibration knob (a lower score band below which Supra is skipped) that
defaults to "no behavior change" — the operator tunes it with real log data.

## Current state

`server.py:172-186` — Supra generation, untimed:

```python
def _supra_complexity(prompt: str) -> int:
    model, tokenizer = _load_supra()
    fmt = f"Task: {prompt}\nAnalysis: "
    inputs = tokenizer(fmt, return_tensors="pt")
    import torch
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=128, do_sample=False,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
        )
    gen = tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True,
    ).strip()
    return _parse_supra_complexity(gen)
```

`server.py:188-205` — the decision path:

```python
def _decide(prompt: str) -> tuple[str, float, int | None]:
    # Take tail of prompt (~15k chars) so routing evaluates the latest user request & context
    trimmed_prompt = prompt[-15000:] if len(prompt) > 15000 else prompt
    try:
        r = _load_router()
        score = float(r.calculate_strong_win_rate(trimmed_prompt))
        supra_complexity = None
        if score >= THRESHOLD:
            return "expensive", score, supra_complexity
        if SUPRA_ENABLED:
            supra_complexity = _supra_complexity(trimmed_prompt)
            if supra_complexity >= SUPRA_THRESHOLD:
                return "expensive", score, supra_complexity
        return "cheap", score, supra_complexity
    except Exception as err:
        print(f"Router decision failed ({err}); defaulting to expensive", flush=True)
        return "expensive", 1.0, None
```

`server.py:230-243` — `_log` writes one JSON line per decision to
`~/.config/llm-router/logs/decisions.log` with `ts, router, threshold, score,
supra_complexity, decision, model, ttfb_ms, prompt[:200]`. `ttfb_ms` covers
the whole decision + upstream TTFB, so Supra's share is currently
unmeasurable.

`_test_scores.sh` — golden smoke set (8 prompts: hello-world → Paxos/BFT)
that prints score + decision from the response headers; used below to prove
decisions are unchanged.

## Commands you will need

| Purpose | Command | Expected on success |
|---------|---------|---------------------|
| Compile check | `cd ~/.config/llm-router && .venv/bin/python -m py_compile server.py` | exit 0 |
| Tests | `.venv/bin/python test_server_helpers.py` | exit 0 |
| Golden decisions | `./_test_scores.sh` (router must be running) | 8 rows, decisions sensible (easy→cheap, Paxos→expensive) |

## Scope

**In scope**:
- `server.py` — cache, score-band gate env, `supra_ms` logging
- (Optionally) `README.md` — one line documenting the two new env vars

**Out of scope**:
- Changing the default routing behavior — with defaults set as below, a
  request that was `cheap` before must still be `cheap` (cache is
  transparent; gate is off by default).
- Reducing `max_new_tokens=128` or switching Supra off — tuning, not code;
  the operator decides after seeing `supra_ms` data.
- The model-loading work in plan 003 (landed separately).

## Git workflow

- Branch: `advisor/004-routing-latency-observability`
- Commit style per `git log` (e.g. `fix(router): ...`):
  `perf(router): cache decisions, time supra, add score-band gate`
- Do NOT push or open a PR unless the operator instructed it.

## Steps

### Step 1: Time Supra and log it

In `_supra_complexity`, wrap the generation with timing:

```python
def _supra_complexity(prompt: str) -> tuple[int, int]:
    model, tokenizer = _load_supra()
    fmt = f"Task: {prompt}\nAnalysis: "
    inputs = tokenizer(fmt, return_tensors="pt")
    import torch
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=128, do_sample=False,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
        )
    supra_ms = int((time.time() - t0) * 1000)
    gen = tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True,
    ).strip()
    return _parse_supra_complexity(gen), supra_ms
```

Update `_decide` to use the tuple and thread `supra_ms` through to `_log`
(add a `supra_ms: int | None = None` parameter and a `"supra_ms": supra_ms`
field). The return type of `_decide` becomes
`tuple[str, float, int | None, int | None]` — update the early-return paths
(score ≥ THRESHOLD → `None`; gate-skip → `None`; exception → `None`) and the
handler call site (`server.py:271-274`, where `_decide` is awaited).

**Verify**: `.venv/bin/python -m py_compile server.py` → exit 0; the existing
helper tests still pass. (Tests don't cover `_decide` — the compile + a live
request cover it.)

### Step 2: Add the LRU cache

In `_decide`, before the try block:

```python
from functools import lru_cache

def _decide_uncached(prompt: str) -> tuple[str, float, int | None, int | None]:
    ...  # existing body, minus the cache wrapper
```

(Keep the function name `_decide` for the handler; move the existing body
into `_decide_uncached` and add the wrapper. The cache key is the trimmed
scoring text — the same string that is scored.)

Then the cached entry point:

```python
@lru_cache(maxsize=256)
def _decide_cached(trimmed_prompt: str) -> tuple[str, float, int | None, int | None]:
    return _decide_uncached(trimmed_prompt)


def _decide(prompt: str) -> tuple[str, float, int | None, int | None]:
    trimmed_prompt = prompt[-15000:] if len(prompt) > 15000 else prompt
    return _decide_cached(trimmed_prompt)
```

Notes:
- Cache the **trimmed** text so the 15k-char trim happens once per call.
- `lru_cache` is fine here: the trimmed prompt is a plain str, the cached
  tuple is small, and 256 entries cap memory.
- The `OPENAI_API_KEY`-missing error and the broad except stay in
  `_decide_uncached` — an exception is never cached.

**Verify**: `.venv/bin/python -m py_compile server.py` → exit 0.

### Step 3: Add the score-band gate (default: off)

Add near the other env config (`server.py:36-41`):

```python
SUPRA_MIN_SCORE = float(os.environ.get("ROUTELLM_SUPRA_MIN_SCORE", "0"))
```

In `_decide_uncached`, change the Supra call to skip below the band:

```python
        if SUPRA_ENABLED and score >= SUPRA_MIN_SCORE:
            supra_complexity, supra_ms = _supra_complexity(trimmed_prompt)
            if supra_complexity >= SUPRA_THRESHOLD:
                return "expensive", score, supra_complexity, supra_ms
        return "cheap", score, supra_complexity, supra_ms
```

(`supra_ms` must be initialized `= None` before, so the non-Supra paths still
return it.) With `SUPRA_MIN_SCORE=0` and non-negative scores, behavior is
identical to today. The operator raises it (e.g. `0.05`) to skip Supra for
very-easy prompts — the log makes the effect measurable.

**Verify**: `.venv/bin/python -m py_compile server.py` → exit 0.

### Step 4: Prove behavior is unchanged at defaults

With the router running (restart via `./llm-router.sh`), run the golden set
and compare against a baseline captured *before* this change:

```bash
./_test_scores.sh
```

**Verify**: every line's `decision` column matches the pre-change run for the
same prompt (the script prints `score decision prompt` triples; easy prompts
route cheap, Paxos/BFT prompts route expensive). If the router isn't running,
record that and mark the gate as "verify on next launch" in the status row.

### Step 5: Optional calibration (operator-facing, not code)

After a few days of traffic with the new `supra_ms` field, run:

```bash
jq -r 'select(.supra_ms != null) | [.score, .supra_ms, .decision] | @tsv' \
  ~/.config/llm-router/logs/decisions.log | sort -n | head -40
```

Look at whether the low-score tail (scores well below `ROUTELLM_THRESHOLD`)
ever flips to `expensive` via Supra (compare `supra_complexity >= 3` rows
against `decision`). If the tail never flips, set `ROUTELLM_SUPRA_MIN_SCORE`
in `~/.zshrc` and re-check `_test_scores.sh` decisions — this step is a
measurement ritual, not a code change, and belongs to the operator.

## Test plan

- Extend `test_server_helpers.py::demo()` with two asserts on pure helpers
  (no model involved):
  - `_decide_cached("same prompt") is _decide_cached("same prompt")` →
    returns the same object (cache hit identity — the tuple is immutable so
    identity is a valid hit marker).
  - `_decide_cached("a") is not _decide_cached("b")` → two entries.
- Existing tests stay green.

## Done criteria

Machine-checkable. ALL must hold:

- [ ] `.venv/bin/python -m py_compile server.py` exits 0
- [ ] `.venv/bin/python test_server_helpers.py` exits 0, including the two cache asserts
- [ ] `grep -n "supra_ms" server.py` matches in `_supra_complexity`, `_decide`, and `_log`
- [ ] `grep -n "lru_cache" server.py` matches
- [ ] `grep -n "SUPRA_MIN_SCORE" server.py` matches (env read + gate use)
- [ ] Golden set decisions unchanged at default config (step 4), or "service not running" recorded
- [ ] `plans/README.md` status row updated

## STOP conditions

Stop and report back (do not improvise) if:

- The `_decide` / `_supra_complexity` code doesn't match the excerpts (drift
  — e.g. plan 003 already changed the loading path).
- The golden-set decisions differ at default config after the change.
- Adding the tuple return breaks the handler call site in a way that isn't
  just the four return points listed in step 1.

## Maintenance notes

- The cache key is the trimmed last-user-message text. If a future change
  makes the routing input richer (e.g. full conversation tail), the cache
  key and `maxsize` must be revisited — cached decisions are only as good as
  their key.
- `lru_cache` never invalidates: a prompt that flips routing intent over
  time (same text, new situation) stays pinned. If that ever matters, switch
  to a TTL cache — not before.
- A reviewer should confirm step 1 updated ALL return paths of `_decide`
  (grep for `return` inside `_decide_uncached`).
- Deferred: analyzing `decisions.log` cost savings programmatically — see the
  direction finding D2 in `plans/README.md`.
