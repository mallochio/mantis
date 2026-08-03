# Plan 003: Eager model load at startup with a truthful ready gate

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

- **Priority**: P2
- **Effort**: M
- **Risk**: LOW
- **Depends on**: none
- **Category**: perf
- **Planned at**: commit `aa6d8c4`, 2026-08-03

## Why this matters

The RouteLLM MF router and the Supra-Router-51M model load lazily on the
first request (`_load_router`, `_load_supra`). On a cold start that means the
first request pays torch/transformers imports plus — on first-ever boot — a
Hugging Face checkpoint download, for a stall of many seconds or minutes.
Meanwhile `/healthz` returns `{"ok": true}` unconditionally, so the launcher
(which waits only for the TCP port) reports "router running" while the router
cannot actually route a single request. Loading both models at startup and
gating health on readiness turns "listening" into "actually able to route",
and moves any load failure (missing `OPENAI_API_KEY`, HF unreachable) from
mid-request fallback to a clear startup failure the launcher reports.

## Current state

`server.py:113-126` — lazy router load, called from `_decide`:

```python
_router = None  # lazy global


def _load_router():
    global _router
    if _router is not None:
        return _router
    # mf router calls OpenAI text-embedding-3-small at scoring time; the OpenAI()
    # client is instantiated at import of routellm.routers.similarity_weighted.utils,
    # so the key must be in env before this import runs.
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY required for mf router (embeddings)")
    from routellm.routers.routers import ROUTER_CLS
    cfg = {"checkpoint_path": "routellm/mf_gpt4_augmented"}
    if ROUTER_NAME == "bert":
        cfg = {"checkpoint_path": "routellm/bert_gpt4_augmented"}
    _router = ROUTER_CLS[ROUTER_NAME](**cfg)
    return _router
```

`server.py:147-168` — same pattern for Supra (`_supra_model`/`_supra_tokenizer`
lazy globals; downloads `SupraLabs/Supra-Router-51M` from HF).

`server.py:254-256` — unconditional health:

```python
@app.get("/healthz")
async def healthz():
    return {"ok": True, "router": ROUTER_NAME, "threshold": THRESHOLD,
            "backend": "litellm"}
```

`server.py:305` — the app is created with `app = FastAPI(title="RouteLLM coding-router")`
(no lifespan). `_load_router` raises `RuntimeError` when `OPENAI_API_KEY` is
missing — the launcher (`llm-router.sh:214-219`) already guarantees the key is
present before starting, so a startup failure would surface a launcher bug.

`llm-router.sh:139-141` — readiness check waits for the port only:

```bash
for _ in $(seq 1 50); do
  if lsof -nP -iTCP:"$ROUTELLM_PORT" -sTCP:LISTEN >/dev/null 2>&1; then
    ROUTER_PID=$(lsof -nP -iTCP:"$ROUTELLM_PORT" -sTCP:LISTEN -t 2>/dev/null | head -n1 || true)
    echo "router running pid ${ROUTER_PID:-$(cat logs/server.pid)} on :$ROUTELLM_PORT"
    exit 0
  fi
  sleep 0.1
done
```

## Commands you will need

| Purpose | Command | Expected on success |
|---------|---------|---------------------|
| Compile check | `cd ~/.config/llm-router && .venv/bin/python -m py_compile server.py` | exit 0 |
| Tests | `.venv/bin/python test_server_helpers.py` | exit 0 |
| Bash syntax | `bash -n llm-router.sh` | exit 0 |
| Live health (after manual launch) | `curl -s http://127.0.0.1:5500/healthz` | `{"ok":true,...,"ready":true}` |

## Scope

**In scope**:
- `server.py` — lifespan startup hook, `_READY` flag, healthz body
- `llm-router.sh` — the readiness wait loop (port wait → health-wait)

**Out of scope**:
- `_decide`'s try/except fallback (`server.py:202-204`) — keep it as
  defense-in-depth; loading failures now surface at startup instead.
- Any change to routing logic, thresholds, or the cache plan (004).
- A `ROUTELLM_LAZY_LOAD` escape hatch — not added; if you believe it's
  needed, STOP and report instead of adding config.

## Git workflow

- Branch: `advisor/003-eager-load-ready-gate`
- Commit style per `git log` (e.g. `fix(router): ...`):
  `perf(router): load router and supra at startup; gate healthz on readiness`
- Do NOT push or open a PR unless the operator instructed it.

## Steps

### Step 1: Add a lifespan that loads both models

Replace `app = FastAPI(title="RouteLLM coding-router")` with:

```python
_READY = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load both scoring models before serving so the first request is fast and
    # load failures (missing OPENAI_API_KEY, HF unreachable) fail startup
    # instead of silently degrading routing mid-request.
    try:
        _load_router()
        _load_supra()
    finally:
        global _READY
        _READY = True
    yield


app = FastAPI(title="RouteLLM coding-router", lifespan=lifespan)
```

Notes:
- Add `from contextlib import asynccontextmanager` to the imports.
- `_load_supra` downloads `SupraLabs/Supra-Router-51M` on first ever boot —
  that download now happens at startup. The launcher's wait loop (step 3)
  must tolerate it; use a generous timeout (60s) rather than the current 5s
  loop.
- If `_load_router` raises (missing key), the exception propagates out of
  lifespan and FastAPI refuses to serve — intended. The `finally` still sets
  `_READY` so healthz can report the failure state. Do NOT swallow the
  exception.
- If `SUPRA_ENABLED == "0"`, skip `_load_supra()` (no generation, no model
  needed): guard the call with the same condition `_supra_complexity` uses.

**Verify**: `.venv/bin/python -m py_compile server.py` → exit 0.

### Step 2: Gate healthz on readiness

Replace the healthz body with:

```python
@app.get("/healthz")
async def healthz():
    return {"ok": _READY, "router": ROUTER_NAME, "threshold": THRESHOLD,
            "backend": "litellm", "ready": _READY}
```

**Verify**: `.venv/bin/python -m py_compile server.py` → exit 0.

### Step 3: Launcher waits for health, not just the port

In `llm-router.sh`, replace the final wait loop (lines 139-141) with one that
polls healthz for `"ready":true` **after** the port is up:

```bash
for _ in $(seq 1 600); do  # 60s — first boot downloads the Supra checkpoint
  if lsof -nP -iTCP:"$ROUTELLM_PORT" -sTCP:LISTEN >/dev/null 2>&1; then
    if curl -fsS --max-time 2 "http://127.0.0.1:$ROUTELLM_PORT/healthz" 2>/dev/null | grep -q '"ready":true'; then
      ROUTER_PID=$(lsof -nP -iTCP:"$ROUTELLM_PORT" -sTCP:LISTEN -t 2>/dev/null | head -n1 || true)
      echo "router running pid ${ROUTER_PID:-$(cat logs/server.pid)} on :$ROUTELLM_PORT"
      exit 0
    fi
  fi
  sleep 0.1
done
```

Note: `curl` and `grep` are already used elsewhere in this script (the
LiteLLM health check at line 58 uses the same pattern).

**Verify**: `bash -n llm-router.sh` → exit 0.

### Step 4: Manual end-to-end check

If the router and LiteLLM are running (check `lsof -nP -iTCP:5500`), restart
via `./llm-router.sh` and observe:

- The script exits 0 and prints `router running ... on :5500` **only after**
  models are loaded.
- `curl -s http://127.0.0.1:5500/healthz` returns `"ok":true,"ready":true`.
- `logs/server.out` shows torch/transformers load activity at boot, and no
  `Router decision failed` lines.
- First request TTFB (see `x-route-score`-bearing response or
  `logs/decisions.log` `ttfb_ms`) is no longer dominated by model loading.

If the service is not currently running, skip this step and note it in the
status row — the done criteria below are still checkable statically.

**Verify**: all four observations hold, or record "service not running" for the last step.

## Test plan

No new tests — this is startup wiring. Existing suite
(`.venv/bin/python test_server_helpers.py`) must stay green. The readiness
behavior is verified by the manual check in step 4 (and by the launcher's
exit code on every future `./llm-router.sh` run).

## Done criteria

Machine-checkable. ALL must hold:

- [ ] `.venv/bin/python -m py_compile server.py` exits 0; `bash -n llm-router.sh` exits 0
- [ ] `.venv/bin/python test_server_helpers.py` exits 0
- [ ] `grep -n "lifespan" server.py` matches (definition + FastAPI arg)
- [ ] `grep -n "asynccontextmanager" server.py` matches
- [ ] `grep -n '"ready":' server.py` matches in healthz
- [ ] `grep -c 'seq 1 600' llm-router.sh` == 1
- [ ] `grep -n '"ready":true' llm-router.sh` matches
- [ ] `plans/README.md` status row updated

## STOP conditions

Stop and report back (do not improvise) if:

- `_load_supra`'s guard condition differs from `SUPRA_ENABLED` (drift in the
  model-loading code, e.g. `_supra_complexity` no longer checks the env var).
- Startup load makes `/healthz` unreachable for the LiteLLM health pattern in
  the script (the script already tolerates curl timeouts — do not add retry
  logic beyond the loop).
- A verification fails twice after a reasonable fix attempt.

## Maintenance notes

- `_READY` is intentionally never reset: if model loading fails at startup,
  uvicorn exits and the launcher restarts it. If that restart loop ever
  spins, check `logs/server.err` for the original load exception (usually a
  missing key or HF unreachable).
- Future routing-model changes (e.g. swapping checkpoint names) must update
  the load functions only — lifespan calls them unchanged.
- Deferred: caching the Supra checkpoint locally (`~/.cache/huggingface`
  already does this) and pre-warming the MF scorer's embedding client.
