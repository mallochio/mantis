# 006 — Fusion context retention & prompt-cache architecture

**Status:** Active improvement plan (quality + cost) — **Phase 1 landed** on branch `fix/fusion-context-cache-plan-56fa`  
**Priority:** P0 for cache/context correctness; P1 for handoff quality; P2 for eval instrumentation  
**Related:** [005 — Fusion hardening](005-fusion-hardening.md) (correctness/safety), [`plans/trinity-prompt-cache.md`](../plans/trinity-prompt-cache.md) (reuse Fusion markup once this lands)

## Problem (observed)

When Fusion runs agentic tests (long tool loops, plan → sidekick → review → follow-up), it:

1. **Loses context** — the sidekick and the lead do not share a durable working state; follow-ups and reviews re-derive intent from truncated text.
2. **Gets confused** — role transitions inject large free-form prompts (`REVIEW_PROMPT` + tool dump + report), reminder nags, and mid-run tool-set changes that fight the models’ format contracts.
3. **Sees nearly zero prompt-cache hits** — provider KV cache requires a **byte-identical prefix on the same model**. Fusion systematically invalidates that prefix on almost every internal hop.

This plan treats prompt cache as **architecture**, not an after-the-fact discount — the same stance as Claude Code, agentcache-style fork sessions, and sticky coding-agent gateways.

## What we already have (keep)

`apps/api/fusion.py` and `apps/api/providers.py` already contain several cache-aware pieces worth preserving:

| Asset | Why it matters |
|---|---|
| Sticky catalog slots (`[fusion] main` / `sidekick`) | Same model across turns on each role |
| Frozen, name-sorted tools (`utils._convert_tools`) | Stable tool JSON order |
| Compaction that **replays the same slot + tools** and appends `COMPACTION_INSTRUCTION` last | Matches Claude Code’s “cache-safe compact” |
| Tool-result prune before drop (`_prune_for_budget` → `_drop_old_groups`) | Microcompact before hard truncate |
| Family-aware breakpoints (`cache_control` / `prompt_cache_options`) | Anthropic + GPT-5.6 markup already wired |
| Reminders as **user** `<system-reminder>` messages | Avoids mutating the system prefix |

Do **not** invent a second cache dialect for Fusion. Extend the existing provider path.

## Root causes in this repo

Diagnosed against `FusionRun._advance` / `FusionCoordinator._call_worker` and popular lead/sidekick harnesses (Claude Code, agentcache, sticky-gateway patterns).

### A. Mid-run tool-set mutation (primary cache killer)

```839:889:apps/api/fusion.py
            available_tools = self.tools if (self.planning_tool_rounds < 2 and self.tools) else None
            ...
                    main_text, main_calls, _ = self._call_main(
                        coordinator, prompt=reminder, tools=None
                    )
```

Planning starts **with tools**, then forces `tools=None` after two rounds or on reminder/budget. Provider tools live in the cached prefix (especially Anthropic). Adding/removing tools mid-conversation is exactly what Claude Code warns against (“Never add or remove tools mid-session”). Plan Mode there keeps the full tool array and uses mode tools / messages instead.

### B. Dual disjoint histories (context killer + dual cold caches)

- Lead: `MAIN_PREAMBLE` + client messages / brief  
- Sidekick: fresh `SIDEKICK_PREAMBLE` + only the text `BRIEF`

Main’s planning tool results never reach the sidekick. The sidekick cannot continue from inspected files, failed commands, or decisions — only from whatever the lead managed to stuff into `BRIEF:`. That is a lossy handoff, not a fork.

Popular patterns:

- **Claude Code Explore / cheap subagent:** lead writes an explicit handoff; child starts cold by design — but the handoff is rich and the lead keeps its own warm cache.
- **Claude Code fork / agentcache:** child inherits **byte-identical** system + tools + history; only the final directive differs → cache hits on the parent prefix.
- Fusion today is the worst of both: cold child **and** a thin brief.

### C. Review loop rebuilds volatile, lossy context

On every sidekick report, Fusion appends a new user blob:

`REVIEW_PROMPT` + truncated `_summarize_sidekick_tool_history()` + report

then often calls main **with tools again**. That:

- grows `main_messages` with near-duplicate tool dumps the lead never saw live;
- places the decision format (`ACCEPT` / `FOLLOW_UP:`) after a wall of noise;
- changes tools presence vs late planning (`tools=None`), busting the lead prefix again.

### D. Compaction is good; silent drop is not

`_compact_replay` is cache-shaped. If it fails or still overflows, `_drop_old_groups` deletes early conversation **while keeping system** — but the *semantic* prefix (client brief, plan decisions, early tool facts) disappears. Models then “forget” and thrash. No structured checkpoint is persisted onto the run object for later roles.

### E. Instrumentation gap

`eval/router_eval.py` already knows `prompt_cache_hit_tokens`, but Fusion has no checked-in cache-hit / prefix-stability harness. Without a SEV-style metric, regressions stay invisible.

### F. Out of scope here (owned by 005)

Admission, unsafe headless, double usage accounting, review parser strictness, tool-id identity — keep those on [005](005-fusion-hardening.md). This plan assumes those correctness fixes continue in parallel; cache work must not weaken them.

## Target architecture

Two **sticky cache namespaces per Fusion run** (same as the Trinity plan’s Worker vs critic split):

| Namespace | Model | Tools | Prefix (immutable for the run) | Volatile tail |
|---|---|---|---|---|
| **Lead** | catalog `main` | **Always** the frozen client tool list (never `None` mid-run) | `MAIN_PREAMBLE` + initial user/history + tools | plan/review directives, tool rounds, reminders |
| **Sidekick** | catalog `sidekick` | Always the same frozen tool list | `SIDEKICK_PREAMBLE` + **structured brief packet** + tools | tool rounds, follow-up directives |

Optional later (P2): a **true fork** path where sidekick shares lead’s prefix on the *same* model for exploration — only when quality data says cold handoff is the bottleneck. Default remains two models (cost ladder), which **cannot** share KV cache across the hop.

### Lead state machine (cache-preserving)

Replace “strip tools to force PLAN/BRIEF” with **Plan Mode style**:

1. Keep `tools=self.tools` on every lead call (empty list only if the client sent none).
2. Cap planning exploration with messages / `tool_choice` / a dedicated `submit_plan` (or strict text contract), **not** by deleting tools.
3. On format failure, append `PLAN_REMINDER_PROMPT` as a user message; do not change tools.
4. Persist structured fields on the run: `plan`, `sidekick_brief`, `checkpoint` (goal / files / decisions / remaining), not only free text in the transcript.

### Sidekick handoff (context-preserving)

Replace “BRIEF string only” with a **brief packet** user message, for example:

```text
<fusion-brief>
goal: ...
plan: ...
constraints: ...
files: ...
decisions: ...
open_questions: ...
evidence: ...   # short excerpts from lead tool results, not a full dump
</fusion-brief>
```

Rules:

- Built once at handoff; follow-ups append `<fusion-follow-up>` messages — never rewrite the system prompt or reshuffle tools.
- Cap evidence size; prefer paths + error snippets over full command logs (microcompact).
- Sidekick final report should be structured enough that review does not need a second lossy tool dump.

### Review without prefix destruction

1. Prefer structured sidekick report (`RESULT` / `CHANGES` / `TESTS` / `RISKS`) over regenerating tool history.
2. Keep review on the **same lead prefix**; only append a short review directive.
3. If lead needs to verify, use tools **with the same tool array** already in the prefix — do not toggle tools on/off between plan and review.
4. On `FOLLOW_UP`, send the sidekick a small delta, not a full re-brief.

### Compaction & checkpoints

1. Keep `_compact_replay` (same slot, same tools, instruction last).
2. After successful compact, **write `run.checkpoint`** from the summary so review/follow-up can refresh the brief packet without relying on dropped messages.
3. Prefer prune → compact → checkpoint refresh over `_drop_old_groups`. Treat hard drop as last resort and log it as a cache/context incident.
4. Reserve a compaction buffer (tokens for instruction + summary output), same lesson as Claude Code.

### Provider / gateway stickiness

- Bifrost path: rely on prefix identity + existing family markup; do not invent OpenRouter-only `prompt_cache_key` on Bifrost bodies (`trinity-prompt-cache.md` already notes this).
- Ensure Fusion runs set a stable `cache_namespace` (today Native runs do; Fusion should pin one per role namespace).
- Failover that changes model mid-role remains a cold-cache event; document and measure it.

## Workstream game plan

### Phase 0 — Measure (before behavior changes)

**Goal:** Prove cache/context failure with numbers, not vibes.

**Status:** Partial — unit harness for lead tool stability, cache namespace, and brief packet added in `tests/test_fusion.py`.

1. Add a Fusion **prefix-stability unit harness**:
   - Two consecutive sidekick tool rounds → identical model id, identical tools JSON, identical message prefix; only the tail grows.
   - Lead plan turn N vs N+1 with tools always present → same tools key.
   - Assert today’s planning path **fails** a “tools never change” invariant (characterization test), then flip it when Phase 1 lands.
2. Add usage assertions where providers report `cache_read` / `prompt_cache_hit_tokens` (skip soft when upstream omits the field).
3. Log per-role: `prompt_tokens`, `cache_hit_tokens`, `tools_present`, `prefix_hash` (hash of system+tools+messages[:-1]).

**Exit:** A failing characterization test + a one-page baseline from a short live smoke (optional).

### Phase 1 — Stop breaking the lead cache (P0)

**Goal:** Near-zero intentional cache busts on the lead slot during a single run.

**Status:** Implemented — lead always sends frozen tools; `tool_choice="none"` blocks calls after budget/reminders.

1. **Never pass `tools=None` on lead** when `self.tools` is non-empty. Use `tool_choice="none"` or a `submit_plan` tool / reminder message to exit planning.
2. Same rule for reminder and review calls.
3. Freeze tools once on `FusionRun.__init__` (already sorted); reuse the same list object for all lead calls.
4. Keep compaction’s `active_tool_choice = "none"` pattern; do not strip tools from the compaction request body.

**Exit:** Unit tests show tools JSON identical across plan → reminder → review → tool round. Live smoke shows non-zero cache reads on turn 2+ for OpenAI/Anthropic lead models.

### Phase 2 — Structured handoff & review (P0/P1)

**Goal:** Sidekick and lead stop “forgetting” mid-task.

**Status:** Partial — `<fusion-brief>` / `<fusion-follow-up>` packets landed; review slim-down still pending.

1. Introduce brief packet + follow-up delta messages (schema above).
2. Have lead fill packet fields from plan + optional checkpoint; include bounded evidence from lead tool results.
3. Require sidekick final report sections; review prompt becomes short and structured.
4. Remove or demote `_summarize_sidekick_tool_history` from the default review path (keep as debug/trace only).
5. Tighten review parsing in coordination with 005 (malformed → retry/error, not soft accept).

**Exit:** Multi-follow-up unit test where sidekick messages retain the original `<fusion-brief>` prefix; review messages stay small; planted facts in the brief survive into the final report.

### Phase 3 — Checkpointed compaction (P1)

**Goal:** Long test runs degrade gracefully instead of amnesia.

1. Persist `run.checkpoint` after compact / after accepted plan.
2. When fitting would drop groups, compact first; if still over budget, refresh sidekick/lead tails from checkpoint rather than silent drop alone.
3. Align retain ratio / max summary tokens with measured context windows for Sol/Luna.

**Exit:** Synthetic long-history test: after compact, lead still sees goal/files/decisions; cache prefix after compact is the new short prefix (expected one-time rebuild).

### Phase 4 — Eval & gates (P2)

**Goal:** Prevent regressions the way Claude Code treats cache hit rate as uptime.

1. Extend Fusion eval / smoke to record cache-hit ratio per role.
2. CI gate (non-flaky unit side): prefix-hash stability tests required green.
3. Optional: cheap live nightly smoke with budget cap; alert if hit ratio collapses vs baseline.
4. Only after Phases 1–2: revisit comparative quality/cost study deferred in 005.

### Phase 5 — Optional fork acceleration (deferred)

Only if Phase 2 handoff quality is still weak **and** cost allows same-model exploration:

- Cache-safe fork of lead → explorer child on the **same** model (placeholder tool results + final directive), à la Claude Code forks / agentcache.
- Keep cheap Luna sidekick for implementation; use fork only for read-only explore.

Do not attempt cross-model cache sharing; it is impossible.

## Explicit non-goals

- Sharing KV cache between Sol lead and Luna sidekick.
- Adding Bifrost support for OpenRouter `prompt_cache_key`.
- Replacing Fusion with Trinity/Ultra.
- Broad prompt rewrites unrelated to cache/context (do those only with eval proof).
- Implementing 005’s P0 headless sandbox work in this plan.

## Suggested implementation order (PRs)

1. **Characterization + metrics** (Phase 0) — tests only / light logging.  
2. **Sticky tools on lead** (Phase 1) — small, high leverage.  
3. **Brief packet + review slim-down** (Phase 2) — largest quality win.  
4. **Checkpoint compaction** (Phase 3).  
5. **Eval gates** (Phase 4).  
6. Reassess fork (Phase 5) from measured gaps.

## Acceptance criteria (overall)

- Lead and sidekick each maintain a frozen (system, tools) pair for the life of a run.
- No Fusion code path removes tools mid-run to coerce formatting.
- Sidekick always receives a structured brief packet; follow-ups are deltas.
- Review does not inject a regenerated full tool transcript by default.
- Compaction remains same-slot, same-tools, instruction-last; checkpoint survives compact.
- Unit suite locks prefix stability; live smoke shows non-zero cache hits on sticky models when the provider reports them.
- 005 correctness invariants (tool id identity, review parse, usage ownership) remain green.

## References (external patterns)

- Anthropic — *Lessons from building Claude Code: Prompt caching is everything* (static→dynamic layout; never mutate tools; cache-safe compact; messages for mode changes).
- Claude Code docs — prompt caching, compact, subagent vs fork semantics.
- agentcache — cache-safe forks, shared prefix across roles, microcompaction, cache-break diffs.
- Sticky coding-agent gateways — session affinity so multi-key routers do not cold-cache every turn.
