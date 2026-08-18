# DeepSeek Harness techniques we can steal

**Status:** research note. Do not vendor Cordis or replace Pi/Mantis with `dsh`. First-party source: `deepseek-ai/deepseek-harness` `99f6f02` (`dsh-0.1.0-rc.7`).

DeepSeek Harness is a full coding agent (tools, sessions, compaction, UI). Mantis is a router plus multi-model coordinators that pause for **client** tools. Adopt the **prefix-stability rules**, not the plugin runtime.

Their cache is automatic byte-prefix matching at the provider. They do **not** send Anthropic `cache_control` or OpenAI `prompt_cache_breakpoint`. Fusion’s family markup is still right for Claude/GPT-5.6 via Bifrost; Gemini/DeepSeek should stay unmarked, matching `dsh`.

## Already in Mantis

- Tool-result head/tail prune (`utils.prune_tool_result`, copied from `dsh-compaction-tool-result-pruner`)
- Repeat-tool guard
- Fusion frozen `MAIN_PREAMBLE` + two-pass prune-before-drop
- Fusion/OpenAI/Anthropic cache dialects; Gemini implicit
- Gateway session pin so `mantis/base` does not hop models mid-conversation
- Upstream streaming

## Worth taking (small, router-safe)

### 1. Canonical tool order — all coordinators + gateway pass-through

`dsh-system-prompt` sorts visible tool schemas lexicographically by name (locale-independent) **before** the request is built. Registration/client order must not matter.

Mantis `_convert_tools` preserves client order. Pi is usually stable, but any client that reshuffles tools busts the prefix on every turn.

**Change:** sort by `function.name` once at run start (Fusion/Trinity/Ultra) and when the gateway forwards `tools`. Freeze that list for the run. Do not rebuild JSON with different key order.

**Where:** `apps/api/utils.py` (`_convert_tools`), optionally gateway `_build_upstream` if it copies tools through.

### 2. Volatile facts as the last user message, never system

`dsh` keeps a fixed system prefix. Time, cwd, approval policy, skill catalogs, and `AGENTS.md` are append-only **user** snapshots, re-injected only when the text actually changed. Time context is **off by default**.

Mantis coordinators already put role text in the last user message. Do not add dates, cwd, or per-step “you are the Thinker” into system. Gateway must not wrap prompts with a timestamped system preamble.

### 3. Compaction as prefix replay — Fusion first, Trinity later

When `dsh` summarizes, it copies the last request’s system + tools + history **verbatim**, same model, and appends the compaction instruction as the **final** user message. A different summarizer model is a cold cache. After the summary lands, a short verbatim tail is kept.

Mantis Fusion currently **drops oldest groups** after pruning. Next step when a window is still too big: one same-model summary call that replays the frozen prefix, then replace the middle, keep a tail. Do not summarize with Luna while the lead is Sol.

Trinity should get this only after Worker pinning (`plans/trinity-prompt-cache.md`).

### 4. Reconstructable requests

`dsh` builds `system + tools + deriveMessages(log)` and deep-freezes it. Listeners must not rewrite the body. Empty usage-only assistant messages are omitted.

Fusion `main_messages` / `sidekick_messages` already append. Trinity still rebuilds every hop with a new model — that is the cache killer, already planned. Ultra workers still see only the last user message; that is a correctness bug, not a cache trick, and should be fixed before Ultra cache work.

### 5. Call-config is part of the cache key

`dsh` treats provider, model, reasoning effort, and sampling as header identity. Changing effort mid-session (gateway `complexity_efforts` ratchet, Fusion review vs plan) is a cold prefix even if text matches. Prefer pinning effort on the sticky Fusion lead; gateway already pins the **model** per session.

## Speed techniques that do **not** belong in Mantis routers

| `dsh` | Why not here |
|---|---|
| Parallel tool pool (`maxParallelToolCalls: 10`) | Tools run in **Pi**, not in Fusion/Trinity. Pi already parallelizes if the model emits several calls. |
| Code Mode SDK / `run_code` | Different product. |
| Jobs, subagents, workflow threads | Coordinators already split lead/sidekick or Worker/Thinker. |
| Write-behind session JSONL | Mantis run store is already separate. |
| Speculative decoding | Not in `dsh` either. |
| `x-deepseek-harness-*` headers | DeepSeek-official adapter only. Useless on Bifrost unless we add a DeepSeek-native route. |

Reasoning passback on **tool-call turns only** (`dsh-llm-deepseek` `serialize.ts`) is worth copying **if** a DeepSeek worker is on the native API. Fusion already preserves `_anthropic_content` / `reasoning_details` for Claude/GPT. Do not invent DeepSeek `reasoning_content` on the OpenAI-compat Bifrost path until a worker uses the official DeepSeek Messages/chat wire.

## Suggested order

1. Sort and freeze tool schemas (all modes). Tiny, cache-safe, no behavior change for Pi if names already unique.
2. Audit system prompts for cwd/time/role leakage (gateway + Fusion + Ultra planner).
3. Fusion: same-model compaction replay when prune-and-drop still overflows (optional).
4. Trinity Worker pin (existing plan). Do not hop Thinker/Verifier.
5. Ultra: give workers full chat history, then the same prefix rules as Fusion.

Do not add OpenRouter `prompt_cache_key` on Bifrost. Do not vendor `dsh` as a router.
