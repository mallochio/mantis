# Trinity prompt cache

**Status:** plan only. Do not implement until Fusion cache markup has landed and been smoke-tested.

**Depends on:** family-aware prompt-cache dispatch in `apps/api/providers.py` (Anthropic `cache_control`, OpenAI explicit breakpoints, Gemini unmarked / implicit). Trinity should reuse that path, not add a third dialect.

## Question

Can Trinity keep its Worker/Thinker/Verifier quality loop while getting Fusion-like prompt-cache hit rates on a sticky model, instead of paying a cold prompt on every inner hop?

## Why cache dies today

Prompt cache is per upstream model. Bytes before the last volatile message must be identical, in the same order, on the same model.

Trinity currently breaks that on every inner step:

1. **Model hop.** `_route()` samples `agent_id` across `slot_order` (seven catalog workers). `_run_model` then calls `slot_models[agent_id]`. Worker on Flash, Thinker on Luna, Verifier on Opus is three cold caches even if the text prefix is identical.
2. **Role prompt is not the only delta.** `_build_messages` shares `SYSTEM_PROMPT` + client history, then appends a role-specific last user message. That last-user pattern is cache-friendly. Tools are not: `_run_model` passes `self.tools` to Thinker and Verifier as well as Worker, so critic calls carry a tool schema they do not use.
3. **Unstable tool JSON.** Tool-call ids are normalized per request, but tool *definitions* must be byte-stable across inner calls. Any per-role `tool_choice` / `response_format` / `controls` that lands in the body before the messages will bust the prefix (Anthropic tools live next to `system` in the native body).
4. **OpenRouter-only stickiness.** `session_id` / `prompt_cache_key` in `_provider_response` apply only when `adapter == "openrouter"`. Live Trinity workers are Bifrost. Those fields are unused there; cache has to come from provider-native prefix identity plus the Fusion family markup.

Hard limits that this plan does not try to lift:

- Cache cannot be shared across models. Flash → Opus is always a cold prompt.
- Bifrost does not honor OpenRouter `prompt_cache_key`. Do not add it on the Bifrost body.

## Target shape

Mirror Fusion: one sticky lead, cheap critics on a second sticky identity, frozen prefix, dialect markup already implemented for Fusion.

| Role | Pin | Tools | Prefix |
|---|---|---|---|
| Worker | First Worker sample of the run (session-sticky). Stay on that slot for later Worker turns and tool rounds. | Yes. Client tools, sorted once, reused verbatim. | `SYSTEM_PROMPT` + client history + tools. Last user message is the Worker role prompt / revision feedback. |
| Thinker | One Luna identity for the run (catalog `conductor` / `gpt-5_6-luna` unless overridden). | No. | Same system + history. No tools in the body. Last user message is `THINKER_PROMPT`. |
| Verifier | Same Luna identity as Thinker. | No. | Same system + history as Thinker. Last user message is `VERIFICATION_PROMPT`. |

Two cache namespaces per run is expected and fine: Worker-with-tools on model A, critics-without-tools on Luna. Do not try to make Thinker share Worker’s cache.

Rare quality upgrade (e.g. Flash Worker → Opus/Sol mid-run) is allowed. Treat that turn as a cold-cache cost, then re-pin. Do not resample every step.

## Implementation (when scheduled)

All of this is Trinity-only. Fusion, Ultra, and the gateway router stay out of scope.

### 1. Session-sticky Worker

In `TrinityRun`, remember `worker_slot` on the first Worker route. Later Worker steps and tool continuations use that slot, ignoring a newly sampled `agent_id`. Keep the router’s role decision; only freeze the Worker *model*.

Store the pin on the run object so a tool-paused resume does not resample.

### 2. Pin Thinker / Verifier

Add a catalog `[trinity]` block analogous to `[fusion]`, for example `thinker = "gpt-5_6-luna"` and `verifier = "gpt-5_6-luna"`. Default both to the existing conductor Luna slot. `_model_name` for those roles returns the pin, not `slot_models[agent_id]`.

### 3. Frozen prefix

Keep `_build_messages` as: system + non-system client history + one trailing role user message. Do not fold the role prompt into system. Do not prepend per-step “you are the Thinker” system text.

Tool rounds on Worker must append assistant/tool messages *after* that frozen prefix, same as Fusion `main_messages`.

### 4. Tools only on Worker

`_run_model` already zeros `tool_choice` / `response_format` / `controls` for non-Worker roles. Also pass `tools=None` (or `[]`) for Thinker and Verifier so the Anthropic native body does not include a tools array. Worker keeps `self.tools`.

Serialize `self.tools` once at run start (`sort_keys` / stable key order) and reuse that list object’s JSON. Do not rebuild from the client body on each inner call.

### 5. Reuse Fusion cache markup

Do not add Trinity-specific breakpoint code. Inner completions already go through `providers._build_request`:

- `bedrock/anthropic/claude-*` / `claude-*` → `cache_control` (chat or native Messages)
- `gpt-5.6-*` / `o3*` / `o4*` chat → `prompt_cache_options` + `prompt_cache_breakpoint`
- `gemini-*` → no extra fields (implicit prefix cache)
- master switch `MANTIS_CACHE_BREAKPOINTS`; OpenAI-only override `MANTIS_OPENAI_CACHE_BREAKPOINTS`

### 6. Trim like Fusion, last

If Worker history blows the window, prune tool-result bodies before dropping oldest non-system groups, so the system + early client history prefix survives. Copy Fusion’s two-pass prune; do not invent a third trimmer.

## Acceptance

- Unit: two Worker calls in one run send the same model id and the same message prefix (system + history); only the last user message / tool tail may change.
- Unit: Thinker and Verifier share one Luna slot and are built with no `tools` key (or empty) on both Chat Completions and native Anthropic bodies.
- Unit: Gemini Worker bodies still have no `cache_control` / `prompt_cache_options` (no regression from Fusion).
- Unit: a forced Worker upgrade changes model once, then stays pinned.
- Smoke (`pi --session-id`, tools off): Trinity still recalls a planted token on turn 2.
- Smoke with tools: second Worker tool round on the sticky model shows non-zero cached prompt tokens when the provider reports them (OpenAI/Anthropic). Gemini may report cache implicitly or not at all; absence of 400s is the Gemini bar.
- Quality: do not require a SWE-rebench win in the first patch. Fail the patch if multi-turn recall or tool continuation regresses.

## Out of scope

- Sharing cache across Worker and critics, or across Flash and Opus.
- Adding OpenRouter `prompt_cache_key` to Bifrost requests.
- Changing the trained router’s role policy, other than ignoring `agent_id` for model selection after the pin.
- Ultra conductor history (separate bug: workers only see the last user message).
