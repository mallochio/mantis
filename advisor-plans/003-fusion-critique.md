# Executive summary

Fusion is a functional prototype with passing focused tests, but it is not production-safe for unattended Pi/sidekick workloads. The most important confirmed problems are:

1. **Headless execution is neither a real worktree nor a sandbox.** It runs model-generated `bash -c` commands in an empty temporary directory, with unrestricted host filesystem, environment, process, and network access, then deletes all work. This combines poor coding-task quality with severe host-security risk.
2. **Usage is double-counted.** `providers._provider_response()` attributes usage to the active run, and Fusion adds the same usage again. Reported tokens and derived cost are therefore approximately doubled for successful Fusion calls.
3. **Provider reasoning state is discarded.** Fusion retains only text and reconstructed tool calls, dropping Responses API reasoning items and Anthropic replayable thinking/signature metadata. Tool continuation can consequently lose provider-required state or degrade quality.
4. **Rejected sidekick reports are not added to sidekick history.** A follow-up asks the sidekick to revise feedback without showing it its previous report.
5. **Crash recovery is at-least-once, not exactly-once.** State is persisted before and after an entire advance, not at each provider boundary. A process failure after an upstream response can repeat model calls and tool requests.
6. **Context management is unsafe around tool transcripts.** It ignores tool-schema tokens, accepts arbitrarily large tool outputs, can split assistant/tool-call groups, reserves the catalog maximum output (up to 128k for the checked-in defaults), and has no summarization.
7. **Memory and file-store locking are not safe for all deployment topologies.** Memory uses no run lock; file locks are process-local; Redis leases are not renewed.
8. **Fusion has no checked-in quality/cost evaluation.** The default configuration invokes a strong main model plus a substantial sidekick model and a review after every report, but there is no evidence that this improves the cost/quality frontier over direct Pi, router, Trinity, or Ultra sessions.

No code was modified during this review. The worktree already contained a modification to `scripts/fusion_headless.py` and two unrelated untracked evaluation files.

# Findings

## Critical

### 1. Headless model commands have unrestricted host access

**Files:**  
- `scripts/fusion_headless.py:70-125`
- Particularly `:93-99`

**Observed behavior and code path**

`_execute_tool_call()` executes model-provided text directly:

```python
subprocess.run(["bash", "-c", command], cwd=str(workdir), ...)
```

The temporary working directory is only a current-directory choice. It is not a security boundary. Commands can use absolute paths, `..`, inherited credentials, network clients, process inspection, or destructive host commands.

The subprocess inherits the complete parent environment because no restricted `env` is supplied. In managed mode, that environment is expected to contain provider credentials and Mantis configuration.

**Impact**

- Remote-code execution on the machine running the headless harness.
- Potential provider-key and local-secret exfiltration.
- Host filesystem modification or deletion.
- Network pivoting and supply-chain exposure.
- A malicious prompt, compromised model response, or prompt injection in repository content is sufficient to trigger it.

This is a confirmed implementation property, not merely a hypothetical shell-injection bug: arbitrary shell execution is the intended tool contract.

**Recommendation**

Do not describe or operate this as a sandbox. Before unattended use:

- Execute in a disposable container/VM or OS sandbox.
- Mount only a dedicated repository worktree, preferably with no access outside it.
- Pass a minimal allowlisted environment; never inherit provider/API credentials.
- Disable network by default.
- Apply CPU, memory, process-count, filesystem, and wall-clock limits.
- Run as an unprivileged dedicated user.
- Prefer structured tools over arbitrary shell where possible.
- Require an explicit unsafe opt-in if unsandboxed execution remains available.

**Suggested tests**

- Verify commands cannot read a sentinel outside the mounted worktree.
- Verify provider-key environment variables are absent.
- Verify network access is denied.
- Verify process, memory, and timeout limits.
- Verify symlink and `../` escapes fail.
- Run destructive-command tests only inside a disposable container.

---

## High

### 2. The headless harness does not operate on a project worktree and deletes all results

**File:** `scripts/fusion_headless.py:163-196`, especially `:183`

**Observed behavior and code path**

Despite the module saying it executes tools “in a scratch worktree,” it creates:

```python
tempfile.TemporaryDirectory(prefix="fusion-pi-")
```

No repository is cloned, copied, mounted, or linked into that directory. The sidekick starts in an empty directory, and the directory is deleted when `run_session()` exits.

**Impact**

- Repository repair tasks cannot inspect or modify the intended project.
- Any files created by the sidekick are discarded.
- The final report may claim successful implementation while no durable patch exists.
- Quality-to-cost is particularly poor for Pi/SWE-style coding tasks: expensive model/tool loops can produce no deliverable.
- The name “worktree” and tool description are operationally misleading.

**Recommendation**

Require an explicit repository path and create a real isolated Git worktree:

- `git worktree add --detach <scratch> <revision>`, or copy a supplied source tree when Git is unavailable.
- Return or persist the resulting diff and test report.
- Preserve failed worktrees by default for diagnosis, or provide `--keep-worktree`.
- Validate cleanliness/revision before execution.
- Distinguish output/artifact directories from run-state directories.

**Suggested tests**

- Seed a repository, ask the sidekick to change a file, and assert the resulting diff is preserved.
- Verify the original checkout is unchanged.
- Verify `--keep-worktree` and cleanup behavior on success, failure, timeout, and interruption.
- Fail early when no repository is supplied for a repository-editing task.

---

### 3. Fusion usage and cost accounting are double-counted

**Files:**  
- `apps/api/providers.py:646-652`
- `apps/api/fusion.py:277-281`
- `apps/api/fusion.py:295-299`
- API exposure at `apps/api/fusion.py:364-374`, `:536-550`

**Observed behavior and code path**

`advance_fusion_run()` installs the Fusion run as `active_run` at `fusion.py:515`. Then:

1. `providers._provider_response()` calls `run.add_usage(...)` at `providers.py:648-652`.
2. `_call_main()` and `_call_sidekick()` call `self.add_usage(...)` again with the same usage at `fusion.py:281` and `:298`.

The tests monkeypatch `FusionCoordinator._call_worker`, bypassing the provider-level accounting and therefore do not detect this.

**Impact**

- Fusion API token totals are approximately doubled.
- Per-model usage and cached-token totals are doubled.
- Any cost derived from `usage_models` is overstated.
- Cost/quality comparisons against other harnesses are invalid.
- Alerts and budget controls based on reported usage can fire incorrectly.

Retries that fail before returning usage remain a separate under-accounting concern; the double count applies to successful returned calls.

**Recommendation**

Choose exactly one accounting owner. The least disruptive option is to rely on `providers._provider_response()` while `active_run` is set and remove Fusion’s second `add_usage()` calls. Alternatively, prevent provider auto-accounting and have Fusion account explicitly, but do not mix both.

Expose per-role/per-model cost and cached-token metrics from one canonical accumulator.

**Suggested tests**

- Run `_call_worker()` through a mocked provider response without replacing `_call_worker`, and assert one unit of usage is recorded exactly once.
- Cover main, sidekick, failover, missing usage, cached tokens, and reasoning-token details.
- Assert final Fusion aggregate equals the sum of successful upstream responses.

---

### 4. Provider-specific reasoning/thinking continuation state is discarded

**Files:**  
- `apps/api/fusion.py:190-217`
- `apps/api/fusion.py:254-280`
- `apps/api/providers.py:409-460`
- `apps/api/provider_protocols.py:53-97`, `:182-232`
- `apps/api/anthropic_protocols.py:50-86`, `:155-185`

**Observed behavior and code path**

Provider adapters can return:

- `reasoning` and `reasoning_details` for OpenAI/Responses-style models;
- `_anthropic_content` and `_anthropic_tool_ids` for native Anthropic continuation.

Fusion extracts only:

```python
text = ...
tcs = msg.get("tool_calls") or []
```

It then reconstructs an assistant message with only `content` and `tool_calls`. The original provider metadata is never appended to either Fusion transcript.

Consequences by provider:

- Responses API reasoning items are not reinserted by `chat_messages_to_input()`, because `reasoning_details` is absent.
- Anthropic’s replayable content and tool-ID mapping are absent, so `chat_to_anthropic()` falls back to reconstructed text/tool blocks.
- Signed or encrypted thinking continuity is lost.
- Reasoning summaries are also unavailable to later Fusion turns.

**Impact**

- Follow-up quality can degrade because prior hidden state is unavailable.
- Some providers/models may reject tool continuation when required signed thinking blocks are missing.
- A failover or resumed session can lose provider-specific identity/state even though text history survives.
- Reasoning tokens are charged but their continuation value is discarded.

**Recommendation**

Have `_call_worker()` return a canonical assistant message, not just text and calls. Persist only provider-approved replay metadata:

- `reasoning_details` required for Responses continuation;
- `_anthropic_content` and tool-ID mappings required for Anthropic replay;
- annotations/citations only when needed.

Keep public report text separate from internal continuation state. Define explicit redaction and retention rules for opaque/encrypted blocks.

**Suggested tests**

- Responses model: tool call with a reasoning item, resume, and assert the item is included exactly once.
- Anthropic model: signed-thinking/tool-use response, resume, and assert replayable blocks and original provider IDs survive.
- File-store round trip for both formats.
- Ensure internal blocks are not exposed by Fusion API responses or activity logs.
- Test failover behavior when continuation state is provider-bound.

---

### 5. Sidekick follow-ups omit the report being revised

**File:** `apps/api/fusion.py:425-455`

**Observed behavior and code path**

When a sidekick returns tool calls, the assistant message is appended at `:428`. When it returns a report without tool calls, the report is sent to the main model but is never appended to `sidekick_messages`.

If the main rejects it, only the feedback is appended:

```python
self.sidekick_messages.append({"role": "user", "content": feedback})
```

The next sidekick call therefore sees its earlier brief/tool history and the feedback, but not its own rejected report.

**Impact**

- The sidekick cannot reliably revise specific claims or omissions.
- It may regenerate from scratch, repeat mistakes, or duplicate work.
- Prompt-cache continuity is less useful because the semantically necessary prior output is missing.
- Follow-up cost increases while revision quality decreases.

**Recommendation**

Append every sidekick assistant response before branching on tool calls versus report. Preserve the complete canonical assistant message, including provider continuation metadata from Finding 4.

Structure follow-up text explicitly, for example: “Lead review feedback for your preceding report: …”.

**Suggested tests**

- Reject the first report and assert it appears immediately before the feedback in the next sidekick request.
- Cover multiple rejection cycles and file-store resume.
- Assert the report is present exactly once.

---

### 6. Crash recovery can repeat provider calls and tool requests

**Files:**  
- `apps/api/fusion.py:501-526`
- State transitions at `:404-455`
- Persistence implementation at `apps/api/runs.py:178-218`

**Observed behavior and code path**

`advance_fusion_run()` persists `in_flight += 1` before running, then persists the final state in `finally`. It does not durably checkpoint after individual model calls or before/after each externally visible side effect.

Examples:

- Crash after a provider completed but before final `_put_run()` causes the prior state to be reloaded and the model call repeated.
- Crash after a sidekick tool-call response but before persistence loses the pending calls; a retry can produce different calls.
- Crash during a multi-follow-up loop can repeat multiple main/sidekick turns.
- The caller’s `request_id` only protects a completed event already stored in the run.

**Impact**

- Duplicate provider spend.
- Divergent reports or tool requests after restart.
- Tool execution may be repeated if the client loses a follow-up response and does not retain/reuse the same request ID.
- Exactly-once semantics are not provided despite resumable persistence.

**Recommendation**

Implement a durable turn journal/state machine:

1. Persist an operation/turn ID and intended request before calling the provider.
2. Supply an upstream idempotency key where supported.
3. Persist the canonical provider result before applying the next transition.
4. Persist pending tool calls before returning them.
5. Bind `request_id` to a hash of the request payload and reject reuse with different results.
6. Document semantics as at-least-once where upstream idempotency is unavailable.

**Suggested tests**

Inject process-failure hooks:

- Before provider call.
- After provider response, before state transition.
- After pending calls are set, before API response.
- After tool results are accepted, before sidekick continuation.
- Before final persistence.

Then reload the file store and assert no duplicate transition or explicitly documented at-least-once behavior.

---

### 7. Tool-result validation permits unknown IDs and does not canonicalize ordering

**Files:**  
- `apps/api/fusion.py:329-340`
- `apps/api/fusion.py:259-268`
- API validation at `apps/api/api.py:774-787`

**Observed behavior and code path**

The API rejects duplicate result IDs, and Fusion verifies every pending ID is present. However, it does not reject extra IDs. It then appends every supplied result in client order, including unknown IDs.

Direct Python callers can also provide duplicates because the internal validator does not reject them.

**Impact**

- Unknown tool outputs create orphan tool messages and may cause provider protocol errors.
- Client-controlled ordering can differ from assistant tool-call ordering.
- Direct callers can duplicate results.
- Retries with the same logical result set but different order are not normalized.
- Correlation is only checked for coverage, not exact identity.

**Recommendation**

Require exact set equality between pending IDs and result IDs, reject duplicates internally, and append results in pending-call order. Optionally validate result payload hashes for idempotency.

Do not synthesize missing IDs. Provider calls lacking IDs should fail before exposure rather than fall back to `call_0`, which can collide across turns (`fusion.py:211`).

**Suggested tests**

- Extra ID, duplicate ID, empty ID, repeated provider ID, reordered results.
- Multiple simultaneous calls resumed in reverse client order.
- Same `request_id` with a different payload must return conflict, not a cached unrelated event.
- Resume after persistence with exact ordering preserved.

---

### 8. Context trimming can produce invalid or incomplete tool transcripts

**File:** `apps/api/fusion.py:135-180`

**Observed behavior and code path**

Trimming preserves the first message and greedily walks backward. It treats each message independently and can retain a tool result without its assistant tool call, or vice versa.

It also:

- Assumes the first message is a system message.
- Breaks entirely on the first newest message that exceeds remaining budget instead of considering older smaller units.
- Returns the system message even when it alone exceeds the budget.
- Uses a global configured window rather than a resolved per-model input limit.
- Does not account for tools/schema tokens.
- Uses a rough four-characters-per-token estimate.
- Has no summarization or explicit truncation marker.

**Impact**

- Provider 400 responses from malformed assistant/tool ordering.
- Loss of the original brief or critical tool context.
- Unexpected quality collapse near the context limit.
- Requests can still exceed the actual provider context window.
- Cacheable prefixes change abruptly when trimming starts.

**Recommendation**

Represent history as atomic turn groups:

- Assistant tool-call message plus all correlated tool results must be retained or removed together.
- Always retain the task/brief, not merely list element zero.
- Use model-aware tokenization or a conservative provider-specific estimator.
- Include tool definitions and provider metadata in the estimate.
- Resolve context and output limits per selected model.
- Summarize old completed turns into a stable summary message before dropping them.
- Fail clearly when system/task/tool schema alone exceeds the budget.

**Suggested tests**

- Multiple tool rounds under a forced small budget.
- Oversized single tool result.
- Tool schemas that dominate input size.
- System message exceeding budget.
- Verify no orphan tool messages after trimming.
- Verify both default configured models’ actual request limits.

---

### 9. Unbounded tool output can exhaust memory, storage, and context

**Files:**  
- `scripts/fusion_headless.py:112-118`
- `apps/api/api.py:170-171`, `:766-778`
- `apps/api/fusion.py:259-268`
- Contrast with native run truncation at `apps/api/runs.py:799-821`, `:1128-1150`

**Observed behavior and code path**

The headless harness captures complete stdout and stderr in memory. The Fusion API accepts large result content up to the global 50 MiB request-body limit. Fusion persists the full content and sends it back to the model.

Unlike Trinity/Conductor, Fusion does not use `RUN_MAX_MSG_BYTES` when storing tool results.

**Impact**

- A single `find`, test log, binary dump, or accidental recursive output can consume tens of MiB in memory and pickle/Redis storage.
- Every subsequent sidekick call may repay that input cost.
- Large results can exceed context limits and trigger the unsafe trimming behavior above.
- API worker latency and file persistence time increase sharply.

**Recommendation**

Apply layered limits:

- Stream subprocess output with a byte cap.
- Preserve head and tail with an explicit truncation marker.
- Set per-result and per-run byte limits substantially below the HTTP body cap.
- Store large artifacts separately and send a bounded summary/reference to the model.
- Enforce limits in the API and state machine, not only the harness.

**Suggested tests**

- Multi-megabyte stdout/stderr.
- Binary/non-UTF-8 output.
- Timeout after partial output.
- API rejection/truncation behavior.
- Persistence size and context accounting after truncation.

---

### 10. Fusion endpoints bypass global concurrency capacity controls

**Files:**  
- Chat semaphore: `apps/api/api.py:170-182`, `:693-756`
- Fusion routes: `apps/api/api.py:802-824`

**Observed behavior and code path**

`/v1/chat/completions` acquires `_capacity`. The three Fusion endpoints do not. A delegate can synchronously perform planning plus sidekick model calls, and a follow-up can perform several sidekick/main rounds under one HTTP request.

**Impact**

- Authenticated clients can create unbounded concurrent expensive Fusion calls.
- Provider connection pools, thread pools, memory, and run stores can be exhausted.
- Fusion load is not governed by `MANTIS_MAX_CONCURRENT_REQUESTS`.
- Headless sessions can produce prolonged requests without admission control.

**Recommendation**

Add dedicated Fusion admission limits, preferably:

- Global request capacity.
- Per-API-key and per-run limits.
- Separate main/sidekick provider concurrency quotas.
- Queue timeout and `429/Retry-After`.
- Metrics for queued, active, rejected, and duration.

Ensure capacity is released on all exceptions and disconnects.

**Suggested tests**

- Concurrent delegate/follow-up calls exceeding capacity.
- Capacity release after provider errors, cancellation, and client disconnect.
- Per-run contention behavior.

---

## Medium

### 11. Review parsing accepts ambiguous or malformed responses

**File:** `apps/api/fusion.py:319-327`

**Observed behavior**

Any word-boundary occurrence of `ACCEPT` wins, including text such as “do not ACCEPT.” Any response not matching `FOLLOW_UP:` defaults to acceptance.

**Impact**

Malformed, verbose, truncated, or contradictory main output can silently approve poor sidekick work. This biases toward false acceptance and lowers quality.

**Recommendation**

Use a typed structured response or require exact full-string matching:

- Exact `ACCEPT`; or
- Exact `FOLLOW_UP:` with non-empty feedback.
- Treat everything else as a protocol error or bounded repair retry.

**Suggested tests**

Cover “do not ACCEPT,” `ACCEPTED`, empty follow-up, mixed accept/follow-up, markdown wrappers, truncated output, and whitespace/case policy.

---

### 12. Output reservation is excessively large and not role-specific

**Files:**  
- `apps/api/fusion.py:182-200`
- Checked-in defaults: `config/catalog.toml:136-141`
- Worker limits around `config/catalog.toml:70-99`

**Observed behavior**

Fusion requests the model’s full catalog maximum output and subtracts it from the configured context window. The checked-in sidekick has a 128k catalog maximum; that is treated as the normal output cap, not merely a provider ceiling.

The same strategy is used for a short main plan, an `ACCEPT` review, and a sidekick coding turn.

**Impact**

- Large latency and runaway-output exposure.
- Unnecessarily reduced input budget.
- Main review calls reserve orders of magnitude more output than needed.
- Cost predictability is poor.
- If the provider interprets maximum output as a strong allocation signal, quality/latency may worsen.

**Recommendation**

Separate provider ceilings from orchestration budgets. Configure role/phase caps, for example:

- Main plan: a few thousand tokens.
- Main review: hundreds to low thousands.
- Sidekick turn/report: a bounded task-appropriate cap.
- Explicit reasoning budget where the provider supports it.

Reserve output per phase and model context limit.

**Suggested tests**

Assert phase-specific request limits and correct input budgets for each configured model.

---

### 13. Errors become terminal HTTP 200 responses with weak diagnostics

**Files:**  
- `apps/api/fusion.py:342-362`, `:379-392`
- `apps/api/api.py:802-824`

**Observed behavior**

All state/provider exceptions are caught and converted to `status: "error"`. The public event omits the top-level error message, although activity may contain an `error` field. The API still returns HTTP 200.

Malformed tool results permanently poison the run rather than returning a client error while preserving `awaiting_tools`.

**Impact**

- Monitoring based on HTTP status misses failures.
- Clients must inspect an application status field.
- Recoverable client mistakes become unrecoverable runs.
- Diagnostics are inconsistent: error detail is absent at the top level but may leak through activity.

**Recommendation**

Distinguish:

- Validation/conflict errors: 400/409, no state transition.
- Busy/capacity: 429.
- Provider/transient errors: 502/503 with retryability.
- Genuine terminal run failure: typed error event.

Include a sanitized error code and retryability flag. Keep raw provider text server-side only.

**Suggested tests**

Missing/extra results, unknown run, wrong run type, provider timeout, provider 4xx/5xx, lock timeout, and retry after validation failure.

---

### 14. File and memory locking are topology-dependent and unsafe

**Files:**  
- `apps/api/fusion.py:493-498`, `:507-526`
- `apps/api/runs.py:187-250`

**Observed behavior**

- Memory store returns `nullcontext()`.
- File locks are an in-process dictionary of `threading.Lock`, explicitly assuming a single process.
- Status reads use the same lock only for file/Redis; memory reads can race.
- Redis lock leases have a fixed timeout and no renewal.

`request_lock` serializes follow-ups with request IDs on the same in-memory object, but does not protect status reads, delegate persistence, or cross-process file-store access.

**Impact**

- Multi-worker Uvicorn with file storage can concurrently load and overwrite the same pickle.
- Memory status can observe partially mutated state.
- Redis work exceeding the lease can overlap with a second worker.
- Last-writer-wins persistence can lose events and usage.

**Recommendation**

- Enforce single-process mode for file storage or use OS file locks.
- Use a per-run lock for memory storage.
- Renew Redis leases or use optimistic version/CAS checks.
- Add a persisted run revision and reject stale writes.
- Document supported worker topology in readiness checks.

**Suggested tests**

Concurrent follow-ups and status calls in threads and separate processes; Redis lease-expiry simulation; stale-write detection.

---

### 15. Follow-up semantics are incomplete and the `message` parameter is dead

**Files:**  
- Internal parameter: `apps/api/fusion.py:379-399`, `:462-468`, `:501-506`
- API request: `apps/api/api.py:774-778`

**Observed behavior**

`message` is threaded through several signatures but never used by `_advance()`. The API exposes only non-empty `tool_results`, so a lead/client cannot issue a textual follow-up after completion or report review.

The only follow-up implemented is internal main-model feedback; terminal runs simply return their existing event.

**Impact**

- API contract and implementation diverge.
- External lead/Pi workflows cannot use Fusion as a resumable sidekick service in the originally described manner.
- Dead parameters increase maintenance ambiguity.

**Recommendation**

Either remove `message` from all signatures or implement explicit mutually exclusive follow-up variants:

- Tool-result continuation while `awaiting_tools`.
- Textual revision request in a documented review/completed state.
- Cancellation.

Define valid state transitions and return 409 for invalid transitions.

**Suggested tests**

Message-only follow-up, tools-only follow-up, both/none, completed/error/cancelled states, and persistence across restart.

---

### 16. Configuration precedence and reload behavior are misleading

**File:** `apps/api/fusion.py:62-122`

**Observed behavior**

Catalog values take precedence over environment variables because `_load().get(...)` is evaluated first. `_FUSION_CONFIG` caches parsed TOML indefinitely, while `_fusion_config()` claims it is “reloaded on first call in a new process.”

Numeric values are not bounded: negative follow-ups or context windows are accepted.

**Impact**

- Operators may believe environment overrides are effective when checked-in/catalog settings silently win.
- Runtime catalog changes are not picked up.
- Invalid context/follow-up values produce surprising behavior.

**Recommendation**

Define and document precedence explicitly, conventionally environment over file. Validate:

- `max_follow_ups >= 0`
- sensible context minimum/maximum
- model slots exist and support the required protocol.

Either remove the reload claim or implement mtime/version-based reload.

**Suggested tests**

Environment/catalog precedence, catalog edits after initialization, negative/zero values, unknown slots, malformed TOML, and concurrent initialization.

---

### 17. Delegate creation is not idempotent

**Files:**  
- `apps/api/api.py:759-764`, `:802-807`
- `apps/api/fusion.py:529-533`

**Observed behavior**

The delegate request has no client request ID. A network timeout after run creation causes a retry to create and begin a second run.

**Impact**

Duplicate planning/sidekick spend and orphaned runs are likely under ambiguous network failures.

**Recommendation**

Accept an idempotency key or client-supplied validated run ID, persist request-to-run mapping atomically, and return the existing run for matching payloads. Reject key reuse with different payloads.

**Suggested tests**

Concurrent duplicate delegates, retry after simulated response loss, and key/payload conflicts across file and Redis stores.

---

### 18. Model failover weakens model/session stickiness

**Files:**  
- `apps/api/providers.py:499-516`, `:533-643`
- Fusion slot selection: `apps/api/fusion.py:128-133`

**Observed behavior**

Fusion roles are fixed to catalog slots under normal operation, which is good for cache locality. However, provider transient failures can fail over to any configured pool worker. The next turn again starts from the original slot rather than persisting the successful fallback.

Provider-specific reasoning state may also be incompatible with the fallback, as discussed above.

**Impact**

- Prompt caches are cold on failover.
- Subsequent turns can oscillate back to the original model.
- Behavior and quality can vary by turn.
- Provider-bound reasoning continuation may not be portable.

**Recommendation**

Persist the selected runtime model/provider for the role after failover, at least for the duration of a run, subject to health policy. Restrict failover to protocol-compatible models and reset/summarize provider-bound state explicitly when migration is unavoidable.

**Suggested tests**

Transient failure followed by another turn, provider/protocol migration, cache namespace stability, and persisted model affinity after restart.

---

## Low

### 19. Activity summaries may disclose task/report/error fragments

**Files:**  
- `apps/api/fusion.py:282-288`, `:299-305`, `:345-359`, `:448-454`
- `apps/api/runs.py:420-448`
- API exposure at `apps/api/fusion.py:364-374`, `:536-550`

**Observed behavior**

The first 200 characters of model outputs and feedback are retained in activity and returned by status/follow-up responses. Error details may also be retained. Run pickles contain full briefs, reports, tool outputs, and any future reasoning metadata.

**Impact**

Authenticated users with the shared API key and a run ID can see operational snippets. Persistent file/Redis compromise exposes full task data. If reasoning metadata is added later without a policy, sensitive hidden state could be accidentally exposed.

**Recommendation**

Use opaque activity labels by default. Put textual summaries behind an explicit debug/detail authorization level, redact secrets, and define run-store encryption/retention requirements. Never return raw hidden reasoning or signed/encrypted thinking blocks.

**Suggested tests**

Secret-pattern redaction, default response fields, debug authorization, and persistence permissions.

---

### 20. Headless logging/output files lack a privacy policy

**File:** `scripts/fusion_headless.py:155-160`, `:229-234`

**Observed behavior**

The first 80 characters of tool arguments are printed, and the complete final event can be written with default filesystem permissions. Tool arguments may contain secrets or sensitive paths.

**Impact**

Terminal logs and output artifacts can disclose credentials or proprietary task details.

**Recommendation**

Redact common secret patterns, log tool name and argument hash by default, and create output files with mode `0600`. Add an explicit verbose/debug option for raw arguments.

**Suggested tests**

Arguments containing tokens, passwords, authorization headers, and sensitive paths; verify file mode.

---

### 21. Static typing has a checked-in Fusion error

**File:** `apps/api/fusion.py:473-476`

**Observed behavior**

Targeted mypy reports `advance_idempotent()` returning `Any` from `request_events`, despite declaring `dict[str, Any]`.

**Impact**

Small immediate impact, but it indicates the typed event contract is not enforced end-to-end.

**Recommendation**

Type `request_events` and event objects consistently, or cast/validate cached events. Prefer `TypedDict` or Pydantic models for internal state-machine events.

**Suggested tests**

Static check in CI for all three reviewed files.

# Cache and cost assessment by harness

## `/v1/fusion/*` direct endpoint

**Routing and model stickiness**

Fusion does not route each role through the public `model="mantis/base"` auto-router. It directly calls the configured catalog slots:

- Main: `gpt-5_6-sol`
- Sidekick: `gpt-5_6-luna`

This provides stable logical role assignment unless provider failover activates. It also means Fusion does not benefit from request-level cheap/expensive routing based on task complexity.

**Stable prefixes**

The two histories generally have stable roots:

- Main: fixed system preamble plus original user brief.
- Sidekick: fixed system preamble plus main-generated sidekick brief.

`_prompt_cache_namespace()` hashes through the first user message plus tools, so its fallback namespace is stable across later turns for each role. Deterministic message serialization and tool-ID normalization also generally preserve prefix bytes.

However:

- Main and sidekick intentionally have different cache namespaces.
- Main-generated sidekick briefs can vary even for identical user briefs.
- Trimming abruptly changes the sent prefix.
- Missing sidekick report history reduces semantic continuity.
- Failover changes model/provider and therefore cache locality.

**Provider cache support**

- OpenRouter requests receive stable `session_id` and `prompt_cache_key`.
- OpenRouter Responses requests also receive ephemeral `cache_control`.
- OpenRouter Chat requests for Anthropic-prefixed models get message breakpoints.
- Native Anthropic Messages requests do **not** appear to receive explicit cache-control breakpoints in this path.
- Non-OpenRouter providers do not receive Fusion-specific session/cache keys.

Provider cache metrics can be captured through usage details, but Fusion currently doubles their accounting.

**Cost profile**

A typical successful tool-using run requires at least:

1. Main planning.
2. Sidekick tool request.
3. Sidekick continuation/report.
4. Main review.

Every rejection adds at least one sidekick call and one main review. With `max_follow_ups=3`, spend can grow substantially. The broad output caps and repeated full-history sends amplify cost.

There is no Fusion-specific eval or benchmark checked in demonstrating that this additional spend improves patch quality.

## `scripts/fusion_headless.py`

This uses `/v1/fusion/*`, so its provider-cache characteristics are the same as above. HTTP connection reuse is present through one `httpx.Client`, but no router session header is relevant to these custom routes.

Its operational quality-to-cost ratio is currently poor for repository tasks because:

- It starts in an empty directory.
- It discards all generated files.
- It has no patch/artifact output.
- It has no retry policy.
- It can spend up to ten tool iterations plus internal review loops.

The increased client timeout from the pre-existing worktree modification reduces premature HTTP timeout risk but does not add server-side progress, retry, or idempotent recovery.

## Pi or another client using `/v1/chat/completions` with `model="mantis/base"`

This is not Fusion. It goes through the router proxy and can use `X-Route-Session`, `metadata.session_id`, or `user` for router affinity. This path is likely cheaper for simple tasks because it can select a single endpoint model and avoid mandatory main planning/review calls.

It also has explicit global concurrency admission control, unlike Fusion.

## `mantis/trinity` / `mantis/ultra`

These are separate orchestration modes with more mature native run machinery, including bounded tool-result persistence and established response metadata handling. They are not substitutes automatically selected by Fusion and need direct comparative evaluation.

## Overall cache/cost conclusion

Fusion’s logical prefixes and OpenRouter cache keys are reasonably designed, but the benefit is undermined by:

- double-counted telemetry;
- dropped provider continuation state;
- model failover without persisted affinity;
- unsafe context trimming;
- missing sidekick reports;
- no explicit native-Anthropic breakpoint handling;
- very large output reservations.

No checked-in evidence supports a claim that Fusion currently offers a favorable quality-to-cost ratio.

# Reasoning-trace/API exposure assessment

## Capture and continuation

- Provider adapters capture Responses reasoning items and Anthropic replayable content.
- Fusion discards them at `_call_worker()`.
- Therefore reasoning is neither correctly persisted nor continued in Fusion.
- Token details may still show reasoning-token spend, but the trace/state itself is lost.

## Duplication

Fusion itself does not currently duplicate reasoning blocks because it drops them. If canonical metadata retention is added, tests must prevent double insertion during serialization, normalization, retry, and file-store reload.

## Persistence

Current Fusion pickles do not retain provider reasoning metadata. They do retain full prompts, reports, tool outputs, API request-event snapshots, and activity snippets.

If provider reasoning state is persisted to repair continuation, opaque encrypted/signed blocks should be stored internally but treated as sensitive provider state, not user-facing text.

## Logging and API exposure

The Fusion response model does not contain `reasoning` or `reasoning_details`, which is a positive default. Unlike the chat SSE path, Fusion has no mechanism that emits reasoning deltas.

Activity does expose short content summaries and error details. Those are not hidden chain-of-thought, but they can contain sensitive task material.

## Privacy/security recommendation

Maintain a strict separation:

- **Public:** report, status, sanitized usage, pending tool calls.
- **Internal replay state:** provider reasoning items, signatures, encrypted content, original provider tool IDs.
- **Operator debug:** sanitized metadata only, gated by authorization and configuration.

Never place raw replay state in activity, normal logs, output JSON, or the Fusion API schema.

# Concurrency, persistence, and follow-up assessment

- **Idempotency:** Follow-up responses are cached by request ID, but the ID is not bound to request content. Reusing an ID with different tool results returns the old event silently. Delegate is not idempotent.
- **Locking:** Redis has a distributed lease; file locks are process-local; memory has no outer run lock. File mode is safe only under the undocumented single-process assumption.
- **Persistence:** File writes use temp file plus `os.replace`, which is atomic at the file replacement level and uses mode `0600`. There is no fsync/directory sync guarantee.
- **Crash recovery:** State is persisted only around the whole advance, so provider calls are at-least-once.
- **Retries:** Provider failover handles configured transient failures, but the headless HTTP client has no retry/backoff behavior. Retrying delegate can duplicate runs.
- **Follow-up semantics:** Only tool-result continuation is exposed. The threaded `message` parameter is unused. Validation errors terminally poison the run.
- **Lifecycle:** There is a status route but no Fusion cancellation/deletion route. Completed and errored runs remain until store TTL/sweeping.
- **Observability:** Activity and aggregate usage exist, but there are no Fusion admission, duration, retry, cache-hit-ratio, run-state, or failure-reason metrics. Provider activity is produced internally, but the custom endpoints do not stream progress.

# Tool-result ordering/deduplication assessment

Confirmed strengths:

- The API rejects duplicate `tool_call_id` values in one request.
- Fusion checks that every pending call receives a result.
- The headless harness preserves pending-list order during normal execution.
- Request IDs prevent a completed follow-up from being advanced twice when the exact same ID is reused.

Confirmed weaknesses:

- Unknown extra IDs are accepted and appended.
- Internal/direct callers can submit duplicates.
- Results are appended in client order rather than pending-call order.
- Request IDs are not bound to payloads.
- Generated fallback call IDs such as `call_0` are not globally unique across turns.
- There is no durable execution ledger proving whether a tool call was already executed after client/server failure.
- Oversized results are neither truncated nor artifactized.
- Provider-specific original tool IDs are discarded by Fusion.

The desired invariant should be: exact one-to-one set equality, canonical pending order, stable server-owned ID, payload hash, and persisted execution/acknowledgment state.

# Validation performed

The repository was found at `/Users/sid/Personal/other/mantis`; the initially supplied working directory was empty.

## Worktree state

Command:

```bash
git status --short
```

Result before review:

```text
 M scripts/fusion_headless.py
?? eval/report-xroutebench-binary.md
?? eval/results-xroutebench-binary.json
```

The existing `fusion_headless.py` changes increase the HTTP timeout from 120 to 600 seconds and add `--output`. I did not modify or revert them.

## Ruff

Command:

```bash
.venv/bin/ruff check apps/api/fusion.py apps/api/api.py scripts/fusion_headless.py
```

Result:

```text
All checks passed!
```

## Mypy

Command:

```bash
.venv/bin/mypy apps/api/fusion.py apps/api/api.py scripts/fusion_headless.py
```

Result: failed with one targeted error:

```text
apps/api/fusion.py:476: error: Returning Any from function declared to return "dict[str, Any]"  [no-any-return]
Found 1 error in 1 file (checked 3 source files)
```

## Fusion-specific tests

Command:

```bash
.venv/bin/pytest -q --no-cov tests/test_fusion.py
```

Result:

```text
11 passed, 1 warning in 1.19s
```

The warning is a Starlette deprecation concerning `httpx`/`TestClient`, not a Fusion failure.

## Fusion plus relevant provider/protocol tests

Command:

```bash
.venv/bin/pytest -q --no-cov \
  tests/test_fusion.py \
  tests/test_parity_upstream.py \
  tests/test_provider_protocols.py \
  tests/test_anthropic_protocols.py
```

Result:

```text
48 passed, 1 warning in 1.06s
```

## Limitations

- `--no-cov` was used for targeted diagnostics because repository pytest defaults enforce a global 90% coverage threshold, which is not meaningful for a small selected test set.
- No network-dependent live provider calls or paid evaluation were performed.
- No Redis integration was run.
- No Fusion-specific quality/cost eval harness or manifest was found.
- The existing tests mock `_call_worker`, so they do not exercise the confirmed double-accounting path or reasoning-metadata loss.
- No destructive sandbox test was performed because the harness is not actually sandboxed.

# Prioritized remediation plan

## Quick wins

1. **Disable or clearly gate unsandboxed headless shell execution.**
2. **Require a real repository/worktree and preserve a patch artifact.**
3. **Remove duplicate Fusion `add_usage()` calls and add an integration-level accounting test.**
4. **Append every sidekick report to sidekick history before review.**
5. **Reject extra/duplicate tool IDs internally and canonicalize result order.**
6. **Cap and truncate tool output in both the harness and Fusion API/state machine.**
7. **Use exact/structured main review parsing; never default malformed output to acceptance.**
8. **Add Fusion endpoint concurrency admission control.**
9. **Bind follow-up request IDs to payload hashes; add delegate idempotency.**
10. **Fix the targeted mypy error and introduce typed event definitions.**
11. **Add phase-specific output caps instead of catalog maximums.**
12. **Return typed, sanitized errors with appropriate HTTP statuses without poisoning runs on client validation errors.**

## Structural changes

1. **Persist canonical provider assistant messages**, including only the replay metadata required for Responses and Anthropic continuation.
2. **Introduce a durable turn journal** with operation IDs, checkpoints around provider calls, and explicit at-least-once/exactly-once semantics.
3. **Replace message-by-message trimming with model-aware atomic-turn compaction and summarization.**
4. **Make locking safe for deployment topology:** memory per-run locks, OS file locks or single-process enforcement, Redis lease renewal/CAS revisions.
5. **Persist role-level selected provider/model affinity after failover.**
6. **Define a versioned Fusion protocol** covering delegate, tool suspension, textual follow-up, cancellation, terminal states, errors, and artifact retrieval.
7. **Build a Fusion evaluation harness** comparing:
   - direct Pi/router;
   - Fusion main+sidekick;
   - Trinity;
   - Ultra;
   - cheap versus strong sidekick configurations.
   
   Record patch success, tests passed, latency, actual provider cost, fresh/cached input tokens, tool rounds, retries, and human/automated quality.
8. **Harden observability:** per-role latency/cost/cache metrics, state-transition counters, retry/failover metrics, queue depth, run age, and sanitized correlation IDs.
9. **Define security and retention policy** for prompts, tool outputs, artifacts, provider replay metadata, run pickles, logs, and output files.