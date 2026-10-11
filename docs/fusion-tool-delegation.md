# Fusion tool delegation

Fusion is an optional built-in lead/sidekick loop behind `mantis/fusion`. It uses the
same OpenAI Chat Completions endpoint as Base. A client does not need a Fusion-specific
agent runtime.

## Configuration

Fusion has one lead and one persistent sidekick. The lead delegates through private
server-resolved tools; there is no delegation-mode switch or multi-lane worker engine.
Keep models, provider bindings, reasoning effort, routing pools, and budgets in the
existing `[fusion]` table in `config/catalog.toml`. Restart the stack after changing
runtime configuration; no new model name is needed.

## Who executes which tools

The lead receives client OpenAI function tools plus two private server tools:

- `sidekick({"message": "complete brief", "block": true})` delegates to one persistent
  sidekick. The brief must include the goal, current context, constraints, and checks.
- `read_subagent({"block": true, "timeout": 60})` reads its latest report or waits for
  background work. The timeout is in seconds.

Mantis resolves those delegation calls internally. Do not register or execute them
in the client. If a client already declares either name, Mantis renames its private
tool deterministically, for example to `fusion_sidekick`; the client tool remains
available under its original name.

Ordinary lead tool calls still return to the harness with `finish_reason="tool_calls"`.
The harness executes them and sends back the assistant message and `role="tool"`
results with the exact returned IDs. A sidekick call also returns to the harness if
it has no matching server tool. Private delegation calls are not part of that public
tool-call stream. A plain lead reply completes the run; the lead can answer a small
chat turn without delegating.

The lead owns planning, review, user-facing decisions, commits, pushes, pull requests,
CI, and secrets. The sidekick handles bounded mechanical implementation and checks.
It retains earlier handoffs but does not see user messages directly. The lead must
pass new requirements explicitly and review actual diffs, command output, and files.

Blocking delegation is the default. `block=false` starts background work only when
all sidekick tools can execute on the server. Otherwise Mantis waits and reports
that limitation. Another `sidekick` call during background work sends an interrupt
for the next sidekick turn. A completed background report is delivered to the lead
as a notification. Do not poll `read_subagent` in a loop.

## Shared workspace

A workspace is an existing absolute directory on the **Mantis API machine**. Local
harnesses can share their repository; sending a remote path does not make it available.

Export the allowlist before starting the API:

```bash
export MANTIS_FUSION_WORKSPACE_ROOTS="$HOME/projects"
```

Multiple roots use the platform path separator (`:` on macOS and Linux). Mantis
resolves paths, including symlinks, and requires the directory to be under an
allowlisted root. With no roots configured, no shared workspace is allowed. Prefer
a narrow project root rather than your whole home directory.

For new Fusion chat runs, workspace selection uses this precedence:

1. Explicit `tool_options.workspace`.
2. `X-Mantis-Workspace` request header.
3. Directory announced in a system or developer prompt.
4. `MANTIS_FUSION_DEFAULT_WORKSPACE`, then `[fusion].default_workspace`.

Recognized prompt forms include `Working directory: /absolute/path`,
`Current working directory: /absolute/path`, `cwd: /absolute/path`,
`<cwd>/absolute/path</cwd>`, and an `<env>` block with `PWD=/absolute/path`.

An invalid explicit workspace rejects the request. An invalid header, prompt, or
default is logged and ignored. Selection does not fall through to a lower-priority
candidate after that failure. Use an explicit path when working in the intended
repository must be guaranteed. Discovery applies only when creating a new chat run,
not when resuming an existing one.

With a valid workspace, the sidekick gets built-in file, shell, and Python tools by
default, even without `server_execution=true`. `tool_options.enabled` can narrow
that set, for example to `["files", "shell"]`. If no shared workspace is selected,
`server_execution=true` plus enabled bundles uses a per-run directory under
`MANTIS_FUSION_WORKSPACE_ROOT` (default `.mantis/fusion`). Otherwise tools execute in
the client. Lead tools remain client-owned in all cases.

### Trust boundary

The local executor is for trusted callers and tasks. **A workspace is not a security
sandbox.** File tools check path containment, but shell and Python code run with the
API process's OS permissions. A working directory and reduced environment do not
isolate the host. Use real OS/container isolation before accepting untrusted execution.

The sidekick policy reserves authority actions for the lead. A regex also denies
common `git commit`, `git push`, and `gh pr`/`gh release` commands. This is policy
support, not secure containment. It does not prove that other commands or programs
cannot perform those actions. Review delegated changes and outputs before publishing.

## Harness-neutral request

This OpenAI SDK request example is not a live verification result. Replace the
workspace with an existing directory under the API's allowlist.

```python
import os
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8088/v1",
    api_key=os.environ["MANTIS_API_KEY"],
)
response = client.chat.completions.create(
    model="mantis/fusion",
    messages=[{
        "role": "user",
        "content": "Inspect this project, fix the failing test, and report actual checks.",
    }],
    extra_body={
        "tool_options": {
            "workspace": "/absolute/path/to/project",
            "enabled": ["files", "shell"],
        },
        "budget": {"max_turns": 32, "timeout_ms": 600000},
    },
    extra_headers={"x-mantis-session-id": "project-fix-1"},
)
print(response.choices[0].message.content)
```

This request has no client tools; only delegated work can execute tools. For lead
actions, include normal `tools` schemas and implement the tool-result loop. Preserve
the returned assistant message and `tool_call_id` values; do not replace IDs.

## Primary client: prime-agent

Keep the local `mantis` provider configured as described in the [README](../README.md#prime-agent),
then choose its `fusion` model:

```bash
cd /absolute/path/to/project
prime-agent --mode text --provider mantis --model fusion --no-session \
  -p 'Fix the failing test, run focused checks, and report the diff and output.'
```

Allowlist that directory on the API before starting. A prime-agent-style system
prompt containing `Working directory:` is recognized without a private client
extension. For session reuse, the transport
must send a stable `x-mantis-session-id` or `metadata.session_id`; do not assume a
local harness session automatically supplies that identity.

A compatible transport can send this shape with its real tool schemas:

```json
{
  "model": "mantis/fusion",
  "messages": [
    {"role": "system", "content": "Working directory: /absolute/path/to/project"},
    {"role": "user", "content": "Inspect the failure and fix it."}
  ],
  "tools": [{"type": "function", "function": {
    "name": "bash",
    "parameters": {"type": "object", "properties": {
      "command": {"type": "string"}
    }, "required": ["command"]}
  }}],
  "stream": true,
  "parallel_tool_calls": true
}
```

## OpenCode-compatible request shape

Use an OpenAI-compatible provider pointed at `http://127.0.0.1:8088/v1` with
`MANTIS_API_KEY` and public model `mantis/fusion`. Keep its existing client tools.
Mantis accepts system/developer messages, text content parts, replayed assistant
reasoning fields, and ordinary tool-result messages. Directory context can use:

```json
{
  "model": "mantis/fusion",
  "messages": [
    {"role": "system", "content": "<env>\nPWD=/absolute/path/to/project\n</env>"},
    {"role": "developer", "content": "Keep changes narrow and run focused tests."},
    {"role": "user", "content": [{"type": "text", "text": "Fix the failing test."}]}
  ],
  "metadata": {"session_id": "project-fix-1"},
  "stream": true
}
```

If the client cannot send extra body fields, use prompt discovery or have its
transport add `X-Mantis-Workspace`. Use `x-mantis-session-id` explicitly
for Fusion; Base's `x-opencode-session` alias is not a Fusion session contract.
These shapes describe API compatibility, not an end-to-end test of a client version.

## Sessions and continuation

`X-Mantis-Run-Id` resumes the same run for a later user turn. Tool-result continuation
uses the Fusion-encoded IDs returned by the API. For live SSE, a new run ID may not
be available in the initial HTTP headers; preserve the final `mantis_event` and
returned tool-call IDs as well as any run header.

`x-mantis-session-id` serves a different purpose: it can restore completed sidekick
history when a new run is created in the same API process. `metadata.session_id` or
`metadata.sessionId` also works; an explicit session header takes precedence. Without
an identity, the request is stateless for this cache. Use a distinct identity for
each conversation, not an account-wide ID.

The cache holds up to 128 identities and 256 messages per identity; context fitting
may trim further. It is process-local, so a restart or a different API worker loses
access. File/Redis run storage does not make it or background workers durable.
Retain client transcripts if restart recovery matters.

## Streaming, timeouts, and cancellation

With `stream=true`, lead content and client calls stream as OpenAI-style SSE deltas.
Private delegation calls and sidekick output are not public tool-call deltas. Streams
send comment keepalives and end with `[DONE]`. Assemble fragmented tool names and
arguments before executing a call.

Set `x-mantis-return-reasoning: true` and `x-mantis-trace-scope: main` to request live
lead reasoning. Text is available as both `delta.reasoning` and
`delta.reasoning_content`; they are aliases, so display one rather than both.
`false` disables raw reasoning. `orchestration` and `sidekick` trace scopes select
diagnostic completed-event traces; live reasoning is limited to main/orchestration
scope. Keep reasoning and diagnostic traces private because they can contain task
or tool data.

Run budgets support `max_turns`, `max_tokens`, and `timeout_ms`. Catalog run and
sidekick guards also apply. `read_subagent.timeout` bounds a wait, not the lifetime
of the background task; a timed-out wait does not cancel it. Its catalog default is
`handoff_wait_timeout_s` (2700 seconds when absent). Built-in shell/Python calls
currently have a 30-second command timeout.

A disconnected SSE consumer signals cancellation to provider-stream checks.
Background stop requests are cooperative at sidekick turn boundaries. Completion
asks background work to stop. Budget exhaustion closes unanswered calls with errors.
Neither is an immediate kill of every local command or child process. Do not rely on disconnect or timeout to roll back edits; inspect the
workspace and any long-running processes before retrying.

## Multi-seed A/B runs

`scripts/run_routed_ab.py` runs live cases and writes observations. It makes provider
calls and executes local commands. Use trusted fixtures in a dedicated workspace,
review the configured checks first, and approve the provider spend separately.
The current Fusion arm is `fusion-tool`. Historical Fusion arm labels remain in
saved records for review, but the live runner no longer selects them.

Export the workspace allowlist **in the API process's environment**, then start or
restart the API. Run the CLI from the repository with the project environment:

```bash
export MANTIS_FUSION_WORKSPACE_ROOTS="$PWD/.scratch/routed-ab"
uv run python scripts/run_routed_ab.py \
  --base-url http://127.0.0.1:8088 \
  --variants mantis-base,fusion-tool \
  --seeds 3 \
  --checks eval/routed-harness-checks.json \
  --workspace-root "$PWD/.scratch/routed-ab" \
  --out eval/results-routed-ab.jsonl \
  --transcripts-dir eval/runs/routed-ab
```

Other current arms are `efficient-only`, `capable-only`, and `mantis-base`.
Direct arms use provider credentials independently of `MANTIS_API_KEY`. Seeds label
repetitions, not deterministic provider seeds. Inspect actual selected models.

Case checks record `check_command`, `check_output`, and `check_passed` separately from
prose verdicts. Fixture checks require the configured exit status and output marker;
other cases use exact/contains rules. Review evidence, not just a model's report.

Workspaces remain at `<workspace-root>/<variant>/<case>/seed<N>`. Per-seed observation
files remain at `<transcripts-dir>/<variant>--<case>--s<N>.json`, alongside JSONL run
records. Keep them for review; they may contain source, prompts, and tool output.
Do not assume an observation file contains every provider request or hidden turn.

Rerunning the same command skips keys already recorded as `(variant, case_id, seed)`;
old seedless records count as seed 1. This resumes unrecorded cases, not an interrupted
model conversation. `--force` reruns recorded keys, appends duplicate records, and
reuses workspaces; fixture copying does not delete old files. For a clean new
comparison, choose new output, transcript, and workspace paths.

`scripts/ab_report.py` summarizes cost, tokens, rounds, check rates, and optional
assessments with mean/stdev and same-seed paired deltas. Observations alone do not
establish a quality/cost claim or end-to-end parity with another harness. Tool-only
delegation is the current runtime, not a benchmark conclusion. No live test is
claimed here.
