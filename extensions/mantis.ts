/**
 * mantis — per-turn orchestration mode switching for pi.
 * /mantis off|trinity|conductor|auto
 * Exposes native provider models (mantis/trinity, mantis/conductor, mantis/auto)
 * and native tool-call steps for background worker/verifier turns.
 */

import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { createHash, randomUUID } from "node:crypto";
import { Text } from "@mariozechner/pi-tui";
import { Static, Type } from "@sinclair/typebox";
import {
  createAssistantMessageEventStream,
  type Api,
  type AssistantMessage,
  type AssistantMessageEventStream,
  type Context,
  type Message,
  type Model,
  type SimpleStreamOptions,
  type TextContent,
  type ToolCall,
  type ToolResultMessage,
} from "@mariozechner/pi-ai";
import type {
  AgentToolResult,
  ExtensionAPI,
  ExtensionCommandContext,
  ExtensionContext,
} from "@mariozechner/pi-coding-agent";

type Mode = "off" | "trinity" | "conductor" | "auto";

interface ChatMessage {
  role: "user" | "assistant" | "system" | "tool";
  content: string;
  tool_call_id?: string;
  tool_calls?: Array<{
    id: string;
    type: "function";
    function: { name: string; arguments: string };
  }>;
}

interface MantisStep {
  turn: number;
  agent_id: number;
  role: string;
  reply: string;
  prompt?: string;
  model_name?: string;
  output?: string;
}

type StreamEvent =
  | { type: "step-start"; turn: number; role: string; agent_id: number; model_name: string; prompt: string }
  | { type: "step-end"; turn: number; role: string; agent_id: number; model_name: string; prompt: string; reply: string }
  | { type: "result"; text: string; trace: string; coordinator: string; mantis_steps: MantisStep[] }
  | { type: "error"; error: string };

const WORKER_NAMES: Record<number, string> = {
  0: "gemini-3.6-flash-high",
  1: "gpt-5.6-luna-max",
  2: "gpt-5.6-sol-medium",
  3: "deepseek-v4-flash-0731-xhigh",
  4: "claude-opus-5-medium",
  5: "claude-sonnet-5-medium",
  6: "gemini-3.1-pro-preview-high",
};

function getApiKey(): string {
  if (process.env.MANTIS_API_KEY) return process.env.MANTIS_API_KEY;
  if (process.env.LITELLM_KEY) return process.env.LITELLM_KEY;

  const envPaths = [
    path.join(process.cwd(), ".env"),
    path.join(os.homedir(), ".config", "mantis", ".env"),
    path.join(os.homedir(), ".env"),
  ];

  for (const envPath of envPaths) {
    try {
      if (fs.existsSync(envPath)) {
        const content = fs.readFileSync(envPath, "utf-8");
        for (const line of content.split("\n")) {
          const trimmed = line.trim();
          if (trimmed.startsWith("#") || !trimmed.includes("=")) continue;
          const [key, ...valParts] = trimmed.split("=");
          const k = key.trim();
          const v = valParts.join("=").trim().replace(/^["']|["']$/g, "");
          if (k === "MANTIS_API_KEY" || k === "LITELLM_KEY") {
            if (v) return v;
          }
        }
      }
    } catch {
      // ignore
    }
  }

  return "";
}

export function getMantisContextWindow(): number {
  const envVal = process.env.MANTIS_CONTEXT_WINDOW;
  if (envVal) {
    const trimmed = envVal.trim();
    if (/^\d+$/.test(trimmed)) {
      const parsed = parseInt(trimmed, 10);
      if (parsed > 0) {
        return parsed;
      }
    }
  }
  return 256000;
}

function getMantisUrl(): string {
  return process.env.MANTIS_URL ?? "http://127.0.0.1:8088/v1";
}

function getRouterUrl(): string {
  return process.env.MANTIS_ROUTER_URL ?? "http://127.0.0.1:5500/v1";
}

const AUTO_THRESHOLD = parseInt(process.env.MANTIS_AUTO_THRESHOLD ?? "6", 10);

const ROUTING_LOG_DIR = path.join(os.homedir(), ".config", "mantis");
const ROUTING_LOG_PATH = path.join(ROUTING_LOG_DIR, "routing-log.jsonl");

let activeMode: Mode = "off";
export const warmed = new Set<string>();

function logRouting(score: number, coordinator: string, text: string) {
  try {
    fs.mkdirSync(ROUTING_LOG_DIR, { recursive: true });
    const entry = {
      ts: Math.floor(Date.now() / 1000),
      score,
      coordinator,
      task_prefix: text.slice(0, 60),
    };
    fs.appendFileSync(ROUTING_LOG_PATH, JSON.stringify(entry) + "\n");
  } catch {
    // logging must never break turn
  }
}

function getTextFromContent(content: string | unknown[]): string {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  return content
    .filter((c: any) => c.type === "text" || c.type === "thinking")
    .map((c: any) => (c.type === "thinking" ? c.thinking ?? "" : c.text ?? ""))
    .filter((t) => t.trim())
    .join("\n");
}

function toRunMessages(context: Context): { messages: ChatMessage[]; lastUserContent: string } {
  // Native-tool runs do not dump a 96 KB repository snapshot into the prompt;
  // the selected model inspects the live repository through Pi tools instead.
  const messages: ChatMessage[] = [];
  if (context.systemPrompt) messages.push({ role: "system", content: context.systemPrompt });
  let lastUserContent = "";
  for (const m of context.messages) {
    if (m.role === "user") {
      const text = getTextFromContent(m.content);
      if (text.trim()) {
        messages.push({ role: "user", content: text });
        lastUserContent = text;
      }
    } else if (m.role === "assistant") {
      const text = getTextFromContent(m.content);
      const toolCalls = m.content
        .filter((item): item is ToolCall => item.type === "toolCall" && item.name !== "mantis_step")
        .map((item) => ({
          id: item.id,
          type: "function" as const,
          function: { name: item.name, arguments: JSON.stringify(item.arguments) },
        }));
      if (text.trim() || toolCalls.length > 0) {
        messages.push({ role: "assistant", content: text, ...(toolCalls.length ? { tool_calls: toolCalls } : {}) });
      }
    } else if (m.role === "toolResult" && m.toolName !== "mantis_step") {
      messages.push({
        role: "tool",
        tool_call_id: m.toolCallId,
        content: getTextFromContent(m.content),
      });
    }
  }
  return { messages, lastUserContent };
}

type TrailingKind = "none" | "real" | "step";
type RunToolResult = { tool_call_id: string; content: string; is_error: boolean };
type TrailingResults = { kind: TrailingKind; ids: string[]; results: RunToolResult[] };
type PendingRun = { runId: string; kind: Exclude<TrailingKind, "none">; expectedIds: string[] };

function detectTrailing(messages: Message[]): TrailingResults {
  const trailing: Message[] = [];
  let i = messages.length - 1;
  while (i >= 0 && messages[i].role === "toolResult") {
    trailing.push(messages[i]);
    i--;
  }
  trailing.reverse();
  const isToolResult = (m: Message): m is ToolResultMessage => m.role === "toolResult";
  const real: ToolResultMessage[] = trailing.filter(
    (m): m is ToolResultMessage => isToolResult(m) && m.toolName !== "mantis_step",
  );
  const steps: ToolResultMessage[] = trailing.filter(
    (m): m is ToolResultMessage => isToolResult(m) && m.toolName === "mantis_step",
  );
  if (real.length > 0 && steps.length > 0) {
    throw new Error("mixed mantis tool results are not supported");
  }
  if (real.length > 0) {
    return {
      kind: "real",
      ids: real.map((m) => m.toolCallId),
      results: real.map((m) => ({
        tool_call_id: m.toolCallId,
        content: getTextFromContent(m.content),
        is_error: m.isError,
      })),
    };
  }
  if (steps.length > 0) {
    return { kind: "step", ids: steps.map((m) => m.toolCallId), results: [] };
  }
  return { kind: "none", ids: [], results: [] };
}

function authHeaders(): Record<string, string> {
  const key = getApiKey();
  return {
    "Content-Type": "application/json",
    ...(key ? { Authorization: `Bearer ${key}` } : {}),
  };
}

const MAX_RUN_RESPONSE_BYTES = 1_000_000;

async function readBoundedJson(res: Response): Promise<Record<string, any>> {
  const declared = Number(res.headers.get("content-length") ?? 0);
  if (declared > MAX_RUN_RESPONSE_BYTES) throw new Error("mantis response exceeds size limit");
  if (!res.body) return {};
  const reader = res.body.getReader();
  const chunks: Uint8Array[] = [];
  let total = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    total += value.byteLength;
    if (total > MAX_RUN_RESPONSE_BYTES) {
      await reader.cancel();
      throw new Error("mantis response exceeds size limit");
    }
    chunks.push(value);
  }
  const bytes = new Uint8Array(total);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.byteLength;
  }
  if (bytes.length === 0) return {};
  const parsed: unknown = JSON.parse(new TextDecoder().decode(bytes));
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
    throw new Error("mantis response must be a JSON object");
  }
  return parsed as Record<string, any>;
}

async function createRun(
  coordinator: string,
  messages: ChatMessage[],
  tools: Context["tools"],
  runId: string,
  signal?: AbortSignal,
): Promise<{ run_id: string }> {
  const res = await fetch(`${getMantisUrl()}/runs`, {
    method: "POST",
    headers: authHeaders(),
    signal,
    body: JSON.stringify({ model: coordinator, messages, tools, run_id: runId }),
  });
  const data = await readBoundedJson(res);
  if (!res.ok || typeof data.run_id !== "string" || !data.run_id) {
    throw new Error(`mantis run create failed (HTTP ${res.status}): ${data?.error ?? JSON.stringify(data)}`);
  }
  return { run_id: data.run_id };
}

type RunEvent =
  | { type: "tool_calls"; tool_calls: Array<{ id: string; name: string; arguments: Record<string, unknown> }> }
  | { type: "step_complete"; turn: number; agent_id: number; role: string; reply: string; prompt: string; model_name?: string }
  | { type: "final"; text: string }
  | { type: "error"; error: string };

class MantisTransportError extends Error {}

async function continueRun(
  runId: string,
  toolResults: RunToolResult[] | undefined,
  requestId: string | undefined,
  signal?: AbortSignal,
): Promise<RunEvent> {
  const body = requestId === undefined && toolResults === undefined
    ? ""
    : JSON.stringify({ tool_results: toolResults, request_id: requestId });
  let res: Response;
  try {
    res = await fetch(`${getMantisUrl()}/runs/${encodeURIComponent(runId)}/continue`, {
      method: "POST",
      headers: authHeaders(),
      signal,
      body,
    });
  } catch (error: any) {
    throw new MantisTransportError(error?.message ?? String(error));
  }
  const data = await readBoundedJson(res);
  if (!res.ok) throw new Error(`mantis run continue failed (HTTP ${res.status}): ${data?.error ?? JSON.stringify(data)}`);
  if (!data || typeof data.type !== "string") throw new Error("invalid mantis run event");
  return data as RunEvent;
}

async function deleteRun(runId: string): Promise<void> {
  const res = await fetch(`${getMantisUrl()}/runs/${encodeURIComponent(runId)}`, {
    method: "DELETE",
    headers: authHeaders(),
    signal: AbortSignal.timeout(5_000),
  });
  if (!res.ok) {
    // best effort cleanup; ignore failures
  }
}

async function supraScore(text: string, signal?: AbortSignal): Promise<number> {
  const apiKey = getApiKey();
  const routerUrl = getRouterUrl();
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 30_000);
  const onAbort = () => controller.abort();
  signal?.addEventListener("abort", onAbort, { once: true });
  if (signal?.aborted) controller.abort();
  try {
    const probeText = text.slice(0, 500);
    const res = await fetch(`${routerUrl}/chat/completions`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${apiKey}` },
      body: JSON.stringify({
        model: "auto",
        messages: [{ role: "user", content: probeText }],
        max_tokens: 1,
      }),
      signal: controller.signal,
    });
    return parseInt(res.headers.get("x-route-supra-complexity") ?? "1", 10);
  } catch {
    return 1;
  } finally {
    clearTimeout(timeout);
    signal?.removeEventListener("abort", onAbort);
  }
}

async function chooseCoordinator(lastUserContent: string, signal?: AbortSignal): Promise<"trinity" | "conductor"> {
  const score = await supraScore(lastUserContent, signal);
  const coordinator = score >= AUTO_THRESHOLD ? "conductor" : "trinity";
  logRouting(score, coordinator, lastUserContent);
  return coordinator;
}

export async function warm(coordinator: string, ctx: ExtensionContext, timeoutMs: number = 30000) {
  if (warmed.has(coordinator)) return;
  const apiKey = getApiKey();
  const mantisUrl = getMantisUrl();

  if (!apiKey) {
    ctx.ui.notify?.("mantis: API key not set. Set MANTIS_API_KEY or LITELLM_KEY in environment or .env", "warning");
  }

  ctx.ui.notify?.(`mantis: warming ${coordinator} (first call may download weights)…`, "info");

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const res = await fetch(`${mantisUrl}/warm?mode=${encodeURIComponent(coordinator)}`, {
      headers: apiKey ? { Authorization: `Bearer ${apiKey}` } : {},
      signal: controller.signal,
    });
    if (res.ok) {
      warmed.add(coordinator);
      ctx.ui.notify?.(`mantis: ${coordinator} ready`, "info");
    } else {
      let errDetail = `status ${res.status}`;
      try {
        const data = await res.json();
        if (data.error) errDetail = data.error;
      } catch {
        // use status
      }
      ctx.ui.notify?.(`mantis: warm failed (${coordinator}): ${errDetail}`, "warning");
    }
  } catch (e: any) {
    const msg = e.name === "AbortError" ? "request timed out" : (e.message || String(e));
    ctx.ui.notify?.(`mantis: warm failed (${coordinator}): ${msg}`, "warning");
  } finally {
    clearTimeout(timer);
  }
}

export async function* streamOrchestrate(
  coordinator: string,
  messages: ChatMessage[],
  signal?: AbortSignal,
  timeoutSeconds: number = 300,
): AsyncGenerator<StreamEvent> {
  const apiKey = getApiKey();
  const mantisUrl = getMantisUrl();

  if (!apiKey) {
    throw new Error("MANTIS_API_KEY / LITELLM_KEY is not set in environment or .env file");
  }

  const controller = new AbortController();
  let isTimeout = false;
  let isUserAbort = false;

  const timeoutId = setTimeout(() => {
    isTimeout = true;
    controller.abort();
  }, timeoutSeconds * 1000);

  const onAbort = () => {
    isUserAbort = true;
    controller.abort();
  };

  if (signal) {
    if (signal.aborted) {
      isUserAbort = true;
      controller.abort();
    } else {
      signal.addEventListener("abort", onAbort, { once: true });
    }
  }

  const formatError = (err: any): Error => {
    if (isTimeout) {
      return new Error(`mantis request timed out after ${timeoutSeconds} seconds`);
    }
    if (isUserAbort || signal?.aborted || err?.name === "AbortError" || err?.message === "aborted") {
      return new Error("mantis request aborted by user");
    }
    if (err instanceof Error) {
      return err;
    }
    return new Error(String(err));
  };

  try {
    let res: Response;
    try {
      res = await fetch(`${mantisUrl}/chat/completions`, {
        method: "POST",
        headers: { "Content-Type": "application/json", Authorization: `Bearer ${apiKey}` },
        body: JSON.stringify({
          model: coordinator,
          messages,
          stream: true,
        }),
        signal: controller.signal,
      });
    } catch (e: any) {
      if (isTimeout) {
        throw new Error(`mantis request timed out after ${timeoutSeconds} seconds`);
      }
      if (isUserAbort || signal?.aborted || e?.name === "AbortError" || e?.message === "aborted") {
        throw new Error("mantis request aborted by user");
      }
      throw new Error(`mantis backend disconnected: ${e?.message || String(e)}`);
    }

    if (!res.ok) {
      const body = await res.text();
      throw new Error(`mantis provider failure (HTTP ${res.status}): ${body}`);
    }

    // Older orchestrator images ignore `stream: true` and return one regular
    // OpenAI completion. Accept that response without requiring a backend rebuild.
    if (res.headers.get("content-type")?.includes("application/json")) {
      const body = (await res.json()) as any;
      const text = body.choices?.[0]?.message?.content;
      if (typeof text !== "string") {
        throw new Error("mantis provider failure: backend returned an invalid completion");
      }
      yield {
        type: "result",
        text,
        trace: body.usage?.mantis_trace ?? body.usage?.fugu_trace ?? "",
        coordinator,
        mantis_steps: body.mantis_steps ?? body.choices?.[0]?.message?.mantis_steps ?? [],
      };
      return;
    }

    const reader = res.body?.getReader();
    if (!reader) {
      throw new Error("mantis backend disconnected: empty response body");
    }

    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      let readResult: ReadableStreamReadResult<Uint8Array>;
      try {
        readResult = await reader.read();
      } catch (e: any) {
        throw formatError(
          e?.name === "TypeError" || e?.name === "FetchError"
            ? new Error(`mantis backend disconnected: ${e?.message || String(e)}`)
            : e,
        );
      }
      const { done, value } = readResult;
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() ?? "";
      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed) continue;
        try {
          const data = JSON.parse(trimmed);
          if (data.type === "error") {
            throw new Error(`mantis provider failure: ${data.error}`);
          }
          yield data as StreamEvent;
        } catch (e: any) {
          if (e.message?.startsWith("mantis provider failure:")) {
            throw e;
          }
          throw new Error("Malformed NDJSON event from mantis backend: " + trimmed);
        }
      }
    }
    if (buffer.trim()) {
      const trimmed = buffer.trim();
      try {
        const data = JSON.parse(trimmed);
        if (data.type === "error") {
          throw new Error(`mantis provider failure: ${data.error}`);
        }
        yield data as StreamEvent;
      } catch (e: any) {
        if (e.message?.startsWith("mantis provider failure:")) {
          throw e;
        }
        throw new Error("Malformed NDJSON event from mantis backend: " + trimmed);
      }
    }
  } catch (err: any) {
    throw formatError(err);
  } finally {
    clearTimeout(timeoutId);
    if (signal) {
      signal.removeEventListener("abort", onAbort);
    }
  }
}

function emitText(stream: AssistantMessageEventStream, text: string, output: AssistantMessage) {
  const contentIndex = output.content.length;
  output.usage.output += Math.ceil(text.length / 4);
  output.usage.totalTokens = output.usage.input + output.usage.output;
  output.content.push({ type: "text", text: "" });
  stream.push({ type: "text_start", contentIndex, partial: output });
  const block = output.content[contentIndex] as TextContent;
  block.text = text;
  stream.push({ type: "text_delta", contentIndex, delta: text, partial: output });
  stream.push({ type: "text_end", contentIndex, content: text, partial: output });
}

const MAX_TOOL_CACHE = 512;
const MAX_PENDING_TOOL_CALLS = 512;
const toolReplyCache = new Map<string, MantisStep>();
const pendingByToolCallId = new Map<string, PendingRun>();
const consumedToolCallIds = new Map<string, true>();
const MAX_RUN_EVENTS = 200;

function pruneMap<K, V>(map: Map<K, V>, maxSize: number) {
  while (map.size > maxSize) {
    const first = map.keys().next().value;
    if (first !== undefined) map.delete(first);
  }
}

function registerPending(runId: string, ids: string[], kind: PendingRun["kind"]): void {
  if (ids.length === 0 || ids.length !== new Set(ids).size) {
    throw new Error("mantis emitted invalid or duplicate tool call ids");
  }
  if (pendingByToolCallId.size + ids.length > MAX_PENDING_TOOL_CALLS) {
    throw new Error("too many pending mantis tool calls");
  }
  const pending: PendingRun = { runId, kind, expectedIds: ids };
  for (const id of ids) {
    if (pendingByToolCallId.has(id) || consumedToolCallIds.has(id)) {
      throw new Error(`duplicate mantis tool call id: ${id}`);
    }
    pendingByToolCallId.set(id, pending);
  }
}

function consumePending(trailing: TrailingResults): PendingRun | undefined {
  if (trailing.kind === "none") return undefined;
  if (trailing.ids.length === 0 || trailing.ids.length !== new Set(trailing.ids).size) {
    throw new Error("duplicate or empty mantis tool results");
  }
  if (trailing.ids.some((id) => consumedToolCallIds.has(id))) {
    throw new Error("mantis tool results were already consumed");
  }
  const matches = new Set(trailing.ids.map((id) => pendingByToolCallId.get(id)));
  if (matches.size !== 1 || matches.has(undefined)) {
    throw new Error("mantis tool results refer to no active run");
  }
  const pending = matches.values().next().value as PendingRun;
  if (
    pending.kind !== trailing.kind ||
    pending.expectedIds.length !== trailing.ids.length ||
    pending.expectedIds.some((id) => !trailing.ids.includes(id))
  ) {
    throw new Error("mantis tool results do not match the pending run");
  }
  for (const id of pending.expectedIds) {
    pendingByToolCallId.delete(id);
    if (pending.kind === "step") toolReplyCache.delete(id);
    consumedToolCallIds.set(id, true);
  }
  pruneMap(consumedToolCallIds, MAX_PENDING_TOOL_CALLS);
  return pending;
}

function restorePending(pending: PendingRun): void {
  for (const id of pending.expectedIds) {
    consumedToolCallIds.delete(id);
    pendingByToolCallId.set(id, pending);
  }
}

function clearPendingRun(runId: string): void {
  for (const [id, pending] of pendingByToolCallId) {
    if (pending.runId === runId) pendingByToolCallId.delete(id);
  }
}

function mantisStreamSimple(
  model: Model<Api>,
  context: Context,
  options?: SimpleStreamOptions,
): AssistantMessageEventStream {
  const stream = createAssistantMessageEventStream();
  let runId: string | null = null;
  let resumedPending: PendingRun | undefined;

  (async () => {
    const output: AssistantMessage = {
      role: "assistant",
      content: [],
      api: model.api,
      provider: model.provider,
      model: model.id,
      usage: {
        input: 0,
        output: 0,
        cacheRead: 0,
        cacheWrite: 0,
        totalTokens: 0,
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
      },
      stopReason: "stop",
      timestamp: Date.now(),
    };

    try {
      if (options?.signal?.aborted) throw new Error("aborted");

      const mode = model.id as Mode;
      const trailing = detectTrailing(context.messages);
      const pending = consumePending(trailing);
      resumedPending = pending;
      let toolResults: RunToolResult[] | undefined;
      let stepAck = false;

      if (pending) {
        runId = pending.runId;
        if (trailing.kind === "real") toolResults = trailing.results;
        else stepAck = true;
      }

      if (runId === null) {
        const { messages: backendMessages, lastUserContent } = toRunMessages(context);
        output.usage.input = Math.ceil(
          backendMessages.reduce((chars, message) => chars + message.content.length, 0) / 4,
        );
        output.usage.totalTokens = output.usage.input;
        if (output.usage.input > model.contextWindow - model.maxTokens) {
          throw new Error(`input token count ${output.usage.input} exceeds the context window of this model`);
        }
        if (!lastUserContent.trim()) throw new Error("No user message to process");
        const coordinator = mode === "auto" ? await chooseCoordinator(lastUserContent, options?.signal) : mode;
        if (options?.signal?.aborted) throw new Error("aborted");
        runId = randomUUID().replaceAll("-", "");
        const created = await createRun(
          coordinator, backendMessages, context.tools, runId, options?.signal,
        );
        runId = created.run_id;
      }
      if (!runId) throw new Error("failed to start mantis run");

      stream.push({ type: "start", partial: output });

      let stepCount = 0;
      let round = 0;
      while (true) {
        if (options?.signal?.aborted) throw new Error("aborted");
        const requestId = resumedPending
          ? createHash("sha256").update([...resumedPending.expectedIds].sort().join("\0")).digest("hex")
          : undefined;
        const ev = await continueRun(
          runId, stepAck ? undefined : toolResults, requestId, options?.signal,
        );
        resumedPending = undefined;
        stepAck = false;
        toolResults = undefined;
        round += 1;
        if (round > MAX_RUN_EVENTS) throw new Error(`mantis run exceeded ${MAX_RUN_EVENTS} events`);

        if (ev.type === "tool_calls") {
          if (!Array.isArray(ev.tool_calls) || ev.tool_calls.length === 0) {
            throw new Error("invalid mantis tool_calls event");
          }
          const activeTools = new Set((context.tools ?? []).map((tool) => tool.name));
          if (ev.tool_calls.some((c) => (
            !c || typeof c.id !== "string" || !c.id ||
            typeof c.name !== "string" || !activeTools.has(c.name) ||
            typeof c.arguments !== "object" || c.arguments === null || Array.isArray(c.arguments)
          ))) {
            throw new Error("invalid or unavailable mantis tool call");
          }
          registerPending(runId, ev.tool_calls.map((call) => call.id), "real");
          for (const c of ev.tool_calls) {
            const toolCall: ToolCall = {
              type: "toolCall",
              id: c.id,
              name: c.name,
              arguments: c.arguments ?? {},
            };
            const contentIndex = output.content.length;
            output.content.push(toolCall as any);
            stream.push({ type: "toolcall_start", contentIndex, partial: output });
            stream.push({ type: "toolcall_end", contentIndex, toolCall, partial: output });
          }
          output.stopReason = "toolUse";
          stream.push({ type: "done", reason: "toolUse", message: output });
          stream.end?.();
          return;
        } else if (ev.type === "step_complete") {
          if (
            typeof ev.turn !== "number" || typeof ev.agent_id !== "number" ||
            typeof ev.role !== "string" || typeof ev.reply !== "string" ||
            typeof ev.prompt !== "string"
          ) {
            throw new Error("invalid mantis step_complete event");
          }
          const reply = ev.reply.trim() ? ev.reply : "(no response)";
          const step: MantisStep = {
            turn: ev.turn,
            agent_id: ev.agent_id,
            role: ev.role,
            reply,
            prompt: ev.prompt,
            model_name: ev.model_name,
            output: reply,
          };
          const toolCallId = `mantis_${output.timestamp}_${stepCount++}_${Math.random().toString(36).slice(2, 8)}`;
          toolReplyCache.set(toolCallId, step);
          pruneMap(toolReplyCache, MAX_TOOL_CACHE);
          registerPending(runId, [toolCallId], "step");
          const toolCall: ToolCall = {
            type: "toolCall",
            id: toolCallId,
            name: "mantis_step",
            arguments: {
              turn: step.turn,
              role: step.role,
              agent_id: step.agent_id,
              model_name: step.model_name,
              prompt: step.prompt,
            },
          };
          const contentIndex = output.content.length;
          output.content.push(toolCall as any);
          stream.push({ type: "toolcall_start", contentIndex, partial: output });
          stream.push({ type: "toolcall_end", contentIndex, toolCall, partial: output });
          output.stopReason = "toolUse";
          stream.push({ type: "done", reason: "toolUse", message: output });
          stream.end?.();
          return;
        } else if (ev.type === "final") {
          if (typeof ev.text !== "string") throw new Error("invalid mantis final event");
          emitText(stream, ev.text || "(no response)", output);
          output.stopReason = "stop";
          stream.push({ type: "done", reason: "stop", message: output });
          stream.end?.();
          void deleteRun(runId).catch(() => {});
          return;
        } else if (ev.type === "error") {
          if (typeof ev.error !== "string") throw new Error("invalid mantis error event");
          throw new Error(ev.error);
        } else {
          throw new Error("unexpected mantis run event");
        }
      }
    } catch (err: any) {
      if (err instanceof MantisTransportError && resumedPending && !options?.signal?.aborted) {
        restorePending(resumedPending);
      } else if (runId) {
        clearPendingRun(runId);
        void deleteRun(runId).catch(() => {});
      }
      output.stopReason = options?.signal?.aborted ? "aborted" : "error";
      output.errorMessage = err.message;
      emitText(stream, `mantis error: ${err.message}`, output);
      stream.push({
        type: "error",
        reason: output.stopReason as "aborted" | "error",
        error: output,
      });
      stream.end?.();
    }
  })();

  return stream;
}

export default function (pi: ExtensionAPI) {
  // Ensure pi's auth resolver can find a MANTIS_API_KEY even if the user only
  // configured MANTIS_API_KEY / LITELLM_KEY in the environment or .env file.
  const apiKey = getApiKey();
  if (apiKey) {
    process.env.MANTIS_API_KEY = apiKey;
  }

  // 1. Register mantis provider with custom streamSimple implementation.
  const contextWindow = getMantisContextWindow();
  const models = [
    {
      id: "trinity",
      name: "mantis: trinity",
      reasoning: true,
      input: ["text"] as ("text" | "image")[],
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
      contextWindow,
      maxTokens: 16384,
    },
    {
      id: "conductor",
      name: "mantis: conductor",
      reasoning: true,
      input: ["text"] as ("text" | "image")[],
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
      contextWindow,
      maxTokens: 16384,
    },
    {
      id: "auto",
      name: "mantis: auto",
      reasoning: true,
      input: ["text"] as ("text" | "image")[],
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
      contextWindow,
      maxTokens: 16384,
    },
  ];

  pi.registerProvider("mantis", {
    name: "mantis",
    baseUrl: getMantisUrl(),
    apiKey: "MANTIS_API_KEY",
    api: "mantis" as Api,
    models,
    streamSimple: mantisStreamSimple,
  });

  // 2. Register native mantis_step tool used by the provider to expose worker turns.
  const mantisStepSchema = Type.Object({
    turn: Type.Number({ description: "Step turn index" }),
    role: Type.String({ description: "Role: Worker, Thinker, Verifier, or Planner" }),
    agent_id: Type.Number({ description: "Worker slot id (0..6)" }),
    model_name: Type.Optional(Type.String({ description: "Model name" })),
    prompt: Type.String({ description: "Prompt sent to the worker" }),
  });
  type MantisStepToolParams = Static<typeof mantisStepSchema>;

  pi.registerTool({
    name: "mantis_step",
    label: "Mantis Worker Step",
    description: "Internal TRINITY/Conductor worker turn exposed as a native tool call",
    parameters: mantisStepSchema,
    executionMode: "sequential",

    renderCall(args: MantisStepToolParams, theme: any) {
      const role = args.role ?? "Worker";
      const agentId = args.agent_id ?? 0;
      const modelName = args.model_name ?? WORKER_NAMES[agentId] ?? `slot-${agentId}`;

      let roleColor: "accent" | "success" | "warning" = "accent";
      if (role === "Verifier") roleColor = "success";
      if (role === "Thinker") roleColor = "warning";

      const title = theme.fg("toolTitle", theme.bold(`[mantis ${role}]`));
      const details = theme.fg("muted", ` slot #${agentId} (${modelName}) — step ${args.turn ?? 0}`);
      return new Text(`${title}${details}`, 0, 0);
    },

    renderResult(result: AgentToolResult<MantisStep>, { expanded }: { expanded?: boolean }, theme: any) {
      if (!expanded) return new Text("", 0, 0);
      const outputStr = result.content?.[0]?.type === "text" ? result.content[0].text : JSON.stringify(result.content);
      return new Text(`\n${theme.fg("toolOutput", outputStr)}`, 0, 0);
    },

    async execute(toolCallId: string, params: MantisStepToolParams): Promise<AgentToolResult<MantisStep>> {
      const step = toolReplyCache.get(toolCallId);
      if (step) {
        return {
          content: [{ type: "text", text: step.reply }],
          details: step,
        };
      }
      const fallback: MantisStep = {
        turn: params.turn,
        agent_id: params.agent_id,
        role: params.role,
        reply: "",
        prompt: params.prompt,
        model_name: params.model_name,
        output: "",
      };
      return {
        content: [{ type: "text", text: "" }],
        details: fallback,
      };
    },
  });

  // 3. Register slash command for mode switching.
  const switchMode = async (m: Mode, ctx: ExtensionCommandContext) => {
    if (!["off", "trinity", "conductor", "auto"].includes(m)) {
      ctx.ui.notify?.("Usage: /mantis off|trinity|conductor|auto", "error");
      return;
    }
    activeMode = m;
    ctx.ui.setStatus?.("mantis", m === "off" ? undefined : `mantis:${m}`);
    ctx.ui.notify?.(`mantis mode: ${m}`, "info");

    if (m !== "off") {
      const foundModel = ctx.modelRegistry?.find?.("mantis", m === "auto" ? "auto" : m);
      if (foundModel) {
        try {
          await pi.setModel(foundModel);
        } catch {
          // ignore
        }
      }
      await warm(m === "auto" ? "trinity" : m, ctx);
    }
  };

  pi.registerCommand("mantis", {
    description: "Set orchestration mode: off | trinity | conductor | auto",
    handler: async (args: string, ctx: ExtensionCommandContext) => {
      await switchMode(args.trim().toLowerCase() as Mode, ctx);
    },
  });

  // 4. Track model selections natively.
  pi.on("model_select", async (event: any, ctx: ExtensionContext) => {
    if (event.model?.provider === "mantis") {
      const m = event.model.id as Mode;
      if (["trinity", "conductor", "auto"].includes(m)) {
        activeMode = m;
        ctx.ui.setStatus?.("mantis", `mantis:${m}`);
      }
    }
  });
}
