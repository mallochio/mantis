/**
 * mantis — per-turn orchestration mode switching for pi.
 * /mantis off|trinity|conductor|auto
 * Exposes native provider models (mantis/trinity, mantis/conductor, mantis/auto)
 * and native tool-call steps for background worker/verifier turns.
 */

import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
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
} from "@mariozechner/pi-ai";
import type {
  AgentToolResult,
  ExtensionAPI,
  ExtensionCommandContext,
  ExtensionContext,
} from "@mariozechner/pi-coding-agent";

type Mode = "off" | "trinity" | "conductor" | "auto";

interface ChatMessage {
  role: "user" | "assistant" | "system";
  content: string;
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
  if (process.env.FUGU_API_KEY) return process.env.FUGU_API_KEY;
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
          if (k === "MANTIS_API_KEY" || k === "FUGU_API_KEY" || k === "LITELLM_KEY") {
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

function getMantisUrl(): string {
  return process.env.MANTIS_URL ?? process.env.FUGU_URL ?? "http://127.0.0.1:8088/v1";
}

function getRouterUrl(): string {
  return process.env.MANTIS_ROUTER_URL ?? process.env.FUGU_ROUTER_URL ?? "http://127.0.0.1:5500/v1";
}

const AUTO_THRESHOLD = parseInt(process.env.MANTIS_AUTO_THRESHOLD ?? process.env.FUGU_AUTO_THRESHOLD ?? "6", 10);

const ROUTING_LOG_DIR = path.join(os.homedir(), ".config", "mantis");
const ROUTING_LOG_PATH = path.join(ROUTING_LOG_DIR, "routing-log.jsonl");

let activeMode: Mode = "off";
const warmed = new Set<string>();

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

const REPO_CONTEXT_FILES = ["README.md", "package.json", "pyproject.toml", "Cargo.toml", "go.mod"];
const REPO_CONTEXT_IGNORED = new Set([".git", ".scratch", "node_modules", "dist", "build", "__pycache__", ".venv"]);

function getRepositoryContext(cwd = process.cwd()): string {
  try {
    const entries = fs.readdirSync(cwd, { withFileTypes: true })
      .filter((entry) => !REPO_CONTEXT_IGNORED.has(entry.name))
      .sort((a, b) => a.name.localeCompare(b.name));
    const tree: string[] = [];
    for (const entry of entries) {
      tree.push(entry.name + (entry.isDirectory() ? "/" : ""));
      if (!entry.isDirectory()) continue;
      try {
        for (const child of fs.readdirSync(path.join(cwd, entry.name), { withFileTypes: true }).slice(0, 40)) {
          if (!REPO_CONTEXT_IGNORED.has(child.name)) {
            tree.push(`  ${child.name}${child.isDirectory() ? "/" : ""}`);
          }
        }
      } catch {
        // A partial tree is still useful if one directory is unreadable.
      }
    }

    const contextFiles = [...REPO_CONTEXT_FILES];
    try {
      const packageJson = JSON.parse(fs.readFileSync(path.join(cwd, "package.json"), "utf8"));
      for (const extension of [packageJson.main, ...(packageJson.pi?.extensions ?? [])]) {
        if (typeof extension === "string" && !contextFiles.includes(extension)) contextFiles.push(extension);
      }
    } catch {
      // Not every repository is a Pi package.
    }

    let remaining = 96_000;
    const files: string[] = [];
    for (const name of contextFiles) {
      const filePath = path.resolve(cwd, name);
      if (!filePath.startsWith(cwd + path.sep) || !fs.existsSync(filePath) || remaining <= 0) continue;
      const content = fs.readFileSync(filePath, "utf8").slice(0, remaining);
      files.push(`--- ${name} ---\n${content}`);
      remaining -= content.length;
    }
    return `Working directory: ${cwd}\n\nRepository tree (two levels):\n${tree.join("\n")}\n\n${files.join("\n\n")}`;
  } catch {
    return `Working directory: ${cwd}`;
  }
}

interface BackendMessagesResult {
  messages: ChatMessage[];
  key: string;
  lastUserContent: string;
}

function toBackendMessages(context: Context): BackendMessagesResult {
  const messages: ChatMessage[] = [{
    role: "system",
    content: [context.systemPrompt, getRepositoryContext()].filter(Boolean).join("\n\n"),
  }];
  let lastUserContent = "";

  const contextMessages = context.messages;

  for (const m of contextMessages) {
    if (m.role === "user") {
      const text = getTextFromContent(m.content);
      if (text.trim()) {
        messages.push({ role: "user", content: text });
        lastUserContent = text;
      }
    } else if (m.role === "assistant") {
      const text = getTextFromContent(m.content as any);
      if (text.trim()) {
        messages.push({ role: "assistant", content: text });
      }
    } else if (m.role === "toolResult") {
      // Skip internal mantis_step results; other tool results become synthetic user messages.
      if (m.toolName === "mantis_step") continue;
      const text = getTextFromContent(m.content);
      if (text.trim()) {
        messages.push({ role: "user", content: `[Tool Result ${m.toolName}]: ${text}` });
      }
    }
  }

  const key = JSON.stringify(messages);
  return { messages, key, lastUserContent };
}

async function supraScore(text: string): Promise<number> {
  const apiKey = getApiKey();
  const routerUrl = getRouterUrl();
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
    });
    return parseInt(res.headers.get("x-route-supra-complexity") ?? "1", 10);
  } catch {
    return 1;
  }
}

async function chooseCoordinator(lastUserContent: string): Promise<"trinity" | "conductor"> {
  const score = await supraScore(lastUserContent);
  const coordinator = score >= AUTO_THRESHOLD ? "conductor" : "trinity";
  logRouting(score, coordinator, lastUserContent);
  return coordinator;
}

async function warm(coordinator: string, ctx: ExtensionContext) {
  if (warmed.has(coordinator)) return;
  const apiKey = getApiKey();
  const mantisUrl = getMantisUrl();

  if (!apiKey) {
    ctx.ui.notify?.("mantis: API key not set. Set MANTIS_API_KEY or LITELLM_KEY in environment or .env", "warning");
  }

  ctx.ui.notify?.(`mantis: warming ${coordinator} (first call may download weights)…`, "info");

  try {
    const health = await fetch(`${mantisUrl}/models`, {
      headers: { Authorization: `Bearer ${apiKey}` },
    });
    if (health.ok) {
      warmed.add(coordinator);
      ctx.ui.notify?.(`mantis: ${coordinator} ready`, "info");
      return;
    }
  } catch {
    // fall through
  }

  try {
    await fetch(`${mantisUrl}/chat/completions`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${apiKey}` },
      body: JSON.stringify({
        model: coordinator,
        messages: [{ role: "user", content: "ping" }],
        max_tokens: 1,
      }),
    });
    warmed.add(coordinator);
    ctx.ui.notify?.(`mantis: ${coordinator} ready`, "info");
  } catch (e: any) {
    ctx.ui.notify?.(`mantis: warm failed (${coordinator}): ${e.message}`, "warning");
  }
}

async function* streamOrchestrate(
  coordinator: string,
  messages: ChatMessage[],
  signal?: AbortSignal,
): AsyncGenerator<StreamEvent> {
  const apiKey = getApiKey();
  const mantisUrl = getMantisUrl();

  if (!apiKey) {
    throw new Error("MANTIS_API_KEY / LITELLM_KEY is not set in environment or .env file");
  }

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 300_000);
  if (signal) {
    signal.addEventListener("abort", () => controller.abort(), { once: true });
  }

  try {
    const res = await fetch(`${mantisUrl}/chat/completions`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${apiKey}` },
      body: JSON.stringify({
        model: coordinator,
        messages,
        stream: true,
      }),
      signal: controller.signal,
    });

    if (!res.ok) {
      const body = await res.text();
      throw new Error(`mantis backend HTTP ${res.status}: ${body}`);
    }

    // Older orchestrator images ignore `stream: true` and return one regular
    // OpenAI completion. Accept that response without requiring a backend rebuild.
    if (res.headers.get("content-type")?.includes("application/json")) {
      const body = await res.json() as any;
      const text = body.choices?.[0]?.message?.content;
      if (typeof text !== "string") {
        throw new Error("mantis backend returned an invalid completion");
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
      throw new Error("mantis backend returned an empty response body");
    }

    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() ?? "";
      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed) continue;
        try {
          const data = JSON.parse(trimmed);
          yield data as StreamEvent;
        } catch {
          // Ignore malformed NDJSON lines.
        }
      }
    }
    if (buffer.trim()) {
      try {
        yield JSON.parse(buffer.trim()) as StreamEvent;
      } catch {
        // Ignore trailing malformed line.
      }
    }
  } finally {
    clearTimeout(timeout);
  }
}

async function collectStream<T>(gen: AsyncGenerator<T>): Promise<T[]> {
  const items: T[] = [];
  for await (const item of gen) items.push(item);
  return items;
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

const MAX_SESSION_CACHE = 32;
const sessionCache = new Map<string, { finalText: string; finalTrace: string; steps: MantisStep[] }>();

const MAX_TOOL_CACHE = 128;
const toolReplyCache = new Map<string, MantisStep>();

function pruneMap<K, V>(map: Map<K, V>, maxSize: number) {
  while (map.size > maxSize) {
    const first = map.keys().next().value;
    if (first !== undefined) map.delete(first);
  }
}

function mantisStreamSimple(
  model: Model<Api>,
  context: Context,
  options?: SimpleStreamOptions,
): AssistantMessageEventStream {
  const stream = createAssistantMessageEventStream();

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
      if (options?.signal?.aborted) {
        throw new Error("aborted");
      }

      const mode = model.id as Mode;
      const { messages: backendMessages, key, lastUserContent } = toBackendMessages(context);
      output.usage.input = Math.ceil(backendMessages.reduce((chars, message) => chars + message.content.length, 0) / 4);
      output.usage.totalTokens = output.usage.input;

      const cached = sessionCache.get(key);
      if (cached) {
        stream.push({ type: "start", partial: output });
        emitText(stream, cached.finalText, output);
        output.stopReason = "stop";
        stream.push({ type: "done", reason: "stop", message: output });
        stream.end?.();
        return;
      }

      if (!lastUserContent.trim()) {
        throw new Error("No user message to process");
      }

      const coordinator = mode === "auto" ? await chooseCoordinator(lastUserContent) : mode;
      const events = await collectStream(streamOrchestrate(coordinator, backendMessages, options?.signal));

      const steps: MantisStep[] = [];
      let finalText = "";
      let finalTrace = "";

      for (const ev of events) {
        if (ev.type === "step-start") {
          // Only stored via step-end; no-op here.
        } else if (ev.type === "step-end") {
          const reply = ev.reply?.trim() ? ev.reply : "(no response)";
          steps.push({
            turn: ev.turn,
            agent_id: ev.agent_id,
            role: ev.role,
            reply,
            prompt: ev.prompt,
            model_name: ev.model_name,
            output: reply,
          });
        } else if (ev.type === "result") {
          finalText = ev.text;
          finalTrace = ev.trace;
          // Fallback if the backend did not emit per-step events.
          if (steps.length === 0) {
            for (const s of ev.mantis_steps || []) {
              const reply = s.reply?.trim() ? s.reply : "(no response)";
              steps.push({ ...s, reply, output: reply });
            }
          }
        } else if (ev.type === "error") {
          throw new Error(ev.error);
        }
      }

      sessionCache.set(key, { finalText, finalTrace, steps });
      pruneMap(sessionCache, MAX_SESSION_CACHE);

      stream.push({ type: "start", partial: output });

      if (steps.length === 0) {
        emitText(stream, finalText || "(no response)", output);
        output.stopReason = "stop";
        stream.push({ type: "done", reason: "stop", message: output });
        stream.end?.();
        return;
      }

      for (let i = 0; i < steps.length; i++) {
        const step = steps[i];
        const toolCallId = `mantis_${output.timestamp}_${i}_${Math.random().toString(36).slice(2, 8)}`;
        toolReplyCache.set(toolCallId, step);
        pruneMap(toolReplyCache, MAX_TOOL_CACHE);

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
        stream.push({
          type: "toolcall_end",
          contentIndex,
          toolCall,
          partial: output,
        });
      }

      output.stopReason = "toolUse";
      stream.push({ type: "done", reason: "toolUse", message: output });
      stream.end?.();
    } catch (err: any) {
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
  // configured FUGU_API_KEY / LITELLM_KEY in the environment or .env file.
  const apiKey = getApiKey();
  if (apiKey) {
    process.env.MANTIS_API_KEY = apiKey;
  }

  // 1. Register mantis provider with custom streamSimple implementation.
  const models = [
    {
      id: "trinity",
      name: "mantis: trinity",
      reasoning: true,
      input: ["text"] as ("text" | "image")[],
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
      contextWindow: 256000,
      maxTokens: 16384,
    },
    {
      id: "conductor",
      name: "mantis: conductor",
      reasoning: true,
      input: ["text"] as ("text" | "image")[],
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
      contextWindow: 256000,
      maxTokens: 16384,
    },
    {
      id: "auto",
      name: "mantis: auto",
      reasoning: true,
      input: ["text"] as ("text" | "image")[],
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
      contextWindow: 256000,
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
    role: Type.String({ description: "Role: Worker, Thinker, or Verifier" }),
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

    renderResult(result: AgentToolResult<MantisStep>, _options: any, theme: any) {
      const outputStr = result.content?.[0]?.type === "text" ? result.content[0].text : JSON.stringify(result.content);
      const formatted = theme.fg("toolOutput", outputStr);
      return new Text(formatted, 0, 0);
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
