/**
 * mantis — per-turn orchestration mode switching for pi.
 * /mantis off|trinity|conductor|auto (alias: /fugu)
 * Exposes native provider models (mantis/trinity, mantis/conductor, mantis/auto)
 * and native tool call steps for background worker turns.
 */

import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { Text } from "@mariozechner/pi-tui";
import { Type } from "@sinclair/typebox";

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
}

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

function buildMessagesHistory(ctx: any, currentText: string): ChatMessage[] {
  const messages: ChatMessage[] = [];
  try {
    const branch = ctx.sessionManager?.getBranch() ?? [];
    for (const entry of branch) {
      const msg = entry.message;
      if (!msg) continue;

      if (msg.role === "user") {
        let text = "";
        if (typeof msg.content === "string") {
          text = msg.content;
        } else if (Array.isArray(msg.content)) {
          text = msg.content
            .filter((c: any) => c.type === "text")
            .map((c: any) => c.text)
            .join("\n");
        }
        if (text.trim()) {
          messages.push({ role: "user", content: text });
        }
      } else if (msg.role === "assistant") {
        let text = "";
        if (typeof msg.content === "string") {
          text = msg.content;
        } else if (Array.isArray(msg.content)) {
          text = msg.content
            .filter((c: any) => c.type === "text")
            .map((c: any) => c.text)
            .join("\n");
        }
        if (text.trim()) {
          messages.push({ role: "assistant", content: text });
        }
      } else if (msg.role === "toolResult") {
        let text = "";
        if (typeof msg.content === "string") {
          text = msg.content;
        } else if (Array.isArray(msg.content)) {
          text = msg.content
            .filter((c: any) => c.type === "text")
            .map((c: any) => c.text)
            .join("\n");
        }
        if (text.trim()) {
          messages.push({ role: "user", content: `[Tool Result ${msg.toolName}]: ${text}` });
        }
      } else if (msg.role === "custom" && msg.customType === "mantis-result") {
        // Persist the assistant result into the conversation history so the next
        // user turn can see it.
        let text = "";
        if (typeof msg.content === "string") {
          text = msg.content;
        } else if (Array.isArray(msg.content)) {
          text = msg.content
            .filter((c: any) => c.type === "text")
            .map((c: any) => c.text)
            .join("\n");
        }
        if (text.trim()) {
          messages.push({ role: "assistant", content: text });
        }
      }
    }
  } catch {
    // ignore
  }

  const last = messages[messages.length - 1];
  if (!last || last.role !== "user" || last.content !== currentText) {
    messages.push({ role: "user", content: currentText });
  }

  return messages;
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

async function warm(coordinator: string, ctx: any) {
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

async function orchestrate(
  coordinator: string,
  messages: ChatMessage[],
  ctx: any,
): Promise<{ text: string; steps: MantisStep[] }> {
  await warm(coordinator, ctx);
  const apiKey = getApiKey();
  const mantisUrl = getMantisUrl();

  if (!apiKey) {
    throw new Error("MANTIS_API_KEY / LITELLM_KEY is not set in environment or .env file");
  }

  const res = await fetch(`${mantisUrl}/chat/completions`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Authorization: `Bearer ${apiKey}` },
    body: JSON.stringify({
      model: coordinator,
      messages,
    }),
    signal: AbortSignal.timeout(300_000),
  });
  if (!res.ok) throw new Error(`mantis backend HTTP ${res.status}`);
  const data = await res.json();
  const text = data.choices?.[0]?.message?.content ?? "(empty response)";
  const steps: MantisStep[] = data.mantis_steps ?? data.choices?.[0]?.message?.mantis_steps ?? data.usage?.mantis_steps ?? [];
  return { text, steps };
}

export default function (pi: any) {
  // 1. Register Providers mantis & fugu with Pi's native model registry
  const models = [
    {
      id: "trinity",
      name: "TRINITY (0.6B router + 7-slot pool)",
      reasoning: true,
      input: ["text"],
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
      contextWindow: 128000,
      maxTokens: 16384,
    },
    {
      id: "conductor",
      name: "Conductor (DAG workflow planner)",
      reasoning: true,
      input: ["text"],
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
      contextWindow: 128000,
      maxTokens: 16384,
    },
    {
      id: "auto",
      name: "Auto (Supra complexity scoring gate)",
      reasoning: true,
      input: ["text"],
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
      contextWindow: 128000,
      maxTokens: 16384,
    },
  ];

  pi.registerProvider("mantis", {
    baseUrl: getMantisUrl(),
    apiKey: "MANTIS_API_KEY",
    api: "openai-completions",
    models,
  });

  pi.registerProvider("fugu", {
    baseUrl: getMantisUrl(),
    apiKey: "MANTIS_API_KEY",
    api: "openai-completions",
    models,
  });

  // 2. Register native tool mantis_step to expose background turns natively
  pi.registerTool({
    name: "mantis_step",
    label: "Mantis Worker Step",
    description: "Executes an internal TRINITY/Conductor worker turn or DAG step",
    parameters: Type.Object({
      turn: Type.Number({ description: "Step turn index" }),
      role: Type.String({ description: "Role: Worker, Thinker, or Verifier" }),
      agent_id: Type.Number({ description: "Worker slot id (0..6)" }),
      model_name: Type.Optional(Type.String({ description: "Model name" })),
      output: Type.String({ description: "Step output content" }),
    }),
    executionMode: "sequential",

    renderCall(args: any, theme: any) {
      const role = args.role ?? "Worker";
      const agentId = args.agent_id ?? 0;
      const modelName = args.model_name ?? WORKER_NAMES[agentId] ?? `slot-${agentId}`;

      let roleColor = "accent";
      if (role === "Verifier") roleColor = "success";
      if (role === "Thinker") roleColor = "warning";

      const title = theme.fg("toolTitle", theme.bold(`[mantis ${role}]`));
      const details = theme.fg("muted", ` slot #${agentId} (${modelName}) — step ${args.turn ?? 0}`);
      return new Text(`${title}${details}`, 0, 0);
    },

    renderResult(result: any, _options: any, theme: any) {
      const outputStr = result.content?.[0]?.type === "text" ? result.content[0].text : JSON.stringify(result.content);
      const formatted = theme.fg("toolOutput", outputStr);
      return new Text(formatted, 0, 0);
    },

    async execute(_toolCallId: string, params: any) {
      return {
        content: [{ type: "text", text: params.output }],
        details: params,
        isError: false,
      };
    },
  });

  // 3. Register Slash Commands for mode switching & model setting
  const switchMode = async (m: Mode, ctx: any) => {
    if (!["off", "trinity", "conductor", "auto"].includes(m)) {
      ctx.ui.notify?.("Usage: /mantis off|trinity|conductor|auto", "error");
      return;
    }
    activeMode = m;
    ctx.ui.setStatus?.("mantis", m === "off" ? undefined : `mantis:${m}`);
    ctx.ui.notify?.(`mantis mode: ${m}`, "info");

    if (m !== "off") {
      const foundModel = ctx.modelRegistry?.find?.("mantis", m === "auto" ? "auto" : m) ??
        ctx.modelRegistry?.find?.("fugu", m === "auto" ? "auto" : m);
      if (foundModel && pi.setModel) {
        try {
          await pi.setModel(foundModel);
        } catch {
          // ignore; the input handler will still route via activeMode
        }
      }
      await warm(m === "auto" ? "trinity" : m, ctx);
    }
  };

  pi.registerCommand("mantis", {
    description: "Set orchestration mode: off | trinity | conductor | auto",
    handler: async (args: string, ctx: any) => {
      await switchMode(args.trim().toLowerCase() as Mode, ctx);
    },
  });

  pi.registerCommand("fugu", {
    description: "Set orchestration mode (alias for /mantis)",
    handler: async (args: string, ctx: any) => {
      await switchMode(args.trim().toLowerCase() as Mode, ctx);
    },
  });

  // 4. Input Handler: Coordinates turns while preserving Native User / Assistant messages
  pi.on("input", async (event: any, ctx: any) => {
    if (
      (event.source !== "interactive" && event.source !== "rpc") ||
      activeMode === "off"
    ) {
      return { action: "continue" };
    }

    let coordinator: "trinity" | "conductor";
    let score: number | undefined;

    ctx.ui.setWorkingMessage?.(`mantis: checking prompt complexity…`);
    ctx.ui.setWorkingIndicator?.({ intervalMs: 120 });

    if (activeMode === "auto") {
      score = await supraScore(event.text);
      if (score <= 2) {
        logRouting(score, "bypass", event.text);
        ctx.ui.setWorkingMessage?.();
        return { action: "continue" };
      }
      coordinator = score >= AUTO_THRESHOLD ? "conductor" : "trinity";
      logRouting(score, coordinator, event.text);
      ctx.ui.setStatus?.("mantis", `mantis:auto→${coordinator} (c${score})`);
    } else {
      coordinator = activeMode;
    }

    ctx.ui.setWorkingMessage?.(`mantis [${coordinator.toUpperCase()}]: orchestrating turns across worker pool…`);

    try {
      const fullMessages = buildMessagesHistory(ctx, event.text);
      const { text, steps } = await orchestrate(coordinator, fullMessages, ctx);

      ctx.ui.setWorkingMessage?.();

      // Emit background worker/thinker/verifier turns as native tool call steps
      if (steps && steps.length > 0) {
        for (const step of steps) {
          const modelName = WORKER_NAMES[step.agent_id] ?? `slot-${step.agent_id}`;
          await pi.sendMessage(
            {
              customType: "mantis-step",
              content: `[mantis ${step.role}] slot #${step.agent_id} (${modelName}):\n${step.reply}`,
              details: {
                turn: step.turn,
                role: step.role,
                agent_id: step.agent_id,
                model_name: modelName,
                output: step.reply,
              },
              display: true,
            },
            { triggerTurn: false },
          );
        }
      }

      // Send response as native assistant message content via sendMessage
      await pi.sendMessage(
        {
          customType: "mantis-result",
          content: text,
          details: { coordinator, score },
          display: true,
        },
        { triggerTurn: false },
      );

      return { action: "handled" };
    } catch (e: any) {
      ctx.ui.setWorkingMessage?.();
      ctx.ui.notify?.(`mantis error: ${e.message} — falling back to normal turn`, "error");
      return { action: "continue" };
    }
  });

  // Track model selections natively
  pi.on("model_select", async (event: any, ctx: any) => {
    if (event.model?.provider === "mantis" || event.model?.provider === "fugu") {
      const m = event.model.id as Mode;
      if (["trinity", "conductor", "auto"].includes(m)) {
        activeMode = m;
        ctx.ui.setStatus?.("mantis", `mantis:${m}`);
      }
    }
  });
}
