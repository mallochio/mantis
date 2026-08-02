/**
 * mantis — per-turn orchestration mode switching for pi.
 * /mantis off|trinity|conductor|auto (alias: /fugu)
 * Backend: mantis orchestrator on :8088; complexity gate: llm-router :5500 Supra header.
 */

import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { Text } from "@mariozechner/pi-tui";

type Mode = "off" | "trinity" | "conductor" | "auto";

interface ChatMessage {
  role: "user" | "assistant" | "system";
  content: string;
}

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
      // ignore read errors
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

let mode: Mode = "off";
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
    // logging must never break a turn
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
      } else if (msg.role === "custom") {
        if (msg.customType === "mantis-prompt") {
          const contentStr = typeof msg.content === "string" ? msg.content : JSON.stringify(msg.content);
          if (contentStr.trim()) {
            messages.push({ role: "user", content: contentStr });
          }
        } else if (msg.customType === "mantis-result") {
          const contentStr = typeof msg.content === "string" ? msg.content : JSON.stringify(msg.content);
          if (contentStr.trim()) {
            messages.push({ role: "assistant", content: contentStr });
          }
        }
      } else if (msg.role === "bashExecution") {
        const cmd = msg.command ?? "";
        const out = msg.output ?? "";
        if (cmd || out) {
          messages.push({
            role: "user",
            content: `[Bash Command: ${cmd}]\nOutput:\n${out}`,
          });
        }
      }
    }
  } catch {
    // ignore read errors
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
    return 1; // router down -> treat as simple
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
    // fall through to paid ping
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

async function orchestrate(coordinator: string, messages: ChatMessage[], ctx: any): Promise<string> {
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
  return data.choices?.[0]?.message?.content ?? "(empty response)";
}

export default function (pi: any) {
  // Register custom message renderers for native TUI rendering
  pi.registerMessageRenderer?.("mantis-prompt", (message: any, _options: any, theme: any) => {
    const text = theme.fg("user", `> ${message.content}`);
    return new Text(text, 0, 0);
  });

  pi.registerMessageRenderer?.("mantis-result", (message: any, options: any, theme: any) => {
    const expanded = options?.expanded;
    const coordinator = message.details?.coordinator ?? "trinity";
    const score = message.details?.score;
    const durationMs = message.details?.durationMs;

    let meta = `mantis:${coordinator}`;
    if (score !== undefined) meta += ` | complexity: c${score}`;
    if (durationMs !== undefined) meta += ` | ${durationMs}ms`;

    let text = theme.fg("accent", theme.bold(`🦗 [${meta}]`)) + "\n";
    text += typeof message.content === "string" ? message.content : JSON.stringify(message.content, null, 2);

    if (expanded && message.details) {
      text += "\n\n" + theme.fg("muted", `Details:\n${JSON.stringify(message.details, null, 2)}`);
    }

    return new Text(text, 0, 0);
  });

  const handleCommand = async (args: string, ctx: any) => {
    const m = args.trim().toLowerCase() as Mode;
    if (!["off", "trinity", "conductor", "auto"].includes(m)) {
      ctx.ui.notify?.("Usage: /mantis off|trinity|conductor|auto", "error");
      return;
    }
    mode = m;
    ctx.ui.setStatus?.("mantis", m === "off" ? undefined : `mantis:${m}`);
    ctx.ui.notify?.(`mantis mode: ${m}`, "info");
    if (m === "trinity") await warm("trinity", ctx);
    if (m === "conductor") await warm("conductor", ctx);
    if (m === "auto") await warm("trinity", ctx);
  };

  pi.registerCommand("mantis", {
    description: "Set orchestration mode: off | trinity | conductor | auto",
    handler: handleCommand,
  });

  pi.registerCommand("fugu", {
    description: "Set orchestration mode (alias for /mantis)",
    handler: handleCommand,
  });

  pi.on("input", async (event: any, ctx: any) => {
    if (
      (event.source !== "interactive" && event.source !== "rpc") ||
      mode === "off"
    ) {
      return { action: "continue" };
    }

    let coordinator: "trinity" | "conductor";
    let score: number | undefined;

    ctx.ui.setWorkingMessage?.(`mantis: checking prompt complexity…`);
    ctx.ui.setWorkingIndicator?.({ intervalMs: 120 });

    if (mode === "auto") {
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
      coordinator = mode;
    }

    ctx.ui.setWorkingMessage?.(`mantis [${coordinator.toUpperCase()}]: orchestrating turns across worker pool…`);

    const startTime = Date.now();
    try {
      const fullMessages = buildMessagesHistory(ctx, event.text);
      const result = await orchestrate(coordinator, fullMessages, ctx);
      const durationMs = Date.now() - startTime;

      ctx.ui.setWorkingMessage?.();

      // Record user input into session tree so it is visible and preserved in multi-turn history
      await pi.sendMessage(
        {
          customType: "mantis-prompt",
          content: event.text,
          display: true,
        },
        { triggerTurn: false },
      );

      // Record orchestrator result into session tree with metadata details
      await pi.sendMessage(
        {
          customType: "mantis-result",
          content: result,
          details: { coordinator, score, durationMs },
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
}
