/**
 * fugu — per-turn orchestration mode switching for pi.
 * /fugu off|trinity|conductor|auto
 * Backend: OpenFugu on :8088; complexity gate: llm-router :5500 Supra header.
 */

type Mode = "off" | "trinity" | "conductor" | "auto";

const FUGU_URL = process.env.FUGU_URL ?? "http://127.0.0.1:8088/v1";
const ROUTER_URL = process.env.FUGU_ROUTER_URL ?? "http://127.0.0.1:5500/v1";
const API_KEY = process.env.FUGU_API_KEY ?? "sk-fugu-local";
const AUTO_THRESHOLD = parseInt(process.env.FUGU_AUTO_THRESHOLD ?? "4", 10);

let mode: Mode = "off";
const warmed = new Set<string>();

async function supraScore(text: string): Promise<number> {
  try {
    const res = await fetch(`${ROUTER_URL}/chat/completions`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${API_KEY}` },
      body: JSON.stringify({
        model: "auto",
        messages: [{ role: "user", content: text }],
        max_tokens: 1,
      }),
    });
    return parseInt(res.headers.get("x-route-supra-complexity") ?? "1", 10);
  } catch {
    return 1; // router down → treat as simple, stay on trinity/bypass
  }
}

async function warm(coordinator: string, ctx: any) {
  if (warmed.has(coordinator)) return;
  ctx.ui.notify?.(`fugu: warming ${coordinator} (first call may download weights)…`, "info");
  try {
    await fetch(`${FUGU_URL}/chat/completions`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${API_KEY}` },
      body: JSON.stringify({
        model: coordinator,
        messages: [{ role: "user", content: "ping" }],
        max_tokens: 1,
      }),
    });
    warmed.add(coordinator);
    ctx.ui.notify?.(`fugu: ${coordinator} ready`, "info");
  } catch (e: any) {
    ctx.ui.notify?.(`fugu: warm failed (${coordinator}): ${e.message}`, "warning");
  }
}

async function orchestrate(coordinator: string, task: string, ctx: any): Promise<string> {
  await warm(coordinator, ctx);
  const res = await fetch(`${FUGU_URL}/chat/completions`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Authorization: `Bearer ${API_KEY}` },
    body: JSON.stringify({
      model: coordinator,
      messages: [{ role: "user", content: task }],
    }),
  });
  if (!res.ok) throw new Error(`fugu backend HTTP ${res.status}`);
  const data = await res.json();
  return data.choices?.[0]?.message?.content ?? "(empty response)";
}

export default function (pi: any) {
  pi.registerCommand("fugu", {
    description: "Set orchestration mode: off | trinity | conductor | auto",
    handler: async (args: string, ctx: any) => {
      const m = args.trim().toLowerCase() as Mode;
      if (!["off", "trinity", "conductor", "auto"].includes(m)) {
        ctx.ui.notify?.("Usage: /fugu off|trinity|conductor|auto", "error");
        return;
      }
      mode = m;
      ctx.ui.setStatus?.("fugu", m === "off" ? "" : `fugu:${m}`);
      ctx.ui.notify?.(`fugu mode: ${m}`, "info");
      if (m === "trinity") await warm("trinity", ctx);
      if (m === "conductor") await warm("conductor", ctx);
      if (m === "auto") await warm("trinity", ctx);
    },
  });

  pi.on("input", async (event: any, ctx: any) => {
    // support both TUI (interactive) and headless RPC (rpc) sessions
    if (
      (event.source !== "interactive" && event.source !== "rpc") ||
      mode === "off"
    ) {
      return { action: "continue" };
    }

    let coordinator: "trinity" | "conductor";
    if (mode === "auto") {
      const score = await supraScore(event.text);
      if (score <= 2) return { action: "continue" };
      coordinator = score >= AUTO_THRESHOLD ? "conductor" : "trinity";
      ctx.ui.setStatus?.("fugu", `fugu:auto→${coordinator} (c${score})`);
    } else {
      coordinator = mode;
    }

    try {
      const result = await orchestrate(coordinator, event.text, ctx);
      await pi.sendMessage(
        { customType: "fugu-result", content: result, display: true },
        { triggerTurn: false },
      );
      return { action: "handled" };
    } catch (e: any) {
      ctx.ui.notify?.(`fugu error: ${e.message} — falling back to normal turn`, "error");
      return { action: "continue" };
    }
  });
}
