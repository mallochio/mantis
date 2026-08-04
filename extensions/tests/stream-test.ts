import mantisExtension from "../mantis.ts";
import type { Context, Message } from "@mariozechner/pi-ai";

process.env.MANTIS_API_KEY = "test-key";
process.env.MANTIS_AUTO_THRESHOLD = "6";

const providers = new Map<string, any>();
const tools = new Map<string, any>();
const commands = new Map<string, any>();
const pi = {
  registerProvider: (name: string, config: any) => providers.set(name, config),
  registerTool: (tool: any) => tools.set(tool.name, tool),
  registerMessageRenderer: () => {},
  registerCommand: (name: string, config: any) => commands.set(name, config),
  on: () => {}, sendMessage: () => {}, setModel: async () => true,
  getFlag: () => undefined, unregisterProvider: () => {},
  events: { on: () => {}, off: () => {}, emit: () => {} },
};
mantisExtension(pi as any);

const provider = providers.get("mantis");
const stepTool = tools.get("mantis_step");
if (!provider?.streamSimple || !stepTool || !commands.has("mantis")) throw new Error("extension registration failed");
const theme = { fg: (_: string, text: string) => text, bold: (text: string) => text };
const rendered = stepTool.renderResult(
  { content: [{ type: "text", text: "hidden" }] }, { expanded: false }, theme,
).render(80);
if (rendered.length !== 0) throw new Error("step not collapsed");

const requests: Array<{ method: string; url: string; body?: any }> = [];
const scripts = new Map<string, any[]>();
const createdByPrompt = new Map<string, string>();
const promptByRun = new Map<string, string>();
const failedRetries = new Set<string>();
const deleted: Array<{ id: string; signalAborted: boolean }> = [];
let markCancelEntered!: () => void;
let markCancelDeleted!: () => void;
const cancelEntered = new Promise<void>((resolve) => { markCancelEntered = resolve; });
const cancelDeleted = new Promise<void>((resolve) => { markCancelDeleted = resolve; });

function response(value: unknown, status = 200, headers?: HeadersInit) {
  return new Response(JSON.stringify(value), { status, headers: { "Content-Type": "application/json", ...headers } });
}

(globalThis as any).fetch = async (rawUrl: string, init: RequestInit = {}) => {
  const url = rawUrl.toString();
  const method = init.method ?? "GET";
  const body = typeof init.body === "string" && init.body ? JSON.parse(init.body) : undefined;
  requests.push({ method, url, body });

  if (url.includes(":5500/") && url.endsWith("/chat/completions")) {
    return response({}, 200, { "x-route-supra-complexity": "9" });
  }
  if (url.endsWith("/warm") || url.includes("/warm?")) return response({ status: "ready" });
  if (url.endsWith("/runs") && method === "POST") {
    if (
      !Array.isArray(body?.messages) || !Array.isArray(body?.tools) ||
      typeof body?.run_id !== "string" || body.run_id.length !== 32
    ) return response({ error: "bad create" }, 400);
    const prompt = body.messages.at(-1)?.content ?? "";
    const id = body.run_id;
    createdByPrompt.set(prompt, id);
    promptByRun.set(id, prompt);
    if (prompt.includes("oversized")) {
      return new Response("{}", { status: 200, headers: { "Content-Length": "1000001" } });
    }
    if (prompt.includes("native")) scripts.set(id, [
      { type: "tool_calls", tool_calls: [{ id: "read-native", name: "read", arguments: { path: "README.md" } }] },
      { type: "step_complete", turn: 0, role: "Worker", agent_id: 1, model_name: "worker", prompt, reply: "worker answer" },
      { type: "step_complete", turn: 1, role: "Verifier", agent_id: 2, model_name: "verifier", prompt: "verify", reply: "ACCEPT" },
      { type: "final", text: "done" },
    ]);
    else if (prompt === "A" || prompt === "B") scripts.set(id, [
      { type: "tool_calls", tool_calls: [{ id: `read-${prompt}`, name: "read", arguments: { path: `${prompt}.txt` } }] },
      { type: "final", text: prompt },
    ]);
    else if (prompt.includes("cancel")) scripts.set(id, ["wait"]);
    else if (prompt.includes("retry")) scripts.set(id, [
      { type: "tool_calls", tool_calls: [{ id: "read-retry", name: "read", arguments: { path: "README.md" } }] },
      { type: "final", text: "retry ok" },
    ]);
    else if (prompt.includes("malformed")) scripts.set(id, [
      { type: "tool_calls", tool_calls: [{ id: "bad", name: "not-active", arguments: {} }] },
    ]);
    else scripts.set(id, [
      { type: "step_complete", turn: 0, role: "Planner", agent_id: 0, model_name: "planner", prompt, reply: "plan" },
      { type: "final", text: body.model },
    ]);
    return response({ run_id: id });
  }
  if (url.includes("/runs/") && url.endsWith("/continue") && method === "POST") {
    const id = url.split("/runs/")[1].split("/")[0];
    const script = scripts.get(id);
    if (!script) return response({ error: "unknown run" }, 404);
    if (promptByRun.get(id)?.includes("retry") && body?.tool_results && !failedRetries.has(id)) {
      failedRetries.add(id);
      throw new TypeError("fetch failed");
    }
    const event = script.shift();
    if (event === "wait") {
      markCancelEntered();
      return await new Promise<Response>((_resolve, reject) => {
        const abort = () => reject(new DOMException("aborted", "AbortError"));
        if (init.signal?.aborted) abort();
        else init.signal?.addEventListener("abort", abort, { once: true });
      });
    }
    if (!event) return response({ error: "script exhausted" }, 500);
    return response(event);
  }
  if (url.includes("/runs/") && method === "DELETE") {
    const id = url.split("/runs/")[1];
    deleted.push({ id, signalAborted: init.signal?.aborted ?? false });
    if (id === createdByPrompt.get("cancel task")) markCancelDeleted();
    return response({ deleted: true });
  }
  return response({ error: `unexpected request: ${method} ${url}` }, 404);
};

const activeTools = [
  { name: "read", description: "read", parameters: { type: "object", properties: {} } },
  { name: "bash", description: "bash", parameters: { type: "object", properties: {} } },
] as any;
const model = (id = "trinity") => ({ id, provider: "mantis", api: "mantis", name: `mantis: ${id}`, contextWindow: 256000, maxTokens: 16384 });
const user = (text: string): Message => ({ role: "user", content: text, timestamp: Date.now() });
const context = (text: string, trailing: Message[] = []): Context => ({
  systemPrompt: "PI_SYSTEM_CONTEXT", messages: [user(text), ...trailing], tools: activeTools,
});
const toolResult = (call: any, text: string): Message => ({
  role: "toolResult", toolCallId: call.id, toolName: call.name,
  content: [{ type: "text", text }], isError: false, timestamp: Date.now(),
});
async function collect(selectedModel: any, ctx: Context, signal?: AbortSignal) {
  const events: any[] = [];
  for await (const event of provider.streamSimple(selectedModel, ctx, { apiKey: "test-key", signal })) events.push(event);
  return events;
}
const emittedCall = (events: any[]) => events.find((event) => event.type === "toolcall_end")?.toolCall;
const finalText = (events: any[]) => events.find((event) => event.type === "text_end")?.content;
const errorText = (events: any[]) => events.find((event) => event.type === "error")?.error?.errorMessage;

// Real native tool call -> Pi result -> collapsed role steps -> exactly one final answer.
const first = await collect(model(), context("native task"));
const readCall = emittedCall(first);
if (readCall?.name !== "read") throw new Error("backend tool call was not emitted natively");
const workerEvents = await collect(model(), context("native task", [toolResult(readCall, "file contents")]));
const workerStep = emittedCall(workerEvents);
if (workerStep?.name !== "mantis_step" || workerStep.arguments.role !== "Worker") throw new Error("worker step missing");
const verifierEvents = await collect(model(), context("native task", [toolResult(workerStep, "worker answer")]));
const verifierStep = emittedCall(verifierEvents);
if (verifierStep?.arguments.role !== "Verifier") throw new Error("verifier step missing");
const finalEvents = await collect(model(), context("native task", [toolResult(verifierStep, "ACCEPT")]));
const allNativeEvents = [...first, ...workerEvents, ...verifierEvents, ...finalEvents];
if (finalText(finalEvents) !== "done" || allNativeEvents.filter((event) => event.type === "text_end").length !== 1) {
  throw new Error("exactly-one final answer failed");
}
const nativeContinue = requests.find((request) => request.url.includes("/continue") && request.body?.tool_results);
if (
  nativeContinue?.body.tool_results[0]?.tool_call_id !== "read-native" ||
  nativeContinue.body.tool_results[0]?.is_error !== false
) throw new Error("native result not forwarded");

// Replays fail closed without creating another run.
const createsBeforeReplay = requests.filter((request) => request.url.endsWith("/runs") && request.method === "POST").length;
const replay = await collect(model(), context("native task", [toolResult(verifierStep, "ACCEPT")]));
const createsAfterReplay = requests.filter((request) => request.url.endsWith("/runs") && request.method === "POST").length;
if (!errorText(replay)?.includes("already consumed") || createsAfterReplay !== createsBeforeReplay) throw new Error("replay did not fail closed");

// Concurrent sessions resume their own run, even in reverse order.
const [eventsA, eventsB] = await Promise.all([
  collect(model(), context("A")),
  collect(model(), context("B")),
]);
const callA = emittedCall(eventsA); const callB = emittedCall(eventsB);
const resultB = await collect(model(), context("B", [toolResult(callB, "b")]));
const resultA = await collect(model(), context("A", [toolResult(callA, "a")]));
if (finalText(resultA) !== "A" || finalText(resultB) !== "B") throw new Error("concurrent runs crossed");

// A lost continuation response restores correlation and reuses the idempotency key.
const retryStart = await collect(model(), context("retry task"));
const retryCall = emittedCall(retryStart);
const retryFailure = await collect(model(), context("retry task", [toolResult(retryCall, "contents")]));
if (!errorText(retryFailure)?.includes("fetch failed")) throw new Error("transport failure not surfaced");
const retryFinal = await collect(model(), context("retry task", [toolResult(retryCall, "contents")]));
if (finalText(retryFinal) !== "retry ok") throw new Error("continuation retry did not resume");
const retryRequests = requests.filter((request) => request.url.includes(`${createdByPrompt.get("retry task")}/continue`) && request.body?.request_id);
if (retryRequests.length !== 2 || retryRequests[0].body.request_id !== retryRequests[1].body.request_id) {
  throw new Error("continuation retry was not idempotent");
}

// Cancellation sends an independent DELETE despite the caller signal being aborted.
const controller = new AbortController();
const cancelling = collect(model(), context("cancel task"), controller.signal);
await cancelEntered;
controller.abort();
const cancelled = await cancelling;
await cancelDeleted;
const cancelledRunId = createdByPrompt.get("cancel task");
const cancelDelete = deleted.find((entry) => entry.id === cancelledRunId);
if (!cancelled.some((event) => event.type === "error" && event.reason === "aborted") || !cancelDelete || cancelDelete.signalAborted) {
  throw new Error("cancel cleanup failed");
}

// Malformed/unavailable tools and oversized responses are rejected and cleaned up.
const malformed = await collect(model(), context("malformed task"));
if (!errorText(malformed)?.includes("invalid or unavailable")) throw new Error("invalid backend event accepted");
const oversized = await collect(model(), context("oversized response"));
if (!errorText(oversized)?.includes("size limit")) throw new Error("oversized backend response accepted");

// Conductor and Auto route through the same resumable API.
const conductorFirst = await collect(model("conductor"), context("conductor task"));
const conductorStep = emittedCall(conductorFirst);
if (conductorStep?.name !== "mantis_step") throw new Error("conductor planner step missing");
const conductorFinal = await collect(model("conductor"), context("conductor task", [toolResult(conductorStep, "plan")]));
if (finalText(conductorFinal) !== "conductor") throw new Error("conductor lifecycle failed");
const autoFirst = await collect(model("auto"), context("auto task"));
const autoCreate = requests.filter((request) => request.url.endsWith("/runs") && request.method === "POST").at(-1);
if (emittedCall(autoFirst)?.name !== "mantis_step" || autoCreate?.body?.model !== "conductor") throw new Error("auto routing failed");

console.log("✓ native tools, replay, concurrency, cancellation, validation, Trinity, Conductor, and Auto");
