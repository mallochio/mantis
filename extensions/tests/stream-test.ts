/**
 * End-to-end streaming test for the mantis pi extension.
 *
 * Mocks the Mantis backend with an NDJSON stream, then exercises the
 * provider's streamSimple implementation. Verifies that worker turns are
 * emitted as native tool calls and that the final answer is returned on
 * the follow-up provider call after tool execution.
 */

import { writeFileSync, unlinkSync } from "node:fs";
import { TextEncoder } from "node:util";
import mantisExtension, { getMantisContextWindow, streamOrchestrate, warm, warmed } from "../mantis.ts";
import type { Context, Message } from "@mariozechner/pi-ai";

process.env.MANTIS_API_KEY = "test-key";

const commands = new Map<string, any>();
const handlers = new Map<string, any[]>();
const tools = new Map<string, any>();
const providers = new Map<string, any>();

const pi = {
  registerProvider: (name: string, config: any) => {
    providers.set(name, config);
  },
  registerTool: (tool: any) => {
    tools.set(tool.name, tool);
  },
  registerMessageRenderer: () => {},
  registerCommand: (name: string, config: any) => {
    commands.set(name, config);
  },
  on: (event: string, handler: any) => {
    if (!handlers.has(event)) handlers.set(event, []);
    handlers.get(event)!.push(handler);
  },
  sendMessage: () => {},
  setModel: async () => true,
  getFlag: () => undefined,
  unregisterProvider: () => {},
  events: { on: () => {}, off: () => {}, emit: () => {} },
};

mantisExtension(pi as any);

const mantisCmd = commands.get("mantis");
if (!mantisCmd) {
  console.error("FAIL: /mantis command not registered");
  process.exit(1);
}

const mantisProvider = providers.get("mantis");
if (!mantisProvider?.streamSimple) {
  console.error("FAIL: mantis provider with streamSimple not registered");
  process.exit(1);
}
if (mantisProvider.models.some((model: any) => model.contextWindow !== 256000)) {
  console.error("FAIL: not all mantis models advertise a 256k context window");
  process.exit(1);
}

const mantisStepTool = tools.get("mantis_step");
if (!mantisStepTool) {
  console.error("FAIL: mantis_step tool not registered");
  process.exit(1);
}

const renderTheme = {
  fg: (_color: string, text: string) => text,
  bold: (text: string) => text,
};
const renderResult = { content: [{ type: "text", text: "intermediate worker output" }] };
if (mantisStepTool.renderResult(renderResult, { expanded: false }, renderTheme).render(80).length !== 0) {
  console.error("FAIL: mantis_step output is not collapsed by default");
  process.exit(1);
}
if (!mantisStepTool.renderResult(renderResult, { expanded: true }, renderTheme).render(80).join("\n").includes("intermediate worker output")) {
  console.error("FAIL: expanded mantis_step output is hidden");
  process.exit(1);
}
console.log("✓ /mantis command, provider, and collapsed native mantis_step tool registered");

const commandCtx = {
  cwd: process.cwd(),
  hasUI: true,
  ui: {
    notify: (msg: string) => console.log("[notify]", msg),
    setStatus: (_key: string, _text: string | undefined) => {},
    setWorkingMessage: (msg?: string) => console.log("[working]", msg),
    setWorkingIndicator: () => {},
    setWorkingVisible: (_visible: boolean) => {},
  },
  modelRegistry: {
    find: (provider: string, id: string) => ({ provider, id, name: `${provider}/${id}` }),
  },
};

const encoder = new TextEncoder();
const backendRequests: any[] = [];
const streamLines = [
  JSON.stringify({
    type: "step-start",
    turn: 0,
    role: "Worker",
    agent_id: 4,
    model_name: "claude-opus-5-medium",
    prompt: "Implement a small helper that reverses a string",
  }) + "\n",
  JSON.stringify({
    type: "step-end",
    turn: 0,
    role: "Worker",
    agent_id: 4,
    model_name: "claude-opus-5-medium",
    prompt: "Implement a small helper that reverses a string",
    reply: "def reverse(s): return s[::-1]",
  }) + "\n",
  JSON.stringify({
    type: "step-end",
    turn: 1,
    role: "Worker",
    agent_id: 2,
    model_name: "gpt-5.6-sol-medium",
    prompt: "Retry the helper",
    reply: "",
  }) + "\n",
  JSON.stringify({
    type: "result",
    text: "def reverse(s):\n    return s[::-1]",
    trace: "Worker(4)→Verifier(1):verifier_accept",
    coordinator: "trinity",
    mantis_steps: [
      {
        turn: 0,
        agent_id: 4,
        role: "Worker",
        reply: "def reverse(s): return s[::-1]",
        prompt: "Implement a small helper that reverses a string",
        model_name: "claude-opus-5-medium",
      },
    ],
  }) + "\n",
];

(globalThis as any).fetch = async (_url: string, init?: any) => {
  const url = _url.toString();
  if (url.includes("/warm")) {
    const auth = init?.headers?.Authorization;
    if (!auth || !auth.includes("Bearer")) {
      return new Response(JSON.stringify({ error: "unauthorized" }), {
        status: 401,
        headers: { "Content-Type": "application/json" },
      });
    }
    const mode = new URL(url, "http://127.0.0.1:8088").searchParams.get("mode");
    if (mode !== "trinity" && mode !== "conductor") {
      return new Response(JSON.stringify({ error: `unknown mode: ${mode}` }), {
        status: 400,
        headers: { "Content-Type": "application/json" },
      });
    }
    return new Response(JSON.stringify({ status: "ready", mode }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  }
  if (url.endsWith("/models")) {
    return new Response(JSON.stringify({ data: [] }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  }
  if (init && typeof init.body === "string" && init.body.includes('"stream":true')) {
    backendRequests.push(JSON.parse(init.body));
    if (init.body.includes("Say hi")) {
      return new Response(JSON.stringify({
        choices: [{ message: { content: "Hello!" } }],
        usage: { fugu_trace: "Worker(0):max_turns" },
      }), { status: 200, headers: { "Content-Type": "application/json" } });
    }
    return new Response(
      new ReadableStream({
        start(controller) {
          for (const line of streamLines) {
            controller.enqueue(encoder.encode(line));
          }
          controller.close();
        },
      }),
      { status: 200, headers: { "Content-Type": "application/x-ndjson" } },
    );
  }
  return new Response(JSON.stringify({ choices: [{ message: { content: "pong" } }] }), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  });
};

await mantisCmd.handler("trinity", commandCtx);

const model = {
  id: "trinity",
  provider: "mantis",
  api: "mantis",
  name: "mantis: trinity",
  contextWindow: 256000,
  maxTokens: 16384,
};

const userMessage: Message = {
  role: "user",
  content: "Implement a small helper that reverses a string",
  timestamp: Date.now(),
};

const context1: Context = { systemPrompt: "PI_SYSTEM_CONTEXT", messages: [userMessage] };
const stream1 = mantisProvider.streamSimple(model, context1, { apiKey: "test-key" });
const events1: any[] = [];
for await (const ev of stream1) events1.push(ev);

const sentSystemContext = backendRequests[0]?.messages?.[0]?.content ?? "";
if (!sentSystemContext.includes("PI_SYSTEM_CONTEXT") ||
    !sentSystemContext.includes("Repository tree") ||
    !sentSystemContext.includes("getMantisContextWindow")) {
  console.error("FAIL: Pi and repository source context were not sent to the backend");
  process.exit(1);
}

const start1 = events1.find((e) => e.type === "start");
if (!start1) {
  console.error("FAIL: first stream did not emit start");
  process.exit(1);
}

const toolcallEnds = events1.filter((e) => e.type === "toolcall_end");
const toolcallEnd = toolcallEnds[0];
if (toolcallEnds.length !== 2) {
  console.error("FAIL: stream did not expose every backend turn, got:", toolcallEnds.length);
  process.exit(1);
}

const done1 = events1.find((e) => e.type === "done");
if (!done1 || done1.reason !== "toolUse") {
  console.error("FAIL: first stream did not end with reason toolUse, got:", done1?.reason);
  process.exit(1);
}

const toolCall = toolcallEnd.toolCall;
if (toolCall.name !== "mantis_step") {
  console.error("FAIL: expected tool call to mantis_step, got:", toolCall.name);
  process.exit(1);
}

if (toolCall.arguments.role !== "Worker" || toolCall.arguments.model_name !== "claude-opus-5-medium") {
  console.error("FAIL: tool call arguments mismatch:", toolCall.arguments);
  process.exit(1);
}

console.log("✓ First stream emitted native mantis_step tool call");

// Simulate pi executing the tool call.
const toolResult = await mantisStepTool.execute(toolCall.id, toolCall.arguments, undefined, undefined, {});

const expectedReply = "def reverse(s): return s[::-1]";
const resultText = toolResult.content?.find((c: any) => c.type === "text")?.text;
if (resultText !== expectedReply) {
  console.error("FAIL: mantis_step execute returned wrong text, got:", resultText);
  process.exit(1);
}

const emptyToolResult = await mantisStepTool.execute(
  toolcallEnds[1].toolCall.id,
  toolcallEnds[1].toolCall.arguments,
  undefined,
  undefined,
  {},
);
const emptyResultText = emptyToolResult.content?.find((c: any) => c.type === "text")?.text;
if (emptyResultText !== "(no response)") {
  console.error("FAIL: empty backend turn was hidden, got:", emptyResultText);
  process.exit(1);
}
console.log("✓ mantis_step exposed successful and empty worker turns");

// Second provider call: pi sends the tool result back and expects the final answer.
const toolResultMessage: Message = {
  role: "toolResult",
  toolCallId: toolCall.id,
  toolName: "mantis_step",
  content: [{ type: "text", text: expectedReply }],
  isError: false,
  timestamp: Date.now(),
};

const secondToolResultMessage: Message = {
  role: "toolResult",
  toolCallId: toolcallEnds[1].toolCall.id,
  toolName: "mantis_step",
  content: [{ type: "text", text: "(no response)" }],
  isError: false,
  timestamp: Date.now(),
};
const context2: Context = {
  systemPrompt: "PI_SYSTEM_CONTEXT_CHANGED_AFTER_ACCEPT",
  messages: [userMessage, toolResultMessage, secondToolResultMessage],
};

// Simulate a background coding agent mutating repository context between Pi's
// tool-call response and its follow-up provider call.
const mutationPath = ".mantis-bridge-mutation.tmp";
writeFileSync(mutationPath, "changed after verifier ACCEPT\n");
const requestsBeforeFollowup = backendRequests.length;
const stream2 = mantisProvider.streamSimple(model, context2, { apiKey: "test-key" });
const events2: any[] = [];
try {
  for await (const ev of stream2) events2.push(ev);
} finally {
  unlinkSync(mutationPath);
}
if (backendRequests.length !== requestsBeforeFollowup) {
  console.error("FAIL: repository mutation caused a second orchestration after ACCEPT");
  process.exit(1);
}
if (events2.some((e) => e.type === "toolcall_start" || e.type === "toolcall_end")) {
  console.error("FAIL: follow-up emitted post-ACCEPT worker steps");
  process.exit(1);
}

const textEnd = events2.find((e) => e.type === "text_end");
if (!textEnd) {
  console.error("FAIL: second stream did not emit text_end");
  process.exit(1);
}

if (textEnd.content !== "def reverse(s):\n    return s[::-1]") {
  console.error("FAIL: final text mismatch, got:", textEnd.content);
  process.exit(1);
}

const done2 = events2.find((e) => e.type === "done");
if (!done2 || done2.reason !== "stop") {
  console.error("FAIL: second stream did not end with reason stop, got:", done2?.reason);
  process.exit(1);
}
if (done2.message.usage.input <= 0 || done2.message.usage.totalTokens <= done2.message.usage.input) {
  console.error("FAIL: mantis did not report estimated context usage");
  process.exit(1);
}

const finalAnswerEvents = [...events1, ...events2].filter((e) => e.type === "text_end");
if (finalAnswerEvents.length !== 1) {
  console.error("FAIL: accepted orchestration did not emit exactly one final answer");
  process.exit(1);
}
console.log("✓ ACCEPT bridged exactly one final answer despite repository mutation");

// A replayed post-tool follow-up must fail closed instead of launching another run.
const reqsBeforeStream3 = backendRequests.length;
const stream3Events: any[] = [];
for await (const ev of mantisProvider.streamSimple(model, context2, { apiKey: "test-key" })) {
  stream3Events.push(ev);
}
if (backendRequests.length !== reqsBeforeStream3 ||
    !stream3Events.some((e) => e.type === "error" && e.error?.errorMessage?.includes("already consumed"))) {
  console.error("FAIL: replayed tool results restarted orchestration instead of failing closed");
  process.exit(1);
}
console.log("✓ Replayed tool-result follow-up cannot restart an accepted run");

// 2. Test mode scoping: identical messages under different modes do not collide
const modelConductor = { ...model, id: "conductor", name: "mantis: conductor" };
const modeUserMsg: Message = { role: "user", content: "Mode scope check", timestamp: Date.now() };
const modeContext: Context = { systemPrompt: "PI_SYSTEM_CONTEXT", messages: [modeUserMsg] };

// Call trinity -> populates cache for trinity
const modeStream1 = mantisProvider.streamSimple(model, modeContext, { apiKey: "test-key" });
for await (const _ of modeStream1) {}

const reqsBeforeConductor = backendRequests.length;
// Call conductor with same context -> should miss cache and call backend
const modeStream2 = mantisProvider.streamSimple(modelConductor, modeContext, { apiKey: "test-key" });
for await (const _ of modeStream2) {}

if (backendRequests.length !== reqsBeforeConductor + 1) {
  console.error("FAIL: conductor mode collided with trinity mode cache entry");
  process.exit(1);
}
console.log("✓ Same messages under different modes do not collide in cache");

// 3. Test no-step responses are never cached
const legacyContext: Context = {
  messages: [{ role: "user", content: "Say hi", timestamp: Date.now() }],
};
const reqsBeforeLegacy1 = backendRequests.length;
const legacyEvents1: any[] = [];
for await (const ev of mantisProvider.streamSimple(model, legacyContext, { apiKey: "test-key" })) {
  legacyEvents1.push(ev);
}
if (backendRequests.length !== reqsBeforeLegacy1 + 1) {
  console.error("FAIL: first no-step call did not invoke backend");
  process.exit(1);
}

const reqsBeforeLegacy2 = backendRequests.length;
const legacyEvents2: any[] = [];
for await (const ev of mantisProvider.streamSimple(model, legacyContext, { apiKey: "test-key" })) {
  legacyEvents2.push(ev);
}
if (backendRequests.length !== reqsBeforeLegacy2 + 1) {
  console.error("FAIL: no-step response was cached");
  process.exit(1);
}

const legacyText = legacyEvents2.find((e) => e.type === "text_end")?.content;
if (legacyText !== "Hello!") {
  console.error("FAIL: regular JSON completion fallback returned:", legacyText);
  process.exit(1);
}
console.log("✓ No-step responses are never cached");

const requestsBeforeOverflow = backendRequests.length;
const overflowEvents: any[] = [];
for await (const ev of mantisProvider.streamSimple(model, {
  messages: [{ role: "user", content: "x".repeat(1_100_000), timestamp: Date.now() }],
}, { apiKey: "test-key" })) {
  overflowEvents.push(ev);
}
const overflow = overflowEvents.find((e) => e.type === "error");
if (!overflow?.error?.errorMessage?.includes("exceeds the context window") ||
    backendRequests.length !== requestsBeforeOverflow) {
  console.error("FAIL: oversized context was sent to the backend");
  process.exit(1);
}
console.log("✓ Oversized context fails locally for Pi/Slipstream compaction recovery");

// 4. Test incremental tool call emission prior to stream finish
let pushChunk!: (str: string) => void;
let closeStream!: () => void;
(globalThis as any).fetch = async () => new Response(
  new ReadableStream({
    start(controller) {
      pushChunk = (str: string) => controller.enqueue(encoder.encode(str));
      closeStream = () => controller.close();
    },
  }),
  { status: 200, headers: { "Content-Type": "application/x-ndjson" } },
);

const incContext: Context = {
  systemPrompt: "sys",
  messages: [{ role: "user", content: "Incremental test", timestamp: Date.now() }],
};
const incStream = mantisProvider.streamSimple(model, incContext, { apiKey: "test-key" });
const incReader = incStream[Symbol.asyncIterator]();

const startEv = (await incReader.next()).value;
if (startEv.type !== "start") {
  console.error("FAIL: incremental test missing start event");
  process.exit(1);
}

// Push step-end before result
pushChunk(JSON.stringify({
  type: "step-end",
  turn: 0,
  role: "Worker",
  agent_id: 1,
  prompt: "p0",
  reply: "r0",
}) + "\n");

const tcStartEv = (await incReader.next()).value;
const tcEndEv = (await incReader.next()).value;

if (tcStartEv.type !== "toolcall_start" || tcEndEv.type !== "toolcall_end") {
  console.error("FAIL: tool call was not emitted incrementally upon step-end");
  process.exit(1);
}

pushChunk(JSON.stringify({ type: "result", text: "done", trace: "", coordinator: "trinity", mantis_steps: [] }) + "\n");
closeStream();

const doneEv = (await incReader.next()).value;
if (doneEv.type !== "done") {
  console.error("FAIL: incremental stream did not end with done");
  process.exit(1);
}
console.log("✓ Native tool call events are emitted incrementally as step-end arrives");

// 5. Test chunk boundary reassembly across stream packets
const fullLine = JSON.stringify({
  type: "result",
  text: "chunk boundary test",
  trace: "",
  coordinator: "trinity",
  mantis_steps: [],
}) + "\n";
const chunkA = fullLine.slice(0, 15);
const chunkB = fullLine.slice(15);

(globalThis as any).fetch = async () => new Response(
  new ReadableStream({
    start(controller) {
      controller.enqueue(encoder.encode(chunkA));
      controller.enqueue(encoder.encode(chunkB));
      controller.close();
    },
  }),
  { status: 200, headers: { "Content-Type": "application/x-ndjson" } },
);

const cbContext: Context = {
  systemPrompt: "sys",
  messages: [{ role: "user", content: "Chunk boundary test", timestamp: Date.now() }],
};
const cbEvents: any[] = [];
for await (const ev of mantisProvider.streamSimple(model, cbContext, { apiKey: "test-key" })) {
  cbEvents.push(ev);
}
const cbTextEnd = cbEvents.find((e) => e.type === "text_end");
if (cbTextEnd?.content !== "chunk boundary test") {
  console.error("FAIL: Chunk boundary reassembly failed, got:", cbTextEnd);
  process.exit(1);
}
console.log("✓ Chunk boundaries across stream packets handled correctly");

// 6. Test malformed NDJSON line rejection
(globalThis as any).fetch = async () => new Response(
  new ReadableStream({
    start(controller) {
      controller.enqueue(encoder.encode("{bad-json-line\n"));
      controller.close();
    },
  }),
  { status: 200, headers: { "Content-Type": "application/x-ndjson" } },
);

const malformedContext: Context = {
  systemPrompt: "sys",
  messages: [{ role: "user", content: "Malformed line test", timestamp: Date.now() }],
};
const malformedEvents: any[] = [];
for await (const ev of mantisProvider.streamSimple(model, malformedContext, { apiKey: "test-key" })) {
  malformedEvents.push(ev);
}
const malformedErr = malformedEvents.find((e) => e.type === "error");
if (!malformedErr?.error?.errorMessage?.includes("Malformed NDJSON event from mantis backend")) {
  console.error("FAIL: Malformed line was not rejected, got:", malformedErr);
  process.exit(1);
}
console.log("✓ Malformed NDJSON line rejected with explicit error");

// 7. Test truncated stream rejection (EOF without result)
(globalThis as any).fetch = async () => new Response(
  new ReadableStream({
    start(controller) {
      controller.enqueue(encoder.encode(JSON.stringify({
        type: "step-end",
        turn: 0,
        role: "Worker",
        agent_id: 1,
        prompt: "p",
        reply: "r",
      }) + "\n"));
      controller.close();
    },
  }),
  { status: 200, headers: { "Content-Type": "application/x-ndjson" } },
);

const truncContext: Context = {
  systemPrompt: "sys",
  messages: [{ role: "user", content: "Truncated stream test", timestamp: Date.now() }],
};
const truncEvents: any[] = [];
for await (const ev of mantisProvider.streamSimple(model, truncContext, { apiKey: "test-key" })) {
  truncEvents.push(ev);
}
const truncErr = truncEvents.find((e) => e.type === "error");
if (!truncErr?.error?.errorMessage?.includes("Stream ended without terminal result event")) {
  console.error("FAIL: Truncated stream was not rejected, got:", truncErr);
  process.exit(1);
}
console.log("✓ Truncated stream (EOF without result) rejected with error");

// 8. Test backend error event propagation
(globalThis as any).fetch = async () => new Response(
  new ReadableStream({
    start(controller) {
      controller.enqueue(encoder.encode(JSON.stringify({
        type: "error",
        error: "Backend worker process crash",
      }) + "\n"));
      controller.close();
    },
  }),
  { status: 200, headers: { "Content-Type": "application/x-ndjson" } },
);

const errContext: Context = {
  systemPrompt: "sys",
  messages: [{ role: "user", content: "Backend error test", timestamp: Date.now() }],
};
const errEvents: any[] = [];
for await (const ev of mantisProvider.streamSimple(model, errContext, { apiKey: "test-key" })) {
  errEvents.push(ev);
}
const backendErr = errEvents.find((e) => e.type === "error");
if (!backendErr?.error?.errorMessage?.includes("Backend worker process crash")) {
  console.error("FAIL: Backend error event was not propagated, got:", backendErr);
  process.exit(1);
}
console.log("✓ Backend error event handled and propagated");

// 9. Test duplicate result event rejection
(globalThis as any).fetch = async () => new Response(
  new ReadableStream({
    start(controller) {
      controller.enqueue(encoder.encode(JSON.stringify({
        type: "result",
        text: "res 1",
        trace: "",
        coordinator: "trinity",
        mantis_steps: [],
      }) + "\n"));
      controller.enqueue(encoder.encode(JSON.stringify({
        type: "result",
        text: "res 2",
        trace: "",
        coordinator: "trinity",
        mantis_steps: [],
      }) + "\n"));
      controller.close();
    },
  }),
  { status: 200, headers: { "Content-Type": "application/x-ndjson" } },
);

const dupContext: Context = {
  systemPrompt: "sys",
  messages: [{ role: "user", content: "Duplicate result test", timestamp: Date.now() }],
};
const dupEvents: any[] = [];
for await (const ev of mantisProvider.streamSimple(model, dupContext, { apiKey: "test-key" })) {
  dupEvents.push(ev);
}
const dupErr = dupEvents.find((e) => e.type === "error");
if (!dupErr?.error?.errorMessage?.includes("Duplicate terminal result event received")) {
  console.error("FAIL: Duplicate result event was not rejected, got:", dupErr);
  process.exit(1);
}
console.log("✓ Duplicate terminal result event rejected");

// 10. Test cancellation handling via AbortSignal
const abortController = new AbortController();
abortController.abort();

const cancelContext: Context = {
  systemPrompt: "sys",
  messages: [{ role: "user", content: "Cancellation test", timestamp: Date.now() }],
};
const cancelEvents: any[] = [];
for await (const ev of mantisProvider.streamSimple(model, cancelContext, { apiKey: "test-key", signal: abortController.signal })) {
  cancelEvents.push(ev);
}
const cancelErr = cancelEvents.find((e) => e.type === "error");
if (cancelErr?.reason !== "aborted") {
  console.error("FAIL: Cancellation test failed, got:", cancelErr);
  process.exit(1);
}
console.log("✓ Cancellation handled with aborted reason");

// Auto mode must abort while waiting for routing and never start orchestration.
const fetchBeforeAutoCancellation = (globalThis as any).fetch;
let autoRouteRequests = 0;
(globalThis as any).fetch = async (_url: string, init?: any) => {
  autoRouteRequests++;
  return new Promise((_resolve, reject) => {
    const rejectAbort = () => {
      const error = new Error("aborted");
      error.name = "AbortError";
      reject(error);
    };
    if (init?.signal?.aborted) rejectAbort();
    else init?.signal?.addEventListener("abort", rejectAbort, { once: true });
  });
};
const autoAbortController = new AbortController();
const autoCancelEvents: any[] = [];
const autoCancelRun = (async () => {
  for await (const ev of mantisProvider.streamSimple(
    { ...model, id: "auto" },
    { messages: [{ role: "user", content: "Auto cancellation test", timestamp: Date.now() }] },
    { apiKey: "test-key", signal: autoAbortController.signal },
  )) autoCancelEvents.push(ev);
})();
await new Promise((resolve) => setTimeout(resolve, 0));
autoAbortController.abort();
await autoCancelRun;
if (autoRouteRequests !== 1 || autoCancelEvents.find((e) => e.type === "error")?.reason !== "aborted") {
  console.error("FAIL: Auto cancellation started orchestration after routing abort", autoRouteRequests, autoCancelEvents);
  process.exit(1);
}
(globalThis as any).fetch = fetchBeforeAutoCancellation;
console.log("✓ Auto cancellation stops during routing without orchestration");

// 11. Test MANTIS_CONTEXT_WINDOW parsing, override, fallback, and boundary overflow
delete process.env.MANTIS_CONTEXT_WINDOW;
delete process.env.FUGU_CONTEXT_WINDOW;
if (getMantisContextWindow() !== 256000) {
  console.error("FAIL: default MANTIS_CONTEXT_WINDOW should be 256000, got:", getMantisContextWindow());
  process.exit(1);
}
console.log("✓ Default MANTIS_CONTEXT_WINDOW returns 256000");

process.env.MANTIS_CONTEXT_WINDOW = "128000";
if (getMantisContextWindow() !== 128000) {
  console.error("FAIL: MANTIS_CONTEXT_WINDOW env override failed, got:", getMantisContextWindow());
  process.exit(1);
}
console.log("✓ MANTIS_CONTEXT_WINDOW environment override works");

for (const invalid of ["invalid", "-100", "0", "128k", "  ", "100.5"]) {
  process.env.MANTIS_CONTEXT_WINDOW = invalid;
  if (getMantisContextWindow() !== 256000) {
    console.error(`FAIL: invalid value '${invalid}' should fall back to 256000, got:`, getMantisContextWindow());
    process.exit(1);
  }
}
delete process.env.MANTIS_CONTEXT_WINDOW;
console.log("✓ Invalid MANTIS_CONTEXT_WINDOW values fall back to default 256000");

// Test overflow boundary error for context window calibration knob
const customCalibratedModel = { ...model, contextWindow: 10000, maxTokens: 2000 };
// contextWindow - maxTokens = 8000 tokens.
// 8005 tokens (> 8000) -> overflow error
const overflowBoundaryContext: Context = {
  systemPrompt: "sys",
  messages: [{ role: "user", content: "x".repeat(32020), timestamp: Date.now() }],
};

const overflowBoundaryEvents: any[] = [];
for await (const ev of mantisProvider.streamSimple(customCalibratedModel, overflowBoundaryContext, { apiKey: "test-key" })) {
  overflowBoundaryEvents.push(ev);
}
const boundaryErr = overflowBoundaryEvents.find((e) => e.type === "error");
if (!boundaryErr?.error?.errorMessage?.includes("exceeds the context window of this model")) {
  console.error("FAIL: overflow boundary error failed, got:", boundaryErr);
  process.exit(1);
}
console.log("✓ MANTIS_CONTEXT_WINDOW overflow boundary error enforced");

// 12. Test warm-up endpoint integration for Trinity and Conductor readiness
warmed.clear();
const warmNotifies: string[] = [];
const warmCtx = {
  ui: {
    notify: (msg: string) => warmNotifies.push(msg),
  },
};

await warm("trinity", warmCtx as any);
if (!warmed.has("trinity") || !warmNotifies.some((n) => n.includes("trinity ready"))) {
  console.error("FAIL: warm('trinity') failed or missing readiness notification, got:", warmNotifies);
  process.exit(1);
}
console.log("✓ Warm-up endpoint works for Trinity readiness");

await warm("conductor", warmCtx as any);
if (!warmed.has("conductor") || !warmNotifies.some((n) => n.includes("conductor ready"))) {
  console.error("FAIL: warm('conductor') failed or missing readiness notification, got:", warmNotifies);
  process.exit(1);
}
console.log("✓ Warm-up endpoint works for Conductor readiness");

// 13. Test idempotent second warm
const notifyCountBeforeSecondWarm = warmNotifies.length;
await warm("trinity", warmCtx as any);
if (warmNotifies.length !== notifyCountBeforeSecondWarm) {
  console.error("FAIL: second warm('trinity') emitted duplicate notification instead of returning early");
  process.exit(1);
}
console.log("✓ Idempotent second warm returns early without duplicate request/notification");

// 14. Test warm-up auth failure
warmed.clear();
const originalFetch = (globalThis as any).fetch;
(globalThis as any).fetch = async () => new Response(JSON.stringify({ error: "unauthorized" }), { status: 401 });
const authErrNotifies: string[] = [];
await warm("trinity", { ui: { notify: (m: string) => authErrNotifies.push(m) } } as any);
if (warmed.has("trinity") || !authErrNotifies.some((n) => n.includes("warm failed") && n.includes("unauthorized"))) {
  console.error("FAIL: warm auth failure test failed, got:", authErrNotifies);
  process.exit(1);
}
console.log("✓ Warm-up handles backend auth failure gracefully without marking mode ready");

// 15. Test warm-up timeout via AbortController
warmed.clear();
(globalThis as any).fetch = async (_url: string, init?: any) => {
  return new Promise((_resolve, reject) => {
    init?.signal?.addEventListener("abort", () => {
      const err = new Error("The operation was aborted");
      err.name = "AbortError";
      reject(err);
    });
  });
};
const timeoutNotifies: string[] = [];
await warm("trinity", { ui: { notify: (m: string) => timeoutNotifies.push(m) } } as any, 50);
if (warmed.has("trinity") || !timeoutNotifies.some((n) => n.includes("warm failed") && n.includes("timed out"))) {
  console.error("FAIL: warm timeout test failed, got:", timeoutNotifies);
  process.exit(1);
}
(globalThis as any).fetch = originalFetch;
console.log("✓ Warm-up timeout via AbortController cancels fetch and notifies user");

// 16. Test distinct error messages for timeout vs user abort vs backend disconnect vs provider failure
(globalThis as any).fetch = async (_url: string, init?: any) => {
  return new Promise((_resolve, reject) => {
    init?.signal?.addEventListener("abort", () => {
      const err = new Error("The operation was aborted");
      err.name = "AbortError";
      reject(err);
    });
  });
};

// 16a. Timeout error message
let timeoutErrMsg = "";
try {
  for await (const _ of streamOrchestrate("trinity", [{ role: "user", content: "test" }], undefined, 0.05)) {}
} catch (e: any) {
  timeoutErrMsg = e.message;
}
if (!timeoutErrMsg.includes("mantis request timed out after 0.05 seconds")) {
  console.error("FAIL: timeout error message mismatch, got:", timeoutErrMsg);
  process.exit(1);
}
console.log("✓ Distinct error message for request timeout verified");

// 16b. User abort error message
const userAbortController = new AbortController();
const userAbortEvents: any[] = [];
const userAbortPromise = (async () => {
  for await (const ev of mantisProvider.streamSimple(model, cbContext, { apiKey: "test-key", signal: userAbortController.signal })) {
    userAbortEvents.push(ev);
  }
})();
userAbortController.abort();
await userAbortPromise;
const userAbortErr = userAbortEvents.find((e) => e.type === "error");
if (!userAbortErr?.error?.errorMessage?.includes("mantis request aborted by user")) {
  console.error("FAIL: user abort error message mismatch, got:", userAbortErr);
  process.exit(1);
}
console.log("✓ Distinct error message for user abort verified");

// 16c. Backend disconnect error message
(globalThis as any).fetch = async () => {
  throw new TypeError("fetch failed");
};
const disconnectEvents: any[] = [];
for await (const ev of mantisProvider.streamSimple(model, cbContext, { apiKey: "test-key" })) {
  disconnectEvents.push(ev);
}
const disconnectErr = disconnectEvents.find((e) => e.type === "error");
if (!disconnectErr?.error?.errorMessage?.includes("mantis backend disconnected: fetch failed")) {
  console.error("FAIL: backend disconnect error message mismatch, got:", disconnectErr);
  process.exit(1);
}
console.log("✓ Distinct error message for backend disconnect verified");

// 16d. Provider failure error message
(globalThis as any).fetch = async () => new Response("Internal Server Error", { status: 500 });
const providerFailEvents: any[] = [];
for await (const ev of mantisProvider.streamSimple(model, cbContext, { apiKey: "test-key" })) {
  providerFailEvents.push(ev);
}
const providerFailErr = providerFailEvents.find((e) => e.type === "error");
if (!providerFailErr?.error?.errorMessage?.includes("mantis provider failure (HTTP 500)")) {
  console.error("FAIL: provider failure error message mismatch, got:", providerFailErr);
  process.exit(1);
}
console.log("✓ Distinct error message for provider failure verified");

// 17. Test listener and timer cleanup
(globalThis as any).fetch = originalFetch;
let removeListenerCalled = false;
const spySignal = {
  aborted: false,
  addEventListener: (_event: string, _fn: any, _opts: any) => {},
  removeEventListener: (event: string, _fn: any) => {
    if (event === "abort") removeListenerCalled = true;
  },
};
for await (const _ of streamOrchestrate("trinity", [{ role: "user", content: "test" }], spySignal as any, 300)) {}
if (!removeListenerCalled) {
  console.error("FAIL: removeEventListener('abort') was not called in finally block");
  process.exit(1);
}
console.log("✓ AbortSignal listener cleanup in finally block verified");

console.log("\nALL STREAMING TESTS PASSED SUCCESSFULLY!");
