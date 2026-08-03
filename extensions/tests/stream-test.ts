/**
 * End-to-end streaming test for the mantis pi extension.
 *
 * Mocks the Mantis backend with an NDJSON stream, then exercises the
 * provider's streamSimple implementation. Verifies that worker turns are
 * emitted as native tool calls and that the final answer is returned on
 * the follow-up provider call after tool execution.
 */

import { TextEncoder } from "node:util";
import mantisExtension from "../mantis.ts";
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

const mantisStepTool = tools.get("mantis_step");
if (!mantisStepTool) {
  console.error("FAIL: mantis_step tool not registered");
  process.exit(1);
}

console.log("✓ /mantis command, mantis provider, and mantis_step tool registered");

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

await mantisCmd.handler("trinity", commandCtx);

const encoder = new TextEncoder();
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
  if (url.endsWith("/models")) {
    return new Response(JSON.stringify({ data: [] }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  }
  if (init && typeof init.body === "string" && init.body.includes('"stream":true')) {
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

const model = { id: "trinity", provider: "mantis", api: "mantis", name: "mantis: trinity" };

const userMessage: Message = {
  role: "user",
  content: "Implement a small helper that reverses a string",
  timestamp: Date.now(),
};

const context1: Context = { messages: [userMessage] };
const stream1 = mantisProvider.streamSimple(model, context1, { apiKey: "test-key" });
const events1: any[] = [];
for await (const ev of stream1) events1.push(ev);

const start1 = events1.find((e) => e.type === "start");
if (!start1) {
  console.error("FAIL: first stream did not emit start");
  process.exit(1);
}

const toolcallEnd = events1.find((e) => e.type === "toolcall_end");
if (!toolcallEnd) {
  console.error("FAIL: first stream did not emit a toolcall_end event");
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

console.log("✓ mantis_step execute returned the worker reply");

// Second provider call: pi sends the tool result back and expects the final answer.
const toolResultMessage: Message = {
  role: "toolResult",
  toolCallId: toolCall.id,
  toolName: "mantis_step",
  content: [{ type: "text", text: expectedReply }],
  isError: false,
  timestamp: Date.now(),
};

const context2: Context = { messages: [userMessage, toolResultMessage] };
const stream2 = mantisProvider.streamSimple(model, context2, { apiKey: "test-key" });
const events2: any[] = [];
for await (const ev of stream2) events2.push(ev);

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

console.log("✓ Second stream returned final answer after tool execution");
console.log("\nALL STREAMING TESTS PASSED SUCCESSFULLY!");
