/**
 * End-to-end streaming test for the mantis pi extension.
 *
 * Mocks the Mantis backend with an NDJSON stream and verifies that the
 * extension emits a mantis-prompt message before the corresponding
 * mantis-step message and a mantis-result at the end.
 */

import { TextEncoder } from "node:util";
import mantisExtension from "../mantis.ts";

process.env.MANTIS_API_KEY = "test-key";

const sentMessages: any[] = [];
const notifications: { message: string; type?: string }[] = [];
const statuses: Map<string, string | undefined> = new Map();
let workingMessage: string | undefined;
let workingVisible = false;

const commands = new Map<string, any>();
const handlers = new Map<string, any[]>();

const pi = {
  registerProvider: () => {},
  registerTool: () => {},
  registerMessageRenderer: () => {},
  registerCommand: (name: string, config: any) => {
    commands.set(name, config);
  },
  on: (event: string, handler: any) => {
    if (!handlers.has(event)) handlers.set(event, []);
    handlers.get(event)!.push(handler);
  },
  sendMessage: (msg: any, _options?: any) => {
    sentMessages.push(msg);
  },
  setModel: async () => true,
  getFlag: () => undefined,
  unregisterProvider: () => {},
  events: { on: () => {}, off: () => {}, emit: () => {} },
};

mantisExtension(pi as any);

const commandCtx = {
  cwd: process.cwd(),
  hasUI: true,
  ui: {
    notify: (message: string, type?: string) => notifications.push({ message, type }),
    setStatus: (key: string, text: string | undefined) => statuses.set(key, text),
    setWorkingMessage: (message?: string) => {
      workingMessage = message;
    },
    setWorkingVisible: (visible: boolean) => {
      workingVisible = visible;
    },
  },
  modelRegistry: {
    find: (provider: string, id: string) => ({ provider, id, name: `${provider}/${id}` }),
  },
  sessionManager: { getBranch: () => [] },
  signal: undefined,
  isIdle: () => true,
  abort: () => {},
  hasPendingMessages: () => false,
  shutdown: () => {},
  getContextUsage: () => undefined,
  compact: () => {},
  getSystemPrompt: () => "",
  waitForIdle: async () => {},
  newSession: async () => ({ cancelled: false }),
  reload: async () => {},
} as any;

// 1. Set mode to trinity
const mantisCmd = commands.get("mantis");
if (!mantisCmd) {
  console.error("FAIL: /mantis command not registered");
  process.exit(1);
}
await mantisCmd.handler("trinity", commandCtx);
if (statuses.get("mantis") !== "mantis:trinity") {
  console.error("FAIL: status not set to mantis:trinity");
  process.exit(1);
}
console.log("✓ /mantis trinity set status to", statuses.get("mantis"));

// 2. Mock fetch to return an NDJSON stream
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

const mockFetch = async (_url: any, init?: any) => {
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
(globalThis as any).fetch = mockFetch;

// 3. Execute the input handler
const inputHandlers = handlers.get("input") || [];
if (inputHandlers.length === 0) {
  console.error("FAIL: input handler not registered");
  process.exit(1);
}

const inputCtx = {
  ...commandCtx,
  sessionManager: {
    getBranch: () => [
      {
        message: {
          role: "user",
          content: "Implement a small helper that reverses a string",
        },
      },
    ],
  },
};

const result = await inputHandlers[0](
  { type: "input", text: "Implement a small helper that reverses a string", source: "interactive" },
  inputCtx,
);

if (result?.action !== "handled") {
  console.error("FAIL: input handler did not return handled, got:", result);
  process.exit(1);
}

// 4. Verify emitted messages
const prompts = sentMessages.filter((m) => m.customType === "mantis-prompt");
const steps = sentMessages.filter((m) => m.customType === "mantis-step");
const results = sentMessages.filter((m) => m.customType === "mantis-result");

if (prompts.length !== 1) {
  console.error("FAIL: expected 1 mantis-prompt, got", prompts.length);
  process.exit(1);
}
if (steps.length !== 1) {
  console.error("FAIL: expected 1 mantis-step, got", steps.length);
  process.exit(1);
}
if (results.length !== 1) {
  console.error("FAIL: expected 1 mantis-result, got", results.length);
  process.exit(1);
}

if (prompts[0].content !== "Implement a small helper that reverses a string") {
  console.error("FAIL: prompt content mismatch");
  process.exit(1);
}
if (steps[0].details.model_name !== "claude-opus-5-medium") {
  console.error("FAIL: step model_name mismatch");
  process.exit(1);
}
if (results[0].content !== "def reverse(s):\n    return s[::-1]") {
  console.error("FAIL: final result mismatch");
  process.exit(1);
}

console.log("✓ Stream delivered prompt before step and result at the end");
console.log("\nALL STREAMING TESTS PASSED SUCCESSFULLY!");
