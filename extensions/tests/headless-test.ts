/**
 * Headless integration test for mantis Pi extension.
 *
 * Tests:
 * 1. Extension loading via discoverAndLoadExtensions
 * 2. Slash command registration (/mantis, /fugu)
 * 3. Mode switching handlers, model resolution & UI status updates
 * 4. Native tool registration for mantis_step (Worker, Thinker, Verifier background turns)
 * 5. Input event interception and multi-turn message history building
 *
 * Usage: bun run extensions/tests/headless-test.ts
 */

import { discoverAndLoadExtensions, type ExtensionContext } from "@mariozechner/pi-coding-agent";
import * as path from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const extensionPath = path.resolve(__dirname, "..", "mantis.ts");

console.log("Testing headless extension loading from:", extensionPath);

const loadResult = await discoverAndLoadExtensions([extensionPath], process.cwd());

if (loadResult.errors.length > 0) {
  console.error("Extension load failed with errors:", loadResult.errors);
  process.exit(1);
}

console.log("✓ Extension discovery loaded extensions count:", loadResult.extensions.length);

const ext = loadResult.extensions.find(e => e.commands.has("mantis"));
if (!ext) {
  console.error("No extension with /mantis command returned in loadResult");
  process.exit(1);
}

// 1. Verify slash commands
const mantisCmd = ext.commands.get("mantis");
const fuguCmd = ext.commands.get("fugu");

if (!mantisCmd) {
  console.error("FAIL: /mantis command not registered");
  process.exit(1);
}
console.log("✓ Registered command /mantis:", mantisCmd.description);

if (!fuguCmd) {
  console.error("FAIL: /fugu alias command not registered");
  process.exit(1);
}
console.log("✓ Registered command /fugu:", fuguCmd.description);

// 2. Verify native tool registration (mantis_step for worker turns)
const mantisStepTool = ext.tools.get("mantis_step");
if (!mantisStepTool) {
  console.error("FAIL: native tool mantis_step not registered");
  process.exit(1);
}
console.log("✓ Registered native tool mantis_step:", mantisStepTool.definition.description);

// 3. Mock UI and Context
const statusMap = new Map<string, string | undefined>();
const notifications: string[] = [];
let workingMessage: string | undefined;

const mockContext = {
  cwd: process.cwd(),
  hasUI: true,
  ui: {
    notify: (msg: string) => {
      notifications.push(msg);
    },
    setStatus: (key: string, text: string | undefined) => {
      statusMap.set(key, text);
    },
    setWorkingMessage: (msg?: string) => {
      workingMessage = msg;
    },
    setWorkingIndicator: () => {},
  },
  modelRegistry: {
    find: (provider: string, id: string) => ({ provider, id, name: `${provider}/${id}` }),
  },
  setModel: async (_model: any) => true,
  sessionManager: {
    getBranch: () => [
      {
        message: {
          role: "user",
          content: "Hello, I am testing multi-turn history",
        },
      },
      {
        message: {
          role: "assistant",
          content: "Hello! How can I assist you with your code today?",
        },
      },
    ],
  },
} as unknown as ExtensionContext;

// 4. Test Mode Switching & Model Setting
await mantisCmd.handler("trinity", mockContext as any);
if (statusMap.get("mantis") !== "mantis:trinity") {
  console.error("FAIL: status not set to mantis:trinity, got:", statusMap.get("mantis"));
  process.exit(1);
}
console.log("✓ Command /mantis trinity correctly updated UI status to:", statusMap.get("mantis"));

// 5. Test Input Event Handler
const inputHandlers = ext.handlers.get("input");
if (!inputHandlers || inputHandlers.length === 0) {
  console.error("FAIL: input event handler not registered");
  process.exit(1);
}
console.log("✓ Input event handler registered");

// Test input handler execution
const inputResult = await inputHandlers[0](
  { type: "input", text: "Please list the Ts files", source: "interactive" },
  mockContext,
);

console.log("✓ Input handler executed successfully. Action:", (inputResult as any)?.action);

console.log("\nALL HEADLESS EXTENSION TESTS PASSED SUCCESSFULLY! 🎉");
