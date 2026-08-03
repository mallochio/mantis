/**
 * Headless integration test for mantis Pi extension.
 *
 * Tests:
 * 1. Extension loading via discoverAndLoadExtensions
 * 2. Slash command registration (/mantis only)
 * 3. Mode switching handlers, model resolution & UI status updates
 * 4. Native tool registration for mantis_step
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

const ext = loadResult.extensions.find(
  e => e.path === extensionPath || e.resolvedPath === extensionPath,
);
if (!ext) {
  console.error("No extension matching", extensionPath, "returned in loadResult");
  console.error("Loaded paths:", loadResult.extensions.map(e => e.path || e.resolvedPath));
  process.exit(1);
}

// 1. Verify slash command
const mantisCmd = ext.commands.get("mantis");
if (!mantisCmd) {
  console.error("FAIL: /mantis command not registered");
  process.exit(1);
}
console.log("✓ Registered command /mantis:", mantisCmd.description);

const fuguCmd = ext.commands.get("fugu");
if (fuguCmd) {
  console.error("FAIL: /fugu alias should not be registered");
  process.exit(1);
}
console.log("✓ /fugu alias not registered");

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
      console.log("Working message:", msg);
    },
    setWorkingIndicator: () => {},
    setWorkingVisible: (_visible: boolean) => {},
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

console.log("\nALL HEADLESS EXTENSION TESTS PASSED SUCCESSFULLY!");
