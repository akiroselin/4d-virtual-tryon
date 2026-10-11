/**
 * Regression test for the "output disappears after /clear" bug.
 *
 * Reproduces the App's <Static> usage in a real Ink render (debug mode →
 * deterministic, no throttling) and verifies that after the history SHRINKS
 * (a /clear), newly-appended items still reach the static output.
 *
 * The original bug was caused by remounting <Static> via a changing `key`:
 * Ink's reconciler nulls `rootNode.staticNode` when the old internal_static
 * node is removed, so appends after the remount stopped printing. The fix is
 * to never remount and let <Static>'s own length-change layout effect reset
 * its cursor. This test guards that fix.
 */
import React from "react";
import { PassThrough } from "node:stream";
import { Box, Static, Text, render } from "ink";

let failures = 0;
function assert(cond: boolean, label: string): void {
  console.log(`${cond ? "  \u2713" : "  \u2717"} ${label}`);
  if (!cond) failures++;
}

interface Item {
  key: string;
  text: string;
}

// External controller so the test can drive state without a TTY.
let setItemsExternal: ((items: Item[]) => void) | null = null;

function Harness(): React.ReactNode {
  const [items, setItems] = React.useState<Item[]>([{ key: "welcome", text: "WELCOME" }]);
  setItemsExternal = setItems;
  return (
    <Box flexDirection="column">
      <Static items={items}>
        {(item) => (
          <Box key={item.key}>
            <Text>{item.text}</Text>
          </Box>
        )}
      </Static>
    </Box>
  );
}

async function tick(): Promise<void> {
  // Let React flush + Ink's layout effects + a render cycle run.
  await new Promise((r) => setTimeout(r, 30));
}

async function main(): Promise<void> {
  const fakeStdout = new PassThrough() as unknown as NodeJS.WriteStream;
  let captured = "";
  (fakeStdout as unknown as PassThrough).on("data", (chunk) => {
    captured += chunk.toString();
  });
  // Ink reads columns/rows off the stream.
  (fakeStdout as unknown as { columns: number }).columns = 80;
  (fakeStdout as unknown as { rows: number }).rows = 24;

  const instance = render(<Harness />, {
    stdout: fakeStdout,
    debug: true, // deterministic: writes fullStaticOutput + output each frame
    exitOnCtrlC: false,
  });

  await tick();
  assert(captured.includes("WELCOME"), "welcome banner printed initially");

  // Grow: simulate a normal conversation.
  setItemsExternal?.([
    { key: "welcome", text: "WELCOME" },
    { key: "u0", text: "USER_FIRST" },
    { key: "a1", text: "ASSISTANT_FIRST" },
  ]);
  await tick();
  assert(captured.includes("USER_FIRST"), "user message printed");
  assert(captured.includes("ASSISTANT_FIRST"), "assistant reply printed");

  // /clear → history shrinks back to just the banner.
  setItemsExternal?.([{ key: "welcome", text: "WELCOME" }]);
  await tick();

  const beforeClearLen = captured.length;

  // New conversation after clear — THIS is what used to vanish.
  setItemsExternal?.([
    { key: "welcome", text: "WELCOME" },
    { key: "u0", text: "USER_AFTER_CLEAR" },
  ]);
  await tick();
  setItemsExternal?.([
    { key: "welcome", text: "WELCOME" },
    { key: "u0", text: "USER_AFTER_CLEAR" },
    { key: "a1", text: "ASSISTANT_AFTER_CLEAR" },
  ]);
  await tick();

  const afterClearOutput = captured.slice(beforeClearLen);
  assert(afterClearOutput.includes("USER_AFTER_CLEAR"), "user message AFTER /clear printed");
  assert(afterClearOutput.includes("ASSISTANT_AFTER_CLEAR"), "assistant reply AFTER /clear printed");

  instance.unmount();
  instance.cleanup();

  console.log(failures === 0 ? "\nALL PASS" : `\n${failures} FAILED`);
  process.exit(failures === 0 ? 0 : 1);
}

void main();
