/**
 * Smoke for the Ctrl+O transcript mouse-wheel capture.
 *
 * Two layers:
 *   1. wheelScrollDelta() unit cases — SGR button decoding (up/down/horizontal/
 *      modifier variants / non-wheel input).
 *   2. End-to-end through Ink: a PassThrough fake stdin injects real SGR wheel
 *      sequences so we exercise the actual useInput → setScroll path, and a fake
 *      stdout captures the DEC 1000/1006 enable escapes on open and the matching
 *      disable escapes on close/unmount.
 *
 * Run: npm run test:transcript-wheel
 */
import React from "react";
import { PassThrough } from "node:stream";
import { Box, Text, render } from "ink";
import { useTranscript, wheelScrollDelta } from "../ui/hooks/useTranscript.js";

const MOUSE_ON = "\u001B[?1000h\u001B[?1006h";
const MOUSE_OFF = "\u001B[?1006l\u001B[?1000l";

// Raw stdin sequences (with ESC) — what the terminal sends and Ink parses.
const WHEEL_UP = "\u001B[<64;5;5M";
const WHEEL_DOWN = "\u001B[<65;5;5M";
const WHEEL_LEFT = "\u001B[<66;5;5M";

// ESC-stripped form — what Ink hands to useInput, i.e. what wheelScrollDelta()
// receives. The unit cases below feed this form directly.
const stripEsc = (s: string) => s.replace(/^\u001B/, "");

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

let failures = 0;
function check(label: string, ok: boolean): void {
  process.stdout.write(`${ok ? "\u001b[32m[pass]\u001b[0m" : "\u001b[31m[FAIL]\u001b[0m"} ${label}\n`);
  if (!ok) failures++;
}

function unitTests(): void {
  process.stdout.write("=== wheelScrollDelta unit cases (ESC-stripped form) ===\n");
  check("wheel up → -3", wheelScrollDelta(stripEsc(WHEEL_UP)) === -3);
  check("wheel down → +3", wheelScrollDelta(stripEsc(WHEEL_DOWN)) === 3);
  check("horizontal wheel → 0", wheelScrollDelta(stripEsc(WHEEL_LEFT)) === 0);
  check("shift+wheel up → -3 (modifier ignored)", wheelScrollDelta("[<68;5;5M") === -3);
  check("custom step honored", wheelScrollDelta(stripEsc(WHEEL_DOWN), 5) === 5);
  check("plain key ('j') → 0", wheelScrollDelta("j") === 0);
  check("arrow-ish CSI → 0", wheelScrollDelta("[A") === 0);
  check("empty → 0", wheelScrollDelta("") === 0);
}

// Renders the current scroll so the harness can read it back from stdout.
function Harness({ open }: { open: boolean }): React.ReactNode {
  const lines = React.useMemo(() => Array.from({ length: 100 }, (_, i) => `line ${i}`), []);
  const { scroll } = useTranscript({ open, lines, viewportHeight: 10, onClose: () => {} });
  return (
    <Box>
      <Text>{`SCROLL=${scroll}`}</Text>
    </Box>
  );
}

function lastScroll(captured: string): number | null {
  const matches = [...captured.matchAll(/SCROLL=(\d+)/g)];
  const last = matches.at(-1);
  return last ? Number(last[1]) : null;
}

async function e2eTests(): Promise<void> {
  process.stdout.write("\n=== end-to-end through Ink (fake stdin/stdout) ===\n");

  let captured = "";
  const fakeStdout = new PassThrough();
  fakeStdout.on("data", (c) => (captured += c.toString()));
  (fakeStdout as unknown as { columns: number }).columns = 80;
  (fakeStdout as unknown as { rows: number }).rows = 24;
  (fakeStdout as unknown as { isTTY: boolean }).isTTY = true;

  const fakeStdin = new PassThrough();
  (fakeStdin as unknown as { isTTY: boolean }).isTTY = true;
  (fakeStdin as unknown as { setRawMode: () => void }).setRawMode = () => {};
  (fakeStdin as unknown as { ref: () => void }).ref = () => {};
  (fakeStdin as unknown as { unref: () => void }).unref = () => {};

  const instance = render(<Harness open={true} />, {
    stdin: fakeStdin as unknown as NodeJS.ReadStream,
    stdout: fakeStdout as unknown as NodeJS.WriteStream,
    debug: true,
    exitOnCtrlC: false,
  });

  await sleep(60);
  check("enabled mouse tracking on open", captured.includes(MOUSE_ON));
  // Opens at the bottom: max(0, 100 - 10) = 90.
  check("opens at bottom (scroll=90)", lastScroll(captured) === 90);

  fakeStdin.write(WHEEL_UP);
  await sleep(40);
  check("wheel up scrolls toward top (90 → 87)", lastScroll(captured) === 87);

  fakeStdin.write(WHEEL_UP);
  await sleep(40);
  check("wheel up again (87 → 84)", lastScroll(captured) === 84);

  fakeStdin.write(WHEEL_DOWN);
  await sleep(40);
  check("wheel down scrolls toward bottom (84 → 87)", lastScroll(captured) === 87);

  fakeStdin.write(WHEEL_LEFT);
  await sleep(40);
  check("horizontal wheel is ignored (stays 87)", lastScroll(captured) === 87);

  const beforeClose = captured.length;
  instance.rerender(<Harness open={false} />);
  await sleep(40);
  check("disabled mouse tracking on close", captured.slice(beforeClose).includes(MOUSE_OFF));

  instance.unmount();
  instance.cleanup();
}

async function main(): Promise<void> {
  unitTests();
  await e2eTests();
  process.stdout.write(
    failures === 0
      ? "\n\u001b[32mAll transcript-wheel checks passed.\u001b[0m\n"
      : `\n\u001b[31m${failures} check(s) failed.\u001b[0m\n`,
  );
  process.exit(failures === 0 ? 0 : 1);
}

void main();
