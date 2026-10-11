/**
 * Smoke for the Bash live-progress heartbeat.
 *
 * The bug: a running Bash card only re-rendered when an output chunk arrived,
 * so a silent/long command froze at "Running… (0s)". Source ticks once a second
 * (TaskOutput.startPolling → onProgress) regardless of output. This verifies our
 * store now does the same:
 *
 *   1. start a command that emits NO output → we still get ≥2 emits over ~2.5s
 *      (the per-second heartbeat).
 *   2. complete the command → the heartbeat stops (no further emits).
 *   3. a separate command's output still flows through (streaming intact).
 *
 * Run: npm run test:bash-heartbeat
 */
import {
  appendBashProgress,
  completeBashProgress,
  startBashProgress,
  subscribeBashProgress,
  clearAllBashProgress,
} from "../state/bashProgressStore.js";

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

let failures = 0;
function check(label: string, ok: boolean, detail = ""): void {
  process.stdout.write(
    `${ok ? "\u001b[32m[pass]\u001b[0m" : "\u001b[31m[FAIL]\u001b[0m"} ${label}${detail ? `  (${detail})` : ""}\n`,
  );
  if (!ok) failures++;
}

async function main(): Promise<void> {
  process.stdout.write("=== bash progress heartbeat ===\n");

  const counts = new Map<string, number>();
  const unsub = subscribeBashProgress((id, snap) => {
    if (snap === null) return;
    counts.set(id, (counts.get(id) ?? 0) + 1);
  });

  // 1. Silent command — no appendBashProgress calls at all.
  const SILENT = "silent-cmd";
  startBashProgress(SILENT, 300_000);
  await sleep(2500);
  const silentEmits = counts.get(SILENT) ?? 0;
  // 1 start emit + ~2 heartbeats over 2.5s.
  check("silent command keeps ticking (≥3 emits incl. start)", silentEmits >= 3, `${silentEmits} emits`);

  // 2. Complete → heartbeat must stop.
  completeBashProgress(SILENT);
  const afterComplete = counts.get(SILENT) ?? 0;
  await sleep(1500);
  const stillTicking = (counts.get(SILENT) ?? 0) - afterComplete;
  check("heartbeat stops after completion", stillTicking === 0, `${stillTicking} extra emits`);

  // 3. Streaming still works for a chatty command.
  const CHATTY = "chatty-cmd";
  startBashProgress(CHATTY, 300_000);
  const baseline = counts.get(CHATTY) ?? 0;
  appendBashProgress(CHATTY, "building module-1\n");
  await sleep(150); // clear the 100ms throttle window
  appendBashProgress(CHATTY, "building module-2\n");
  await sleep(150);
  const streamed = (counts.get(CHATTY) ?? 0) - baseline;
  check("output chunks still emit (streaming intact)", streamed >= 2, `${streamed} emits`);
  completeBashProgress(CHATTY);

  unsub();
  clearAllBashProgress();

  process.stdout.write(
    failures === 0
      ? "\n\u001b[32mAll bash-heartbeat checks passed.\u001b[0m\n"
      : `\n\u001b[31m${failures} check(s) failed.\u001b[0m\n`,
  );
  process.exit(failures === 0 ? 0 : 1);
}

void main();
