/**
 * UI notice bus — a tiny pub/sub channel for transient, non-fatal notices
 * (a flaky MCP server, a malformed config) that originate OUTSIDE React.
 *
 * The problem it solves: helpers like `logWarn` used to `console.error`
 * straight to stderr. While the Ink REPL is running that text interleaves
 * with the rendered frame and corrupts the carefully laid-out UI. Instead,
 * once the REPL is live we route those messages here; the App subscribes and
 * renders them as quiet, aligned dim lines above the input box.
 *
 * When no UI is active (headless `--dump-system-prompt`, piped output, or
 * before the first frame) callers fall back to stderr — see `logWarn`.
 */

export type UiNoticeTone = "info" | "warn" | "error";

export interface UiNotice {
  id: number;
  tone: UiNoticeTone;
  text: string;
}

let nextId = 1;
const notices: UiNotice[] = [];
const listeners = new Set<() => void>();

// Whether an interactive Ink frame is on screen. Toggled by the entrypoint
// right before/after render() so `logWarn` knows where to send output.
let uiActive = false;

// Keep the live frame bounded — we only ever show the most recent few.
const MAX_NOTICES = 8;

function emit(): void {
  for (const l of listeners) l();
}

export function setUiActive(active: boolean): void {
  uiActive = active;
}

export function isUiActive(): boolean {
  return uiActive;
}

export function pushUiNotice(tone: UiNoticeTone, text: string): void {
  notices.push({ id: nextId++, tone, text });
  if (notices.length > MAX_NOTICES) {
    notices.splice(0, notices.length - MAX_NOTICES);
  }
  emit();
}

export function getUiNotices(): readonly UiNotice[] {
  return notices;
}

export function subscribeUiNotices(listener: () => void): () => void {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

export function clearUiNotices(): void {
  if (notices.length === 0) return;
  notices.length = 0;
  emit();
}
