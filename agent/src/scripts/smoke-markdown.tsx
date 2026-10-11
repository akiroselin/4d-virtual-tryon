/**
 * Stage 24 markdown smoke test — verifies markdownToAnsi + highlight +
 * stable-prefix split work end to end (ANSI is hard to assert exactly, so
 * we check structural properties + no crashes).
 */
import chalk from "chalk";
// Force a color level so the ANSI assertions hold even when stdout is a
// pipe (CI / `npm run`). The markdown module shares this chalk singleton.
chalk.level = 3;

import { markdownToAnsi, hasMarkdownSyntax } from "../ui/markdown/markdownToAnsi.js";
import { highlightCode } from "../ui/markdown/highlight.js";
import { splitStablePrefix } from "../ui/markdown/Markdown.js";

let failures = 0;
function assert(cond: boolean, label: string): void {
  console.log(`${cond ? "  \u2713" : "  \u2717"} ${label}`);
  if (!cond) failures++;
}

const ESC = "\u001B";

console.log("[1] fast path");
assert(hasMarkdownSyntax("# Heading") === true, "detects heading");
assert(hasMarkdownSyntax("plain sentence with no markup") === false, "plain text → no markdown");
assert(markdownToAnsi("just plain text") === "just plain text", "plain text returned unchanged");

console.log("\n[2] inline + block styling produces ANSI");
const bold = markdownToAnsi("hello **world**");
assert(bold.includes(ESC), "bold emits ANSI escapes");
assert(bold.includes("world"), "bold keeps the text");

const heading = markdownToAnsi("## Title here");
assert(heading.includes("Title here"), "heading text preserved");
assert(heading.includes(ESC), "heading emits ANSI");

console.log("\n[3] code block highlighting");
const code = markdownToAnsi("```ts\nconst x: number = 1;\n```");
assert(code.includes("const"), "code content preserved");
const hl = highlightCode("const x = 1;", "js");
assert(typeof hl === "string" && hl.includes("const"), "highlightCode returns string");
assert(highlightCode("???", "no-such-lang").length > 0, "unknown language degrades gracefully");

console.log("\n[4] list + link");
const list = markdownToAnsi("- one\n- two\n- three");
assert(list.includes("one") && list.includes("three"), "list items rendered");
assert(list.includes("\u2022"), "bullet glyph present");
const link = markdownToAnsi("[Anthropic](https://example.com)");
assert(link.includes("Anthropic"), "link label rendered");

console.log("\n[5] stable-prefix split");
const open = splitStablePrefix("done para\n\nstart of ```js\ncode");
assert(open.tail.startsWith("```"), "open code fence becomes tail");
assert(open.stable.includes("done para"), "completed paragraph is stable");

const para = splitStablePrefix("first paragraph\n\nsecond inc");
assert(para.stable === "first paragraph", "stable ends at last paragraph break");
assert(para.tail === "second inc", "tail is the incomplete trailing block");

const none = splitStablePrefix("no breaks yet");
assert(none.stable === "" && none.tail === "no breaks yet", "nothing stable until a break");

console.log(failures === 0 ? "\nALL PASS" : `\n${failures} FAILED`);
process.exit(failures === 0 ? 0 : 1);
