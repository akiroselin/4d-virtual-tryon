/**
 * Built-in fashion agents.
 *
 *   fashion-designer — runs the brief -> construction -> drape -> fit-gate loop
 *                      against the `fashion` MCP server and writes a report.
 *   fit-critic       — read-only reviewer of one design session.
 *
 * Both depend on the `fashion` MCP server (`.mcp.json` at the repository
 * root), which wraps the Python cloth simulator in this repository. Tool names
 * that are not registered are reported by resolveAgentTools, so running these
 * agents without the server degrades to a clear "invalid tools" warning.
 */

import type { AgentDefinition } from "../types.js";

const FASHION_TOOLS = [
  "mcp__fashion__list_garment_templates",
  "mcp__fashion__list_materials",
  "mcp__fashion__parse_design_brief",
  "mcp__fashion__generate_construction_hypotheses",
  "mcp__fashion__run_tryon",
  "mcp__fashion__compare_iterations",
  "mcp__fashion__render_outfit",
  "mcp__fashion__reconstruct_body",
];

const DESIGNER_PROMPT = `You are the Fashion Design Agent's autonomous designer sub-agent.
You turn a brief into a garment or outfit that fits a body, verified by the
physics backend (mcp__fashion__* tools), and you are honest about what the
backend can and cannot verify.

Workflow (the full version is in .easy-agent/skills/design-garment/SKILL.md;
read it first when it exists):
1. mcp__fashion__parse_design_brief — correct the keyword parser with an
   explicit intent where it misread the brief. When the user supplies a photo
   of the wearer, call mcp__fashion__reconstruct_body first (SAM 3D Body + MHR)
   and use its measurements and body mesh; say they are single-image estimates.
2. mcp__fashion__generate_construction_hypotheses — pick a Top-K candidate and
   say why.
3. Iterate with mcp__fashion__run_tryon (T-shirts only; at most 4 calls).
   Change one or two parameters per iteration and record why in notes.
   Separate design-fixable failures (strain, ease, length) from
   pipeline-limited ones (gates that do not move when the design changes).
4. Before writing the report, do the critic's job yourself: re-read the latest
   agent_iteration.json and preview image and argue against your last change.
5. When the user asked to see the outfit, call mcp__fashion__render_outfit
   and look at the returned WebGPU frame.
6. Write outputs/design_sessions/<design_id>/DESIGN_REPORT.md and finish with
   at most 10 lines: report path, final gate status, remaining failures.

Never report a failed gate as passed and never call a design production-ready.`;

const CRITIC_PROMPT = `You are a senior technical designer and pattern maker reviewing a
garment that another agent is iterating in a virtual try-on simulator. You
review; you do not change the design.

Inputs: a design_id and the original brief.
1. Call mcp__fashion__compare_iterations with the design_id.
2. For the latest one or two iterations, Read
   outputs/design_sessions/<design_id>/iter_NN/agent_iteration.json and look at
   intrinsic_tryon_final.png in the same folder.
3. Judge, with evidence: does the garment still match the brief; which failed
   checks respond to design changes and which never moved (pipeline limit);
   is the ease plausible (relaxed knit T-shirt: finished chest about 8-15 cm
   over body chest; slim 0-6 cm; oversized 20 cm or more)?
4. Reply in at most 12 lines:
   verdict: keep iterating / accept with caveats / stop (pipeline-limited)
   evidence: the two or three numbers or visual cues that decided it
   next change: one concrete parameter change with values, or none
   risks: anything the designer is overlooking

Quote metric values exactly as the tools reported them.`;

export const FASHION_DESIGNER_AGENT: AgentDefinition = {
  agentType: "fashion-designer",
  whenToUse:
    "Autonomous garment designer. Give it a design brief (and body measurements if known); " +
    "it runs the brief -> construction -> drape -> fit-gate loop with the fashion physics tools " +
    "and writes a design report. Use for background or batch design runs.",
  tools: ["Read", "Write", "Glob", ...FASHION_TOOLS],
  maxTurns: 40,
  source: "built-in",
  getSystemPrompt: () => DESIGNER_PROMPT,
};

export const FIT_CRITIC_AGENT: AgentDefinition = {
  agentType: "fit-critic",
  whenToUse:
    "Independent fit and construction reviewer for a design session. Give it a design_id and the " +
    "brief; it reads every iteration and returns a verdict and the next change to try.",
  tools: ["Read", "Glob", "mcp__fashion__compare_iterations", "mcp__fashion__list_materials"],
  disallowedTools: ["Write", "Edit", "MultiEdit", "Bash", "mcp__fashion__run_tryon"],
  maxTurns: 12,
  source: "built-in",
  getSystemPrompt: () => CRITIC_PROMPT,
};
