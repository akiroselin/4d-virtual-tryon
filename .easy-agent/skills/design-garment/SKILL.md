---
name: design-garment
description: Design a garment from a brief by iterating pattern, drape simulation and fit gates until it passes or the budget runs out.
when_to_use: The user describes a garment to design, asks to fit or size a garment to a body, or wants a design iterated against the try-on simulator.
argument-hint: "<design brief, optionally with body measurements in cm>"
---

# Agentic garment design loop

Brief: $ARGUMENTS

You are the design lead. The `fashion` MCP server gives you the tools; the
`fit-critic` agent gives you an independent review. Work in this loop and keep
every decision traceable to a tool result.

## 1. Understand the brief

1. Call `mcp__fashion__parse_design_brief` with the brief. Read the parsed
   intent critically: the parser is a keyword heuristic. Re-call it with an
   explicit `intent` object to fix anything it misread (garment type,
   silhouette, sleeve, neckline, fabric, style).
2. If the brief names a fabric the solver does not know, call
   `mcp__fashion__list_materials` and pick the closest card; say which.
3. Collect body measurements from the brief (height, chest, waist, shoulder,
   hip, upper arm, in cm). If none are given, say you are using the default
   adult avatar. Do not invent measurements and present them as the user's.
4. Choose a short `design_id` (letters, digits, `-`, `_`).

## 2. Explore construction options

Call `mcp__fashion__generate_construction_hypotheses` with the intent and body.
Summarise the Top-K candidates in one short table (fit variant, score, key
graded measurements, warnings). Pick one and say why.

If `end_to_end_simulation` is false for this garment type, stop here: deliver
the construction hypotheses as the result, state plainly that this build only
drapes T-shirts, and do not call `run_tryon`.

## 3. Iterate (at most 4 `run_tryon` calls unless the user asks for more)

For each iteration:

1. Call `mcp__fashion__run_tryon` with `design_id`, intent, body, and either no
   `size_cm` (let the body grade it) or explicit finished measurements from the
   chosen hypothesis. Put what changed and why in `notes`.
   Use `steps: 32, resolution: 19` while exploring; raise to `64/23` only for
   the final confirmation run.
2. Read `fit_validation`, `preflight.failed_checks`, `preflight.recommendations`
   and `key_metrics`, and look at the preview image.
3. Classify every failure:
   - **design-fixable**: material strain too high (garment too small or too
     tight a silhouette for the body), sleeve coverage, ease, length.
     Change one or two parameters at a time: size_cm, silhouette, fabric.
   - **pipeline-limited**: failures that do not move when the design changes,
     e.g. `max_seam_gap_m` / `initial_seam_closure` on the measurement-only
     proxy avatar (the four-panel fallback). Do not keep iterating on these;
     report them and suggest the remedy (a commercial body mesh via
     `body_mesh`, or the implicit-shell route in `examples/run_anny_cloth4d.py`).
4. After the second iteration, delegate a review: run the `fit-critic` agent
   with the `design_id` and the brief. Weigh its verdict, then decide the next
   change.
5. Stop when the gates pass, when two consecutive iterations stop improving
   the design-fixable metrics, or when the budget is spent.

## 4. Deliver

Call `mcp__fashion__compare_iterations` and write
`outputs/design_sessions/<design_id>/DESIGN_REPORT.md` containing:

- the brief and final structured intent;
- body measurements used, marked as given or default;
- the chosen construction hypothesis and the final garment measurements (cm);
- an iteration table (change made, gate status, p95 strain, seam gap);
- remaining failures, split into design-fixable and pipeline-limited;
- artifact paths (OBJ, preview PNG, GIF, preflight JSON).

Then give the user a short summary with the report path. Never describe a
failed gate as passed, and never call the result production-ready: the solver
is uncalibrated and the gates are this build's geometric checks.
