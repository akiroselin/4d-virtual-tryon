# 4D Virtual Try-On — agent notes

Research prototype that turns a garment brief or product image into a
pattern, dresses a body, simulates cloth and runs fit/physics gates. This file
is loaded automatically by Easy Agent (and other tools that read `AGENTS.md`).

## Layout

- `aigc_fashion/` — the library. Entry points: `pipeline.run_pipeline`
  (brief -> pattern -> drape -> gates, T-shirt only),
  `construction_hypotheses.generate_physical_structure_hypotheses` (all
  templates), `pleated_skirt`, `anny_cloth4d_simulation`, `unified_cloth4d_scene`.
- `fashion_agent/` — the agent layer. `tools.py` wraps the library as JSON tools;
  `mcp_server.py` serves them over stdio MCP (dependency-free).
- `.mcp.json`, `.easy-agent/` — Easy Agent wiring: the `fashion` MCP server,
  the `/design-garment` skill, and the `fashion-designer` / `fit-critic` agents.
- `assets/pattern_templates/pattern_library.json` — garment template ontology.
- `examples/`, `scripts/` — CLI runners; `docs/` — design notes; `web/` — WebGPU viewer.
- `tests/` — pytest suite.

## Commands

```bash
python3 -m pip install -r requirements.txt
python3 -m pytest -q                       # tests needing local-only assets are skipped
python3 fashion_agent/mcp_server.py        # MCP server (stdio); normally started by Easy Agent
python3 examples/run_mvp.py --prompt "relaxed cotton t-shirt" --steps 32
python3 -m http.server 8769                # then open /anny-cloth-webgpu.html (WebGPU demo)
```

## Local-only inputs (gitignored)

`assets/body_meshes/`, `assets/mocap/`, `outputs/`, `work/` (SewFormer,
CortexODE, Genesis checkouts) are not in the repository. Code that needs them
must take the path as an argument; never hard-code a personal absolute path.
The optional `genesis_geometric` package is preferred when importable;
otherwise `aigc_fashion/mesh_operators.py` provides the same operators.

## Rules for changes

- Keep claims honest: if a learned model, checkpoint or external backend did
  not run, outputs must say so (see `implementation_status` in summaries).
- The material metric (rest state) and the dressed 3D state are separate
  contracts; do not let physics or contact silently change the rest metric.
- New agent tools go in `fashion_agent/tools.py` (function + JSON schema in
  `TOOLS`) with a test in `tests/test_fashion_agent.py`.
- Run `python3 -m pytest -q` before committing.
