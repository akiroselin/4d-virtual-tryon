# AIGC Fashion Prototype

> OpenAI Build Week project: **4D Virtual Try-On**. Codex with GPT-5.6 was used
> to translate the research proposal into executable geometry, material,
> contact, 4D scene-graph, testing, and WebGPU-rendering modules. The primary
> build session is `019f390f-b11c-7372-95b7-2026cd58af33`. See
> [DEVPOST_SUBMISSION.md](DEVPOST_SUBMISSION.md) for the project story and a
> concise implemented-versus-next roadmap.

This is the first runnable implementation of the working solution:

1. Parse text or JSON into a structured garment intent.
2. Route the garment category to a license-aware pattern template/provider.
3. Generate a pattern-first manufacturing atlas with explicit panels and seam interfaces.
4. Import a commercial-safe body mesh, or use the measurement-only fallback proxy.
5. Fuse VTO silhouette, landmark, occlusion, size, topology, seam, strain, and collision evidence in a preflight gate.
6. Keep the 2D material rest state and dressed 3D state as separate contracts.
7. Convert the atlas to one continuous body-SDF T-shirt shell for simulation, while retaining the panel/seam solver as a comparison backend.
8. Export OBJ, GIF, PNG, preflight, and JSON summary artifacts.

The current MVP intentionally keeps FashionCLIP, GarmentDiffusion, and aesthetic
scoring as stable interfaces/placeholders. The working core is the garment state
contract, pattern mesh generation, generic body mesh support, and intrinsic
physics rollout.

The current intrinsic rollout is still an experimental, uncalibrated textile
solver. Its default Anny path now preserves an immutable 3D shell rest metric,
uses pure body-SDF contact, and has no runtime seam springs. Genesis FEM plus IPC
remains the target high-fidelity validator on a supported Linux/`pyuipc` host.

## Fashion Design Agent

`agent/` is a fork of [Easy Agent](https://github.com/ConardLi/easy-agent)
(MIT, copied from upstream commit `876c5ca`) turned into a garment-design
agent. The Python code in this repository is its physics backend, reached
through the `fashion` MCP server.

| Piece | Path | Role |
|---|---|---|
| Agent CLI | `agent/`, `bin/fashion-agent` | Fashion identity prompt, built-in `fashion-designer` and `fit-critic` agents, 30-minute MCP tool timeout (`EASY_AGENT_MCP_TOOL_TIMEOUT`) |
| Physics MCP server `fashion` | `fashion_agent/`, `.mcp.json` | `parse_design_brief`, `generate_construction_hypotheses`, `run_tryon`, `compare_iterations`, `list_garment_templates`, `list_materials`, `list_outfit_scenes`, `simulate_outfit`, `render_outfit` |
| Skill `/design-garment` | `.easy-agent/skills/design-garment/SKILL.md` | Brief → hypotheses → try-on → critique → revise → render → `DESIGN_REPORT.md` |
| WebGPU viewer | `anny-cloth-webgpu.html`, `web/` | Three.js `WebGPURenderer` playback of simulated outfits |
| Headless capture | `scripts/capture_webgpu.mjs` | Renders frames in Chromium and fails unless the backend is WebGPU (hardware GPU, else SwiftShader) |
| Body from a photo | `aigc_fashion/body_reconstruction.py`, `scripts/sam3d_body_infer.py` | SAM 3D Body → MHR identity + bone scales → A-pose body mesh and measurements (`reconstruct_body` tool) |
| Physics notes | `docs/PHYSICS_PRINCIPLES_zh.md` | How the cloth simulation works (Chinese) |

```bash
python3 -m pip install -r requirements.txt   # physics backend (+ `pip install anny roma` for the Anny body)
npm install                                  # playwright-core for WebGPU capture
export ANTHROPIC_AUTH_TOKEN=...              # or a provider profile in ~/.easy-agent/settings.json
bin/fashion-agent                            # builds agent/ on first run; trust the folder
> /design-garment relaxed white cotton tee for a 128 cm child, then render the outfit
```

`.mcp.json` starts the server with `python3`; point `command` at the
interpreter that has the requirements if `python3` on `PATH` does not.

### Body reconstruction from one image (SAM 3D Body + MHR)

[SAM 3D Body](https://github.com/facebookresearch/sam-3d-body) predicts the
[Momentum Human Rig](https://github.com/facebookresearch/MHR) parameters of each
person in a photo. `reconstruct_body` keeps the person's 45 identity
coefficients and 68 bone scales, drops the photographed pose, evaluates MHR in
its rest pose (already an A-pose, arms about 48° down) and returns an OBJ in
metres (+Y up, +Z front), height / chest / waist / hip / shoulder estimates from
mesh cross-sections, and a preview image. `run_tryon` accepts the mesh directly.

- MHR (code and model assets): Apache-2.0. `ensure_mhr_model()` downloads the
  release assets into `work/mhr/` (or set `MHR_MODEL_PATH`).
- SAM 3D Body (code and weights): SAM License; the checkpoints are gated on
  Hugging Face (request access to `facebook/sam-3d-body-dinov3`). Install it in
  its own environment per its INSTALL.md, then set `SAM3D_PYTHON`, `SAM3D_REPO`
  and `SAM3D_CHECKPOINT`. Without it, the tool also accepts a saved prediction
  JSON written by `scripts/sam3d_body_infer.py`.
- Single-view estimates: metric scale comes from the model's priors and clothes
  hide the body, so confirm key measurements with the wearer.
- Verified here with MHR's mean body (`docs/images/mhr_mean_body_apose.png`):
  172.7 cm, chest 100.5, waist 95.2, hip 109.5, shoulder 40.4 cm. SAM 3D Body
  itself was not run in this environment (its Hugging Face checkpoints are gated
  and Hugging Face is not reachable from it).

### Demo outfit rendered with Three.js on WebGPU

![Cropped tee and pleated skirt, WebGPU](docs/images/webgpu_demo_outfit_front.png)

`web/scenes/agent-demo-crop-tee-pleated-skirt` was produced end to end in this
repository from two synthetic product images (`scripts/make_demo_inputs.py`):
image-fitted T-shirt pattern → implicit-shell T-shirt dynamics → pleated-skirt
dynamics on the Anny child body (`scripts/export_anny_body_obj.py`), exported
with `scripts/export_garment_runs_scene.py` and rendered by
`scripts/capture_webgpu.mjs` (WebGPU backend; here Chromium's SwiftShader
adapter). The cropped tee ends 3.2 cm above the waistband, so the two garments
were simulated separately and need no inter-garment contact. Overlapping layers
go through `examples/run_unified_anny_cloth4d.py`; on a 4-core CPU that joint
solve did not finish in usable time for this outfit (and with walking enabled it
stopped at its contact-convergence check), so it is not used here. Both garment
rollouts are short settling runs (0.072 s and 0.24 s) with uncalibrated
(`material_floor`) constitutive scaling.

Open it interactively with `python3 -m http.server 8769` and
`http://127.0.0.1:8769/anny-cloth-webgpu.html?scene=agent-demo-crop-tee-pleated-skirt`.

Headless WebGPU capture of any scene:

```bash
CHROMIUM_PATH=/path/to/chrome node scripts/capture_webgpu.mjs \
  --scene anny-skirt-walk-one-step-2s-contactfix-v10-webgpu --yaws 0,35 --out outputs/renders
```

## Run

All commands run from the repository root. Paths under `assets/body_meshes/`,
`assets/mocap/`, `outputs/` and `work/` are local-only (gitignored) and must be
created or downloaded first; the image-driven examples now take
`--texture-image` / `--shirt-texture` / `--skirt-image` as required arguments.

```bash
python3 -m pip install -r requirements.txt
python3 examples/run_mvp.py \
  --prompt "oversized minimal streetwear short sleeve cotton t-shirt" \
  --steps 64 \
  --resolution 23
```

The original build machine's absolute paths still appear in some commands
below; read `/opt/homebrew/anaconda3/bin/python3` as `python3` and
`outputs/aigc_fashion_prototype/` as the repository root.

Commercial-safe Anny body mesh path:

```bash
ANNY_CACHE_DIR=outputs/aigc_fashion_prototype/work/anny_cache \
WARP_CACHE_PATH=outputs/aigc_fashion_prototype/work/warp_cache \
/opt/homebrew/anaconda3/bin/python3 outputs/aigc_fashion_prototype/scripts/export_anny_body_obj.py \
  --output outputs/aigc_fashion_prototype/assets/body_meshes/anny_default.obj \
  --height-cm 178 \
  --topology default

/opt/homebrew/anaconda3/bin/python3 outputs/aigc_fashion_prototype/examples/run_mvp.py \
  --texture-image "/Users/muwusah/Documents/AI/WorldModel/Genesis/2d/tshirt.png" \
  --fit-from-image \
  --learned-pattern \
  --body-mesh outputs/aigc_fashion_prototype/assets/body_meshes/anny_default.obj \
  --strict-preflight \
  --out-dir outputs/aigc_fashion_prototype/outputs/anny_tshirt_run
```

The body adapter infers the up/front axes from an upright mesh and records the
resolved mapping in `summary.json`. Explicit axis flags remain available for
posed or unusually oriented meshes.

Continuous-shell NeuralClothSim run:

```bash
cd /Users/muwusah/Documents/Codex/2026-07-06/5/outputs/aigc_fashion_prototype
/opt/homebrew/anaconda3/bin/python3 examples/run_anny_cloth4d.py \
  --simulation-topology implicit_shell \
  --implicit-shell-target-faces 700 \
  --out-dir outputs/anny_implicit_shell_hypernode_kl
```

Use `--simulation-topology seam_atlas --placement-mode arrangement_points` to
retain the previous panel-plus-runtime-seam baseline. The default
`implicit_shell` representation is one genus-0 manifold with four physical
boundary loops (neck, hem, and two cuffs); it has no seam constraints during
simulation. The learned panel atlas remains the manufacturing and material
provenance representation. See
[docs/IMPLICIT_TSHIRT_SHELL_2026.md](docs/IMPLICIT_TSHIRT_SHELL_2026.md).

Anny's authored axes are Blender `Z`-up with the character facing `-Y`. The
Anny-specific commands therefore default to `--body-source-up-axis z
--body-source-front-axis -y`; bounding-box axis inference cannot determine the
front-axis sign.

## Genesis Nyx rendering

The final cloth state can be exported as Nyx-ready PBR GLBs on any host:

```bash
/opt/homebrew/anaconda3/bin/python3 examples/render_anny_cloth4d_nyx.py \
  --prepare-only \
  --rollout outputs/anny_implicit_shell_crew_neck_texture_v9/e3_anchor_constrained_rollout.npy \
  --out-dir outputs/anny_implicit_shell_crew_neck_texture_v9/nyx
```

The export contains a Z-up Anny GLB, a continuous T-shirt GLB, embedded
Disney-style PBR materials, a unified white cotton base for collar/body/sleeves,
and a separate front-print primitive. On an x86-64 Linux or Windows NVIDIA host:

```bash
python -m pip install "gs-nyx-plugin==0.1.1"
python examples/render_anny_cloth4d_nyx.py \
  --rollout outputs/anny_implicit_shell_crew_neck_texture_v9/e3_anchor_constrained_rollout.npy \
  --out-dir outputs/anny_implicit_shell_crew_neck_texture_v9/nyx \
  --spp 128
```

Nyx 0.1.1 has no Apple Silicon backend. It requires an NVIDIA GPU with compute
capability 7.0+, CUDA 12.9+, and driver 575+, so this macOS host can prepare and
validate the scene but cannot produce genuine Nyx pixels. The older PIL renderer
is retained only as a geometry/debug preview; its per-face shading branches were
the cause of the collar colour discontinuity.

Default outputs are written to:

`/Users/muwusah/Documents/Codex/2026-07-06/5/outputs/aigc_fashion_prototype/outputs/mvp_run`

The current SMPL-free child Anny T-shirt pipeline uses:

- the official SewFormer checkpoint as a research-only pattern proposal;
- panel, stitch-reference, and stitch-length validation before meshing;
- a license-aware router that prefers project-owned or MIT GarmentCode programs;
- a mature-VTO preflight before physics, including semantic landmarks, protected body regions, occlusion order, body grading, material strain, seams, collision, and sleeve coverage;
- an avatar-scaled four-panel pattern program retained for historical comparison when the learned proposal is invalid;
- matched sleeve-cap/armhole seam lengths before dressing;
- topology-aware sleeve coordinates and seam-preserving sleeve refinement;
- a legacy collision-prefitted intrinsic metric and radial sleeve/body SDF collision,
  retained only for comparison.

The current refined child result is in:

`/Users/muwusah/Documents/Codex/2026-07-06/5/outputs/aigc_fashion_prototype/outputs/anny_child_intrinsic_refined_final`

For the supplied white-background product image, SewFormer currently predicts
front/back top and skirt panels with no sleeve panels or stitches. The pipeline
records that result, rejects it as non-sewable, and generates a validated child
T-shirt pattern instead. This is deliberate: product images do not provide a
metric scale, so garment scale comes from the avatar and the image contributes
silhouette proportions and texture.

The four-panel fallback is no longer considered production-ready. A whole-sleeve
T-shirt pattern is valid in principle; preflight now accepts either whole sleeves or
mapped front/back sleeve patches. This candidate is still rejected because its
canonical placement has excessive material strain and body penetration, catching
the cause of the earlier underarm bulge before the physics solver runs.
See [docs/MATURE_VTO_PREFLIGHT.md](docs/MATURE_VTO_PREFLIGHT.md).
The wider 2025-2026 VTO and digital-garment review, licensing boundaries, and
revised architecture are in
[docs/VTO_LANDSCAPE_2026.md](docs/VTO_LANDSCAPE_2026.md).
The authoritative global architecture decision is in
[docs/END_TO_END_ARCHITECTURE_2026.md](docs/END_TO_END_ARCHITECTURE_2026.md).
The CortexODE/Neural ODE transfer analysis and its single-image identifiability
limits are in
[docs/NEURAL_DYNAMICAL_SURFACE_RECONSTRUCTION_2026.md](docs/NEURAL_DYNAMICAL_SURFACE_RECONSTRUCTION_2026.md).
The authoritative NeuralClothSim/CortexODE route audit is in
[docs/NDF_CORTEXODE_CLOTH_ROUTE_AUDIT_2026.md](docs/NDF_CORTEXODE_CLOTH_ROUTE_AUDIT_2026.md),
and the implemented fold/contact/intersection event contract is in
[docs/DYNAMIC_EVENT_HANDLER_2026.md](docs/DYNAMIC_EVENT_HANDLER_2026.md).
The implemented geometric-learning replacements, Genesis differentiability
audit, Fast-Marching topology gate, and Genesis-constrained 4D NDF objective are in
[docs/GENESIS_GEOMETRIC_DIFFERENTIABLE_STACK_2026.md](docs/GENESIS_GEOMETRIC_DIFFERENTIABLE_STACK_2026.md).
The NeuralClothSim representation is now a cloth/body-anchor E(3)-equivariant
field rather than a raw world-coordinate vector decoder. Its definition, geometric
conditioning bridge, proof, and tests are in
[docs/E3_ANCHOR_NEURALCLOTH4D_2026.md](docs/E3_ANCHOR_NEURALCLOTH4D_2026.md).
The scene runtime now adds a material-point Hyper-network NODE constitutive
field on Kirchhoff-Love membrane and bending kinematics, with dynamic
equilibrium and free-edge traction losses. Its equations, contracts, and limits
are in
[docs/HYPERNODE_KIRCHHOFF_LOVE_CLOTH_2026.md](docs/HYPERNODE_KIRCHHOFF_LOVE_CLOTH_2026.md).
The transfer audit for dHCP, DDT-from-CSR, FetalCSR, DDCSR, and BrainNet is in
[docs/CSR_TO_GARMENT_SURFACE_TRANSFER_2026.md](docs/CSR_TO_GARMENT_SURFACE_TRANSFER_2026.md).
The executable typed-geometry manifest rejects untyped Cartesian features in
every shared or per-scene learned module while preserving coordinate access for
numerical physics, SDF queries, observation operators, and rendering. The 4D
scene graph uses three-hop hierarchy-aware spatiotemporal equivariant attention:
learned messages consume relative displacement, relative time, hierarchy-level
difference, and typed relation IDs rather than absolute world coordinates.

The runtime phase state is now the product
`T*Emb(S_cloth,R3) x T*SE(3) x C_contact`: cloth stores immutable material UV,
the spatial embedding `q`, and canonical momentum `p=Mv`; articulated bodies
use SE(3) poses and body cotangent momenta; contact stores signed SDF gaps,
normals, and an SO(2) charge-1 tangent momentum field. Cloth substeps use a
canonical kick-drift symplectic update. A localized SDF event applies a momentum
jump/reset and bounded gauge-equivariant friction before metric projection.
The authoritative potential is still the analytic Hyper-NODE Kirchhoff-Love
shell plus gravity. No trained learned-Hamiltonian head is claimed yet.

The runtime additionally stores an immutable reference map
`xi=(sheet_id,u,v)`, its shell metric, and a separate Spin(3) quaternion surface
frame. All interactions now enter a typed generalized-flow ledger: E(3)
force-velocity power, SO(2) complex tangent contact, conservative/dissipative/
control/constraint roles, and measure-valued impulse work for SDF resets and
metric projection. Event-localization trial work is rolled back before the
accepted trajectory is audited.

The same shell is now lifted into a bundle-valued empirical measure
`mu_vertex direct-sum mu_face direct-sum mu_hinge`. Vertex atoms carry mass and
momentum, face atoms carry rest-area-weighted deformation gradient/metric/frame,
and hinge atoms carry Kirchhoff-Love curvature. The runtime advances the vertex
measure with symplectic kick-drift, applies SDF collision as a momentum-fibre
push-forward, and evaluates membrane and bending energies on their original
face/hinge quadrature. It does not collapse folded layers into a scalar world
density. Every rollout reports mass/material-coordinate drift and the complete
real measure Hamiltonian; the formal contract is documented in
[docs/GEOMETRIC_PHASE_GRAPH_2026.md](docs/GEOMETRIC_PHASE_GRAPH_2026.md).
During joint field/Hyper-NODE fitting, each three-frame sample now also receives
measure-continuity, Hamiltonian power-balance, passivity, and dissipative-sign
losses. These are logged separately from trajectory RMSE and the older
vertex-level variational PINN terms.

After mesh-motion pretraining, the field can now be refined as a variational
phase PINN. The loss combines discrete Hamilton-d'Alembert stationarity,
Kirchhoff-Love force balance, local work-energy balance, SDF contact KKT, and
friction/damping dissipation. Metric projection remains a hard post-step
material constraint. This is action-stationarity training, not direct
minimization of the scalar action value.

Personality and emotion are represented as invariant, time-varying high-level
phase context rather than material properties. New 16-channel scene models put
OCEAN personality on the person/root node and VAD emotion on the body node, then
propagate context through `person -> body -> garment -> panel/contact` with the
three-layer hierarchy-aware attention stack. A separate bounded behavioral
control port modulates body cadence, stride, sway, and gesture; cloth receives
the resulting work through moving-body SDF contact and momentum reset. The
provided OCEAN/VAD-to-gait mapping is an engineering prior, not a validated
psychological measurement model. Existing 8-channel checkpoints remain
loadable and use the physical body-control/contact path only.

## Image-Conditioned Pleated Skirt

The skirt route classifies a catalog image into a supported construction prior,
fits a continuous annular shell to Anny waist/hip/thigh arrangement points, and
stores the waistband and pressed box pleats in the immutable rest surface. The
runtime shell has waist and hem boundaries but no seam springs; its manufacturing
provenance remains separate from the simulation representation. It uses the same
body-SDF contact, intrinsic Laplace-Beltrami propagation, Hyper-NODE
Kirchhoff-Love losses, E(3)-equivariant per-scene 4D field, and Nyx GLB export as
the T-shirt route.

Each run also exports a manufacturing bundle containing a construction record,
a GarmentCode-compatible JSON pattern, and an SVG with outer folds, underfolds,
grain direction, seam allowance, pleat uptake, and hidden-construction
hypotheses. The official MIT-licensed Design2GarmentCode source is probed as an
optional image provider. If its Qwen base/LoRA weights and a user-supplied API
credential are absent, the run records `deterministic_family_prior`; it does not
claim that a learned model produced the pattern. Internal pleat-fold extensions
are compiled into the welded intrinsic rest shell because stock GarmentCode JSON
does not encode the complete box-pleat folding operation.

```bash
/opt/homebrew/anaconda3/bin/python3 examples/run_anny_pleated_skirt.py \
  --image /absolute/path/to/skirt.webp \
  --out-dir outputs/anny_pleated_skirt
```

To retrain only the 4D field from an already validated physics teacher:

```bash
/opt/homebrew/anaconda3/bin/python3 examples/run_anny_pleated_skirt.py \
  --image /absolute/path/to/skirt.webp \
  --teacher-rollout /absolute/path/to/physics_teacher_rollout.npy \
  --out-dir outputs/anny_pleated_skirt_retrained
```

Single-view images determine garment family, silhouette, pleat frequency, and
appearance; body anchors supply metric scale because an uncalibrated catalog
image cannot identify real waist circumference or length by itself.

### CortexODE intrinsic-shell route

The optional `--shell-reconstruction cortexode` route follows the CortexODE
ordering rather than starting from a final panel mesh:

1. a CortexODE-style 3D U-Net predicts a closed garment occupancy proxy;
2. a signed distance transform and the official Bazin Fast Marching backend
   produce a connected genus-0 level set;
3. Marching Cubes extracts the corrected closed proxy;
4. panel-library semantics reopen `waist/hem`, or
   `neck/hem/left_cuff/right_cuff`, and compile panel adjacency into a sewing
   graph;
5. a spectrally bounded, multiscale-SDF-conditioned Neural ODE deforms the
   intrinsic shell under Kirchhoff-Love energy, equilibrium, free-edge
   traction, body-contact, and optional Genesis trajectory supervision.

The closed proxy is only a topology-correction device. Semantic openings stay
free boundaries. On a welded shell, side/shoulder/armhole links become
structural rest-length edges and cross-link hinges; they are not zero-gap seam
springs. Exact production panels remain an optional manufacturing output, while
garment family, opening topology, material metric, grain, and fold constraints
remain required physical information.

### FFN panel instance grower

`aigc_fashion.panel_instance_grower` adapts Google Flood-Filling Networks to
canonical 2D sewing panels. A dense image encoder predicts an unknown number of
panel seeds and roles. For each seed, a recurrent local cell updates a panel
mask, seam-endpoint field, and stop probability, while a frontier queue moves
the field of view until the complete connected panel instance is recovered.
The result contains one SDF and boundary per panel and is passed to the sewing
graph and physical-validation stages.

The image route is activated only with a trained checkpoint:

```python
summary = run_pipeline(
    ...,
    texture_image="garment.png",
    panel_growth_checkpoint="panel_ffn.pt",
)
```

The checkpoint must contain `model`, `role_names`, and `feature_channels`.
Without it, the project does not claim FFN image inference. The separate
GarmentCode teacher diagnostic validates the scheduler and creates training
targets without pretending that a known pattern was recovered from an image:

```bash
/opt/homebrew/anaconda3/bin/python3 examples/run_panel_growth_teacher_diagnostic.py \
  --pattern /absolute/path/to/garmentcode.json \
  --out-dir outputs/panel_growth_teacher
```

### Joint simulation-ready intrinsic shell

`aigc_fashion.simulation_ready_shell` joins the previously separate inference
branches into one trainable graph:

1. The FFN image encoder, seed head, and recurrent local cell recover a variable
   number of canonical panel instances.
2. SewFormer panel roles and stitch adjacency are encoded by a sewing-graph
   message-passing network. External role descriptors provide an open-vocabulary
   path instead of requiring every garment class to fit a fixed label list.
3. Panel tokens and image features are lifted into the body-aligned volume and
   condition a 3D U-Net that predicts a closed proxy SDF, seam field, and
   spatial material field.
4. The differentiable volume loss applies SDF/occupancy supervision, Eikonal
   regularity, body clearance, semantic seam/opening constraints, constitutive
   residuals, and an optional Genesis FEM/MPM teacher SDF.
5. CortexODE topology correction extracts the proxy, restores semantic openings
   and sewing links, and deforms the resulting fixed-topology intrinsic shell.
   `assess_simulation_ready_shell` then rejects non-manifold, disconnected,
   degenerate, wrongly opened, or body-intersecting outputs.

This architecture reduces dependence on exact production-pattern recovery, but
does not make single-view hidden construction identifiable. Generalisation to
new categories requires training over compositional panel/sewing grammars,
multi-view or synthetic supervision, diverse bodies/materials, and hard physics
validation. The implementation is trainable and tested; no all-category trained
checkpoint is claimed yet.

### Body-graded construction hypotheses

`aigc_fashion.construction_hypotheses` treats the pattern/process database as
the authority for rest metric, panel topology, sewing graph, openings and
material priors. Body measurements grade that structure into Top-K fit
candidates. Missing measurements remain explicitly marked anthropometric priors
with uncertainty instead of being presented as observations.

```bash
/opt/homebrew/anaconda3/bin/python3 examples/run_construction_hypotheses.py \
  --garment-type t-shirt --height-cm 128 --chest-cm 64 \
  --waist-cm 56 --shoulder-cm 30 --upper-arm-cm 21 \
  --top-k 3 --out outputs/construction_hypotheses.json
```

CortexODE is now guarded as a current-embedding refinement stage. Its output
must retain face connectivity, material rest coordinates, panel roles, sewing
interfaces and UVs exactly. It cannot repair a wrong construction hypothesis by
silently changing the garment's manufactured structure.

Every CortexODE RK4 update is followed by a differentiable material-metric
projection. The immutable construction edge lengths define material-family
compression and tension intervals: knit profiles receive a wider elastic
envelope, while woven, linen and denim profiles are kept near inextensible.
Pinned vertices remain fixed. A hard audit runs after every projected step and
rejects mutually incompatible body placement, pinning and panel dimensions.
These family envelopes are engineering priors and must be replaced by calibrated
bias-extension measurements for production material matching.

### NeuralClothSim physical-time 4D dynamics

`aigc_fashion.neuralclothsim_dynamics` turns the scene-optimised cloth field
into an actual state trajectory `(x(t), v(t))`. Each physical substep evaluates
the latent-conditioned Hyper-NODE Kirchhoff-Love membrane and bending forces,
adds gravity, damping, explicit seam forces and a bounded E(3)-equivariant
residual, then advances the state with semi-implicit integration. The residual
network is not allowed to replace the constitutive shell law.

Before integration, one robust scene-wide scalar calibrates the uncalibrated
Hyper-NODE modulus against a target P95 shell acceleration. That scalar remains
fixed throughout the rollout; forces are never normalized per frame. The report
labels very small scales as `constitutive_recalibration_required`. This makes
the current engineering fallback explicit until measured fabric trajectories
can identify physical constitutive parameters. The learned E(3) residual also
has an independent acceleration bound.

Body contact is queried against the Anny triangle mesh. A signed-gap crossing
is localized on the physical time segment, recorded by the explicit event
detector, and causes the solver to split and restart the substep at contact.
At the active contact patch, `GaugeEquivariantContactCNN` projects relative
velocity into arbitrary local tangent frames, performs SO(2)-equivariant message
passing with parallel transport, and emits a bounded gauge-invariant friction
scale. The analytic SDF normal, non-penetration projection, and Coulomb friction
direction remain authoritative. A fresh network is exactly neutral, so an
untrained contact model reproduces the calibrated analytic solver rather than
injecting random impulses. Contact response and clearance projection are then
followed by the material-specific hard metric projection, so the manufactured
rest metric remains authoritative during motion. The final physical trajectory
is saved before the continuous 4D field is refitted to it.

The current hybrid mode variable covers contact enter/exit and stick/slip state.
Ordinary contact changes the active contact graph only; it does not remesh the
garment. Seam rupture, material tearing, and cutting are intentionally outside
this phase.

The dynamics report records CCD events, minimum body gap, material strain audit,
maximum speed, clipped and unclipped acceleration, and the fraction of vertex
updates that reached the acceleration cap. A nontrivial cap fraction means the
constitutive calibration or initial drape is not yet trustworthy even if the
animation looks stable. This single-garment rollout still lacks cloth
self-collision and calibrated fabric measurements; shared moving-body and
multi-layer handling are provided by the unified scene below. Genesis FEM/MPM
remains an optional trajectory-distillation and validation source; it is not
reported as executed unless that backend ran.

### Unified continuous 4D scene

`aigc_fashion.unified_cloth4d_scene` advances the Anny body, T-shirt, and skirt
on one physical clock. Every garment retains its own immutable intrinsic metric,
Hyper-NODE material latent, and metric projector. Body and garment-garment gap
zero-crossings are explicit events; an event localizes a root and restarts the
shared step. Moving-body contact uses the same analytic SE(3) trajectory for
collision queries, attachment points, saved body geometry, and rendering.
Garments have no artificial body pins: the T-shirt collar/shoulders and the
skirt waistband are not attached to avatar vertices. Retention comes from a
signed moving-body SDF, boundary-edge quadrature on openings, closed garment
topology, and a post-projection static/Coulomb friction pass. Inside-body
samples remain negative; their normals are never flipped into an unsigned
distance.

The saved field is a continuous material-surface embedding
`X_g(u,v,t) -> R3`: barycentric interpolation is used inside intrinsic triangles
and cubic Hermite position/velocity interpolation between physical time knots.
This is different from synchronizing independent 3D rollouts by normalized
frame index.

```bash
python examples/run_unified_anny_cloth4d.py
```

The validated demo window is currently 0.16 seconds (40 internal 4 ms steps,
9 physical knots, and 25 queried render frames). It uses sampled-vertex
inter-garment narrow phase rather than full vertex-face/edge-edge IPC. Extending
the accepted time window requires VF/EE IPC and a better coupled contact/metric
solver, not looser material gates.

## Test

```bash
python3 -m pip install -r requirements.txt
python3 -m pytest -q
```

Tests that read local-only assets (body meshes, mocap, prior outputs, the
CortexODE checkout) are reported as skipped on a clean clone.

## License Boundary

The Anny default body model is the commercial-safe body asset used here. Do not
substitute Anny-Fit or Multi-HMR checkpoints in a commercial product without a
separate license. The checked-out SewFormer code/checkpoint has no verified
commercial license in this project, so its generated patterns must remain an
experimental input until that license is resolved.

For a future image-only preview branch, FASHN VTON v1.5 is the current
commercial-capable candidate: its published inference code and model weights
are Apache-2.0. It is not yet a runtime dependency here, and its pixels will be
used only as soft silhouette and occlusion evidence, never as metric cloth.

## Next Implementation Targets

- Define the project-owned Natural Garment Schema with observed/inferred/unknown
  evidence states and interval-valued uncertainty.
- Train the joint FFN + sewing graph + 3D U-Net model on GarmentCodeData-like
  rendered patterns, multi-view shells, body SDFs, and validated FEM/IPC
  trajectories; report held-out garment-family and compositional-role results.
- Replace one-shot routing with Top-K GarmentCode-compatible candidates and joint
  silhouette, landmark, fit, seam, collision, and license ranking.
- Replace the current hand-authored T-shirt geometry with a fabrication-faithful,
  interface-complete deterministic candidate and pass strict preflight.
- Add inverse-drape optimization over semantic 2D pattern parameters.
- Add PatternGSL as a research-only Top-K provider after its promised artifacts
  are released and their licenses are audited.
- Benchmark Apache-2.0 GarmentGPT behind an optional provider adapter after its
  model, base-model, training-data, and dependency terms are audited.
- Add an Image2Garment-style material attribute-to-physics estimator using owned or
  cleared calibration data and uncertainty intervals.
- Add real GarmentCode-backed templates for skirts, shirts, dresses, pants, and jackets.
- Add Genesis FEM cloth plus IPC as the high-fidelity contact/reference backend;
  keep PBD as a frozen regression baseline rather than teacher truth. IPC remains
  a forward validator until its Genesis `couple_grad()` path is implemented.
- Rebuild intrinsic v2 around an immutable 2D metric, orthotropic membrane/shear,
  rest-dihedral bending, stable integration, and robust contact. Train any learned
  residual with physical energy plus IPC data, not PBD alone.
- Replace the unified 4D scene's sampled-vertex inter-garment narrow phase with
  VF/EE IPC, then extend the strictly accepted physical-time window.
- Connect the implemented event state machine to body SDF, VF/EE CCD, DCD contour
  extraction, and DOPRI5 dense-output root refinement.
- Connect the implemented material latent, bounded hypernetwork, spectral
  CortexODE, intrinsic DiffusionNet layer, E(3) cloth prior, and mandatory
  Genesis-constrained NDF objective to a garment that passes G0-G4.
- Register every new shared spatial learner in the typed-geometry manifest and
  benchmark EGNN/PaiNN/SEGNN/Clifford routes with matched parameters, data, and FLOPs.
- Run the directional-gradient manifest over constitutive, rollout, coupling,
  contact, sensor, and raster stages before calling the stack fully differentiable.
- After valid G0-G4 garments and IPC references exist, benchmark the spectrally
  bounded CortexODE only as a canonical-dressing warm start and inverse-control
  accelerator, never as a direct single-image physical truth model.
- Keep measurement-first Anny as the primary commercial-safe body route. Evaluate
  SAM 3D Body to MHR for optional image HMR after legal review, with user
  measurements overriding monocular shape estimates.
- Do not use SMPL/SMPL-X, Multi-HMR, or Anny-Fit in commercial production unless separately licensed.
- Add FASHN VTON v1.5 as optional visual evidence, then add FIT-style size stress
  tests that score visual fidelity separately from physical fit.
