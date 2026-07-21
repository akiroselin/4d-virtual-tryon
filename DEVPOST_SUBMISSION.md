# 4D Virtual Try-On

## Inspiration

Generative fashion tools can create compelling images, but an image does not tell a customer whether a garment fits, collides with the body, or moves like its stated material. We built 4D Virtual Try-On to connect image-led garment design with simulation-ready geometry and physically interpretable motion.

## What it does

The prototype turns a garment concept into an intrinsic shell around a commercial-safe Anny avatar, then simulates a T-shirt and pleated skirt during a captured walking motion. Its WebGPU viewer plays the body and both garments on one physical clock and preserves the T-shirt print as part of the deforming surface.

The current demo includes:

- pattern semantics and construction hypotheses for garment topology;
- a CortexODE-inspired SDF-to-topology-to-deformation path;
- remeshing-aware intrinsic operators based on the Laplace-Beltrami spectrum;
- a Hyper-NODE material field using Kirchhoff-Love membrane and bending terms;
- a 4D cloth/body scene graph with E(3)-equivariant message passing;
- a symplectic phase-space update with gravity and body-root momentum transport;
- body-SDF event detection, collision impulses, and gauge-frame contact friction;
- layered T-shirt/skirt contact and a Three.js WebGPU renderer.

## How we built it

The project is written in Python, PyTorch, NumPy, SciPy, trimesh, and Three.js. A garment first becomes an immutable material reference map plus a deforming 3D shell. Hyper-NODE predicts spatially varying constitutive response, while analytic Kirchhoff-Love terms keep membrane and bending behavior physically interpretable. Cloth state is represented by position and canonical momentum; the runtime advances it with a symplectic update and treats gravity, prescribed body motion, contact, and damping as explicit open-system ports.

The avatar, T-shirt, skirt, and contact patches share a hierarchy-aware 4D scene graph. Learned messages use relative geometric quantities rather than absolute raw coordinates. Body contact is detected from a signed-distance query and resolved as a momentum event with local SO(2) tangent frames. The final demo disables curvature-driven AMR to show that the intrinsic formulation still runs on a fixed, lower-cost mesh.

Codex with GPT-5.6 was the main engineering partner throughout the project. It helped turn the research brief into executable modules, inspect and integrate upstream projects, derive invariance and Hamiltonian contracts, write focused tests, diagnose visual failures from repeated WebGPU runs, and iterate on contact, transport, material, gait, and rendering behavior. The majority of the implementation history is in Codex session `019f390f-b11c-7372-95b7-2026cd58af33`.

## Challenges we ran into

The hardest failure was not rendering but coupling. Early garments looked like rigid shells, drifted behind the moving body, penetrated at the underarm and hips, or expanded when contact corrections changed the metric. We separated the immutable material metric from the dressed shape, moved body transport into phase momentum, removed per-step metric projection from the physical rollout, and made contact a localized event instead of a visual offset.

Single-view pattern recovery is also underdetermined. The implementation therefore treats image evidence as a construction hypothesis and combines it with garment-category priors, body measurements, topology checks, and material constraints instead of claiming exact production patterns from one photograph.

## Accomplishments that we're proud of

- A runnable two-garment, moving-body 4D simulation rather than a static try-on render.
- Explicit physical audit reports for contact gaps, strain limits, Hamiltonian terms, generalized power ports, and symmetry structure.
- A fixed-resolution no-AMR path that keeps the same intrinsic material model.
- 32 focused tests passing across the final runtime changes.
- A browser-native Three.js WebGPU result that can be inspected interactively.

## What we learned

Equivariance helps representation and generalization, but it does not replace contact mechanics or a calibrated material law. A Hamiltonian formulation is most useful when external work, damping, prescribed motion, and collision impulses remain explicit. For virtual try-on, topology and construction priors must be established before dynamics; physics cannot repair an invalid sleeve or waistband.

## What's next

Next we will train the multi-rate dynamics head on a larger library of Genesis/IPC teacher trajectories, calibrate Hyper-NODE latents from measured textiles, add triangle-level continuous collision detection for garment layering, and evaluate remeshing invariance across garment categories and body shapes.

## Built with

Python, PyTorch, NumPy, SciPy, trimesh, Three.js, WebGPU, Anny, Neural ODEs, Hyper-NODE, Kirchhoff-Love shells, Laplace-Beltrami operators, E(3)-equivariant graph learning, symplectic integration, signed-distance fields, and Codex with GPT-5.6.

## Run the demo

```bash
python3 -m http.server 8769
```

Open `http://127.0.0.1:8769/anny-cloth-webgpu.html` in a WebGPU-capable browser.

