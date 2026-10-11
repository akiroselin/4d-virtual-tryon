# Dynamic Cloth Event Handler Design (2026-07-17)

## Decision

Keep the sewn material mesh fixed for ordinary folds. Add explicit dynamic state
for contact pairs, visible folds, occlusion relations, and intersection contours.
Only tearing, cutting, fastening, or unfastening may change the material topology;
those operations remain out of scope.

This preserves the project mainline:

```text
one fully specified scene
  -> optimise one multi-chart 4D NeuralClothSim-style NDF
  -> query a continuous garment trajectory
  -> validate and correct contact with CCD/IPC
```

CortexODE remains a separate, spectrally bounded canonical-dressing flow. The
event handler is shared, but the response differs: CortexODE stops and restarts
DOPRI5; the direct 4D NDF adds event-localised samples and active constraints to
its scene objective.

## What Transfers From The Three Papers

| Source | Actual contribution | Transfer into this project | Do not transfer |
|---|---|---|---|
| [Liu et al., Folding Clothing with Topological Dynamics](https://arxiv.org/abs/2504.20720) | A low-dimensional heterogeneous graph of visible region-boundary nodes, crease endpoints, sleeve-axis nodes, region membership, and occlusion, learned for robot jacket folding | A dynamic `FoldGraph`: crease birth/death, region relation updates, visibility/occlusion metadata, and adaptive NDF sampling around stable creases | Material edge collapse, garment remeshing, or the claim that a visual graph is physical topology |
| [Valle et al., Neural ODEs for Image Segmentation with Level Sets](https://arxiv.org/abs/1912.11683) | A Neural ODE evolves an image contour/embedding toward a segmentation zero level set | The mathematical event contract `g(state,time)=0`, bracketed root localisation, and stop/update/restart semantics | A ready-made 3D cloth SDF, self-contact detector, or collision guarantee |
| [ContourCraft, SIGGRAPH 2024](https://arxiv.org/abs/2405.09522) | DCD intersections are joined into open/closed contours; cloth correspondences are split into repulsive and non-repulsive sets; squared contour length is reduced with its quasi-rigid translational gradient | Intersection-contour birth/update/death events, outermost-contour filtering, layer-pair state, selective non-repulsive nodes, and a contour loss for NDF optimisation | Treating all close pairs as repulsive, dropping CCD/IPC validation, or claiming guaranteed intersection-free output |

Liu's use of “topological” refers to a changing perceptual skeleton. A physical
fold changes the embedding and contact graph but not the genus of the sewn
surface. This distinction is essential: CortexODE topology preservation does not
forbid a fold, but a smooth flow alone does not model finite-thickness contact.

## Event Surfaces

Use a positive gap outside contact and an active state at or below zero:

```text
g_body(v,t) = phi_body(x_v(t),t) - h
g_vf(v,f,t) = d_oriented(vertex,face,t) - h
g_ee(a,b,t) = d_oriented(edge_a,edge_b,t) - h
```

Use a positive confidence above the fold threshold:

```text
g_fold(c,t) = confidence(c,t) - threshold
```

`phi_body` is a valid signed field for a watertight body. A sewn T-shirt is an
open surface at the neck, cuffs, and hem, so one global cloth SDF cannot reliably
encode cloth self-contact. Self-contact must use non-adjacent vertex-face and
edge-edge primitive gaps plus CCD. Hysteresis confirms transitions while the
stored event time remains the first zero crossing.

## ContourCraft State

At each event-localised state:

1. Run triangle-triangle DCD and join adjacent intersections into contour
   components.
2. Match components over time by garment-layer pair and overlap of intersecting
   face pairs.
3. Remove nested closed contours from the response set, keeping the outermost
   contour.
4. Mark nodes on open contours as non-repulsive.
5. Mark nodes enclosed by an outermost closed contour as non-repulsive.
6. Keep other close cloth correspondences repulsive.
7. Minimise squared intersection-segment length using only the quasi-rigid
   translational gradient; shell energy continues to resist distortion.

This is a recovery path for an already intersecting state. Broad-phase CCD and
IPC remain the prevention and acceptance path. ContourCraft itself reports large
improvements, not a mathematical zero-intersection guarantee.

## Solver Integration

```text
query trial state at [s0,s1]
  -> broad phase candidate pairs
  -> evaluate body/VF/EE/fold event functions
  -> bracket all sign changes
  -> choose earliest root s_event
  -> refine root using solver dense output or CCD
  -> evaluate DCD contours at s_event
  -> update ContactGraph, FoldGraph, LayerGraph, ContourGraph
  -> apply response
       CortexODE: stop at tau_event and restart DOPRI5
       scene NDF: add t_event samples/constraints and continue optimisation
       Genesis: IPC projection/reference state
```

The event handler must never silently accept a step containing an unresolved
event. “DOPRI5 completed on CPU without RK4 fallback” verifies the selected solver
path only. It does not prove that narrow event surfaces were detected.

## Implemented Contract

[`aigc_fashion/dynamic_events.py`](../aigc_fashion/dynamic_events.py) now provides:

- scalar zero-crossing localisation with configurable active side and hysteresis;
- external CCD-root injection for crossings hidden between same-sign endpoints;
- body, vertex-face, edge-edge, and fold transition types;
- Liu-style crease/region fold-graph birth and death tracking;
- ContourCraft-style contour identity matching across frames;
- open/closed/nested contour handling and a non-repulsive node set;
- `EventBatch.earliest_time` and `restart_required` for solver adapters.

The module is deliberately geometry-backend independent. The remaining adapters
must supply body SDF samples, VF/EE CCD gaps, DCD contour components, and DOPRI5
dense-output root refinement. This keeps the event state testable without falsely
presenting a discrete frame detector as continuous collision detection.

## Acceptance Tests

The current unit tests verify:

- enter and exit root localisation;
- hysteresis without contact chatter;
- fold birth/death without material remeshing;
- contour birth/update/death identity;
- ContourCraft non-repulsive handling for open, closed, and nested contours.

The state-machine test now covers injection of a CCD root hidden between two
positive endpoint gaps. The next geometry integration test must actually move two
triangles through each other between those endpoints. It passes only if the CCD
backend finds the intermediate root, the solver restarts there, and the accepted
trajectory contains no unresolved crossing.
