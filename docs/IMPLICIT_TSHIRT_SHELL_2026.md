# Continuous Implicit T-shirt Shell

## Decision

The pipeline now keeps two garment representations instead of forcing one mesh
to satisfy incompatible manufacturing and simulation requirements.

1. **Manufacturing atlas:** learned SewFormer/GarmentCode-style 2D panels,
   ordered seam interfaces, grain/material coordinates, and editable sizing.
2. **Simulation shell:** one continuous explicit triangle shell extracted from
   a placement-conditioned body-distance field. It has no runtime seam springs.

Removing seams from the simulation representation does not mean that a physical
T-shirt can be manufactured without seams. It means that already-sewn cloth is
simulated as one midsurface, while its cuts and stitches remain provenance data.

## Geometry

The current extractor in `aigc_fashion/implicit_shell.py` uses the Anny body SDF
and learned T-shirt extent to define a garment field conceptually as

\[
\phi_g(x)=d_{body}(x)-r_{ease}, \qquad x\in M_{torso\cup upper-arm}.
\]

The zero-distance body triangulation is used as an adaptive sampling of that
field, restricted by the torso and upper-arm arrangement volumes and displaced
outward by the wearing-ease level. This avoids a dense voxel grid while retaining
the same level-set contract. A production extractor can replace this adaptive
sampling with octree marching tetrahedra without changing downstream physics.

The extracted T-shirt must satisfy:

- one connected component;
- no non-manifold or isolated edges;
- four boundary loops: neck, hem, left cuff, right cuff;
- Euler characteristic `-2` and genus `0`;
- zero body-SDF penetration at vertices and sampled triangle interiors.

Boundary loops are regularized after topology-preserving decimation. The hem is
planar and cuffs are projected to their upper-arm arrangement planes. The neck
mask is generated before extraction from the learned `12.89 cm` neckline width:
the front centre drops `2.6 cm`, while the back drops only slightly, producing a
crew neck instead of a constant-height ring around the avatar's neck.
The sleeve offset tapers from `16 mm` at the shoulder to `8 mm` at the cuff. For
the current learned pattern this reduces the extracted cuff circumference from
about `27 cm` to `22.5 cm`; the learned `19.38 cm` opening is below the feasible
arm-plus-contact circumference and is therefore retained as inverse-fit evidence
rather than forced into an intersecting initial state.

The Anny asset faces Blender `-Y`. Production Anny runs map source
`(x, z, -y)` to simulation `(x, up, front)` explicitly; automatic extent-based
axis selection identifies the depth axis but cannot infer its sign.

## Physics Contract

The continuous shell uses its undeformed 3D surface as the intrinsic material
metric. Each triangle receives a local tangent chart, so Kirchhoff-Love membrane
strain is measured from the curved rest surface rather than a distorted global
UV projection. The 2D learned atlas coordinates still condition the material
Hyper-network NODE.

The runtime combines:

- immutable structural rest lengths and intrinsic Laplace-Beltrami propagation;
- Hyper-network NODE material response at shell material points;
- Kirchhoff-Love membrane and rest-curvature bending energy;
- dynamic equilibrium and free-boundary Piola traction losses;
- cloth/body-anchor E(3)-equivariant 4D field fitting;
- pure body-SDF vertex and sampled-triangle contact;
- explicit SDF zero-crossing event detection.

An empty seam set is an explicit mode switch: semantic front/back panel barriers
are disabled, because applying them to a continuous shell creates artificial
layer separation and severe strain. The retained `seam_atlas` backend continues
to use ordered seam closure and panel-aware collision for comparison.

## Validated Result

The July 2026 Anny child run at 699 faces produced:

| Measure | Panel + runtime seams | Continuous implicit shell |
| --- | ---: | ---: |
| Runtime seam constraints | 76 | 0 |
| Final p95 edge change | +10.7% | +1.6% |
| Final p99 edge change | +18.3% | +3.2% |
| Intersecting cloth faces | 11 | 0 |
| E(3) equivariance max error | not recorded | `1.19e-7 m` |

The physics teacher itself has p95/p99 edge ratios `1.020/1.031`, a final area
ratio of `1.0047`, no penetrating sampled faces, and about `17.5 mm` mean motion
over 12 steps. The 180-step neural field fit reaches about `2.00 mm` temporal
coordinate RMSE while preserving zero final body intersections.

## Reproduce

```bash
/opt/homebrew/anaconda3/bin/python3 examples/run_anny_cloth4d.py \
  --simulation-topology implicit_shell \
  --implicit-shell-target-faces 700 \
  --training-steps 180 \
  --minimum-training-steps 80 \
  --out-dir outputs/anny_implicit_shell_crew_neck_texture_v9
```

The existing panel baseline is preserved with:

```bash
/opt/homebrew/anaconda3/bin/python3 examples/run_anny_cloth4d.py \
  --simulation-topology seam_atlas \
  --placement-mode arrangement_points \
  --out-dir outputs/anny_seam_atlas_baseline
```

## Remaining Boundary

This implementation solves topology, sleeve continuity, initial placement, and
runtime seam-stress failure. It does not infer calibrated cotton parameters from
one RGB image, nor does it yet transfer every learned panel grainline and local
ease value onto the implicit shell. Those are inverse-material and atlas-transfer
problems, not reasons to reintroduce runtime seam springs.

Nyx-ready PBR assets are exported by `examples/render_anny_cloth4d_nyx.py`.
Collar, body, and sleeve use one white-cotton base material, while only the front
print primitive receives the dog texture. Genuine Nyx rendering requires the
official CUDA backend and cannot execute on the current Apple Silicon host.
