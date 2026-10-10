# Cortical Surface Reconstruction Transfer to Garments

## Bottom Line

Fast cortical reconstruction is useful as an amortised geometry-evidence and
initialisation branch. It must not directly define a garment's final topology
or material metric. Cortical models assume one or two fixed genus-0 surfaces;
a T-shirt has multiple open 2D charts, openings, orientation-specific material
axes, and a stitch graph that changes connectivity only during sewing.

The accepted garment route is:

```text
image / multiview evidence
  -> implicit surface and semantic evidence
  -> explicit category-template mesh candidate
  -> projection to learned 2D panel charts and stitch interfaces
  -> topology, seam, size, and strain preflight
  -> Hyper-NODE Kirchhoff-Love 4D field
  -> Genesis IPC validation when executable
```

## Transfer Audit

### Fast dHCP pipeline

[Ma et al.](https://arxiv.org/abs/2405.08783) combine a multiscale
diffeomorphic deformation network, accelerated surface processing, and
edge/area distortion checks. The reported complete GPU pipeline takes 24 s.

Adopt:

- coarse-to-fine panel deformation as an amortised warm start;
- explicit edge and face-area distortion gates;
- GPU-batched mesh operators and a fast quality-control path.

Do not adopt spherical projection or its genus-0 guarantee for clothing.

### DDT-from-CSR / ACDE

[Wu et al., ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Wu_Drawing_Developmental_Trajectory_from_Cortical_Surface_Reconstruction_ICCV_2025_paper.html)
use an age-conditioned ODE with adaptive steps, limited longitudinal pairs,
single-time samples, distillation, and cycle consistency.

Adopt the distinction between a global scene condition and continuous time,
plus coarse paired trajectories followed by single-frame refinement. Do not
learn arbitrary physical-time steps for cloth: contact and topology events must
still bracket, stop, update, and restart the solver. Standard error-controlled
DOPRI5 remains preferable to an unconstrained learned step controller.

### FetalCSR

[FetalCSR, ICLR 2025 workshop](https://iclr.cc/virtual/2025/34333) fuses
positional coordinates, samples along the surface normal, surrounding-volume
features, and tissue masks with multi-input attention.

Direct garment analogue:

- material-coordinate and panel tokens;
- image or multiview features sampled at the projected garment surface;
- body SDF samples along cloth normals;
- local tactile/contact features from the surrounding 3D region;
- garment/body semantic masks.

This belongs in the geometric evidence conditioner, not in the constitutive
hyper-network. Material properties must not change merely because an occlusion
feature changes.

### DDCSR

[DDCSR](https://arxiv.org/abs/2503.03790) uses an implicit voxel-wise
intermediate surface representation followed by explicit mesh prediction.
For garments, this is a good way to produce an observation-space surface
candidate from uncertain visual data. The candidate must then be fitted to a
category-conditioned panel/stitch template; marching or directly predicting a
closed mesh cannot recover manufacturing topology by itself.

### End-to-end Clinical CSR / BrainNet

[Nielsen et al.](https://arxiv.org/abs/2505.14017) use synthetic domain
randomisation and template-mesh deformation for heterogeneous image contrast
and resolution. They report cortical-thickness error decreasing from 0.50 mm
to 0.24 mm relative to RAC. The public
[BrainNet repository](https://github.com/simnibs/brainnet) is GPL-3.0 and uses
CUDA/Kaolin components.

Adopt synthetic domain randomisation across lighting, background, resolution,
viewpoint, body size, material, and occlusion. Replace the spherical template
with a category-specific panel library. Do not embed GPL BrainNet code in the
commercial garment runtime without a compatible licensing decision.

VoxelMorph, SynthSeg, FastSurfer, and DeepCSR remain useful references for
registration, domain randomisation, and segmentation design, but their trained
anatomical priors and topology assumptions are not garment models.

## Current Implementation Decision

The current v4 adds no mock medical-image encoder. Existing DiffusionNet,
E(3)-anchor, scene-graph, body-SDF, and tactile branches remain the geometric
conditioning path. A real latent-diffusion multiview feature adapter can later
fill the FetalCSR-style image slots. The physics branch already consumes the
validated material charts and remains independent of image semantics.
