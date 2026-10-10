# VTO Dressing and Physics Audit (2026-07-17)

## Scope

This audit separates four tasks that are often all called virtual try-on:

1. photorealistic 2D appearance transfer;
2. garment construction recovery;
3. dressing initialization and collision-free settling;
4. physical animation.

Passing one task does not imply that the next task is solved.

## What mature systems actually do

### Image VTO is a visual prior, not a simulator

StableVITON learns latent semantic correspondence between garment and person.
IDM-VTON uses parsing, DensePose, agnostic masks, and diffusion conditioning.
FASHN VTON 1.5 is a maskless pixel-space model. These systems output pixels;
they do not recover a material rest metric, simulation mesh, sewing interfaces,
or calibrated elasticity. Their output is useful as silhouette, texture,
occlusion, and semantic evidence only.

Sources:

- https://openaccess.thecvf.com/content/CVPR2024/papers/Kim_StableVITON_Learning_Semantic_Correspondence_with_Latent_Diffusion_Model_for_Virtual_CVPR_2024_paper.pdf
- https://github.com/yisol/IDM-VTON
- https://github.com/fashn-AI/fashn-vton-1.5

### Robust monocular digitization uses garment-class templates

Majithia et al. first choose a known garment type, then fit a fixed-topology
parametric template from garment boundary landmarks and transfer/inpaint its UV
texture. This avoids asking one network to invent arbitrary watertight topology
from one view, but it does not identify hidden construction uniquely.

Source:

- https://openaccess.thecvf.com/content/WACV2022/html/Majithia_Robust_3D_Garment_Digitization_From_Monocular_2D_Images_for_3D_WACV_2022_paper.html

### GarmentCodeData uses patterns, measurements, staged attachment, and XPBD

GarmentCodeData generates body-measured parametric sewing patterns and drapes
them with an extended XPBD pipeline. Its dressing safeguards are directly
relevant to this project:

- lower garments are initially placed at the waist;
- skirt parts are associated with both legs as one collision group;
- body-part collision filters prevent wide skirts from being caught by arms;
- vertices accidentally inside the body are pulled toward a nearby body surface;
- lower garments receive a waist half-space attachment;
- attachment constraints are released after a fixed number of frames so the
  garment can find its natural rise;
- invalid drapes are detected and rejected rather than retained as training data.

Sources:

- https://igl.ethz.ch/projects/GarmentCodeData/
- https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/07721.pdf
- https://github.com/maria-korosteleva/GarmentCode

### Dress-1-to-3 refines patterns through differentiable simulation

Dress-1-to-3 predicts a coarse sewing pattern, synthesizes multi-view evidence,
and refines the pattern with a differentiable garment simulator. The critical
idea is not merely adding diffusion: hidden 3D geometry is constrained by
multiple generated views while pattern parameters are judged through physical
draping. It returns separated simulation-ready garments rather than one fused
clothed-human surface.

Source:

- https://dress-1-to-3.github.io/

### DiffAvatar jointly optimizes body, pattern, and material

DiffAvatar starts from multi-view capture and a reconstructed target mesh, then
uses differentiable simulation to co-optimize body shape/pose, 2D dress pattern,
and material parameters. This supports a key identifiability boundary: one
uncalibrated catalog image is not enough to claim measured density, bending,
stretch, and damping. Material values require observations of deformation,
multiple views or motions, and inverse simulation.

Source:

- https://openaccess.thecvf.com/content/CVPR2024/papers/Li_DiffAvatar_Simulation-Ready_Garment_Optimization_with_Differentiable_Simulation_CVPR_2024_paper.pdf

### DrapeNet and collision-learning methods solve narrower problems

DrapeNet represents garments with unsigned distance fields and learns a
self-supervised draping deformation. It is useful evidence that implicit garment
representations can cover multiple topologies, but an implicit surface does not
by itself encode manufacturing panels, grain, fold pairs, or material calibration.

ContourCraft adds intersection-contour and repulsion losses to neural cloth
simulation and can recover from missed collisions or poor multi-layer states.
It is a recovery layer, not a substitute for correct garment placement, body
attachments, seam construction, or conservative CCD.

Sources:

- https://openaccess.thecvf.com/content/CVPR2023/papers/De_Luigi_DrapeNet_Garment_Generation_and_Self-Supervised_Draping_CVPR_2023_paper.pdf
- https://dolorousrtur.github.io/contourcraft/

## Diagnosis of our current skirt result

The current prototype violated several of these established practices:

1. It generated a radial sinusoidal shell instead of a waistband plus true box-
   pleat construction with fold correspondences and signed rest angles.
2. The first waist ring had a median body gap of 24.5 mm and was pinned in world
   coordinates, so it appeared to float around the body.
3. Material parameters were image-conditioned guesses expressed through legacy
   prototype scales, not calibrated physical measurements.
4. The physics teacher smoothed absolute curvature, causing pressed rest folds
   to be treated as errors. Rest-curvature subtraction has now been added.
5. A long rollout showed that curvature subtraction alone is insufficient;
   rotation-invariant rest-hinge constraints are also required.
6. The displayed NeuralClothSim surrogate error was too large relative to the
   actual skirt motion. Teacher and surrogate must be shown separately.
7. Increasing teacher substeps exposed a five-timescale error in 4D field time
   and velocity conditioning; the field now uses the visible-frame duration.

## Revised project route

1. Use FASHN/StableVITON-like output only as visual silhouette and occlusion
   evidence.
2. Classify the garment and retrieve Top-K GarmentCode-compatible construction
   programs.
3. For this image, instantiate a waistband, front/back skirt panels, and explicit
   box-pleat fold pairs. Preserve seam, grain, and fold provenance.
4. Scale pattern parameters from body measurements and uncertain image ratios;
   do not infer metric scale from pixels alone.
5. Arrange panels around body placement volumes. Apply temporary waist
   half-space attachment and lower-body collision filters, then release it after
   sewing and initial settling.
6. Use XPBD/IPC with membrane, shear, signed rest-dihedral bending, body CCD,
   self-collision, and drape rejection as the trajectory teacher.
7. Use Dress-1-to-3-style multi-view diffusion evidence to optimize pattern
   parameters. Estimate material only when motion or multi-state observations
   make inverse identification possible; otherwise report intervals/presets.
8. Weld the validated sewn garment into a continuous runtime shell and build its
   intrinsic cotangent Laplacian, mass matrix, spectrum, and rest-hinge operators.
9. Distill only validated physical trajectories into the E(3)-equivariant 4D
   field. Keep the physical teacher as the reference artifact.
10. Add ContourCraft-style intersection contour loss as recovery/training
    supervision, while retaining explicit SDF/CCD event detection.

## Decision

The intrinsic-shell + E(3) 4D field remains useful, but it moves downstream of
construction, dressing, and validated physics. It must not be used to invent a
box-pleated skirt directly from a single image or to replace garment-specific
attachment and rest-fold semantics.
