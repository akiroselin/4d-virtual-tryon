# Same-model, same-garment VTO benchmark

## Fixed inputs

- Person A: the same Anny body wearing the validated physical pleated skirt.
- Person B: the same Anny body wearing the NeuralClothSim4D T-shirt.
- Top product: the original dog-print T-shirt image.
- Bottom product: the original `19_189.webp` pleated-skirt image.
- Target outfit: the dog-print T-shirt and pleated skirt on the same Anny body.

The two FASHN routes intentionally reverse operation order. A mature 2D VTO
should preserve the already worn garment while replacing only the requested
category:

1. Person A + top product, category `tops`, photo type `flat-lay`.
2. Person B + bottom product, category `bottoms`, photo type `flat-lay`.

## Selected mature baselines

FASHN VTON v1.5 is the primary executable baseline because its official code
and weights are Apache-2.0, it is maskless, and it explicitly supports model and
flat-lay garment inputs for tops, bottoms, and one-pieces:
https://github.com/fashn-AI/fashn-vton-1.5

IDM-VTON is a strong garment-fidelity research reference. Its architecture uses
separate high-level and low-level garment encoders, but the repository does not
provide a clear commercial license boundary for this project:
https://idm-vton.github.io/

CatVTON is the lightweight research reference. Its official implementation
reports less than 8 GB VRAM at 1024x768, but code, checkpoints, and demo are
CC BY-NC-SA 4.0 and therefore unsuitable for the commercial route:
https://github.com/Zheng-Chong/CatVTON

## Current observed result

The NeuralClothSim4D result has explicit T-shirt and skirt meshes, separate
material coordinates, body alignment, reusable GLB geometry, and six synchronized
animation frames. The T-shirt/skirt overlap-band nearest-vertex distance is
16.44 mm. Both garments are manifold and the body-intersection audits are zero.

Its visual weaknesses are equally clear: the T-shirt silhouette is too boxy,
the low-resolution shell exposes polygonal shoulders and hem, texture placement
is rectangular, and the short trajectory does not yet show convincing fine
wrinkles or cloth inertia. The skirt preserves its pleat topology but lacks the
high-frequency appearance of a photographed wool blend.

## Comparison criteria

| Criterion | NeuralClothSim4D | Mature 2D diffusion VTO |
|---|---|---|
| Photorealistic pixels | Currently weak | Expected strength |
| Garment print/plaid fidelity | Moderate | FASHN/IDM design target |
| Body/garment occlusion | Explicit geometry, coarse render | Learned image prior |
| Metric size and ease | Explicit SI-unit mesh | Not observable from pixels |
| Body and garment collision | Auditable | No physical guarantee |
| Temporal dynamics | Physical state trajectory | Single-image output unless extended to video |
| Novel viewpoint | Mesh can be re-rendered | Usually view-specific |
| Manufacturing rest metric | Preserved | Absent |

## Execution status

The exact benchmark inputs and official runner are prepared in
`outputs/vto_benchmark_same_model` and
`examples/run_fashn_same_input_benchmark.py`. The mature output has not been
fabricated: this Mac's PyTorch build exposes neither CUDA nor MPS, the released
model is about 2 GB, and the current sandbox blocks the LFS weight transfer and
automatic local-file selection in the public Space. A real result requires the
official weights on a CUDA worker or a FASHN API credential.
