# Hyper-network NODE Kirchhoff-Love Cloth

## Decision

NeuralClothSim remains one scene-optimised 4D field. The new constitutive layer
does not replace physical time with another black-box trajectory network. It
maps each 2D material point to a local constitutive Neural ODE and constrains the
4D field with thin-shell mechanics.

The implementation is in
`aigc_fashion/hypernetwork_kirchhoff_love.py` and is connected to the Anny
runtime in `aigc_fashion/anny_cloth4d_simulation.py`.

## Correct Reading of Taç et al.

[Taç et al. (npj Computational Materials, 2026)](https://www.nature.com/articles/s41524-026-02027-8)
do not use the NODE to advance a mesh in physical time. Their NODE maps a
strain-energy invariant to the derivative of an energy term. A hyper-network
maps each material coordinate `X` to all parameters of that point's local
NODE. The inverse objective enforces equilibrium and prescribed traction.

The cloth adaptation keeps that split:

- physical time belongs to the E(3)-equivariant 4D cloth field;
- NODE depth `tau in [0,1]` belongs to a local constitutive response;
- the hyper-network input is the intrinsic sewing-pattern coordinate, panel
  chart, and one scene material latent;
- all learned affine maps use `SpectralLinear`, and every generated NODE matrix
  receives an additional exact spectral-norm cap.

For a scalar shell strain `e`, the implementation integrates the same Lipschitz
ODE from initial states `e` and `0`, then subtracts the two terminal states.
This enforces a zero response at zero strain while preserving response sign.
Three-point Gauss-Legendre quadrature integrates that response to a non-negative
energy.

## Kirchhoff-Love Adaptation

The learned 2D pattern is immutable material geometry. For each valid triangle,

```text
F = Ds Dm^-1
C = F^T F
e_warp  = 0.5 (C_00 - 1)
e_weft  = 0.5 (C_11 - 1)
e_shear = C_01
e_area  = sqrt(det C) - 1
```

These quantities are frame-indifferent. Four local NODE branches produce the
orthotropic membrane resultants. A fifth branch acts on a discrete
Kirchhoff-Love hinge curvature, using adjacent face-normal change and the
standard material-space dual-area weight. The resulting first Piola surface
stress is converted to nodal internal force without ambient-coordinate
features.

The training objective includes:

```text
M x_ddot - f_membrane - f_bending - f_gravity - f_contact = 0
P N = prescribed free-edge traction = 0
```

The dynamic residual is evaluated directly on field samples at
`x(t-dt), x(t), x(t+dt)`. Body contact uses the Anny collision surface and
seams remain explicit constraints. Degenerate 2D triangles are rejected from
physical quadrature and reported rather than assigned a fabricated metric.

## Contracts and Limits

- Material chart identity is separate from front/back visibility semantics. A
  whole sleeve remains one physical chart even when rendering classifies its
  visible vertices as front or back.
- The present body contact model is a differentiable local-plane approximation;
  Genesis FEM.Cloth plus IPC remains the intended high-fidelity contact teacher.
- A single image and one unforced trajectory do not uniquely identify a textile
  law. Current stiffness bounds are weak cotton-jersey priors. Production
  calibration needs multiple motions plus measured or trusted simulated forces.
- The macOS ARM runtime cannot execute Genesis IPC because no compatible
  `pyuipc` package is installed. Stock Genesis PBD is not substituted.
- The canonical-metric post-corrector remains only a collision/seam stabiliser.
  Every material-law metric and reported KL strain uses the immutable 2D pattern.

## Tests

`tests/test_hypernetwork_kirchhoff_love.py` verifies the stress-free state,
positive energy, zero anchoring, sign preservation, rigid-motion invariance,
E(3)-equivariant forces and tractions, degenerate-face rejection, and end-to-end
gradient flow through dynamic equilibrium.
