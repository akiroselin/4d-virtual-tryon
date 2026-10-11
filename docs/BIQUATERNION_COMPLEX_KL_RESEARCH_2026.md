# Biquaternion Extended Phase and Complex Kirchhoff-Love Research (2026-07-19)

## Decision

A single real quaternion cannot contain vertex displacement, momentum, and
time without information loss. Those variables have seven real components.
The project therefore uses a complex quaternion (biquaternion) chart of the
eight-dimensional extended phase state:

```text
Z_i = (t, delta_q_i) + i (p_t_i, p_i)
sum_i p_t_i + H = 0
```

The additional scalar is momentum conjugate to time. `Z_i` is not an attitude
quaternion and must never be unit-normalized. Body and shell-frame rotations
continue to use unit quaternions or `SE(3)` where appropriate.

The Hamiltonian remains a real scalar. Quaternion/complex coordinates organize
the state and its symmetry; they do not make physical energy quaternion-valued.

## Complex Kirchhoff-Love representation

For a symmetric material-chart strain tensor, define

```text
trace = e11 + e22
deviator = (e11 - e22) + 2 i e12
```

Under an `SO(2)` material-frame rotation, the deviator is a charge-2 complex
field. Its magnitude and the trace are invariant, so an isotropic Hermitian
audit energy can be written as

```text
W = 1/2 K trace^2 + 1/2 G |deviator|^2.
```

The production constitutive energy remains the nonlinear orthotropic
Hyper-NODE Kirchhoff-Love model. Replacing it by the quadratic complex form
would lose warp/weft asymmetry, area response, and nonlinear material behavior.
The complex form is currently an invariant representation and audit path.

The existing hinge model supplies one signed dihedral curvature per edge, not a
full symmetric face-curvature tensor. A complex bending field must wait until a
face-based curvature reconstruction is implemented; inventing a missing tensor
from a scalar hinge angle would be mathematically misleading.

## CortexODE boundary

CortexODE deformation time is reconstruction homotopy time. It may transport
the biquaternion chart and minimize zero-momentum shell energy, but its RK4
samples must not be interpreted as physical momentum. Physical `p`, `p_t`, and
action stationarity start only on trajectories with real timestamps.

## Primary evidence

- Brugnoli et al., *Port-Hamiltonian formulation and symplectic discretization
  of plate models, Part II: Kirchhoff model for thin plates*:
  https://arxiv.org/abs/1809.11136
- Hwu, *Stroh-Like Complex Variable Formalism for the Bending Theory of
  Anisotropic Plates*: https://doi.org/10.1115/1.1600474
- Shen and Leok, *Lie group variational integrators for rigid body problems
  using quaternions*: https://arxiv.org/abs/1705.04404
- Xu and Halse, *Dual Quaternion Variational Integrator for Rigid Body Dynamic
  Simulation*: https://arxiv.org/abs/1611.00616
- Zhang and Zhong, *A weak form quadrature element formulation for
  geometrically exact thin shell analysis*:
  https://doi.org/10.1016/j.compstruc.2018.03.002
- Seslija, van der Schaft, and Scherpen, *Reduction of Stokes-Dirac structures
  and gauge symmetry in port-Hamiltonian systems*:
  https://arxiv.org/abs/1206.3781

These references support quaternionic rotation variables, complex material
fields, and port-Hamiltonian shell structure separately. They do not support
packing all cloth phase variables into one ordinary quaternion.
