# Geometric Phase Graph (2026-07-19)

## Scope

The NeuralClothSim runtime no longer treats the 4D scene as one flat XYZ latent.
Its state is a product of the manifolds appropriate to each physical object:

```text
T*Emb(S_cloth, R3) x T*SE(3)^B x C_contact x R_time
```

This is a unified typed phase system, not one global coordinate chart.

## Cloth phase

Each cloth vertex carries:

```text
(material_uv, q, p=Mv, lumped_mass)
```

`material_uv` and the rest metric are immutable. The embedding `q` is an affine
E(3) point and `p` is canonical momentum. Learned scalar networks do not consume
the components of `q` or `p` directly.

The current step is canonical symplectic Euler:

```text
p_next = p + dt F
q_next = q + dt M^-1 p_next
```

The conservative force is supplied by the Hyper-NODE Kirchhoff-Love membrane
and bending energy. Gravity is an external potential; damping, friction,
acceleration safety caps, and metric projection are explicit non-Hamiltonian
ports or constraints.

## Reference-map fibre state

The cloth phase now separates material memory from spatial orientation:

```text
xi = (sheet_id, u, v)                         immutable material label
F = partial x / partial xi                    deformation gradient
g = F^T F                                    stretch/shear metric
q_frame in Spin(3)                           local rotational fibre
```

`xi`, the inverse material matrices, and the rest metric are copied from the
simulation-ready intrinsic shell and never advected as world XYZ. The unit
quaternion carries only the orthonormal surface frame. It cannot encode stretch
or shear, so these remain in the symmetric metric. A rigid E(3) motion changes
`q_frame` while leaving `g` invariant; pure stretch changes `g` without being
misrepresented as quaternion rotation.

This is the discrete reference-map interpretation used by the runtime. It is
not a claim that quaternion multiplication replaces the deformation gradient.

## Bundle-valued shell measure

The executable state is now also lifted into a graded empirical measure on the
material-spatial bundle. It is deliberately not a scalar density voxelized in
world space:

```text
mu = mu_vertex direct-sum mu_face direct-sum mu_hinge

mu_vertex = sum_i m_i delta_(sheet_i, xi_i, x_i, p_i)
mu_face   = sum_f A0_f delta_(sheet_f, xi_f, x_f, F_f, g_f, q_frame_f)
mu_hinge  = sum_h w_h  delta_(sheet_h, xi_h, x_h, kappa_h)
```

Vertex atoms carry positive lumped mass and canonical momentum. Face atoms use
rest-area quadrature and carry the full deformation gradient, metric, and
Spin(3) frame. Hinge atoms carry discrete Kirchhoff-Love curvature strain.
Keeping these strata separate is essential: averaging membrane or bending
energy into a world-space vertex density would lose the shell's differential
structure. `sheet_id` and `xi=(u,v)` also prevent two spatially adjacent layers
of a fold from being identified as one material point.

The real measure Hamiltonian is assembled without changing quadrature:

```text
H[mu] = integral |p|^2/(2m) dmu_vertex
      + integral W_HyperNODE(F,g,z_m) dmu_face
      + integral B_HyperNODE(kappa,z_m) dmu_hinge
      + U_gravity + U_seam + U_contact.
```

Continuous and impulsive ports share the distributional transport statement

```text
dmu + Div_E(mu[(J-R) delta_H/delta_mu + G_u u] dt
             + mu G_C d_lambda) = 0.
```

The runtime's conservative vertex transport is the canonical kick-drift step,
which exactly preserves atom mass and material labels. SDF contact is the
push-forward `mu_plus=(R_contact)_#mu_minus`; at the event instant the reset
acts only on the momentum fibre. Metric projection and attachment remain
explicit constraint ports. Reports expose atom counts, total mass, reference
area, material-coordinate drift, mass drift, momentum, and every Hamiltonian
term so the representation is auditable during single- and multi-garment runs.

### Measure-level training

The scene-optimised field now evaluates this representation on each physical
three-frame training sample. The incoming and outgoing half-step velocities are

```text
v_minus = (q_k - q_(k-1)) / dt
v_plus  = (q_(k+1) - q_k) / dt,
```

so `mu_k=(q_k,M v_minus)` and `mu_(k+1)=(q_(k+1),M v_plus)` provide a
non-vacuous canonical momentum residual. The loss is

```text
L_measure = L_continuity
          + alpha L_power-balance
          + beta  rho(relu((Delta H/dt - P_supplied)/P0))
          + gamma rho(relu(P_dissipative/P0)),

rho(r) = sqrt(1 + r^2) - 1.
```

`L_continuity` includes mass and material-label conservation, position
transport, and momentum balance. `H` is re-evaluated from native vertex, face,
and hinge quadrature at the next field sample. Gravity remains inside `H`;
smooth normal contact is supplied as a constraint port; friction and Rayleigh
damping must have non-positive power. Runtime zero-crossing impulses are not
smoothed into this loss and remain explicit momentum-fibre push-forwards. The
one-sided pseudo-Huber penalty preserves the exact passive feasible set while
preventing an uncalibrated initial field from overwhelming trajectory supervision.
The training history records the total, continuity, power-balance, passivity, and
dissipation-sign terms independently.

## Generalized flows

Interactions share a power contract rather than a single coordinate type:

```text
P = <effort, flow> in R
```

The pairing is an E(3) dot product for force-velocity fields, the real Hermitian
pairing for SO(2) charge-1 tangent fields, and the wrench-twist pairing for an
SE(3) rigid port. Conservative, dissipative, control, constraint, and bounded
learned-residual roles remain explicit. The runtime ledger currently records:

- Hyper-NODE Kirchhoff-Love, seam elasticity, and gravity as conservative flow;
- Rayleigh damping, seam damping, contact damping, and gauge friction as
  dissipative flow;
- moving-body contact work as a control flow;
- E(3) learned residual force as a separately bounded learned flow;
- acceleration caps, attachment constraints, and metric/contact projection as
  constraint flow.

Collision and projection are measure-valued interactions:

```text
p_plus = p_minus + impulse
W_jump = K_relative(p_plus) - K_relative(p_minus).
```

Normal contact impulse remains an E(3) vector. Tangential Coulomb impulse is
projected onto an SO(2) local frame and stored as a complex charge-1 field.
These orthogonal work terms are recorded once each. Event localization uses a
ledger checkpoint, so work from a rejected speculative step is discarded.

## Body and contact fibres

Rigid bodies use poses on `SE(3)` and body momenta in its cotangent space.
Contact uses an SDF scalar, a polar normal, and charge-1 complex tangent momentum
under the local SO(2) gauge. Contact establishment does not change cloth mesh
topology.

At an SDF zero crossing, position is localized at the event and momentum jumps:

```text
p_plus = p_minus + Jn^T lambda_n + Jt^T lambda_t
```

The implemented reset removes inward normal relative velocity and applies a
bounded gauge-equivariant Coulomb friction scale. It is dissipative for a static
body and leaves inactive vertices unchanged.

## Variational phase PINN

After trajectory pretraining, the 4D field is refined with a discrete
Hamilton-d'Alembert PINN residual on three consecutive phase samples. It uses
the canonical momentum defect

```text
r_p = (p_next - p_previous) / dt
      - (F_KL + F_gravity + F_contact + F_friction + F_damping)
```

rather than minimizing the scalar action itself. Hamilton's principle is a
stationarity principle, and friction/contact make this a nonsmooth dissipative
system, so minimizing the action value would be physically wrong. The training
objective contains:

- discrete action stationarity / Euler-Lagrange residual;
- local work-energy balance;
- dimensionless Fischer-Burmeister SDF contact KKT residual;
- non-positive friction and damping power;
- the existing free-edge traction and hard post-step metric projection.

The constrained Hamiltonian is compiled explicitly as

```text
H_c(q,p;z) = 1/2 p^T M^-1 p
             + U_HyperNODE-KL(q; z_material)
             + U_gravity(q)
             + I_Cmetric(q)
```

`I_Cmetric` is zero inside the material-specific strain envelope and infinite
outside it. Training uses a differentiable violation barrier for this indicator;
runtime uses its proximal map, the edge-colored hard metric projection. Both
read the same `material_strain_limits` result. Metric projection is therefore
part of the constrained Hamiltonian formulation without being mislabeled as a
smooth conservative force.

All residual scalars are built from norms and inner products, so simultaneous
E(3) transformation of positions, forces, normals, and body motion leaves the
loss unchanged. Scene hierarchy enters through the existing typed
spatiotemporal conditioning. Multi-garment training can additionally aggregate
per-object residuals with hierarchy weights without flattening their phase
charts.

The pretrained 4D field supplies the warm start. The PINN objective refines its
positions and material-conditioned flow; it does not replace the shell rest
metric or infer material measurements that are absent from the input.

### CortexODE boundary

CortexODE reconstruction time is a deformation homotopy, not physical time.
The intrinsic-shell stage therefore compiles the Hamiltonian contract and uses
its zero-momentum potential/equilibrium terms to select a physically admissible
static embedding. It does not apply inertial action equations between CortexODE
RK4 samples. Discrete action stationarity starts only when three samples have
real timestamps. This prevents reconstruction speed from being misinterpreted
as cloth velocity or momentum.

## Relation to the 4D graph

Hierarchy-aware spatiotemporal attention uses relative displacement, relative
time, hierarchy-level difference, and relation type. It can condition future
Hamiltonian or dissipation coefficients, but the current code does not claim a
trained neural Hamiltonian. Analytic shell energy remains authoritative.

## Intrinsic complex Hamiltonian chart

The 4D graph now has a complex energy-analysis layer, but not one global flat
complex coordinate system. Each typed node keeps the chart of its physical
manifold. A cloth cotangent pair is represented locally by the mass-scaled
Kahler coordinate

```text
Q_i = sqrt(m_i omega_ref) (q_i - q_ref_i)
P_i = p_i / sqrt(m_i omega_ref)
z_i = (Q_i + i P_i) / sqrt(2).
```

This transform is symplectic. Translation disappears through the reference
embedding and an `SO(3)` rotation acts equally on the real and imaginary vector
parts, so no learned scalar has to infer invariance from raw XYZ. The arbitrary
positive `omega_ref` only balances position and momentum units; decoding is
lossless and physical results may not depend on its value.

The physical Hamiltonian is necessarily real and is therefore written as the
non-holomorphic function

```text
H_scene = H(z, z_bar) in R
dot(z) = -i partial H / partial z_bar.
```

A nonconstant real-valued holomorphic Hamiltonian cannot exist on a connected
complex domain. Wirtinger derivatives are consequently the correct complex
calculus here. Node energies, relation energies, and constraint energies are
aggregated by graph hierarchy:

```text
H_scene = sum_v H_v + sum_e H_e + H_constraints.
```

Relation energy is charged to the deeper endpoint for hierarchy diagnostics,
without changing the exact total. Cloth uses the Kahler chart above; rigid
nodes remain on `T*SE(3)` with local Lie-algebra coframes; contact tangents remain
charge-1 SO(2) complex fields. Their real scalar energies can share the ledger
without falsely identifying these different manifolds.

Irreversible effects use a port-Hamiltonian extension

```text
dot(z) = -i partial_(z_bar) H - R partial_(z_bar) H + G u,
R >= 0,
P_port = Re <effort, flow>.
```

Rayleigh damping and friction use `R` or dissipative ports. SDF contact onset
and collision impulses remain event/reset maps because a smooth conservative
Hamiltonian flow cannot represent a discontinuous momentum jump. Hyper-NODE
and metric projection affect the real Hamiltonian through constitutive energy
and a feasible-set indicator/proximal map; they do not make energy complex.

### Behavioral hierarchy and external work

Personality and emotion are E(3)-invariant scalar context, not cloth material
latents. OCEAN personality attaches to the person/root node and time-varying VAD
emotion attaches to the body node. The explicit hierarchy is:

```text
person/root -> body -> garment -> panel
                   \-> contact -> cloth momentum reset
```

Three attention layers allow root information to reach panel nodes without a
flat global latent. The physical route is a bounded body-control port that
modulates cadence, stride, sway, and gesture. Its work enters cloth through body
motion and SDF contact. It never edits rest metric, material latent, shell energy,
or strain limits. The bundled behavioral-to-gait mapping is only an engineering
simulation prior and must be replaced or calibrated for behavioral research.

## Implemented contracts

- `IntrinsicClothPhaseState`: `T*Emb(S_cloth,R3)` state;
- `SE3CanonicalPhaseState`: rigid `T*SE(3)` state;
- `SO2ContactPhaseState`: gauge-typed contact fibre;
- `GeometricPhaseGraphState`: synchronized product state;
- `symplectic_euler_phase_step`: canonical kick-drift update;
- `apply_sdf_contact_momentum_reset`: contact jump map;
- `variational_phase_pinn_residual`: Hamilton-d'Alembert, energy, contact KKT,
  and dissipation residual;
- `KahlerClothPhase`: lossless mass-scaled complex cloth phase chart;
- `assemble_complex_scene_hamiltonian`: hierarchy-aware real energy ledger;
- `wirtinger_hamiltonian_vector_field`: complex Hamiltonian flow;
- `port_hamiltonian_vector_field`: positive-semidefinite dissipation and ports;
- `ReferenceMapFiberBundle`: immutable material map, shell metric, and Spin(3)
  orientation fibre sampled on the evolving embedding;
- `BundleMeasureBuilder`: exact vertex/face/hinge empirical-measure lift of a
  Kirchhoff-Love intrinsic shell;
- `assemble_bundle_measure_hamiltonian`: kinetic, membrane, bending, gravity,
  seam, and contact energy on their native measure strata;
- `symplectic_euler_measure_step`: mass- and material-label-preserving measure
  transport;
- `momentum_fibre_push_forward`: contact reset as an empirical-measure
  push-forward;
- `bundle_measure_port_hamiltonian_loss`: differentiable continuity,
  power-balance, passivity, and dissipation supervision for the 4D field;
- `GeneralizedFlowLedger`: typed continuous power and impulse-work audit with
  dissipative sign checks and event rollback;
- typed-geometry manifest entry `geometric_phase_graph`.

## Remaining work

- replace the current scalar/diagonal complex dissipation coefficient by a
  learned positive-semidefinite intrinsic operator `R`;
- differentiate event time and impulse solves with a saltation/reset Jacobian;
- add rigid-body kinetic energy and coupled cloth-body contact work accounting;
- replace the aggregate inter-garment projection audit with full VF/EE
  pairwise effort-flow ports after IPC integration;
- train any learned Hamiltonian correction only as a bounded residual to the
  analytic shell energy.

## Primary references

- Brugnoli et al., *Port-Hamiltonian formulation and symplectic discretization
  of plate models, Part II: Kirchhoff model for thin plates*:
  https://arxiv.org/abs/1809.11136
- Distributed-parameter port-Hamiltonian PDE representation and the
  Stokes-Dirac construction used here for energy ports:
  https://arxiv.org/abs/1207.4732
- Seslija, van der Schaft, and Scherpen, *Reduction of Stokes-Dirac structures
  and gauge symmetry in port-Hamiltonian systems*:
  https://arxiv.org/abs/1206.3781
- Desai et al., *Symplectic structure-aware Hamiltonian graph neural networks*:
  https://arxiv.org/abs/2309.04885
