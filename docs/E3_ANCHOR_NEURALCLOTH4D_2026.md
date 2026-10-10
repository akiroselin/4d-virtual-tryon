# E(3)-Equivariant Anchor NeuralCloth4D Field (2026-07-17)

## 1. Decision

Replace the direct scene-field vector decoder

```text
(panel, material coordinate, physical time, world xyz, context) -> R3 displacement
```

with an anchor field whose learned networks emit invariant scalars only. Every
3D output is assembled from declared polar-vector (`1o`) bases. The field remains
one optimized multi-chart 4D representation per fixed cloth/body/material/motion
scene, but changing the global Euclidean frame no longer changes its physical
prediction.

The implementation is
[`e3_anchor_field.py`](../aigc_fashion/e3_anchor_field.py). The integration bridge
is [`cloth4d_geometric_bridge.py`](../aigc_fashion/cloth4d_geometric_bridge.py).

## 2. Field Definition

For panel `p`, immutable material coordinate `xi`, and physical time `t`, define
a canonical affine reference from same-panel cloth anchors:

```text
rho_qa = softmax_a(-||xi_q - xi_a||^2 / (2 sigma_xi^2)), p_q = p_a
c_bar_q = sum_a rho_qa C_a
```

The vector bases are

```text
r_cloth_qa = C_a - c_bar_q
r_body_qb  = B_b(t) - c_bar_q
n_body_qb  = outward body-SDF gradient at B_b(t)
v_body_qb  = body-anchor velocity
h_touch_qb = reconstructed gauge-free tangential tactile vector
g_ql       = typed query vectors: particle force, scene message, gravity, wind
```

Scalar networks consume only material/time/panel features, anchor scalar
features, norms, dot products, and relative material coordinates. They predict
attention and signed amplitudes. The residual is

```text
d_q = sum_a alpha_qa unit(r_cloth_qa)
    + sum_b beta^r_qb unit(r_body_qb)
    + sum_b beta^n_qb unit(n_body_qb)
    + sum_b beta^v_qb unit(v_body_qb)
    + sum_b beta^h_qb unit(h_touch_qb)
    + sum_l gamma_ql unit(g_ql)
```

and the 4D field is

```text
x_q(t) = c_bar_q + t v_q(0) + B_p(xi_q) I(t) d_q(t)
I(t)   = 1 - exp(-(t / tau_0)^2)
```

Therefore `x(0)=c_bar` and `partial_t x(0)=v(0)` exactly. `B_p` is a true
positional-boundary mask, not a substitute for seam constraints. Smooth
`0.1 mm` direction regularization prevents floating-point noise at coincident
query/anchor locations from becoming a finite unit vector.

## 3. E(3) Contract

For any `Q in O(3)` and translation `a`, transform all affine points and all
polar vectors together:

```text
C' = Q C + a          B' = Q B + a
n' = Q n              v' = Q v
h' = Q h              g' = Q g
```

Material coordinates, panel IDs, physical time, latent material features,
distances, and dot products are unchanged. Hence every scalar coefficient is
unchanged, `c_bar' = Q c_bar + a`, `d' = Q d`, and

```text
x'(p, xi, t) = Q x(p, xi, t) + a.
```

Body normals in this contract are polar SDF-gradient normals. A normal produced
solely by a fixed mesh-winding cross product has different reflection parity and
must be converted to the outward SDF convention before entering the field.

## 4. Mandatory Geometric Branches

### 4.1 Particle dynamics: E(3) message passing

[`geometric_cloth_prior.py`](../aigc_fashion/geometric_cloth_prior.py) implements
the lowest-order e3nn/SEGNN path:

```text
relative displacement x_j - x_i : 1x1o
invariant material/edge features : 0e
message                           : scalar * unit(x_j - x_i)
```

Messages are pairwise antisymmetric, so learned internal force sums to zero. The
force is both a typed query-vector basis for the field and a term in the physical
dynamics residual. It remains a learned residual/prior, not the orthotropic shell
constitutive law. See [EGNN](https://proceedings.mlr.press/v139/satorras21a.html)
and [SEGNN](https://arxiv.org/abs/2110.02905).

### 4.2 Soft-surface features: intrinsic DiffusionNet

[`intrinsic_diffusion.py`](../aigc_fashion/intrinsic_diffusion.py) propagates
cloth-anchor scalar features with the material-surface Laplace-Beltrami spectrum:

```text
H_tau f = Phi exp(-Lambda tau) Phi^T M f.
```

The resulting intrinsic features enter scalar coefficient networks. Ambient raw
xyz never enters this propagation path. DiffusionNet is parameterization-robust
feature transport; shell membrane/shear/bending energies remain separate. See
[DiffusionNet](https://arxiv.org/abs/2012.00888).

### 4.3 Body trajectory: Lie-group diffusion

[`lie_group_diffusion.py`](../aigc_fashion/lie_group_diffusion.py) now provides a
discrete right-invariant SE(3) random-walk DDPM:

```text
T_k = T_(k-1) Exp(sqrt(beta_k) S epsilon)
```

with distinct rotation/translation metric scales, right-trivialized transition
scores, and a reverse geodesic step. It is a local wrapped-Gaussian approximation,
not an exact closed-form SE(3) heat kernel. The score head operates on relative
body-frame twists. `body_anchors_from_se3()` attaches local body anchors to the
generated link-pose trajectory without flattening poses into `R7`. See
[Riemannian Score-Based Generative Modelling](https://arxiv.org/abs/2202.02763).

### 4.4 Tactile field: SO(2) gauge transport

Genesis taxel shear is projected into charge-1 complex tangent features. The
`GaugeTactileVectorEncoder` applies SO(2) parallel transport and shared real
channel mixing, then reconstructs a gauge-independent 3D tangential polar vector.
Changing every local tangent frame independently leaves the field position
unchanged. See [Gauge Equivariant Mesh CNNs](https://arxiv.org/abs/2003.05425).

### 4.5 Whole scene: hierarchy-aware 4D attention

Object, cloth, body, contact, sensor, and temporal-copy nodes form one scene
graph. [`spatiotemporal_scene_graph.py`](../aigc_fashion/spatiotemporal_scene_graph.py)
uses relative displacement, distance, relative physical time, hierarchy-level
difference, and relation type. Edge-conditioned scalar gates and vector
coefficients make hierarchy affect messages even when a receiver has one edge.

```text
scene scalar output -> anchor coefficient context
scene 1o output     -> typed query-vector bank
```

The scene layer reasons over object/time relations. It does not replace
per-edge shell physics or continuous NDF time derivatives.

## 5. Runtime Path

```text
multi-chart material mesh
  -> Laplace-Beltrami spectrum -> intrinsic anchor scalars
sampled cloth graph
  -> relative 1o particle messages -> internal-force vectors
4D hierarchy scene graph
  -> invariant scalars + equivariant vectors
SE(3) body diffusion trajectory
  -> moving body positions/normals/velocities
SO(2) taxel graph
  -> gauge-free body tactile vectors
all typed branches
  -> E3AnchorNeuralClothField
  -> x(p, xi, t), partial_t x, partial_tt x
  -> shell/seam/contact/Genesis/observation objective
  -> explicit zero-crossing event detector and IPC validation
```

`query_conditioned_cloth4d()` is the single bridge call from all geometric
branches into the field. Field outputs pass directly to
`scene_ndf_physics_objective()`; autograd computes velocity and acceleration with
respect to physical time.

## 6. Topology, Seams, And Events

- Cloth anchors never change panel identity or immutable material coordinate.
- Same-panel anchor interpolation cannot silently merge front/back/sleeve charts.
- Seams remain explicit paired atlas maps and losses.
- Ordinary folds and contact do not remesh or change material topology.
- The existing SDF/VF/EE zero-crossing detector still creates event-localized
  samples, active contacts, fold-graph updates, and IPC correction requests.
- Anchor attention is not a collision solver and cannot approve penetration.

## 7. Verified Properties

The tests cover:

- translation, proper rotation, and reflection equivariance;
- invariance to cloth/body/query-vector anchor permutation;
- exact initial position and velocity plus finite second time derivatives;
- end-to-end gradients through the Genesis-constrained NDF objective;
- spectral caps on every affine coefficient path;
- DiffusionNet, particle-message, tactile, and scene-graph bridge composition;
- independent SO(2) tangent-frame changes;
- global-left SE(3) equivariance of diffusion and body-anchor attachment;
- global scene-time shift invariance and effective hierarchy conditioning.

## 8. Remaining Production Work

This implements the representation and its geometric contracts. It does not yet
train a real T-shirt scene. Production still requires a G0-G4-valid sewn shirt,
calibrated orthotropic shell v2, moving Anny/MHR link anchors, sparse anchor
neighborhoods for scale, robust self/body contact, Genesis trajectory IO,
differentiable observations, and comparison with real cloth plus IPC. The old
coordinate NDF/CortexODE remains only as a frozen ablation baseline.
