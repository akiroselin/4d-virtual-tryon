from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Protocol

import torch
from torch import Tensor


class ContactSampleLike(Protocol):
    closest_points: Tensor
    normals: Tensor
    body_velocities: Tensor | None


ContactQuery = Callable[[Tensor, float], ContactSampleLike]


@dataclass(frozen=True)
class PhasePotentialTerms:
    """Real-valued energy densities integrated on the reference cloth measure."""

    hypernode_membrane: Tensor
    hypernode_bending: Tensor
    hypernode_strain_limit: Tensor
    gravity: Tensor
    seam: Tensor
    trim: Tensor
    contact: Tensor
    contact_minimum_gap: Tensor | None = None
    contact_quadrature_points: int = 0
    # Kept as an audit term. Non-smooth multi-garment contact is exchanged
    # through the shared constraint port, not counted twice in each local H.
    inter_garment_contact: Tensor | None = None
    se3_generative: Tensor | None = None

    @property
    def total(self) -> Tensor:
        return (
            self.hypernode_membrane
            + self.hypernode_bending
            + self.hypernode_strain_limit
            + self.gravity
            + self.seam
            + self.trim
            + self.contact
            + (self.se3_generative if self.se3_generative is not None else 0.0)
        )

    def detached_energies(self) -> dict[str, float]:
        values = {
            "total_potential_j": self.total,
            "hypernode_membrane_j": self.hypernode_membrane,
            "hypernode_bending_j": self.hypernode_bending,
            "hypernode_strain_limit_j": self.hypernode_strain_limit,
            "gravity_j": self.gravity,
            "seam_j": self.seam,
            "trim_j": self.trim,
        "contact_j": self.contact,
        "inter_garment_contact_j": (
            self.inter_garment_contact
            if self.inter_garment_contact is not None
            else self.contact.new_zeros(())
        ),
        }
        if self.se3_generative is not None:
            values["se3_generative_j"] = self.se3_generative
        return {name: float(value.detach()) for name, value in values.items()}


PotentialFunction = Callable[[Tensor], PhasePotentialTerms]
FeasibilityFunction = Callable[[Tensor], bool]
DirectionPreconditioner = Callable[[Tensor], Tensor]


@dataclass(frozen=True)
class IncrementalHamiltonianConfig:
    iterations: int = 6
    line_search_steps: int = 5
    gradient_tolerance_n: float = 2.0e-4
    maximum_iteration_displacement_m: float = 0.003
    armijo_fraction: float = 1.0e-4

    def __post_init__(self) -> None:
        if self.iterations <= 0 or self.line_search_steps <= 0:
            raise ValueError("Hamiltonian solve iteration counts must be positive")
        if self.gradient_tolerance_n <= 0.0:
            raise ValueError("Hamiltonian gradient tolerance must be positive")
        if self.maximum_iteration_displacement_m <= 0.0:
            raise ValueError("Hamiltonian trust radius must be positive")
        if not 0.0 < self.armijo_fraction < 1.0:
            raise ValueError("Hamiltonian Armijo fraction must lie in (0, 1)")


@dataclass(frozen=True)
class IncrementalHamiltonianDiagnostics:
    iterations: int
    converged: bool
    initial_incremental_potential_j: float
    final_incremental_potential_j: float
    maximum_gradient_n: float
    accepted_line_search_fraction: float
    kinetic_energy_j: float
    total_hamiltonian_j: float
    minimum_contact_gap_m: float | None
    contact_quadrature_points: int
    energy_terms: dict[str, float]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class IncrementalHamiltonianStep:
    positions: Tensor
    momenta: Tensor
    velocities: Tensor
    acceleration: Tensor
    diagnostics: IncrementalHamiltonianDiagnostics


def solve_incremental_hamiltonian_step(
    current_positions: Tensor,
    current_velocities: Tensor,
    masses: Tensor,
    *,
    dt: float,
    potential: PotentialFunction,
    direction_preconditioner: DirectionPreconditioner | None = None,
    nonconservative_force: Tensor | None = None,
    moving_reference_velocity: Tensor | None = None,
    moving_reference_displacement: Tensor | None = None,
    moving_reference_next_velocity: Tensor | None = None,
    pinned_vertices: Tensor | None = None,
    pinned_positions: Tensor | None = None,
    maximum_acceleration_m_s2: float | None = None,
    maximum_velocity_m_s: float | None = None,
    differentiable: bool = False,
    feasible: FeasibilityFunction | None = None,
    prefer_transport_target: bool = False,
    config: IncrementalHamiltonianConfig = IncrementalHamiltonianConfig(),
) -> IncrementalHamiltonianStep:
    """Solve one phase-space step by minimizing a discrete incremental action.

    Hyper-NODE and geometric shell terms enter through ``potential``.  The
    inertial term supplies the cotangent-bundle metric, so the returned momentum
    is derived from the same high-dimensional solve rather than reconstructed
    after an explicit force/projection sequence.
    """

    if current_positions.ndim != 2 or current_positions.shape[1] != 3:
        raise ValueError("phase positions must have shape (V,3)")
    if current_velocities.shape != current_positions.shape:
        raise ValueError("phase velocities must match positions")
    if masses.shape != (len(current_positions),) or torch.any(masses <= 0.0):
        raise ValueError("phase masses must be positive with shape (V,)")
    if dt <= 0.0:
        raise ValueError("phase time step must be positive")
    if maximum_acceleration_m_s2 is not None and maximum_acceleration_m_s2 <= 0.0:
        raise ValueError("maximum phase acceleration must be positive")
    if maximum_velocity_m_s is not None and maximum_velocity_m_s <= 0.0:
        raise ValueError("maximum phase velocity must be positive")
    force = (
        torch.zeros_like(current_positions)
        if nonconservative_force is None
        else nonconservative_force
    )
    if force.shape != current_positions.shape:
        raise ValueError("nonconservative force must match phase positions")
    reference_velocity = (
        torch.zeros_like(current_positions)
        if moving_reference_velocity is None
        else moving_reference_velocity.to(current_positions)
    )
    reference_displacement = (
        torch.zeros_like(current_positions)
        if moving_reference_displacement is None
        else moving_reference_displacement.to(current_positions)
    )
    reference_next_velocity = (
        reference_velocity
        if moving_reference_next_velocity is None
        else moving_reference_next_velocity.to(current_positions)
    )
    for name, value in (
        ("moving_reference_velocity", reference_velocity),
        ("moving_reference_displacement", reference_displacement),
        ("moving_reference_next_velocity", reference_next_velocity),
    ):
        if value.shape != current_positions.shape:
            raise ValueError(f"{name} must match phase positions")

    pins = (
        torch.empty(0, dtype=torch.long, device=current_positions.device)
        if pinned_vertices is None
        else pinned_vertices.to(device=current_positions.device, dtype=torch.long)
    )
    fixed = (
        current_positions[pins]
        if pinned_positions is None
        else pinned_positions.to(current_positions)
    )
    if fixed.shape != (len(pins), 3):
        raise ValueError("pinned positions must have shape (P,3)")

    mass = masses.to(current_positions)
    # The canonical momentum is measured in the moving body frame.  The
    # reference displacement is prescribed by the body's SE(3) trajectory;
    # cloth deformation is the residual displacement solved by H(q,p).
    relative_current_velocity = current_velocities - reference_velocity
    ballistic_target = (
        current_positions
        + float(dt) * relative_current_velocity
        + reference_displacement
    )
    inertial_target = (
        ballistic_target
        + float(dt * dt) * force / mass[:, None]
    )

    def project_numerical_trust_region(points: Tensor) -> Tensor:
        result = points
        if maximum_acceleration_m_s2 is not None:
            acceleration_offset = result - ballistic_target
            norm = torch.linalg.vector_norm(
                acceleration_offset, dim=-1, keepdim=True
            )
            # Leave a small float32 guard band because acceleration is recovered
            # from two nearly equal positions after the nonlinear solve.
            radius = 0.999 * float(dt * dt * maximum_acceleration_m_s2)
            result = ballistic_target + acceleration_offset * torch.clamp(
                radius / norm.clamp_min(1.0e-12), max=1.0
            )
        if maximum_velocity_m_s is not None:
            velocity_offset = result - current_positions
            norm = torch.linalg.vector_norm(velocity_offset, dim=-1, keepdim=True)
            radius = float(dt * maximum_velocity_m_s)
            result = current_positions + velocity_offset * torch.clamp(
                radius / norm.clamp_min(1.0e-12), max=1.0
            )
        return result

    inertial_target = project_numerical_trust_region(inertial_target)
    # Keep a pure body-root transport target alongside the force-augmented
    # target.  If the nonlinear material/contact line search cannot accept a
    # force update, the prescribed SE(3) frame motion must still be carried
    # into q_next; otherwise a failed constitutive step makes the garment
    # appear to stick to world space while the body walks away.
    transport_target = project_numerical_trust_region(
        current_positions + reference_displacement
    )
    if len(pins):
        inertial_target = inertial_target.clone()
        inertial_target[pins] = fixed
        transport_target = transport_target.clone()
        transport_target[pins] = fixed
    # Warm-start from the ballistic target when it is admissible.  An
    # unconstrained target can already violate a material envelope after a
    # contact impulse; in that case start at the current admissible phase
    # point instead of making a finite barrier look like a soft spring.
    target_is_feasible = feasible is None or feasible(inertial_target.detach())
    transport_is_feasible = feasible is None or feasible(transport_target.detach())
    # The no-projection material path starts from the rigid body transport
    # target even when an external force has made the ballistic target large.
    # The latter remains in the incremental inertia term, so gravity/contact
    # work is still solved by Hamiltonian descent instead of being discarded.
    if prefer_transport_target:
        candidate = transport_target.clone()
    else:
        candidate = (
            inertial_target.clone()
            if target_is_feasible
            else transport_target.clone()
            if transport_is_feasible and bool(torch.any(reference_displacement.detach()))
            else current_positions.clone()
        )
    if len(pins):
        candidate[pins] = fixed
    initial_objective = None
    final_objective = None
    final_gradient = current_positions.new_tensor(float("inf"))
    accepted_fraction = 0.0
    iterations_run = 0
    converged = False

    def incremental_objective(points: Tensor) -> tuple[Tensor, PhasePotentialTerms]:
        inertia = 0.5 * (
            mass[:, None] * (points - inertial_target).square()
        ).sum() / float(dt * dt)
        terms = potential(points)
        if terms.total.ndim != 0:
            raise ValueError("phase potential terms must reduce to scalars")
        return inertia + terms.total, terms

    for iteration in range(config.iterations):
        variable = candidate if candidate.requires_grad else candidate.detach().requires_grad_(True)
        objective, _terms = incremental_objective(variable)
        if initial_objective is None:
            initial_objective = objective.detach()
        gradient = torch.autograd.grad(
            objective,
            variable,
            create_graph=differentiable,
        )[0]
        if len(pins):
            gradient = gradient.clone()
            gradient[pins] = 0.0
        final_gradient = torch.linalg.vector_norm(gradient.detach(), dim=-1).max()
        iterations_run = iteration + 1
        if float(final_gradient) <= config.gradient_tolerance_n:
            candidate = variable
            final_objective = objective
            converged = True
            break

        raw_direction = -float(dt * dt) * gradient / mass[:, None]
        direction = raw_direction
        if direction_preconditioner is not None:
            direction = direction_preconditioner(direction)
            if direction.shape != current_positions.shape:
                raise ValueError(
                    "phase direction preconditioner must preserve the state shape"
                )

        def bounded_free_direction(value: Tensor) -> Tensor:
            norm = torch.linalg.vector_norm(value, dim=-1, keepdim=True)
            bounded = value * torch.clamp(
                config.maximum_iteration_displacement_m
                / norm.clamp_min(1.0e-12),
                max=1.0,
            )
            if len(pins):
                bounded = bounded.clone()
                bounded[pins] = 0.0
            return bounded

        direction = bounded_free_direction(direction)
        directional_derivative = (gradient * direction).sum().detach()
        # A geometric preconditioner is useful only when it preserves a
        # descent direction for the actual incremental action. Generalized
        # Laplacian eigenvectors are mass-orthogonal rather than necessarily
        # Euclidean-orthogonal, so an unchecked spectral map can flip this
        # sign and make Armijo accept an energy increase. Fall back to the
        # canonical inverse-mass gradient whenever that happens.
        if float(directional_derivative) >= 0.0:
            direction = bounded_free_direction(raw_direction)
            directional_derivative = (gradient * direction).sum().detach()
        base = objective.detach()
        accepted = None
        accepted_objective = None
        fraction = 1.0
        for _ in range(config.line_search_steps):
            trial = variable + fraction * direction
            trial = project_numerical_trust_region(trial)
            if len(pins):
                trial = trial.clone()
                trial[pins] = fixed
            trial_objective, _trial_terms = incremental_objective(trial)
            sufficient_decrease = base + (
                config.armijo_fraction * fraction * directional_derivative
            )
            trial_is_feasible = feasible is None or feasible(trial.detach())
            if (
                trial_is_feasible
                and bool(torch.isfinite(trial_objective).detach())
                and float(trial_objective.detach()) <= float(sufficient_decrease)
            ):
                accepted = trial
                accepted_objective = trial_objective
                break
            fraction *= 0.5
        if accepted is None:
            # A constrained failure means the proposed action would leave the
            # material manifold. Hold the last feasible phase point. When the
            # body frame is prescribed, that point is the pure SE(3) transport
            # target rather than the old world-space q, so a failed force
            # solve cannot strand the garment behind the avatar.
            fraction = 0.0
            accepted = variable.detach() if not differentiable else variable
            accepted_objective, _accepted_terms = incremental_objective(accepted)
        candidate = accepted if differentiable else accepted.detach()
        final_objective = accepted_objective
        accepted_fraction += fraction

    if final_objective is None:
        final_objective, _unused = incremental_objective(candidate)
    final_terms = potential(candidate)
    relative_velocities = (
        candidate - current_positions - reference_displacement
    ) / float(dt)
    velocities = relative_velocities + reference_next_velocity
    momenta = mass[:, None] * relative_velocities
    acceleration = (velocities - current_velocities) / float(dt)
    kinetic = 0.5 * (mass[:, None] * relative_velocities.square()).sum()
    minimum_gap = (
        None
        if final_terms.contact_minimum_gap is None
        else float(final_terms.contact_minimum_gap.detach())
    )
    diagnostics = IncrementalHamiltonianDiagnostics(
        iterations=iterations_run,
        converged=converged,
        initial_incremental_potential_j=float(
            (final_objective if initial_objective is None else initial_objective).detach()
        ),
        final_incremental_potential_j=float(final_objective.detach()),
        maximum_gradient_n=float(final_gradient),
        accepted_line_search_fraction=(
            accepted_fraction / max(iterations_run, 1)
        ),
        kinetic_energy_j=float(kinetic.detach()),
        total_hamiltonian_j=float((kinetic + final_terms.total).detach()),
        minimum_contact_gap_m=minimum_gap,
        contact_quadrature_points=final_terms.contact_quadrature_points,
        energy_terms=final_terms.detached_energies(),
    )
    return IncrementalHamiltonianStep(
        candidate,
        momenta,
        velocities,
        acceleration,
        diagnostics,
    )


def reference_surface_contact_potential(
    positions: Tensor,
    faces: Tensor,
    rest_areas: Tensor,
    contact_query: ContactQuery,
    *,
    time: float,
    clearance_m: float,
    pressure_stiffness_n_m3: float,
    contact_sample: ContactSampleLike | None = None,
) -> tuple[Tensor, Tensor, int]:
    """Area-measure contact quadrature, convergent under triangle refinement."""

    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("contact faces must have shape (F,3)")
    if rest_areas.shape != (len(faces),) or torch.any(rest_areas <= 0.0):
        raise ValueError("one positive rest area is required per contact face")
    if clearance_m < 0.0 or pressure_stiffness_n_m3 <= 0.0:
        raise ValueError("contact clearance/stiffness is invalid")
    samples = reference_surface_contact_quadrature(positions, faces)
    queried = (
        contact_query(samples.detach(), float(time))
        if contact_sample is None
        else contact_sample
    )
    closest = queried.closest_points.to(samples)
    normals = torch.nn.functional.normalize(queried.normals.to(samples), dim=-1)
    if closest.shape != samples.shape or normals.shape != samples.shape:
        raise ValueError("contact query must return one closest point and normal per sample")
    gap = ((samples - closest) * normals).sum(dim=-1)
    compression = torch.relu(float(clearance_m) - gap)
    penetration = torch.relu(-gap)
    quadrature_weight = (
        rest_areas.to(positions)[:, None].expand(-1, 3).reshape(-1) / 3.0
    )
    energy = (
        0.5
        * float(pressure_stiffness_n_m3)
        * quadrature_weight
        * (
            compression.square()
            # Cloth thickness is a compliant pressure layer, while crossing
            # the body SDF is inadmissible. A steep but differentiable
            # interior penalty keeps the Hamiltonian solve on the outside so
            # the final contact guard does not create a local outward spike.
            + 1000.0 * penetration.square()
        )
    ).sum()
    return energy, gap.min(), len(samples)


def reference_surface_contact_quadrature(
    positions: Tensor,
    faces: Tensor,
) -> Tensor:
    """Three-point reference-face quadrature used by the contact measure."""

    triangles = positions[faces.to(device=positions.device, dtype=torch.long)]
    barycentric = positions.new_tensor(
        (
            (2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0),
            (1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0),
            (1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0),
        )
    )
    return torch.einsum("qk,fkj->fqj", barycentric, triangles).reshape(-1, 3)
