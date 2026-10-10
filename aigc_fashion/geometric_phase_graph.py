from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class IntrinsicClothPhaseState:
    """One cloth chart in canonical phase coordinates ``(material_uv, q, p)``."""

    material_coordinates: Tensor
    positions: Tensor
    momenta: Tensor
    lumped_mass: Tensor
    time: float

    def __post_init__(self) -> None:
        vertex_count = len(self.positions)
        if self.positions.ndim != 2 or self.positions.shape[1] != 3:
            raise ValueError("cloth positions must have shape (V,3)")
        if self.momenta.shape != self.positions.shape:
            raise ValueError("cloth momenta must match positions")
        if self.material_coordinates.shape != (vertex_count, 2):
            raise ValueError("material coordinates must have shape (V,2)")
        if self.lumped_mass.shape != (vertex_count,) or torch.any(self.lumped_mass <= 0.0):
            raise ValueError("lumped cloth mass must be positive with shape (V,)")

    @classmethod
    def from_velocity(
        cls,
        material_coordinates: Tensor,
        positions: Tensor,
        velocities: Tensor,
        lumped_mass: Tensor,
        time: float,
    ) -> IntrinsicClothPhaseState:
        if velocities.shape != positions.shape:
            raise ValueError("cloth velocities must match positions")
        mass = lumped_mass.to(positions)
        return cls(
            material_coordinates.to(positions),
            positions,
            mass[:, None] * velocities,
            mass,
            float(time),
        )

    @property
    def velocities(self) -> Tensor:
        return self.momenta / self.lumped_mass[:, None]

    def kinetic_energy(self) -> Tensor:
        return 0.5 * (
            self.momenta.square().sum(dim=-1) / self.lumped_mass
        ).sum()


@dataclass(frozen=True)
class SE3CanonicalPhaseState:
    """Rigid-body cotangent state on ``T*SE(3)`` in body coordinates."""

    transforms: Tensor
    body_momenta: Tensor
    time: float

    def __post_init__(self) -> None:
        if self.transforms.ndim != 3 or self.transforms.shape[-2:] != (4, 4):
            raise ValueError("rigid transforms must have shape (B,4,4)")
        if self.body_momenta.shape != (len(self.transforms), 6):
            raise ValueError("body momenta must have shape (B,6)")


@dataclass(frozen=True)
class SO2ContactPhaseState:
    """Contact fibre with invariant gaps and charge-1 tangent momentum."""

    signed_gaps: Tensor
    normals: Tensor
    tangent_momenta: Tensor
    active: Tensor
    time: float

    def __post_init__(self) -> None:
        contact_count = len(self.signed_gaps)
        if self.signed_gaps.shape != (contact_count,):
            raise ValueError("signed contact gaps must have shape (C,)")
        if self.normals.shape != (contact_count, 3):
            raise ValueError("contact normals must have shape (C,3)")
        if not torch.is_complex(self.tangent_momenta) or self.tangent_momenta.shape != (
            contact_count,
        ):
            raise ValueError("tangent momenta must be a charge-1 complex field")
        if self.active.shape != (contact_count,):
            raise ValueError("active contact mask must have shape (C,)")


@dataclass(frozen=True)
class GeometricPhaseGraphState:
    """Product-manifold state for the hierarchy-aware 4D scene graph."""

    cloth: tuple[IntrinsicClothPhaseState, ...]
    rigid: SE3CanonicalPhaseState | None
    contact: SO2ContactPhaseState | None
    time: float

    def __post_init__(self) -> None:
        states: tuple[object, ...] = (*self.cloth, self.rigid, self.contact)
        for state in states:
            if state is not None and abs(float(getattr(state, "time")) - self.time) > 1.0e-8:
                raise ValueError("all product-phase components must share physical time")

    def total_kinetic_energy(self) -> Tensor:
        if not self.cloth:
            return torch.tensor(0.0)
        return torch.stack([state.kinetic_energy() for state in self.cloth]).sum()


@dataclass(frozen=True)
class ContactMomentumReset:
    state: IntrinsicClothPhaseState
    impulse: Tensor
    normal_impulse: Tensor
    tangent_impulse: Tensor


def symplectic_euler_phase_step(
    state: IntrinsicClothPhaseState,
    force: Tensor,
    dt: float,
) -> IntrinsicClothPhaseState:
    """Kick then drift in canonical ``(q,p)`` coordinates."""

    if force.shape != state.positions.shape:
        raise ValueError("phase force must match cloth positions")
    if dt <= 0.0:
        raise ValueError("phase time step must be positive")
    momenta = state.momenta + dt * force
    positions = state.positions + dt * momenta / state.lumped_mass[:, None]
    return IntrinsicClothPhaseState(
        state.material_coordinates,
        positions,
        momenta,
        state.lumped_mass,
        state.time + dt,
    )


def apply_sdf_contact_momentum_reset(
    state: IntrinsicClothPhaseState,
    normals: Tensor,
    body_velocities: Tensor,
    active: Tensor,
    friction: Tensor,
) -> ContactMomentumReset:
    """Apply a non-penetrating, dissipative contact jump to canonical momentum."""

    if normals.shape != state.positions.shape or body_velocities.shape != state.positions.shape:
        raise ValueError("contact vectors must match cloth positions")
    if active.shape != state.lumped_mass.shape or friction.shape != state.lumped_mass.shape:
        raise ValueError("contact masks and friction must have shape (V,)")
    normal = torch.nn.functional.normalize(normals.to(state.positions), dim=-1)
    relative = state.velocities - body_velocities.to(state.positions)
    normal_speed = (relative * normal).sum(dim=-1, keepdim=True)
    normal_velocity = normal_speed * normal
    tangent_velocity = relative - normal_velocity
    coefficient = friction.to(state.positions).clamp(0.0, 1.0)
    resolved_relative = (1.0 - coefficient[:, None]) * tangent_velocity
    resolved_relative = resolved_relative + torch.clamp_min(normal_speed, 0.0) * normal
    velocity = torch.where(
        active.to(device=state.positions.device, dtype=torch.bool)[:, None],
        body_velocities.to(state.positions) + resolved_relative,
        state.velocities,
    )
    momenta = state.lumped_mass[:, None] * velocity
    impulse = momenta - state.momenta
    normal_impulse = (impulse * normal).sum(dim=-1, keepdim=True) * normal
    tangent_impulse = impulse - normal_impulse
    return ContactMomentumReset(
        IntrinsicClothPhaseState(
            state.material_coordinates,
            state.positions,
            momenta,
            state.lumped_mass,
            state.time,
        ),
        impulse,
        normal_impulse,
        tangent_impulse,
    )


def canonical_hamiltonian(
    state: IntrinsicClothPhaseState,
    potential_energy: Tensor,
) -> Tensor:
    if potential_energy.ndim != 0:
        raise ValueError("potential energy must be scalar")
    return state.kinetic_energy() + potential_energy
