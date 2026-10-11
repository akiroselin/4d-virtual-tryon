from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import Tensor

from .biquaternion_phase import hamilton_product
from .hypernetwork_kirchhoff_love import KirchhoffLoveShellOutput


def normalize_quaternion(quaternion: Tensor) -> Tensor:
    if quaternion.shape[-1] != 4:
        raise ValueError("quaternion tensors must end in four components")
    return quaternion / torch.linalg.vector_norm(
        quaternion, dim=-1, keepdim=True
    ).clamp_min(1.0e-12)


def quaternion_conjugate(quaternion: Tensor) -> Tensor:
    if quaternion.shape[-1] != 4:
        raise ValueError("quaternion tensors must end in four components")
    return torch.cat((quaternion[..., :1], -quaternion[..., 1:]), dim=-1)


def rotate_vectors(quaternion: Tensor, vectors: Tensor) -> Tensor:
    """Apply the Spin(3) action without choosing a world-coordinate frame."""

    rotation = normalize_quaternion(quaternion).to(vectors)
    while rotation.ndim < vectors.ndim:
        rotation = rotation.unsqueeze(-2)
    pure = torch.cat((torch.zeros_like(vectors[..., :1]), vectors), dim=-1)
    return hamilton_product(
        hamilton_product(rotation, pure), quaternion_conjugate(rotation)
    )[..., 1:]


@dataclass(frozen=True)
class UnitDualQuaternion:
    """Spin double cover of one SE(3) pose: q_r + epsilon q_d."""

    real: Tensor
    dual: Tensor

    def __post_init__(self) -> None:
        if self.real.shape[-1] != 4 or self.dual.shape != self.real.shape:
            raise ValueError("dual quaternion parts must have matching (...,4) shapes")

    @classmethod
    def from_rotation_translation(
        cls, rotation: Tensor, translation: Tensor
    ) -> UnitDualQuaternion:
        if translation.shape[-1] != 3:
            raise ValueError("dual quaternion translation must end in three components")
        real = normalize_quaternion(rotation)
        pure_translation = torch.cat(
            (torch.zeros_like(translation[..., :1]), translation), dim=-1
        )
        dual = 0.5 * hamilton_product(pure_translation, real)
        return cls(real, dual)

    def normalized(self) -> UnitDualQuaternion:
        norm = torch.linalg.vector_norm(self.real, dim=-1, keepdim=True).clamp_min(
            1.0e-12
        )
        real = self.real / norm
        dual = self.dual / norm
        # Projection onto the Study quadric <q_r,q_d>=0.
        dual = dual - (real * dual).sum(dim=-1, keepdim=True) * real
        return UnitDualQuaternion(real, dual)

    @property
    def translation(self) -> Tensor:
        value = self.normalized()
        return 2.0 * hamilton_product(
            value.dual, quaternion_conjugate(value.real)
        )[..., 1:]

    def transform_points(self, points: Tensor) -> Tensor:
        value = self.normalized()
        return rotate_vectors(value.real, points) + value.translation

    def transform_vectors(self, vectors: Tensor) -> Tensor:
        return rotate_vectors(self.normalized().real, vectors)

    def compose(self, right: UnitDualQuaternion) -> UnitDualQuaternion:
        left = self.normalized()
        right = right.normalized()
        return UnitDualQuaternion(
            hamilton_product(left.real, right.real),
            hamilton_product(left.real, right.dual)
            + hamilton_product(left.dual, right.real),
        ).normalized()

    def sign_flipped(self) -> UnitDualQuaternion:
        return UnitDualQuaternion(-self.real, -self.dual)

    def study_residual(self) -> tuple[Tensor, Tensor]:
        unit = self.real.square().sum(dim=-1) - 1.0
        orthogonality = (self.real * self.dual).sum(dim=-1)
        return unit, orthogonality


@dataclass(frozen=True)
class Cloth4DStructureGroup:
    """Typed product group acting on one non-relativistic 4D scene slice."""

    world_pose: UnitDualQuaternion
    contact_spin2: Tensor

    def __post_init__(self) -> None:
        if not torch.is_complex(self.contact_spin2):
            raise ValueError("Spin(2) contact gauge must be represented by a complex phase")
        if torch.any(self.contact_spin2.abs() <= 0.0):
            raise ValueError("Spin(2) contact phase must be non-zero")

    @property
    def normalized_contact_spin2(self) -> Tensor:
        return self.contact_spin2 / self.contact_spin2.abs().clamp_min(1.0e-12)


@dataclass(frozen=True)
class Cloth4DGroupPhaseState:
    positions: Tensor
    momenta: Tensor
    face_spin3: Tensor
    contact_tangent_momenta: Tensor
    time: float

    def __post_init__(self) -> None:
        if self.positions.ndim != 2 or self.positions.shape[1] != 3:
            raise ValueError("group phase positions must have shape (V,3)")
        if self.momenta.shape != self.positions.shape:
            raise ValueError("group phase momenta must match positions")
        if self.face_spin3.ndim != 2 or self.face_spin3.shape[1] != 4:
            raise ValueError("face Spin(3) frames must have shape (F,4)")
        if not torch.is_complex(self.contact_tangent_momenta):
            raise ValueError("contact tangent momenta must be a complex Spin(2) field")

    def acted_on(self, group: Cloth4DStructureGroup) -> Cloth4DGroupPhaseState:
        pose = group.world_pose.normalized()
        rotation = pose.real.to(self.face_spin3)
        frame = rotation.expand_as(self.face_spin3)
        return Cloth4DGroupPhaseState(
            pose.transform_points(self.positions),
            pose.transform_vectors(self.momenta),
            normalize_quaternion(hamilton_product(frame, self.face_spin3)),
            group.normalized_contact_spin2.to(self.contact_tangent_momenta)
            * self.contact_tangent_momenta,
            self.time,
        )


@dataclass(frozen=True)
class StructureGroupTrainingLoss:
    total: Tensor
    force_equivariance: Tensor
    acceleration_equivariance: Tensor
    traction_equivariance: Tensor
    hamiltonian_invariance: Tensor
    noether_linear_momentum: Tensor
    noether_angular_momentum: Tensor

    def detached_metrics(self) -> dict[str, float]:
        return {
            name: float(getattr(self, name).detach())
            for name in asdict(self)
        }


def hypernode_structure_group_loss(
    original: KirchhoffLoveShellOutput,
    transformed: KirchhoffLoveShellOutput,
    original_positions: Tensor,
    *,
    world_rotation: Tensor,
    original_acceleration: Tensor,
    transformed_acceleration: Tensor,
    force_scale_n: float = 2.0e-3,
    acceleration_scale_m_s2: float = 1.0,
    energy_scale_j: float = 1.0e-4,
) -> StructureGroupTrainingLoss:
    """Train Hyper-NODE/KL under the product-group world action.

    Material coordinates and latents stay on the reference bundle. Only spatial
    vectors are rotated, so orthotropic warp/weft directions remain physical.
    """

    if min(force_scale_n, acceleration_scale_m_s2, energy_scale_j) <= 0.0:
        raise ValueError("structure-group loss scales must be positive")
    original_force = (
        original.membrane_internal_force + original.bending_internal_force
    )
    transformed_force = (
        transformed.membrane_internal_force + transformed.bending_internal_force
    )
    rotated_force = rotate_vectors(world_rotation, original_force)
    force_equivariance = (
        (transformed_force - rotated_force) / force_scale_n
    ).square().mean()
    rotated_acceleration = rotate_vectors(world_rotation, original_acceleration)
    acceleration_equivariance = (
        (transformed_acceleration - rotated_acceleration)
        / acceleration_scale_m_s2
    ).square().mean()
    if len(original.free_boundary_traction):
        rotated_traction = rotate_vectors(
            world_rotation, original.free_boundary_traction
        )
        traction_equivariance = (
            (transformed.free_boundary_traction - rotated_traction) / force_scale_n
        ).square().mean()
    else:
        traction_equivariance = original_positions.new_zeros(())
    original_energy = (
        original.membrane_energy.sum()
        + original.strain_limit_energy.sum()
        + original.bending_energy.sum()
    )
    transformed_energy = (
        transformed.membrane_energy.sum()
        + transformed.strain_limit_energy.sum()
        + transformed.bending_energy.sum()
    )
    hamiltonian_invariance = (
        (transformed_energy - original_energy) / energy_scale_j
    ).square()

    resultant = original_force.sum(dim=0)
    center = original_positions.mean(dim=0, keepdim=True)
    torque = torch.linalg.cross(
        original_positions - center,
        original_force,
        dim=-1,
    ).sum(dim=0)
    noether_linear = (resultant / force_scale_n).square().mean()
    noether_angular = (torque / force_scale_n).square().mean()
    total = (
        force_equivariance
        + 0.25 * acceleration_equivariance
        + 0.1 * traction_equivariance
        + 0.25 * hamiltonian_invariance
        + 0.05 * noether_linear
        + 0.05 * noether_angular
    )
    return StructureGroupTrainingLoss(
        total,
        force_equivariance,
        acceleration_equivariance,
        traction_equivariance,
        hamiltonian_invariance,
        noether_linear,
        noether_angular,
    )


def deterministic_training_group(
    step: int,
    reference: Tensor,
) -> Cloth4DStructureGroup:
    """Deterministic SE(3) x Spin(2) augmentation for reproducible training."""

    phase = reference.new_tensor(0.61803398875 * (int(step) + 1))
    axis = torch.stack(
        (torch.cos(phase), torch.sin(phase), torch.cos(2.0 * phase))
    )
    axis = axis / torch.linalg.vector_norm(axis).clamp_min(1.0e-12)
    angle = 0.7 * torch.sin(1.7 * phase)
    rotation = torch.cat(
        (torch.cos(0.5 * angle).reshape(1), torch.sin(0.5 * angle) * axis)
    )
    translation = 0.05 * torch.stack(
        (torch.sin(phase), torch.cos(1.3 * phase), torch.sin(0.7 * phase))
    )
    contact_angle = 1.2 * torch.cos(0.9 * phase)
    contact_phase = torch.polar(torch.ones_like(contact_angle), contact_angle)
    return Cloth4DStructureGroup(
        UnitDualQuaternion.from_rotation_translation(rotation, translation),
        contact_phase,
    )
