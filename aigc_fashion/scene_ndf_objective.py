from __future__ import annotations

from dataclasses import dataclass, fields
from enum import Enum
from typing import Mapping

import torch
from torch import Tensor
import torch.nn.functional as F


class GenesisNDFPriorMode(str, Enum):
    TRAJECTORY_DISTILLATION = "trajectory_distillation"
    ONLINE_DIFFERENTIABLE = "online_differentiable"


@dataclass(frozen=True)
class SceneNDFLossWeights:
    visual_evidence: float = 0.25
    metric_evidence: float = 0.25
    shell: float = 1.0
    dynamics: float = 1.0
    seam: float = 1.0
    contact: float = 10.0
    genesis_position: float = 1.0
    genesis_velocity: float = 0.1
    genesis_online: float = 0.0
    material_prior: float = 1.0e-3

    def __post_init__(self) -> None:
        for item in fields(self):
            if float(getattr(self, item.name)) < 0.0:
                raise ValueError(f"{item.name} weight must be non-negative")


@dataclass(frozen=True)
class SceneNDFLossResult:
    total: Tensor
    velocity: Tensor
    acceleration: Tensor
    raw_terms: Mapping[str, Tensor]
    weighted_terms: Mapping[str, Tensor]


def pointwise_time_derivatives(positions: Tensor, physical_time: Tensor) -> tuple[Tensor, Tensor]:
    """Differentiate a pointwise NDF output with respect to physical time."""

    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape (N, 3)")
    if physical_time.shape not in {(len(positions),), (len(positions), 1)}:
        raise ValueError("physical_time must have shape (N,) or (N, 1)")
    if not physical_time.requires_grad:
        raise ValueError("physical_time must require gradients")
    velocity_components = []
    for axis in range(3):
        derivative = torch.autograd.grad(
            positions[:, axis].sum(),
            physical_time,
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )[0]
        if derivative is None:
            raise ValueError("NDF positions are disconnected from physical_time")
        velocity_components.append(derivative.reshape(-1))
    velocity = torch.stack(velocity_components, dim=-1)

    acceleration_components = []
    for axis in range(3):
        derivative = torch.autograd.grad(
            velocity[:, axis].sum(),
            physical_time,
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )[0]
        if derivative is None:
            raise ValueError("NDF velocity is disconnected from physical_time")
        acceleration_components.append(derivative.reshape(-1))
    return velocity, torch.stack(acceleration_components, dim=-1)


def scene_ndf_physics_objective(
    *,
    positions: Tensor,
    physical_time: Tensor,
    material_latent: Tensor,
    mass_per_vertex: Tensor,
    internal_force: Tensor,
    external_force: Tensor,
    genesis_positions: Tensor | None,
    genesis_velocities: Tensor | None = None,
    genesis_online_residual: Tensor | None = None,
    visual_residual: Tensor | None = None,
    metric_residual: Tensor | None = None,
    shell_residual: Tensor | None = None,
    seam_residual: Tensor | None = None,
    contact_gap: Tensor | None = None,
    genesis_confidence: Tensor | None = None,
    prior_mode: GenesisNDFPriorMode = GenesisNDFPriorMode.TRAJECTORY_DISTILLATION,
    weights: SceneNDFLossWeights = SceneNDFLossWeights(),
) -> SceneNDFLossResult:
    """Constrain one scene-optimized 4D NDF with physical and Genesis evidence."""

    _validate_physics_inputs(positions, mass_per_vertex, internal_force, external_force)
    if weights.genesis_position > 0.0 and genesis_positions is None:
        raise ValueError("Genesis position weight requires Genesis positions")
    if weights.genesis_velocity > 0.0 and genesis_velocities is None:
        raise ValueError("Genesis velocity weight requires Genesis velocities")
    if prior_mode is GenesisNDFPriorMode.ONLINE_DIFFERENTIABLE:
        if weights.genesis_online <= 0.0 or genesis_online_residual is None:
            raise ValueError("online Genesis mode requires a positive online weight and residual")
    genesis_weight = weights.genesis_position + weights.genesis_velocity + weights.genesis_online
    if genesis_weight <= 0.0:
        raise ValueError("a Genesis physics prior is mandatory; semantic/metric evidence alone is invalid")

    velocity, acceleration = pointwise_time_derivatives(positions, physical_time)
    mass = mass_per_vertex.to(device=positions.device, dtype=positions.dtype).reshape(-1, 1)
    dynamics_residual = mass * acceleration - internal_force - external_force
    zero = positions.new_zeros(())
    confidence = (
        torch.ones(len(positions), device=positions.device, dtype=positions.dtype)
        if genesis_confidence is None
        else genesis_confidence.to(device=positions.device, dtype=positions.dtype)
    )
    if confidence.shape != (len(positions),) or torch.any(confidence < 0.0):
        raise ValueError("genesis_confidence must be non-negative with shape (N,)")

    position_error = (
        _confidence_mse(positions, genesis_positions.detach(), confidence)
        if genesis_positions is not None
        else zero
    )
    velocity_error = (
        _confidence_mse(velocity, genesis_velocities.detach(), confidence)
        if genesis_velocities is not None
        else zero
    )
    raw_terms = {
        "visual_evidence": _mean_square(visual_residual, zero),
        "metric_evidence": _mean_square(metric_residual, zero),
        "shell": _mean_square(shell_residual, zero),
        "dynamics": dynamics_residual.square().mean(),
        "seam": _mean_square(seam_residual, zero),
        "contact": F.relu(-contact_gap).square().mean() if contact_gap is not None else zero,
        "genesis_position": position_error,
        "genesis_velocity": velocity_error,
        "genesis_online": _mean_square(genesis_online_residual, zero),
        "material_prior": material_latent.square().mean(),
    }
    weighted_terms = {
        name: getattr(weights, name) * value for name, value in raw_terms.items()
    }
    total = torch.stack(tuple(weighted_terms.values())).sum()
    return SceneNDFLossResult(total, velocity, acceleration, raw_terms, weighted_terms)


def _validate_physics_inputs(
    positions: Tensor,
    mass: Tensor,
    internal_force: Tensor,
    external_force: Tensor,
) -> None:
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape (N, 3)")
    if mass.shape not in {(len(positions),), (len(positions), 1)}:
        raise ValueError("mass_per_vertex must have shape (N,) or (N, 1)")
    if internal_force.shape != positions.shape or external_force.shape != positions.shape:
        raise ValueError("force tensors must match positions")


def _confidence_mse(prediction: Tensor, target: Tensor, confidence: Tensor) -> Tensor:
    if target.shape != prediction.shape:
        raise ValueError("Genesis target must match the NDF prediction")
    target = target.to(device=prediction.device, dtype=prediction.dtype)
    squared = (prediction - target).square().mean(dim=-1)
    return (confidence * squared).sum() / confidence.sum().clamp_min(1.0e-8)


def _mean_square(value: Tensor | None, zero: Tensor) -> Tensor:
    return value.square().mean() if value is not None else zero
