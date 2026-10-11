from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aigc_fashion.scene_ndf_objective import (  # noqa: E402
    SceneNDFLossWeights,
    pointwise_time_derivatives,
    scene_ndf_physics_objective,
)


def _field() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    time = torch.linspace(0.0, 0.3, 5).unsqueeze(-1).requires_grad_(True)
    amplitude = torch.tensor([0.4, -0.2, 0.1], requires_grad=True)
    base = torch.tensor([0.2, 0.3, 0.4]).expand(len(time), -1)
    positions = base + time * amplitude + time.square() * (0.5 * amplitude)
    return positions, time, amplitude


def test_pointwise_ndf_time_derivatives_are_physical_time_derivatives() -> None:
    positions, time, amplitude = _field()
    velocity, acceleration = pointwise_time_derivatives(positions, time)
    expected_velocity = amplitude + time * amplitude
    expected_acceleration = amplitude.expand_as(positions)
    assert torch.allclose(velocity, expected_velocity, atol=1.0e-6)
    assert torch.allclose(acceleration, expected_acceleration, atol=1.0e-6)


def test_scene_ndf_combines_genesis_trajectory_and_physics_not_only_semantics() -> None:
    positions, time, amplitude = _field()
    velocity, acceleration = pointwise_time_derivatives(positions, time)
    material_latent = torch.randn(32, requires_grad=True)
    mass = torch.full((len(time),), 0.2)
    internal_force = mass[:, None] * acceleration
    result = scene_ndf_physics_objective(
        positions=positions,
        physical_time=time,
        material_latent=material_latent,
        mass_per_vertex=mass,
        internal_force=internal_force,
        external_force=torch.zeros_like(positions),
        genesis_positions=positions.detach() + 0.01,
        genesis_velocities=velocity.detach() - 0.02,
        visual_residual=torch.tensor([0.1]),
        metric_residual=torch.tensor([0.05]),
        shell_residual=torch.tensor([0.02]),
        contact_gap=torch.tensor([0.01, -0.005]),
    )
    assert result.raw_terms["dynamics"].item() == pytest.approx(0.0, abs=1.0e-10)
    assert result.raw_terms["genesis_position"].item() > 0.0
    assert result.raw_terms["genesis_velocity"].item() > 0.0
    result.total.backward()
    assert amplitude.grad is not None and torch.isfinite(amplitude.grad).all()
    assert material_latent.grad is not None


def test_semantic_only_scene_field_is_rejected() -> None:
    positions, time, _ = _field()
    weights = SceneNDFLossWeights(
        shell=0.0,
        dynamics=0.0,
        seam=0.0,
        contact=0.0,
        genesis_position=0.0,
        genesis_velocity=0.0,
        genesis_online=0.0,
    )
    with pytest.raises(ValueError, match="Genesis physics prior is mandatory"):
        scene_ndf_physics_objective(
            positions=positions,
            physical_time=time,
            material_latent=torch.zeros(32),
            mass_per_vertex=torch.ones(len(time)),
            internal_force=torch.zeros_like(positions),
            external_force=torch.zeros_like(positions),
            genesis_positions=None,
            weights=weights,
        )
