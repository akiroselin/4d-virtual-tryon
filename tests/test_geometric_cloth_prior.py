from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aigc_fashion.differentiability_contract import (  # noqa: E402
    DifferentiabilityManifest,
    GradientStage,
    ProbeStatus,
    probe_directional_gradient,
)
from aigc_fashion.geometric_cloth_prior import IntrinsicEquivariantClothPrior  # noqa: E402


def _inputs() -> tuple[torch.Tensor, ...]:
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.1, 0.0, 0.0], [0.0, 0.9, 0.1]],
        requires_grad=True,
    )
    velocities = torch.tensor(
        [[0.0, 0.0, 0.0], [0.1, 0.2, 0.0], [-0.1, 0.0, 0.1]]
    )
    edges = torch.tensor([[0, 1, 2], [1, 2, 0]])
    rest = torch.tensor([1.0, 2.0**0.5, 1.0])
    cotangent = torch.tensor([1.0, 0.5, 1.0])
    latent = torch.randn(32, requires_grad=True)
    return positions, velocities, edges, rest, cotangent, latent


def test_prior_is_e3_equivariant_translation_invariant_and_balanced() -> None:
    torch.manual_seed(11)
    model = IntrinsicEquivariantClothPrior(hidden_dim=16)
    positions, velocities, edges, rest, cotangent, latent = _inputs()
    baseline = model(positions, velocities, edges, rest, cotangent, latent)

    rotation = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    translation = torch.tensor([3.0, -2.0, 0.5])
    transformed = model(
        positions @ rotation.T + translation,
        velocities @ rotation.T,
        edges,
        rest,
        cotangent,
        latent,
    )
    assert torch.allclose(
        transformed.internal_force,
        baseline.internal_force @ rotation.T,
        atol=1.0e-5,
        rtol=1.0e-5,
    )
    assert torch.allclose(
        baseline.internal_force.sum(dim=0),
        torch.zeros(3),
        atol=1.0e-6,
    )


def test_prior_backpropagates_to_geometry_and_material_latent() -> None:
    model = IntrinsicEquivariantClothPrior(hidden_dim=12)
    inputs = _inputs()
    output = model(*inputs)
    output.internal_force.square().sum().backward()
    assert inputs[0].grad is not None and torch.isfinite(inputs[0].grad).all()
    assert inputs[-1].grad is not None and torch.isfinite(inputs[-1].grad).all()


def test_directional_probe_and_manifest_block_unverified_stages() -> None:
    point = torch.tensor([0.3, -0.2])
    result = probe_directional_gradient(
        GradientStage.MATERIAL_TO_STRESS,
        lambda value: (value.square() + 0.2 * value).sum(),
        point,
        direction=torch.tensor([1.0, -0.5]),
        epsilon=1.0e-3,
        relative_tolerance=1.0e-3,
    )
    assert result.status is ProbeStatus.PASSED
    manifest = DifferentiabilityManifest({GradientStage.MATERIAL_TO_STRESS: result})
    manifest.require([GradientStage.MATERIAL_TO_STRESS])
    with pytest.raises(RuntimeError, match="raster_render=not_run"):
        manifest.require([GradientStage.MATERIAL_TO_STRESS, GradientStage.RASTER_RENDER])
