from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aigc_fashion.physics_conditioned_flow import (  # noqa: E402
    GenesisMPMPriorMode,
    PhysicsConditionedCortexODE,
    PhysicsLossWeights,
    SceneMaterialLatent,
    SpectralLinear,
    physics_constrained_loss,
)


def _flow_inputs(batch_size: int = 6) -> dict[str, torch.Tensor]:
    return {
        "state": torch.randn(batch_size, 3, requires_grad=True),
        "material_coordinates": torch.rand(batch_size, 2),
        "material_latent": torch.randn(32, requires_grad=True),
    }


def test_flow_is_latent_conditioned_finite_and_differentiable() -> None:
    torch.manual_seed(7)
    model = PhysicsConditionedCortexODE(hidden_dim=16, hidden_layers=2)
    inputs = _flow_inputs()
    velocity = model(inputs["state"], 0.25, **{key: value for key, value in inputs.items() if key != "state"})
    assert velocity.shape == inputs["state"].shape
    assert torch.isfinite(velocity).all()

    changed = model(
        inputs["state"],
        0.25,
        material_coordinates=inputs["material_coordinates"],
        material_latent=inputs["material_latent"] + 1.0,
    )
    assert not torch.allclose(velocity, changed)

    velocity.square().mean().backward()
    assert inputs["state"].grad is not None
    assert inputs["material_latent"].grad is not None
    assert torch.isfinite(inputs["material_latent"].grad).all()


def test_flow_uses_no_absolute_xyz_and_is_e3_equivariant() -> None:
    torch.manual_seed(17)
    model = PhysicsConditionedCortexODE(hidden_dim=16, hidden_layers=2)
    inputs = _flow_inputs()
    sdf_scalar = torch.randn(6, 2)
    sdf_gradient = torch.randn(6, 2, 3)
    body_context = torch.cat(
        (sdf_scalar[..., None], sdf_gradient),
        dim=-1,
    ).reshape(6, 8)
    baseline = model(
        inputs["state"],
        0.4,
        material_coordinates=inputs["material_coordinates"],
        material_latent=inputs["material_latent"],
        body_context=body_context,
    )
    orthogonal = torch.tensor(
        [[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    translation = torch.tensor([2.0, -1.0, 0.5])
    transformed_fields = torch.cat(
        (sdf_scalar[..., None], sdf_gradient @ orthogonal.T),
        dim=-1,
    ).reshape(6, 8)
    transformed = model(
        inputs["state"] @ orthogonal.T + translation,
        0.4,
        material_coordinates=inputs["material_coordinates"],
        material_latent=inputs["material_latent"],
        body_context=transformed_fields,
    )
    assert torch.allclose(transformed, baseline @ orthogonal.T, atol=2e-5, rtol=2e-5)


def test_every_affine_path_is_spectrally_capped() -> None:
    model = PhysicsConditionedCortexODE(hidden_dim=12, hidden_layers=3, spectral_cap=0.7)
    assert not any(isinstance(module, nn.Linear) for module in model.modules())
    layers = [module for module in model.modules() if isinstance(module, SpectralLinear)]
    assert layers
    for layer in layers:
        sigma = torch.linalg.matrix_norm(layer.effective_weight(), ord=2)
        assert sigma.item() <= layer.max_singular_value + 1.0e-5
    assert model.state_lipschitz_upper_bound() <= 0.25 * 0.7**4 + 1.0e-12


def test_scene_material_latent_is_one_optimisable_scene_code() -> None:
    latent = SceneMaterialLatent(latent_dim=32)
    assert latent().shape == (32,)
    assert latent(5).shape == (5, 32)
    assert latent.prior_loss().ndim == 0


def test_physics_loss_penalises_penetration_and_backpropagates() -> None:
    latent = torch.randn(32, requires_grad=True)
    shell = torch.tensor([0.2, -0.1], requires_grad=True)
    result = physics_constrained_loss(
        material_latent=latent,
        shell_residual=shell,
        contact_gap=torch.tensor([0.01, -0.02, -0.03]),
    )
    assert result.raw_terms["contact"].item() == pytest.approx((0.02**2 + 0.03**2) / 3)
    result.total.backward()
    assert latent.grad is not None
    assert shell.grad is not None


def test_genesis_mpm_prior_is_guarded_by_equivalence_gate() -> None:
    latent = torch.zeros(32)
    weights = PhysicsLossWeights(genesis_mpm=0.5)
    with pytest.raises(ValueError, match="enabled prior mode"):
        physics_constrained_loss(material_latent=latent, weights=weights)
    with pytest.raises(ValueError, match="shell-equivalence"):
        physics_constrained_loss(
            material_latent=latent,
            genesis_mpm_mode=GenesisMPMPriorMode.DIFFERENTIABLE,
            genesis_mpm_residual=torch.ones(3),
            weights=weights,
        )

    result = physics_constrained_loss(
        material_latent=latent,
        genesis_mpm_mode=GenesisMPMPriorMode.DISTILLATION,
        genesis_mpm_residual=torch.tensor([1.0, 2.0]),
        mpm_shell_equivalence_passed=True,
        weights=weights,
    )
    assert result.weighted_terms["genesis_mpm"].item() == pytest.approx(1.25)
