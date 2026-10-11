from __future__ import annotations

from pathlib import Path
import sys

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aigc_fashion.e3_anchor_field import E3AnchorNeuralClothField  # noqa: E402
from aigc_fashion.physics_conditioned_flow import SpectralLinear  # noqa: E402
from aigc_fashion.scene_ndf_objective import (  # noqa: E402
    pointwise_time_derivatives,
    scene_ndf_physics_objective,
)


def _model() -> E3AnchorNeuralClothField:
    torch.manual_seed(31)
    return E3AnchorNeuralClothField(
        2,
        material_latent_dim=5,
        query_feature_dim=3,
        cloth_anchor_feature_dim=2,
        body_anchor_feature_dim=1,
        query_vector_feature_dim=2,
        panel_embedding_dim=4,
        hidden_dim=16,
        hidden_layers=2,
        displacement_limit=0.3,
    )


def _inputs() -> dict[str, torch.Tensor]:
    torch.manual_seed(37)
    return {
        "material_coordinates": torch.tensor(
            [[0.1, 0.2], [0.6, 0.3], [0.2, 0.8], [0.3, 0.2], [0.7, 0.7]]
        ),
        "physical_time": torch.linspace(0.1, 0.5, 5)[:, None],
        "panel_ids": torch.tensor([0, 0, 0, 1, 1]),
        "material_latent": torch.randn(5),
        "cloth_anchor_positions": torch.tensor(
            [
                [-0.4, 0.0, 0.2],
                [0.4, 0.0, 0.2],
                [0.0, 0.7, 0.25],
                [-0.35, 0.0, -0.2],
                [0.35, 0.0, -0.2],
                [0.0, 0.7, -0.25],
            ]
        ),
        "cloth_anchor_material_coordinates": torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]] * 2
        ),
        "cloth_anchor_panel_ids": torch.tensor([0, 0, 0, 1, 1, 1]),
        "cloth_anchor_features": torch.randn(6, 2),
        "body_anchor_positions": torch.tensor(
            [[-0.3, 0.2, 0.0], [0.3, 0.2, 0.0], [0.0, 0.8, 0.0], [0.0, -0.2, 0.0]]
        ),
        "body_anchor_features": torch.rand(4, 1),
        "body_anchor_normals": torch.tensor(
            [[-0.7, 0.1, 0.7], [0.7, 0.1, 0.7], [0.0, 0.8, 0.6], [0.0, -1.0, 0.0]]
        ),
        "body_anchor_velocities": torch.randn(4, 3) * 0.1,
        "body_tactile_vectors": torch.randn(4, 3) * 0.05,
        "query_scalar_features": torch.randn(5, 3),
        "query_vectors": torch.randn(5, 3, 3),
        "query_vector_features": torch.randn(5, 3, 2),
        "initial_velocity": torch.randn(5, 3) * 0.03,
        "boundary_weight": torch.tensor([[1.0], [0.8], [1.0], [0.4], [1.0]]),
    }


def _transform_inputs(
    inputs: dict[str, torch.Tensor],
    orthogonal: torch.Tensor,
    translation: torch.Tensor,
) -> dict[str, torch.Tensor]:
    transformed = dict(inputs)
    for name in ("cloth_anchor_positions", "body_anchor_positions", "query_reference_positions"):
        if name not in inputs:
            continue
        transformed[name] = inputs[name] @ orthogonal.T + translation
    for name in (
        "body_anchor_normals",
        "body_anchor_velocities",
        "body_tactile_vectors",
        "query_vectors",
        "initial_velocity",
    ):
        transformed[name] = inputs[name] @ orthogonal.T
    return transformed


def test_anchor_field_is_exactly_e3_equivariant_including_reflection() -> None:
    model = _model()
    inputs = _inputs()
    baseline = model(**inputs)
    orthogonal = torch.tensor(
        [[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    assert torch.linalg.det(orthogonal) < 0.0
    translation = torch.tensor([1.7, -0.8, 0.4])
    moved = model(**_transform_inputs(inputs, orthogonal, translation))
    assert torch.allclose(
        moved.positions,
        baseline.positions @ orthogonal.T + translation,
        atol=3.0e-5,
        rtol=3.0e-5,
    )
    assert torch.allclose(
        moved.residual_vector,
        baseline.residual_vector @ orthogonal.T,
        atol=3.0e-5,
        rtol=3.0e-5,
    )
    assert torch.allclose(moved.cloth_attention, baseline.cloth_attention, atol=2.0e-5)
    assert torch.allclose(moved.body_attention, baseline.body_attention, atol=2.0e-5)
    assert torch.allclose(
        moved.query_vector_attention,
        baseline.query_vector_attention,
        atol=2.0e-5,
    )


def test_anchor_field_is_invariant_to_anchor_bank_permutations() -> None:
    model = _model()
    inputs = _inputs()
    baseline = model(**inputs)
    cloth_permutation = torch.tensor([4, 1, 5, 0, 3, 2])
    body_permutation = torch.tensor([2, 0, 3, 1])
    vector_permutation = torch.tensor([2, 0, 1])
    moved = dict(inputs)
    for name in (
        "cloth_anchor_positions",
        "cloth_anchor_material_coordinates",
        "cloth_anchor_panel_ids",
        "cloth_anchor_features",
    ):
        moved[name] = inputs[name][cloth_permutation]
    for name in (
        "body_anchor_positions",
        "body_anchor_features",
        "body_anchor_normals",
        "body_anchor_velocities",
        "body_tactile_vectors",
    ):
        moved[name] = inputs[name][body_permutation]
    moved["query_vectors"] = inputs["query_vectors"][:, vector_permutation]
    moved["query_vector_features"] = inputs["query_vector_features"][:, vector_permutation]
    permuted = model(**moved)
    assert torch.allclose(permuted.positions, baseline.positions, atol=2.0e-5, rtol=2.0e-5)
    assert torch.allclose(
        permuted.residual_vector,
        baseline.residual_vector,
        atol=2.0e-5,
        rtol=2.0e-5,
    )


def test_anchor_field_enforces_initial_position_and_velocity() -> None:
    model = _model()
    inputs = _inputs()
    time = torch.zeros(5, 1, requires_grad=True)
    initial_velocity = inputs["initial_velocity"].clone()
    inputs["physical_time"] = time
    output = model(**inputs)
    velocity, acceleration = pointwise_time_derivatives(output.positions, time)
    assert torch.allclose(output.positions, output.cloth_reference, atol=1.0e-7)
    assert torch.allclose(velocity, initial_velocity, atol=2.0e-6, rtol=2.0e-6)
    assert torch.isfinite(acceleration).all()


def test_sparse_anchors_can_use_an_exact_typed_query_reference() -> None:
    model = _model()
    inputs = _inputs()
    exact_reference = torch.randn(5, 3)
    inputs["query_reference_positions"] = exact_reference
    inputs["physical_time"] = torch.zeros(5, 1)
    output = model(**inputs)
    assert torch.equal(output.positions, exact_reference)
    assert torch.equal(output.cloth_reference, exact_reference)

    orthogonal = torch.tensor(
        [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]]
    )
    translation = torch.tensor([-0.3, 0.8, 1.2])
    moved = model(**_transform_inputs(inputs, orthogonal, translation))
    assert torch.allclose(
        moved.positions,
        exact_reference @ orthogonal.T + translation,
        atol=2.0e-6,
    )


def test_anchor_field_uses_only_spectrally_capped_affine_paths() -> None:
    model = _model()
    assert not any(isinstance(module, nn.Linear) for module in model.modules())
    layers = [module for module in model.modules() if isinstance(module, SpectralLinear)]
    assert layers
    for layer in layers:
        sigma = torch.linalg.matrix_norm(layer.effective_weight(), ord=2)
        assert sigma <= layer.max_singular_value + 1.0e-5


def test_anchor_field_runs_through_genesis_constrained_ndf_objective() -> None:
    model = _model()
    inputs = _inputs()
    time = inputs["physical_time"].clone().requires_grad_(True)
    latent = inputs["material_latent"].clone().requires_grad_(True)
    body = inputs["body_anchor_positions"].clone().requires_grad_(True)
    inputs.update(
        physical_time=time,
        material_latent=latent,
        body_anchor_positions=body,
    )
    field = model(**inputs)
    velocity, _ = pointwise_time_derivatives(field.positions, time)
    result = scene_ndf_physics_objective(
        positions=field.positions,
        physical_time=time,
        material_latent=latent,
        mass_per_vertex=torch.full((5,), 0.2),
        internal_force=torch.zeros_like(field.positions),
        external_force=torch.zeros_like(field.positions),
        genesis_positions=field.positions.detach() + 0.01,
        genesis_velocities=velocity.detach() - 0.01,
    )
    result.total.backward()
    assert latent.grad is not None and torch.isfinite(latent.grad).all()
    assert body.grad is not None and torch.isfinite(body.grad).all()
    assert any(parameter.grad is not None for parameter in model.parameters())
