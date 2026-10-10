from __future__ import annotations

from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aigc_fashion.cloth4d_geometric_bridge import (  # noqa: E402
    Cloth4DGeometricConditioner,
    GaugeTactileVectorEncoder,
    body_anchors_from_se3,
    query_conditioned_cloth4d,
)
from aigc_fashion.e3_anchor_field import E3AnchorNeuralClothField  # noqa: E402
from aigc_fashion.gauge_tactile import gauge_transform, transform_connection  # noqa: E402
from aigc_fashion.genesis_tactile_adapter import rotate_tangent_gauge  # noqa: E402
from aigc_fashion.lie_group_diffusion import se3_exp  # noqa: E402


def _square_inputs() -> dict[str, torch.Tensor]:
    torch.manual_seed(41)
    positions = torch.tensor(
        [[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0], [0.5, 1.0, 0.1], [-0.5, 1.0, 0.1]]
    )
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]])
    rest = torch.linalg.vector_norm(
        positions[edge_index[1]] - positions[edge_index[0]],
        dim=-1,
    )
    tangent_u = torch.tensor([[1.0, 0.0, 0.0]]).expand(4, -1).clone()
    tangent_v = torch.tensor([[0.0, 1.0, 0.0]]).expand(4, -1).clone()
    return {
        "surface_features": torch.randn(4, 2),
        "laplace_eigenvalues": torch.tensor([0.0, 1.0, 2.0, 3.0]),
        "laplace_eigenvectors": torch.eye(4),
        "vertex_mass": torch.ones(4),
        "particle_positions": positions,
        "particle_velocities": torch.randn(4, 3) * 0.1,
        "particle_edge_index": edge_index,
        "rest_lengths": rest,
        "cotangent_weights": torch.ones(4),
        "material_latent": torch.randn(5),
        "scene_scalar_features": torch.randn(4, 3),
        "scene_positions": positions + torch.tensor([0.2, -0.1, 0.3]),
        "scene_times": torch.tensor([0.0, 0.1, 0.2, 0.3]),
        "scene_hierarchy_levels": torch.tensor([0.0, 1.0, 2.0, 1.0]),
        "scene_edge_index": edge_index,
        "scene_relation_ids": torch.tensor([0, 1, 2, 1]),
        "query_scene_node_ids": torch.arange(4),
        "tactile_features": torch.complex(torch.randn(4, 1), torch.randn(4, 1)),
        "tactile_edge_index": edge_index,
        "tactile_connection_angle": torch.tensor([0.0, 0.1, -0.2, 0.15]),
        "tactile_tangent_u": tangent_u,
        "tactile_tangent_v": tangent_v,
        "tactile_normal_force": torch.rand(4, 1),
    }


def test_tactile_vector_encoder_is_independent_of_local_gauge() -> None:
    torch.manual_seed(43)
    encoder = GaugeTactileVectorEncoder(1)
    inputs = _square_inputs()
    baseline = encoder(
        inputs["tactile_features"],
        inputs["tactile_edge_index"],
        inputs["tactile_connection_angle"],
        inputs["tactile_tangent_u"],
        inputs["tactile_tangent_v"],
    )
    angle = torch.tensor([0.3, -0.4, 0.7, -1.1])
    moved_u, moved_v = rotate_tangent_gauge(
        inputs["tactile_tangent_u"],
        inputs["tactile_tangent_v"],
        angle,
    )
    moved = encoder(
        gauge_transform(inputs["tactile_features"], angle),
        inputs["tactile_edge_index"],
        transform_connection(
            inputs["tactile_connection_angle"],
            inputs["tactile_edge_index"],
            angle,
        ),
        moved_u,
        moved_v,
    )
    assert torch.allclose(moved, baseline, atol=3.0e-5, rtol=3.0e-5)


def test_conditioner_combines_all_geometric_paths_and_is_e3_equivariant() -> None:
    torch.manual_seed(47)
    conditioner = Cloth4DGeometricConditioner(
        2,
        3,
        1,
        material_latent_dim=5,
        hidden_dim=12,
    )
    inputs = _square_inputs()
    baseline = conditioner(**inputs)
    rotation = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    translation = torch.tensor([1.2, -0.6, 0.4])
    moved_inputs = dict(inputs)
    for name in ("particle_positions", "scene_positions"):
        moved_inputs[name] = inputs[name] @ rotation.T + translation
    for name in (
        "particle_velocities",
        "tactile_tangent_u",
        "tactile_tangent_v",
    ):
        moved_inputs[name] = inputs[name] @ rotation.T
    moved = conditioner(**moved_inputs)
    assert torch.allclose(
        moved.cloth_anchor_features,
        baseline.cloth_anchor_features,
        atol=2.0e-5,
        rtol=2.0e-5,
    )
    assert torch.allclose(
        moved.query_scalar_features,
        baseline.query_scalar_features,
        atol=3.0e-5,
        rtol=3.0e-5,
    )
    assert torch.allclose(
        moved.query_vectors,
        baseline.query_vectors @ rotation.T,
        atol=3.0e-5,
        rtol=3.0e-5,
    )
    assert torch.allclose(
        moved.body_tactile_vectors,
        baseline.body_tactile_vectors @ rotation.T,
        atol=3.0e-5,
        rtol=3.0e-5,
    )


def test_scene_conditioning_uses_relative_time_and_hierarchy() -> None:
    torch.manual_seed(53)
    conditioner = Cloth4DGeometricConditioner(
        2,
        3,
        1,
        material_latent_dim=5,
        hidden_dim=12,
    )
    inputs = _square_inputs()
    baseline = conditioner(**inputs)
    shifted = dict(inputs)
    shifted["scene_times"] = inputs["scene_times"] + 19.0
    shifted_output = conditioner(**shifted)
    assert torch.allclose(
        shifted_output.query_scalar_features,
        baseline.query_scalar_features,
        atol=2.0e-5,
        rtol=2.0e-5,
    )
    assert torch.allclose(
        shifted_output.query_vectors,
        baseline.query_vectors,
        atol=2.0e-5,
        rtol=2.0e-5,
    )

    changed = dict(inputs)
    changed["scene_hierarchy_levels"] = torch.tensor([0.0, 0.0, 0.0, 0.0])
    hierarchy_output = conditioner(**changed)
    assert not torch.allclose(
        hierarchy_output.query_scalar_features,
        baseline.query_scalar_features,
    )


def test_se3_body_anchor_adapter_is_equivariant_to_global_left_action() -> None:
    local_positions = torch.tensor(
        [[0.0, 0.0, 0.2], [0.3, 0.1, 0.0], [-0.2, 0.2, 0.1]]
    )
    local_normals = torch.tensor(
        [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]
    )
    local_velocities = torch.tensor(
        [[0.0, 0.1, 0.0], [0.1, 0.0, 0.0], [0.0, -0.1, 0.0]]
    )
    transforms = se3_exp(
        torch.tensor(
            [
                [0.1, 0.0, 0.0, 0.2, 0.0, 0.0],
                [0.0, -0.2, 0.1, -0.3, 0.4, 0.1],
            ]
        )
    )
    ids = torch.tensor([0, 1, 1])
    baseline = body_anchors_from_se3(
        local_positions,
        local_normals,
        transforms,
        ids,
        local_velocities=local_velocities,
    )
    global_transform = se3_exp(torch.tensor([0.2, -0.1, 0.3, 1.0, -0.5, 0.4]))
    moved = body_anchors_from_se3(
        local_positions,
        local_normals,
        global_transform @ transforms,
        ids,
        local_velocities=local_velocities,
    )
    rotation = global_transform[:3, :3]
    translation = global_transform[:3, 3]
    assert torch.allclose(
        moved.positions,
        baseline.positions @ rotation.T + translation,
        atol=2.0e-5,
        rtol=2.0e-5,
    )
    assert torch.allclose(moved.normals, baseline.normals @ rotation.T, atol=2.0e-5)
    assert torch.allclose(moved.velocities, baseline.velocities @ rotation.T, atol=2.0e-5)


def test_full_conditioner_queries_one_unified_anchor_field() -> None:
    torch.manual_seed(59)
    conditioner = Cloth4DGeometricConditioner(
        2,
        3,
        1,
        material_latent_dim=5,
        hidden_dim=12,
    )
    inputs = _square_inputs()
    conditioning = conditioner(**inputs)
    field = E3AnchorNeuralClothField(
        1,
        material_latent_dim=5,
        query_feature_dim=3,
        cloth_anchor_feature_dim=2,
        body_anchor_feature_dim=1,
        query_vector_feature_dim=2,
        hidden_dim=12,
        hidden_layers=2,
    )
    material = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    )
    body_positions = inputs["particle_positions"] + torch.tensor([0.0, 0.0, -0.2])
    body_normals = torch.tensor([[0.0, 0.0, 1.0]]).expand(4, -1)
    output = query_conditioned_cloth4d(
        field,
        conditioning,
        material_coordinates=material,
        physical_time=torch.linspace(0.0, 0.3, 4)[:, None],
        panel_ids=torch.zeros(4, dtype=torch.long),
        material_latent=inputs["material_latent"],
        cloth_anchor_positions=inputs["particle_positions"],
        cloth_anchor_material_coordinates=material,
        cloth_anchor_panel_ids=torch.zeros(4, dtype=torch.long),
        body_anchor_positions=body_positions,
        body_anchor_normals=body_normals,
        body_anchor_velocities=torch.zeros_like(body_positions),
    )
    assert output.positions.shape == (4, 3)
    assert output.query_vector_attention.shape == (4, 2)
    assert torch.isfinite(output.positions).all()
