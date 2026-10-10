import numpy as np
import torch

from aigc_fashion.data import GarmentMesh
from aigc_fashion.simulation_ready_shell import (
    PanelSemanticGuidance,
    PhysicsVolumeSupervision,
    SimulationReadyIntrinsicShellNet,
    assess_simulation_ready_shell,
    simulation_ready_shell_loss,
)


def _guidance(*, descriptors: bool = False) -> PanelSemanticGuidance:
    masks = torch.zeros(1, 3, 16, 16)
    masks[0, 0, 2:14, 2:8] = 1.0
    masks[0, 1, 2:14, 8:14] = 1.0
    masks[0, 2, 6:11, 5:11] = 1.0
    adjacency = torch.tensor([[[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]])
    valid = torch.tensor([[1.0, 1.0, 0.0]])
    seam = torch.zeros(1, 1, 16, 16, 16)
    seam[:, :, :, :, 7:9] = 1.0
    opening = torch.zeros_like(seam)
    opening[:, :, 7:9, 6:10, 6:10] = 1.0
    if descriptors:
        return PanelSemanticGuidance(
            masks,
            adjacency,
            valid,
            role_descriptors=torch.randn(1, 3, 12),
            seam_volume=seam,
            opening_volume=opening,
        )
    return PanelSemanticGuidance(
        masks,
        adjacency,
        valid,
        role_logits=torch.randn(1, 3, 5),
        seam_volume=seam,
        opening_volume=opening,
    )


def test_joint_ffn_semantic_graph_and_3d_unet_are_differentiable() -> None:
    model = SimulationReadyIntrinsicShellNet(
        role_count=5,
        feature_channels=8,
        semantic_dim=8,
        descriptor_dim=12,
        lift_channels=3,
        material_channels=2,
    )
    image = torch.rand(1, 3, 32, 32)
    coarse = torch.randn(1, 1, 16, 16, 16) * 0.01
    body_sdf = torch.full_like(coarse, 0.02)
    guidance = _guidance()
    prediction = model(image, coarse, body_sdf, guidance)
    assert prediction.closed_proxy_sdf.shape == coarse.shape
    assert prediction.material_field.shape == (1, 2, 16, 16, 16)
    assert prediction.panel_tokens.shape == (1, 3, 8)
    assert torch.count_nonzero(prediction.panel_tokens[:, 2]) == 0

    target = torch.randn_like(coarse) * 0.01
    physics = PhysicsVolumeSupervision(
        residual=torch.rand_like(coarse) * 0.1,
        teacher_sdf=target + 0.001,
        material_target=torch.ones_like(prediction.material_field),
    )
    result = simulation_ready_shell_loss(
        prediction,
        target_sdf=target,
        body_sdf=body_sdf,
        guidance=guidance,
        physics=physics,
        voxel_spacing=(0.01, 0.01, 0.01),
    )
    result.total.backward()
    assert torch.isfinite(result.total)
    assert model.panel_model.image_encoder.network[0].weight.grad is not None
    assert model.volume_network.output2.weight.grad is not None
    assert set(result.terms) == {
        "sdf",
        "occupancy",
        "eikonal",
        "body_contact",
        "seam_alignment",
        "opening",
        "physics_residual",
        "genesis_teacher",
        "material",
        "total",
    }


def test_open_role_descriptors_support_unseen_compositional_roles() -> None:
    model = SimulationReadyIntrinsicShellNet(
        role_count=5,
        feature_channels=8,
        semantic_dim=8,
        descriptor_dim=12,
        lift_channels=2,
        material_channels=2,
    )
    prediction = model(
        torch.rand(1, 3, 32, 32),
        torch.zeros(1, 1, 16, 16, 16),
        torch.ones(1, 1, 16, 16, 16),
        _guidance(descriptors=True),
    )
    assert prediction.panel_tokens.shape == (1, 3, 8)
    assert torch.isfinite(prediction.closed_proxy_sdf).all()


def test_simulation_ready_gate_checks_openings_and_body_intersections() -> None:
    mesh = GarmentMesh(
        vertices=np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float32,
        ),
        faces=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32),
        pinned_vertices=np.empty(0, dtype=np.int32),
        panel_labels=["implicit_shell"] * 4,
        seam_edges=[],
    )
    accepted = assess_simulation_ready_shell(mesh, expected_boundary_loops=1, semantic_link_count=2)
    assert accepted.passed
    assert accepted.semantic_link_count == 2

    rejected = assess_simulation_ready_shell(
        mesh,
        expected_boundary_loops=4,
        body_intersecting_faces=3,
    )
    assert not rejected.passed
    assert "semantic opening count mismatch" in rejected.reasons
    assert "body intersections" in rejected.reasons
