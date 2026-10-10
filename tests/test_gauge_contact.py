import torch

from aigc_fashion.gauge_contact import GaugeEquivariantContactCNN
from aigc_fashion.genesis_tactile_adapter import (
    rotate_tangent_gauge,
    tangent_frames_from_normals,
)


def _contact_patch() -> tuple[torch.Tensor, ...]:
    normals = torch.tensor(
        [
            [0.0, 1.0, 0.0],
            [0.0, 0.995, 0.1],
            [0.1, 0.995, 0.0],
            [0.05, 0.997, 0.05],
        ],
        dtype=torch.float32,
    )
    normals = torch.nn.functional.normalize(normals, dim=-1)
    relative_velocity = torch.tensor(
        [
            [0.4, -0.1, 0.2],
            [0.3, -0.2, -0.1],
            [-0.2, -0.05, 0.25],
            [0.15, 0.02, -0.3],
        ]
    )
    penetration = torch.tensor([0.002, 0.001, 0.0025, 0.0])
    active = penetration > 0.0
    edges = torch.tensor([[0, 1, 2, 0], [1, 2, 0, 3]])
    return relative_velocity, normals, penetration, active, edges


def test_contact_cnn_is_invariant_to_independent_tangent_gauge_rotations() -> None:
    torch.manual_seed(101)
    model = GaugeEquivariantContactCNN(5)
    # Exercise a non-neutral trained head while retaining the bounded output.
    torch.nn.init.normal_(model.friction_head.weight, std=0.2)
    velocity, normals, penetration, active, edges = _contact_patch()
    tangent_u, tangent_v = tangent_frames_from_normals(normals)
    baseline = model(
        velocity,
        normals,
        penetration,
        active,
        edges,
        clearance_m=0.003,
        tangent_u=tangent_u,
        tangent_v=tangent_v,
    )
    angle = torch.tensor([0.7, -0.4, 1.2, -0.9])
    moved_u, moved_v = rotate_tangent_gauge(tangent_u, tangent_v, angle)
    moved = model(
        velocity,
        normals,
        penetration,
        active,
        edges,
        clearance_m=0.003,
        tangent_u=moved_u,
        tangent_v=moved_v,
    )
    assert torch.allclose(moved.friction_scale, baseline.friction_scale, atol=2e-5, rtol=2e-5)
    assert torch.allclose(moved.tangent_speed, baseline.tangent_speed, atol=2e-6, rtol=2e-6)


def test_fresh_contact_cnn_is_neutral_and_bounded() -> None:
    model = GaugeEquivariantContactCNN(4, maximum_log_friction_scale=0.2)
    velocity, normals, penetration, active, edges = _contact_patch()
    response = model(
        velocity,
        normals,
        penetration,
        active,
        edges,
        clearance_m=0.003,
    )
    assert torch.equal(response.friction_scale, torch.ones_like(response.friction_scale))
    assert response.active_edge_count == 6
    assert torch.isfinite(response.tangent_speed).all()
