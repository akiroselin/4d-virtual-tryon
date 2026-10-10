import torch

from aigc_fashion.hypernetwork_kirchhoff_love import (
    HyperNetworkKirchhoffLoveShell,
    build_kirchhoff_love_shell_geometry,
)
from aigc_fashion.structure_group_4d import (
    Cloth4DGroupPhaseState,
    Cloth4DStructureGroup,
    UnitDualQuaternion,
    hypernode_structure_group_loss,
)


def _pose() -> UnitDualQuaternion:
    angle = torch.tensor(0.7)
    rotation = torch.stack(
        (
            torch.cos(0.5 * angle),
            torch.tensor(0.0),
            torch.sin(0.5 * angle),
            torch.tensor(0.0),
        )
    )
    return UnitDualQuaternion.from_rotation_translation(
        rotation, torch.tensor([0.2, -0.1, 0.3])
    )


def test_dual_quaternion_sign_is_the_same_se3_group_element() -> None:
    pose = _pose()
    points = torch.tensor([[0.1, 0.2, 0.3], [-0.2, 0.4, 0.1]])
    expected = pose.transform_points(points)
    actual = pose.sign_flipped().transform_points(points)
    unit, study = pose.study_residual()
    assert torch.allclose(actual, expected, atol=1.0e-6)
    assert torch.allclose(unit, torch.zeros_like(unit), atol=1.0e-6)
    assert torch.allclose(study, torch.zeros_like(study), atol=1.0e-6)


def test_product_group_acts_on_typed_phase_fibres() -> None:
    state = Cloth4DGroupPhaseState(
        positions=torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.2, 0.3]]),
        momenta=torch.tensor([[0.0, 1.0, 0.0], [0.2, 0.0, -0.1]]),
        face_spin3=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        contact_tangent_momenta=torch.tensor([1.0 + 2.0j]),
        time=0.4,
    )
    gauge = torch.polar(torch.tensor(1.0), torch.tensor(0.35))
    group = Cloth4DStructureGroup(_pose(), gauge)
    transformed = state.acted_on(group)
    assert transformed.positions.shape == state.positions.shape
    assert torch.allclose(
        torch.linalg.vector_norm(transformed.momenta, dim=-1),
        torch.linalg.vector_norm(state.momenta, dim=-1),
        atol=1.0e-6,
    )
    assert torch.allclose(
        transformed.contact_tangent_momenta,
        gauge * state.contact_tangent_momenta,
    )
    assert torch.allclose(
        torch.linalg.vector_norm(transformed.face_spin3, dim=-1),
        torch.ones(1),
        atol=1.0e-6,
    )


def test_hypernode_kl_energy_and_forces_follow_se3_structure_group() -> None:
    rest = torch.tensor(
        [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.1, 0.1, 0.0]]
    )
    faces = torch.tensor([[0, 1, 2], [1, 3, 2]])
    geometry = build_kirchhoff_love_shell_geometry(
        rest[:, :2], faces, torch.zeros(4, dtype=torch.long), reference_positions=rest
    )
    model = HyperNetworkKirchhoffLoveShell(
        1, latent_dim=3, hyper_hidden_dim=8, node_hidden_dim=3, node_steps=2
    )
    current = rest.clone()
    current[3, 2] = 0.008
    current = current.requires_grad_(True)
    latent = torch.zeros(3)
    original = model(current, geometry, latent, create_graph=True)
    pose = _pose()
    transformed_positions = pose.transform_points(current.detach()).requires_grad_(True)
    transformed = model(
        transformed_positions, geometry, latent, create_graph=True
    )
    acceleration = torch.tensor(
        [[0.1, -0.2, 0.3], [0.0, 0.4, -0.1], [0.2, 0.1, 0.0], [-0.1, 0.0, 0.2]]
    )
    loss = hypernode_structure_group_loss(
        original,
        transformed,
        current,
        world_rotation=pose.real,
        original_acceleration=acceleration,
        transformed_acceleration=pose.transform_vectors(acceleration),
    )
    assert loss.force_equivariance < 1.0e-5
    assert loss.acceleration_equivariance < 1.0e-10
    assert loss.hamiltonian_invariance < 1.0e-5
