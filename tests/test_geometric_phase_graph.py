import pytest
import torch

from aigc_fashion.geometric_phase_graph import (
    GeometricPhaseGraphState,
    IntrinsicClothPhaseState,
    SE3CanonicalPhaseState,
    SO2ContactPhaseState,
    apply_sdf_contact_momentum_reset,
    canonical_hamiltonian,
    symplectic_euler_phase_step,
)


def _cloth() -> IntrinsicClothPhaseState:
    positions = torch.tensor(
        [[0.0, 0.1, 0.0], [0.2, 0.1, 0.0], [0.0, -0.1, 0.1]]
    )
    velocity = torch.tensor(
        [[0.2, -0.4, 0.1], [-0.1, -0.2, 0.0], [0.3, 0.1, -0.2]]
    )
    return IntrinsicClothPhaseState.from_velocity(
        torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        positions,
        velocity,
        torch.tensor([0.2, 0.3, 0.25]),
        0.0,
    )


def test_symplectic_phase_step_is_e3_equivariant_and_translation_free() -> None:
    state = _cloth()
    force = torch.tensor(
        [[0.1, -0.3, 0.2], [-0.2, 0.1, 0.0], [0.05, 0.2, -0.1]]
    )
    baseline = symplectic_euler_phase_step(state, force, 0.01)
    orthogonal = torch.tensor(
        [[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    translation = torch.tensor([1.0, -2.0, 0.5])
    moved = IntrinsicClothPhaseState(
        state.material_coordinates,
        state.positions @ orthogonal.T + translation,
        state.momenta @ orthogonal.T,
        state.lumped_mass,
        state.time,
    )
    transformed = symplectic_euler_phase_step(moved, force @ orthogonal.T, 0.01)
    assert torch.allclose(
        transformed.positions,
        baseline.positions @ orthogonal.T + translation,
        atol=1e-6,
    )
    assert torch.allclose(transformed.momenta, baseline.momenta @ orthogonal.T, atol=1e-6)
    assert transformed.kinetic_energy() == pytest.approx(float(baseline.kinetic_energy()))


def test_zero_force_preserves_kinetic_hamiltonian() -> None:
    state = _cloth()
    stepped = symplectic_euler_phase_step(state, torch.zeros_like(state.positions), 0.02)
    before = canonical_hamiltonian(state, torch.tensor(1.25))
    after = canonical_hamiltonian(stepped, torch.tensor(1.25))
    assert after == pytest.approx(float(before))


def test_sdf_contact_is_a_dissipative_momentum_reset() -> None:
    state = _cloth()
    normals = torch.tensor([[0.0, 1.0, 0.0]]).expand(3, -1)
    reset = apply_sdf_contact_momentum_reset(
        state,
        normals,
        torch.zeros_like(state.positions),
        torch.tensor([True, True, False]),
        torch.tensor([0.4, 0.4, 0.4]),
    )
    normal_speed = (reset.state.velocities * normals).sum(dim=-1)
    assert torch.all(normal_speed[:2] >= -1e-7)
    assert torch.equal(reset.state.momenta[2], state.momenta[2])
    assert reset.state.kinetic_energy() <= state.kinetic_energy() + 1e-7
    assert torch.allclose(
        reset.impulse,
        reset.normal_impulse + reset.tangent_impulse,
        atol=1e-7,
    )


def test_product_phase_graph_requires_one_physical_clock() -> None:
    cloth = _cloth()
    rigid = SE3CanonicalPhaseState(torch.eye(4).unsqueeze(0), torch.zeros(1, 6), 0.0)
    contact = SO2ContactPhaseState(
        torch.tensor([0.01]),
        torch.tensor([[0.0, 1.0, 0.0]]),
        torch.complex(torch.tensor([0.2]), torch.tensor([-0.1])),
        torch.tensor([False]),
        0.0,
    )
    graph = GeometricPhaseGraphState((cloth,), rigid, contact, 0.0)
    assert graph.total_kinetic_energy() == pytest.approx(float(cloth.kinetic_energy()))
    with pytest.raises(ValueError, match="share physical time"):
        GeometricPhaseGraphState(
            (cloth,),
            SE3CanonicalPhaseState(torch.eye(4).unsqueeze(0), torch.zeros(1, 6), 0.1),
            contact,
            0.0,
        )
