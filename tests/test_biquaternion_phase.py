import torch

from aigc_fashion.biquaternion_phase import (
    BiquaternionVertexPhase,
    complex_kirchhoff_love_quadratic_energy,
    hamilton_product,
    membrane_strain_to_complex,
)
from aigc_fashion.geometric_phase_graph import IntrinsicClothPhaseState


def test_extended_phase_biquaternion_is_lossless_and_satisfies_energy_shell() -> None:
    positions = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    reference = positions - 0.02
    momentum = torch.tensor([[0.01, 0.02, 0.03], [0.04, 0.05, 0.06]])
    state = IntrinsicClothPhaseState(
        torch.zeros((2, 2)), positions, momentum, torch.tensor([0.2, 0.3]), 0.7
    )
    hamiltonian = torch.tensor(2.4)
    encoded = BiquaternionVertexPhase.from_canonical_phase(
        state, reference, hamiltonian=hamiltonian
    )
    assert torch.allclose(encoded.displacement, positions - reference)
    assert torch.allclose(encoded.momenta, momentum)
    assert torch.allclose(encoded.extended_hamiltonian_constraint(hamiltonian), torch.tensor(0.0))


def test_complex_kl_energy_is_invariant_under_material_frame_rotation() -> None:
    tensor = torch.tensor([[[0.12, 0.04], [0.04, -0.03]]])
    angle = torch.tensor(0.63)
    rotation = torch.stack(
        (
            torch.stack((torch.cos(angle), -torch.sin(angle))),
            torch.stack((torch.sin(angle), torch.cos(angle))),
        )
    )
    rotated = rotation.T @ tensor @ rotation
    base_vector = torch.stack((tensor[:, 0, 0], tensor[:, 1, 1], 2.0 * tensor[:, 0, 1]), dim=-1)
    rotated_vector = torch.stack(
        (rotated[:, 0, 0], rotated[:, 1, 1], 2.0 * rotated[:, 0, 1]), dim=-1
    )
    base = membrane_strain_to_complex(base_vector)
    changed = membrane_strain_to_complex(rotated_vector)
    base_energy = complex_kirchhoff_love_quadratic_energy(
        base, torch.ones(1), bulk_modulus=2.0, shear_modulus=0.7
    )
    changed_energy = complex_kirchhoff_love_quadratic_energy(
        changed, torch.ones(1), bulk_modulus=2.0, shear_modulus=0.7
    )
    assert torch.allclose(base.trace, changed.trace, atol=1.0e-6)
    assert torch.allclose(base.deviator.abs(), changed.deviator.abs(), atol=1.0e-6)
    assert torch.allclose(base_energy, changed_energy, atol=1.0e-6)


def test_hamilton_product_matches_quaternion_basis_rule() -> None:
    i = torch.tensor([0.0, 1.0, 0.0, 0.0])
    j = torch.tensor([0.0, 0.0, 1.0, 0.0])
    assert torch.equal(hamilton_product(i, j), torch.tensor([0.0, 0.0, 0.0, 1.0]))
