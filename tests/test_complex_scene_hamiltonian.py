import torch

from aigc_fashion.complex_scene_hamiltonian import (
    KahlerClothPhase,
    assemble_complex_scene_hamiltonian,
    hermitian_power_pairing,
    port_hamiltonian_vector_field,
    wirtinger_hamiltonian_vector_field,
)
from aigc_fashion.geometric_phase_graph import IntrinsicClothPhaseState


def _phase() -> IntrinsicClothPhaseState:
    return IntrinsicClothPhaseState(
        material_coordinates=torch.tensor([[0.0, 0.0], [1.0, 0.0]]),
        positions=torch.tensor([[0.2, 0.3, -0.1], [1.1, 0.4, 0.2]]),
        momenta=torch.tensor([[0.01, -0.03, 0.02], [0.04, 0.01, -0.02]]),
        lumped_mass=torch.tensor([0.2, 0.35]),
        time=0.4,
    )


def test_kahler_phase_round_trip_is_lossless() -> None:
    state = _phase()
    reference = state.positions - torch.tensor([0.02, -0.01, 0.03])
    phase = KahlerClothPhase.from_canonical_phase(
        state, reference, reference_angular_frequency=12.0
    )
    decoded = phase.to_canonical_phase()
    assert torch.allclose(decoded.positions, state.positions, atol=1.0e-7)
    assert torch.allclose(decoded.momenta, state.momenta, atol=1.0e-7)
    assert torch.allclose(decoded.material_coordinates, state.material_coordinates)


def test_kahler_coordinates_are_e3_covariant_and_translation_free() -> None:
    state = _phase()
    reference = state.positions - torch.tensor([0.02, -0.01, 0.03])
    baseline = KahlerClothPhase.from_canonical_phase(
        state, reference, reference_angular_frequency=8.0
    )
    rotation = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    translation = torch.tensor([4.0, -2.0, 0.5])
    transformed = IntrinsicClothPhaseState(
        state.material_coordinates,
        state.positions @ rotation.T + translation,
        state.momenta @ rotation.T,
        state.lumped_mass,
        state.time,
    )
    changed = KahlerClothPhase.from_canonical_phase(
        transformed,
        reference @ rotation.T + translation,
        reference_angular_frequency=8.0,
    )
    assert torch.allclose(
        changed.coordinates,
        baseline.coordinates @ rotation.to(baseline.coordinates).T,
        atol=1.0e-6,
    )
    assert torch.allclose(
        changed.coordinates.abs().square().sum(),
        baseline.coordinates.abs().square().sum(),
        atol=1.0e-6,
    )


def test_wirtinger_flow_matches_complex_harmonic_oscillator() -> None:
    omega = 3.2
    z = torch.tensor([1.0 + 2.0j, -0.3 + 0.7j], requires_grad=True)
    hamiltonian = omega * z.abs().square().sum()
    flow = wirtinger_hamiltonian_vector_field(hamiltonian, z)
    assert torch.allclose(flow, -1.0j * omega * z)


def test_wirtinger_flow_decodes_to_canonical_cloth_equations() -> None:
    state = _phase()
    reference = state.positions - torch.tensor([0.02, -0.01, 0.03])
    encoded = KahlerClothPhase.from_canonical_phase(
        state, reference, reference_angular_frequency=7.0
    )
    coordinates = encoded.coordinates.detach().requires_grad_(True)
    phase = KahlerClothPhase(
        coordinates=coordinates,
        lumped_mass=encoded.lumped_mass,
        reference_angular_frequency=encoded.reference_angular_frequency,
        reference_positions=encoded.reference_positions,
        material_coordinates=encoded.material_coordinates,
        time=encoded.time,
    )
    decoded = phase.to_canonical_phase()
    stiffness = 4.5
    displacement = decoded.positions - reference
    hamiltonian = decoded.kinetic_energy() + 0.5 * stiffness * displacement.square().sum()
    flow = wirtinger_hamiltonian_vector_field(hamiltonian, coordinates)
    scale = torch.sqrt(encoded.lumped_mass * encoded.reference_angular_frequency)[:, None]
    position_rate = 2.0**0.5 * flow.real / scale
    momentum_rate = 2.0**0.5 * flow.imag * scale
    assert torch.allclose(position_rate, decoded.velocities, atol=1.0e-6)
    assert torch.allclose(momentum_rate, -stiffness * displacement, atol=1.0e-6)


def test_port_dissipation_has_non_positive_energy_rate() -> None:
    z = torch.tensor([1.0 + 2.0j, -0.3 + 0.7j], requires_grad=True)
    hamiltonian = z.abs().square().sum()
    flow = port_hamiltonian_vector_field(
        hamiltonian, z, dissipation=torch.tensor(0.4)
    )
    gradient = torch.autograd.grad(hamiltonian, z, retain_graph=True)[0]
    energy_rate = hermitian_power_pairing(gradient, flow)
    assert float(energy_rate.detach()) < 0.0


def test_hermitian_port_power_is_gauge_invariant() -> None:
    effort = torch.tensor([1.0 + 2.0j, -0.4 + 0.3j])
    flow = torch.tensor([0.2 - 0.7j, 0.5 + 0.1j])
    phase = torch.exp(torch.tensor(0.73j))
    assert torch.allclose(
        hermitian_power_pairing(effort, flow),
        hermitian_power_pairing(phase * effort, phase * flow),
        atol=1.0e-6,
    )


def test_scene_hamiltonian_ledger_preserves_hierarchy_and_total() -> None:
    ledger = assemble_complex_scene_hamiltonian(
        node_energies=torch.tensor([1.0, 2.0, 3.0, 4.0]),
        relation_energies=torch.tensor([0.5, 0.7, 0.2]),
        node_hierarchy_levels=torch.tensor([0, 1, 2, 2]),
        relation_index=torch.tensor([[0, 1, 2], [1, 2, 3]]),
        constraint_energy=torch.tensor(0.3),
    )
    assert torch.allclose(ledger.total, torch.tensor(11.7))
    assert torch.allclose(ledger.hierarchy_energy[0], torch.tensor(1.0))
    assert torch.allclose(ledger.hierarchy_energy[1], torch.tensor(2.5))
    assert torch.allclose(ledger.hierarchy_energy[2], torch.tensor(7.9))


def test_scene_hamiltonian_rejects_genuinely_complex_energy() -> None:
    try:
        assemble_complex_scene_hamiltonian(
            node_energies=torch.tensor([1.0 + 0.2j]),
            relation_energies=torch.empty(0, dtype=torch.complex64),
            node_hierarchy_levels=torch.tensor([0]),
            relation_index=torch.empty((2, 0), dtype=torch.long),
        )
    except ValueError as error:
        assert "real-valued" in str(error)
    else:
        raise AssertionError("complex physical energy must be rejected")
