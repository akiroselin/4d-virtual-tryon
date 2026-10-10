import torch

from aigc_fashion.variational_phase_pinn import (
    VariationalPhasePINNConfig,
    assemble_cloth_hamiltonian,
    compile_cloth_hamiltonian_contract,
    metric_strain_barrier_energy,
    variational_phase_pinn_residual,
)
from aigc_fashion.geometric_phase_graph import IntrinsicClothPhaseState


def _free_particle_residual(positions: tuple[torch.Tensor, ...]):
    count = len(positions[0])
    return variational_phase_pinn_residual(
        positions[0],
        positions[1],
        positions[2],
        torch.ones(count),
        torch.zeros((count, 3)),
        torch.zeros(()),
        dt=0.1,
        config=VariationalPhasePINNConfig(
            velocity_damping_n_s_m=0.0,
            friction_coefficient=0.0,
        ),
    )


def test_constant_velocity_has_zero_action_stationarity_residual() -> None:
    current = torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    velocity = torch.tensor([[0.3, -0.2, 0.1], [0.1, 0.0, -0.4]])
    result = _free_particle_residual(
        (current - 0.1 * velocity, current, current + 0.1 * velocity)
    )
    assert float(result.action_stationarity) < 1.0e-6
    assert float(result.energy_balance) < 1.0e-6


def test_variational_residual_is_e3_invariant() -> None:
    previous = torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.9, 0.2]])
    current = torch.tensor([[0.03, 0.0, 0.0], [0.11, 0.88, 0.18]])
    following = torch.tensor([[0.07, -0.01, 0.0], [0.13, 0.85, 0.15]])
    force = torch.tensor([[0.02, -0.03, 0.01], [-0.02, 0.03, -0.01]])
    mass = torch.tensor([0.4, 0.7])
    baseline = variational_phase_pinn_residual(
        previous,
        current,
        following,
        mass,
        force,
        torch.tensor(0.6),
        dt=0.1,
        external_force=torch.tensor([0.0, -0.2, 0.0]),
    )
    rotation = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    translation = torch.tensor([2.0, -3.0, 0.7])
    transformed = variational_phase_pinn_residual(
        previous @ rotation.T + translation,
        current @ rotation.T + translation,
        following @ rotation.T + translation,
        mass,
        force @ rotation.T,
        torch.tensor(0.6),
        dt=0.1,
        external_force=torch.tensor([0.0, -0.2, 0.0]) @ rotation.T,
    )
    assert torch.allclose(
        baseline.action_stationarity,
        transformed.action_stationarity,
        atol=1.0e-5,
    )
    assert torch.allclose(baseline.energy_balance, transformed.energy_balance, atol=1.0e-5)


def test_contact_kkt_and_dissipation_backpropagate() -> None:
    previous = torch.tensor([[0.0, 0.001, 0.0]], requires_grad=True)
    current = torch.tensor([[0.01, 0.001, 0.0]], requires_grad=True)
    following = torch.tensor([[0.02, 0.001, 0.0]], requires_grad=True)
    result = variational_phase_pinn_residual(
        previous,
        current,
        following,
        torch.ones(1),
        torch.zeros((1, 3)),
        torch.zeros(()),
        dt=0.1,
        contact_gap=torch.tensor([0.001]),
        contact_normals=torch.tensor([[0.0, 1.0, 0.0]]),
    )
    assert float(result.contact_kkt) > 0.0
    assert float(result.contact_multiplier[0]) > 0.0
    friction_power = (result.friction_force[0] * torch.tensor([1.0, 0.0, 0.0])).sum()
    assert float(friction_power.detach()) < 0.0
    result.total.backward()
    assert current.grad is not None
    assert torch.isfinite(current.grad).all()


def test_compiled_hamiltonian_separates_conservative_and_dissipative_terms() -> None:
    contract = compile_cloth_hamiltonian_contract(has_contact=True, has_seams=True)
    assert any("membrane" in term for term in contract.conservative_terms)
    assert any("friction" in term for term in contract.dalembert_ports)
    assert all("friction" not in term for term in contract.conservative_terms)
    assert "immutable xi" in contract.reference_map_fiber_role
    assert "mu_vertex+mu_face+mu_hinge" in contract.bundle_measure_state
    assert "Div_E" in contract.bundle_measure_transport
    assert "momentum fibre" in contract.bundle_measure_contact_reset
    assert "E(3)" in contract.generalized_flow_equation
    assert "measure-valued" in contract.hybrid_event_flow
    state = IntrinsicClothPhaseState.from_velocity(
        torch.zeros((2, 2)),
        torch.tensor([[0.0, 1.0, 0.0], [0.0, 2.0, 0.0]]),
        torch.zeros((2, 3)),
        torch.tensor([0.2, 0.3]),
        0.0,
    )
    hamiltonian = assemble_cloth_hamiltonian(
        state,
        torch.tensor([0.4]),
        torch.tensor([0.1]),
    )
    expected_gravity = 9.81 * (0.2 * 1.0 + 0.3 * 2.0)
    assert torch.allclose(hamiltonian.gravity, torch.tensor(expected_gravity))
    assert torch.allclose(hamiltonian.total, torch.tensor(0.5 + expected_gravity))


def test_metric_barrier_uses_the_same_hard_strain_envelope() -> None:
    rest = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    edges = torch.tensor([[0, 1]])
    rest_lengths = torch.ones(1)
    feasible = metric_strain_barrier_energy(
        rest * torch.tensor([1.1, 1.0, 1.0]),
        edges,
        rest_lengths,
        maximum_tensile_strain=0.18,
        maximum_compressive_strain=0.10,
    )
    stretched = metric_strain_barrier_energy(
        rest * torch.tensor([1.3, 1.0, 1.0]),
        edges,
        rest_lengths,
        maximum_tensile_strain=0.18,
        maximum_compressive_strain=0.10,
    )
    assert float(feasible) == 0.0
    assert float(stretched) > 0.0
