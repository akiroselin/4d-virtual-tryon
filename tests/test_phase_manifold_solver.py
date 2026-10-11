import torch

from aigc_fashion.neuralclothsim_dynamics import BodyContactSample
from aigc_fashion.phase_manifold_solver import (
    IncrementalHamiltonianConfig,
    PhasePotentialTerms,
    reference_surface_contact_potential,
    solve_incremental_hamiltonian_step,
)


def _zero_terms(reference: torch.Tensor, *, hypernode: torch.Tensor) -> PhasePotentialTerms:
    zero = reference.new_zeros(())
    return PhasePotentialTerms(hypernode, zero, zero, zero, zero, zero, zero)


def test_se3_generative_energy_is_part_of_total_hamiltonian() -> None:
    zero = torch.tensor(0.0)
    generative = torch.tensor(0.25)
    terms = PhasePotentialTerms(zero, zero, zero, zero, zero, zero, zero, se3_generative=generative)

    assert terms.total == generative
    assert terms.detached_energies()["se3_generative_j"] == 0.25


def test_hypernode_energy_participates_in_next_phase_frame_solve() -> None:
    position = torch.zeros((1, 3))
    velocity = torch.zeros_like(position)
    mass = torch.ones(1)
    target = torch.tensor([[0.04, 0.0, 0.0]])

    def potential(candidate: torch.Tensor) -> PhasePotentialTerms:
        energy = 0.5 * 20.0 * (candidate - target).square().sum()
        return _zero_terms(candidate, hypernode=energy)

    step = solve_incremental_hamiltonian_step(
        position,
        velocity,
        mass,
        dt=0.1,
        potential=potential,
        config=IncrementalHamiltonianConfig(
            iterations=12,
            line_search_steps=6,
            gradient_tolerance_n=1.0e-6,
            maximum_iteration_displacement_m=0.02,
        ),
    )
    assert step.positions[0, 0] > 0.0
    assert step.momenta[0, 0] > 0.0
    assert step.diagnostics.energy_terms["hypernode_membrane_j"] > 0.0
    assert (
        step.diagnostics.final_incremental_potential_j
        < step.diagnostics.initial_incremental_potential_j
    )


def test_intrinsic_direction_preconditioner_participates_in_phase_solve() -> None:
    position = torch.zeros((2, 3))
    velocity = torch.zeros_like(position)
    mass = torch.ones(2)
    target = torch.tensor([[0.02, 0.0, 0.0], [0.02, 0.0, 0.0]])
    calls = 0

    def potential(candidate: torch.Tensor) -> PhasePotentialTerms:
        energy = 0.5 * 10.0 * (candidate - target).square().sum()
        return _zero_terms(candidate, hypernode=energy)

    def precondition(direction: torch.Tensor) -> torch.Tensor:
        nonlocal calls
        calls += 1
        return direction.mean(dim=0, keepdim=True).expand_as(direction)

    step = solve_incremental_hamiltonian_step(
        position,
        velocity,
        mass,
        dt=0.1,
        potential=potential,
        direction_preconditioner=precondition,
        config=IncrementalHamiltonianConfig(
            iterations=4,
            line_search_steps=4,
            gradient_tolerance_n=1.0e-6,
            maximum_iteration_displacement_m=0.01,
        ),
    )

    assert calls > 0
    assert torch.all(step.positions[:, 0] > 0.0)


def test_non_descent_preconditioner_falls_back_to_inverse_mass_gradient() -> None:
    position = torch.zeros((1, 3))
    velocity = torch.zeros_like(position)
    mass = torch.ones(1)
    target = torch.tensor([[0.02, 0.0, 0.0]])

    def potential(candidate: torch.Tensor) -> PhasePotentialTerms:
        energy = 0.5 * 10.0 * (candidate - target).square().sum()
        return _zero_terms(candidate, hypernode=energy)

    step = solve_incremental_hamiltonian_step(
        position,
        velocity,
        mass,
        dt=0.1,
        potential=potential,
        direction_preconditioner=lambda direction: -direction,
        config=IncrementalHamiltonianConfig(
            iterations=4,
            line_search_steps=4,
            gradient_tolerance_n=1.0e-6,
            maximum_iteration_displacement_m=0.01,
        ),
    )

    assert step.positions[0, 0] > 0.0
    assert (
        step.diagnostics.final_incremental_potential_j
        < step.diagnostics.initial_incremental_potential_j
    )


def test_reference_area_contact_energy_is_invariant_to_uniform_refinement() -> None:
    coarse = torch.tensor(
        [
            [0.0, 0.001, 0.0],
            [1.0, 0.001, 0.0],
            [0.0, 0.001, 1.0],
            [1.0, 0.001, 1.0],
        ]
    )
    coarse_faces = torch.tensor([[0, 1, 2], [1, 3, 2]])
    coarse_areas = torch.full((2,), 0.5)
    fine = torch.tensor(
        [
            [0.0, 0.001, 0.0],
            [0.5, 0.001, 0.0],
            [1.0, 0.001, 0.0],
            [0.0, 0.001, 0.5],
            [0.5, 0.001, 0.5],
            [1.0, 0.001, 0.5],
            [0.0, 0.001, 1.0],
            [0.5, 0.001, 1.0],
            [1.0, 0.001, 1.0],
        ]
    )
    fine_faces = torch.tensor(
        [
            [0, 1, 3], [1, 4, 3],
            [1, 2, 4], [2, 5, 4],
            [3, 4, 6], [4, 7, 6],
            [4, 5, 7], [5, 8, 7],
        ]
    )
    fine_areas = torch.full((8,), 0.125)

    def plane(points: torch.Tensor, _time: float) -> BodyContactSample:
        closest = points.clone()
        closest[:, 1] = 0.0
        normals = torch.zeros_like(points)
        normals[:, 1] = 1.0
        return BodyContactSample(closest, normals)

    coarse_energy, coarse_gap, coarse_count = reference_surface_contact_potential(
        coarse,
        coarse_faces,
        coarse_areas,
        plane,
        time=0.0,
        clearance_m=0.003,
        pressure_stiffness_n_m3=2.0e5,
    )
    fine_energy, fine_gap, fine_count = reference_surface_contact_potential(
        fine,
        fine_faces,
        fine_areas,
        plane,
        time=0.0,
        clearance_m=0.003,
        pressure_stiffness_n_m3=2.0e5,
    )
    assert torch.allclose(coarse_energy, fine_energy, rtol=1.0e-6, atol=1.0e-8)
    assert torch.allclose(coarse_gap, fine_gap)
    assert coarse_count == 6
    assert fine_count == 24
