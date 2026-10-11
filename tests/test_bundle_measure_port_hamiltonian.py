from dataclasses import replace

import pytest
import torch

from aigc_fashion.bundle_measure_port_hamiltonian import (
    BundleMeasureBuilder,
    BundleMeasureHamiltonian,
    assemble_bundle_measure_hamiltonian,
    bundle_measure_continuity_residual,
    bundle_measure_port_hamiltonian_loss,
    momentum_fibre_push_forward,
    symplectic_euler_measure_step,
)
from aigc_fashion.hypernetwork_kirchhoff_love import (
    HyperNetworkKirchhoffLoveShell,
    build_kirchhoff_love_shell_geometry,
)
from aigc_fashion.reference_map_bundle import ReferenceMapFiberBundle


def _fixture() -> tuple[
    BundleMeasureBuilder,
    HyperNetworkKirchhoffLoveShell,
    torch.Tensor,
    torch.Tensor,
]:
    rest = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]
    )
    faces = torch.tensor([[0, 1, 2], [1, 3, 2]])
    geometry = build_kirchhoff_love_shell_geometry(
        rest[:, :2],
        faces,
        torch.zeros(4, dtype=torch.long),
        reference_positions=rest,
    )
    reference = ReferenceMapFiberBundle.from_shell_geometry(geometry, rest)
    builder = BundleMeasureBuilder(
        reference,
        geometry,
        areal_density_kg_m2=0.2,
    )
    model = HyperNetworkKirchhoffLoveShell(
        1,
        latent_dim=4,
        hyper_hidden_dim=8,
        node_hidden_dim=3,
        node_steps=2,
    )
    return builder, model, rest, torch.zeros(4)


def _lift(
    builder: BundleMeasureBuilder,
    model: HyperNetworkKirchhoffLoveShell,
    positions: torch.Tensor,
    latent: torch.Tensor,
    velocities: torch.Tensor | None = None,
):
    velocity = torch.zeros_like(positions) if velocities is None else velocities
    output = model(
        positions,
        builder.geometry,
        latent,
        compute_bending_forces=False,
        create_graph=False,
    )
    state = builder.lift(
        positions,
        velocity,
        latent,
        time=0.0,
        garment_id="shirt",
        shell_output=output,
    )
    return state, output


def test_graded_measure_preserves_exact_mass_area_and_kl_quadrature() -> None:
    builder, model, rest, latent = _fixture()
    state, output = _lift(builder, model, rest, latent)
    hamiltonian = assemble_bundle_measure_hamiltonian(
        state,
        output.membrane_energy,
        output.bending_energy,
        gravity_m_s2=(0.0, 0.0, 0.0),
    )
    report = builder.report(state, hamiltonian)

    assert report.vertex_atoms == 4
    assert report.face_atoms == 2
    assert report.hinge_atoms == 1
    assert report.total_mass_kg == pytest.approx(0.2)
    assert report.reference_area_m2 == pytest.approx(1.0)
    assert report.material_map_immutable
    assert hamiltonian.total >= 0.0


def test_measure_symplectic_step_exactly_satisfies_discrete_continuity() -> None:
    builder, _model, rest, _latent = _fixture()
    velocity = torch.tensor(
        [[0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1], [0.1, 0.1, 0.0]]
    )
    previous = builder.vertex_measure(rest, velocity)
    force = torch.tensor(
        [[0.0, -0.1, 0.0], [0.0, -0.2, 0.0], [0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]]
    )
    current = symplectic_euler_measure_step(previous, force, 0.01)
    residual = bundle_measure_continuity_residual(previous, current, force, 0.01)

    assert torch.allclose(residual.mass, torch.zeros_like(residual.mass))
    assert torch.allclose(
        residual.material_coordinates,
        torch.zeros_like(residual.material_coordinates),
    )
    assert residual.squared_norm() < 1.0e-10


def test_lifted_measure_keeps_e3_motion_out_of_intrinsic_metric_and_energy() -> None:
    builder, model, rest, latent = _fixture()
    velocity = torch.tensor([[0.2, -0.1, 0.3]]).expand_as(rest).clone()
    base, base_output = _lift(builder, model, rest, latent, velocity)
    rotation = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    moved_positions = rest @ rotation.T + torch.tensor([2.0, -1.0, 0.4])
    moved_velocity = velocity @ rotation.T
    moved, moved_output = _lift(
        builder, model, moved_positions, latent, moved_velocity
    )
    base_h = assemble_bundle_measure_hamiltonian(
        base,
        base_output.membrane_energy,
        base_output.bending_energy,
        gravity_m_s2=(0.0, 0.0, 0.0),
    )
    moved_h = assemble_bundle_measure_hamiltonian(
        moved,
        moved_output.membrane_energy,
        moved_output.bending_energy,
        gravity_m_s2=(0.0, 0.0, 0.0),
    )

    assert torch.allclose(moved.face.metric, base.face.metric, atol=1.0e-6)
    assert torch.allclose(moved_h.total, base_h.total, atol=1.0e-6)
    assert torch.equal(moved.vertex.material_coordinates, base.vertex.material_coordinates)


def test_contact_push_forward_changes_only_momentum_fibre() -> None:
    builder, model, rest, latent = _fixture()
    state, _output = _lift(builder, model, rest, latent)
    reset_velocity = torch.zeros_like(rest)
    reset_velocity[:, 1] = 0.4
    reset = momentum_fibre_push_forward(state, reset_velocity, time=0.25)

    assert reset.time == 0.25
    assert torch.equal(reset.vertex.positions, state.vertex.positions)
    assert torch.equal(reset.face.metric, state.face.metric)
    assert torch.equal(reset.vertex.masses, state.vertex.masses)
    assert torch.allclose(reset.vertex.velocities, reset_velocity)


def test_measure_training_loss_accepts_passive_free_transport() -> None:
    builder, model, rest, latent = _fixture()
    velocity = torch.tensor([[0.03, -0.02, 0.01]]).expand_as(rest).clone()
    current, output = _lift(builder, model, rest, latent, velocity)
    dt = 0.02
    following, _ = _lift(
        builder,
        model,
        rest + dt * velocity,
        latent,
        velocity,
    )
    hamiltonian = assemble_bundle_measure_hamiltonian(
        current,
        output.membrane_energy,
        output.bending_energy,
        gravity_m_s2=(0.0, 0.0, 0.0),
    )
    loss = bundle_measure_port_hamiltonian_loss(
        current,
        following,
        torch.zeros_like(rest),
        hamiltonian,
        hamiltonian,
        dt=dt,
    )

    assert loss.continuity < 1.0e-9
    assert loss.energy_balance == 0.0
    assert loss.passivity == 0.0
    assert loss.dissipation == 0.0


def test_measure_training_loss_enforces_energy_passivity_and_dissipation_sign() -> None:
    builder, model, rest, latent = _fixture()
    state, _output = _lift(builder, model, rest, latent)

    def hamiltonian(total: float) -> BundleMeasureHamiltonian:
        zero = rest.new_zeros(())
        return BundleMeasureHamiltonian(
            rest.new_tensor(total), zero, zero, zero, zero, zero, zero
        )

    passive = bundle_measure_port_hamiltonian_loss(
        state,
        state,
        torch.zeros_like(rest),
        hamiltonian(0.2),
        hamiltonian(0.19),
        dt=0.1,
        dissipative_power=rest.new_tensor(-0.1),
    )
    active = bundle_measure_port_hamiltonian_loss(
        state,
        state,
        torch.zeros_like(rest),
        hamiltonian(0.2),
        hamiltonian(0.3),
        dt=0.1,
        dissipative_power=rest.new_tensor(0.1),
    )

    assert passive.energy_balance < 1.0e-8
    assert passive.passivity == 0.0
    assert passive.dissipation == 0.0
    assert active.passivity > 0.0
    assert active.passivity < 1.0e3
    assert active.dissipation > 0.0


def test_welded_vertex_can_overlap_two_panel_charts_without_duplicating_mass() -> None:
    rest = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]
    )
    faces = torch.tensor([[0, 1, 2], [1, 3, 2]])
    geometry = build_kirchhoff_love_shell_geometry(
        rest[:, :2],
        faces,
        torch.zeros(4, dtype=torch.long),
        reference_positions=rest,
    )
    geometry = replace(
        geometry,
        face_panel_features=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
    )
    reference = ReferenceMapFiberBundle.from_shell_geometry(geometry, rest)
    builder = BundleMeasureBuilder(reference, geometry, areal_density_kg_m2=0.2)
    measure = builder.vertex_measure(rest, torch.zeros_like(rest))

    assert torch.equal(reference.face_sheet_ids, torch.tensor([0, 1]))
    assert len(measure.positions) == len(rest)
    assert measure.total_mass() == pytest.approx(0.2)
