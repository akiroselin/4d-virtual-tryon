from __future__ import annotations

from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aigc_fashion.hypernetwork_kirchhoff_love import (  # noqa: E402
    HyperNetworkKirchhoffLoveShell,
    MaterialPointHyperNetwork,
    MonotoneConstitutiveNODE,
    build_kirchhoff_love_shell_geometry,
    kirchhoff_love_dynamics_loss,
)


def _square_geometry():
    material = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        dtype=torch.float32,
    )
    faces = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.long)
    panel_ids = torch.zeros(4, dtype=torch.long)
    return build_kirchhoff_love_shell_geometry(material, faces, panel_ids)


def test_flat_material_state_is_stress_free() -> None:
    geometry = _square_geometry()
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
        requires_grad=True,
    )
    model = HyperNetworkKirchhoffLoveShell(1, latent_dim=3)
    output = model(positions, geometry, torch.zeros(3))
    assert torch.allclose(output.membrane_strains, torch.zeros_like(output.membrane_strains))
    assert torch.allclose(output.membrane_resultants, torch.zeros_like(output.membrane_resultants))
    assert torch.allclose(output.membrane_energy, torch.zeros_like(output.membrane_energy))
    assert torch.allclose(output.bending_energy, torch.zeros_like(output.bending_energy))
    assert torch.allclose(output.free_boundary_traction, torch.zeros_like(output.free_boundary_traction))


def test_constitutive_node_is_zero_anchored_and_sign_preserving() -> None:
    hypernetwork = MaterialPointHyperNetwork(1, latent_dim=3)
    node = MonotoneConstitutiveNODE(integration_steps=4)
    coordinates = torch.zeros((3, 2))
    panels = torch.ones((3, 1))
    parameters = hypernetwork(coordinates, panels, torch.zeros(3))
    strains = torch.tensor([[-0.4], [0.0], [0.4]])
    response = node.response(strains, parameters, (0,))
    energy = node.energy(strains, parameters, (0,))
    assert response[0, 0] < 0.0
    assert response[1, 0] == 0.0
    assert response[2, 0] > 0.0
    assert torch.all(energy >= 0.0)


def test_shell_is_frame_indifferent_and_forces_are_equivariant() -> None:
    geometry = _square_geometry()
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.08, 0.05, 0.10], [1.03, 1.04, 0.18], [-0.02, 0.98, -0.04]],
        requires_grad=True,
    )
    angle = torch.tensor(0.63)
    cosine = torch.cos(angle)
    sine = torch.sin(angle)
    rotation = torch.tensor(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]]
    )
    translation = torch.tensor([0.7, -0.4, 0.2])
    transformed = (positions.detach() @ rotation.T + translation).requires_grad_(True)
    model = HyperNetworkKirchhoffLoveShell(1, latent_dim=3)
    latent = torch.tensor([0.02, -0.01, 0.03])
    baseline = model(positions, geometry, latent)
    moved = model(transformed, geometry, latent)
    assert torch.allclose(baseline.membrane_energy, moved.membrane_energy, atol=3.0e-5)
    assert torch.allclose(baseline.bending_energy, moved.bending_energy, atol=3.0e-7)
    assert torch.allclose(
        baseline.membrane_internal_force @ rotation.T,
        moved.membrane_internal_force,
        atol=4.0e-5,
    )
    assert torch.allclose(
        baseline.free_boundary_traction @ rotation.T,
        moved.free_boundary_traction,
        atol=4.0e-5,
    )


def test_hypernode_intrinsic_edge_barrier_activates_before_hard_limit() -> None:
    geometry = _square_geometry()
    model = HyperNetworkKirchhoffLoveShell(1, latent_dim=3)
    latent = torch.zeros(3)
    moderate = torch.tensor(
        [[0.0, 0.0, 0.0], [1.065, 0.0, 0.0], [1.065, 1.0, 0.0], [0.0, 1.0, 0.0]],
        requires_grad=True,
    )
    near_limit = torch.tensor(
        [[0.0, 0.0, 0.0], [1.074, 0.0, 0.0], [1.074, 1.0, 0.0], [0.0, 1.0, 0.0]],
        requires_grad=True,
    )
    kwargs = {
        "maximum_tensile_strain": 0.075,
        "maximum_compressive_strain": 0.05,
        "strain_limit_onset_fraction": 0.8,
    }
    moderate_output = model(moderate, geometry, latent, **kwargs)
    near_output = model(near_limit, geometry, latent, **kwargs)
    assert moderate_output.strain_limit_utilization.max() < 1.0
    assert moderate_output.strain_limit_energy.sum() > 0.0
    assert near_output.strain_limit_energy.sum() > moderate_output.strain_limit_energy.sum()
    assert torch.linalg.vector_norm(near_output.strain_limit_internal_force) > 0.0
    assert torch.allclose(
        near_output.strain_limit_internal_force.sum(dim=0),
        torch.zeros(3),
        atol=1.0e-4,
    )


def test_degenerate_material_triangles_are_rejected() -> None:
    material = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [0.0, 1.0]],
        dtype=torch.float32,
    )
    faces = torch.tensor([[0, 1, 2], [0, 1, 3]], dtype=torch.long)
    geometry = build_kirchhoff_love_shell_geometry(
        material,
        faces,
        torch.zeros(4, dtype=torch.long),
    )
    assert len(geometry.faces) == 1
    assert geometry.rejected_face_count == 1


def test_curved_reference_surface_supplies_intrinsic_face_metric() -> None:
    material_conditioning = torch.tensor(
        [[0.0, 0.0], [0.5, 0.0], [1.0, 0.0], [1.5, 0.0]],
        dtype=torch.float32,
    )
    reference = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
            [0.5, 1.0, 1.0],
        ],
        dtype=torch.float32,
    )
    faces = torch.tensor([[0, 1, 2], [1, 3, 2]], dtype=torch.long)
    geometry = build_kirchhoff_love_shell_geometry(
        material_conditioning,
        faces,
        torch.zeros(4, dtype=torch.long),
        reference_positions=reference,
    )
    assert geometry.rejected_face_count == 0
    assert len(geometry.faces) == 2
    assert torch.all(geometry.rest_areas > 0.0)

    model = HyperNetworkKirchhoffLoveShell(1, latent_dim=3)
    output = model(reference.clone().requires_grad_(True), geometry, torch.zeros(3))
    assert torch.allclose(
        output.membrane_strains,
        torch.zeros_like(output.membrane_strains),
        atol=1.0e-6,
    )
    assert torch.allclose(
        output.bending_strain,
        torch.zeros_like(output.bending_strain),
        atol=1.0e-6,
    )
    assert torch.allclose(
        output.bending_energy,
        torch.zeros_like(output.bending_energy),
        atol=1.0e-8,
    )


def test_dynamic_equilibrium_loss_backpropagates() -> None:
    geometry = _square_geometry()
    current = torch.tensor(
        [[0.0, 0.0, 0.0], [1.03, 0.0, 0.02], [1.02, 1.0, 0.05], [0.0, 1.0, 0.0]],
        requires_grad=True,
    )
    previous = current.detach() + torch.tensor([0.0, 0.002, 0.0])
    following = current.detach() - torch.tensor([0.0, 0.002, 0.0])
    model = HyperNetworkKirchhoffLoveShell(1, latent_dim=3)
    latent = torch.zeros(3, requires_grad=True)
    loss = kirchhoff_love_dynamics_loss(
        model,
        geometry,
        previous,
        current,
        following,
        latent,
        dt=0.01,
    )
    assert torch.isfinite(loss.total)
    loss.total.backward()
    assert current.grad is not None
    assert latent.grad is not None
    assert torch.all(torch.isfinite(current.grad))
    assert torch.all(torch.isfinite(latent.grad))


def test_prescribed_boundary_traction_is_a_nonzero_neumann_target() -> None:
    geometry = _square_geometry()
    current = torch.tensor(
        [[0.0, 0.0, 0.0], [1.04, 0.0, 0.0], [1.04, 1.0, 0.0], [0.0, 1.0, 0.0]],
        requires_grad=True,
    )
    model = HyperNetworkKirchhoffLoveShell(1, latent_dim=3)
    latent = torch.zeros(3)
    predicted = model(current, geometry, latent).free_boundary_traction.detach()
    zero_target = kirchhoff_love_dynamics_loss(
        model,
        geometry,
        current.detach(),
        current,
        current.detach(),
        latent,
        dt=0.01,
        gravity_m_s2=(0.0, 0.0, 0.0),
    )
    measured_target = kirchhoff_love_dynamics_loss(
        model,
        geometry,
        current.detach(),
        current,
        current.detach(),
        latent,
        dt=0.01,
        gravity_m_s2=(0.0, 0.0, 0.0),
        prescribed_free_boundary_traction=predicted,
    )
    assert zero_target.traction > 0.0
    assert torch.allclose(measured_target.traction, torch.zeros_like(measured_target.traction))
