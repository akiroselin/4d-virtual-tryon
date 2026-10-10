import torch

from aigc_fashion.data import MaterialSpec
from aigc_fashion.hypernetwork_kirchhoff_love import (
    HyperNetworkKirchhoffLoveShell,
    build_kirchhoff_love_shell_geometry,
)
from aigc_fashion.neuralclothsim_dynamics import (
    BodyContactSample,
    NeuralCloth4DDynamics,
    NeuralCloth4DDynamicsConfig,
)


def _edges(faces: torch.Tensor) -> torch.Tensor:
    pairs = set()
    for face in faces.tolist():
        for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            pairs.add(tuple(sorted((a, b))))
    return torch.tensor(sorted(pairs), dtype=torch.long)


def _system(
    rest: torch.Tensor,
    material_coordinates: torch.Tensor,
    pinned: torch.Tensor,
    *,
    config: NeuralCloth4DDynamicsConfig,
    retention_edges: torch.Tensor | None = None,
) -> tuple[NeuralCloth4DDynamics, torch.Tensor]:
    faces = torch.tensor([[0, 1, 2], [1, 3, 2]], dtype=torch.long)
    panel_ids = torch.zeros(4, dtype=torch.long)
    edges = _edges(faces)
    lengths = torch.linalg.vector_norm(rest[edges[:, 1]] - rest[edges[:, 0]], dim=-1)
    geometry = build_kirchhoff_love_shell_geometry(
        material_coordinates,
        faces,
        panel_ids,
        reference_positions=rest,
    )
    model = HyperNetworkKirchhoffLoveShell(
        1,
        latent_dim=4,
        hyper_hidden_dim=8,
        node_hidden_dim=3,
        node_steps=2,
    )
    dynamics = NeuralCloth4DDynamics(
        model,
        geometry,
        rest_positions=rest,
        edge_index=edges.T,
        rest_lengths=lengths,
        cotangent_weights=torch.ones_like(lengths),
        pinned_vertices=pinned,
        retention_edges=retention_edges,
        material=MaterialSpec(name="cotton jersey knit"),
        config=config,
    )
    return dynamics, torch.zeros(4)


def test_physical_time_rollout_keeps_pins_and_hard_material_metric() -> None:
    rest = torch.tensor(
        [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, -0.1, 0.0], [0.1, -0.1, 0.0]]
    )
    dynamics, latent = _system(
        rest,
        rest[:, :2],
        torch.tensor([0, 1]),
        config=NeuralCloth4DDynamicsConfig(
            dt=0.002,
            substeps_per_frame=2,
            maximum_acceleration_m_s2=5.0,
            metric_projection_iterations=20,
        ),
    )
    rollout = dynamics.rollout(rest, latent, frame_count=5)
    assert rollout.positions.shape == (5, 4, 3)
    assert rollout.velocities.shape == rollout.positions.shape
    assert torch.equal(rollout.positions[:, [0, 1]], rest[[0, 1]].expand(5, -1, -1))
    assert rollout.positions[-1, 2:, 1].mean() < rest[2:, 1].mean()
    assert rollout.report.metric_projection["violating_edges"] == 0
    assert rollout.report.physical_duration_s == 0.016
    assert rollout.report.maximum_acceleration_m_s2 <= 5.0 + 1.0e-6
    assert rollout.report.maximum_unclamped_acceleration_m_s2 > 5.0
    assert rollout.report.acceleration_cap_hits > 0
    assert 0.0 < rollout.report.acceleration_cap_fraction <= 1.0
    assert 0.0 < rollout.report.constitutive_force_scale <= 1.0
    assert rollout.report.constitutive_calibration_status in {
        "near_calibrated",
        "scene_rescaled",
        "constitutive_recalibration_required",
    }
    assert rollout.report.generalized_flow["dissipative_sign_violations"] == 0
    assert rollout.report.generalized_flow["representation_count"]["e3_vector"] > 0
    assert rollout.report.reference_map_fiber["material_map_immutable"] is True
    measure = rollout.report.bundle_measure_port_hamiltonian
    assert measure["vertex_atoms"] == 4
    assert measure["face_atoms"] == 2
    assert measure["material_map_immutable"] is True
    assert measure["mass_drift_kg"] < 1.0e-8
    assert "total" in measure["hamiltonian_j"]


def test_body_zero_crossing_splits_step_and_restarts_contact_solver() -> None:
    rest = torch.tensor(
        [[0.0, 0.02, 0.0], [0.1, 0.02, 0.0], [0.0, 0.02, 0.1], [0.1, 0.02, 0.1]]
    )
    dynamics, latent = _system(
        rest,
        rest[:, [0, 2]],
        torch.empty(0, dtype=torch.long),
        config=NeuralCloth4DDynamicsConfig(
            dt=0.01,
            substeps_per_frame=2,
            maximum_acceleration_m_s2=40.0,
            contact_clearance_m=0.003,
            metric_projection_iterations=8,
            shell_force_scale=0.0,
        ),
    )

    def plane_contact(positions: torch.Tensor, _time: float) -> BodyContactSample:
        closest = positions.clone()
        closest[:, 1] = 0.0
        normals = torch.zeros_like(positions)
        normals[:, 1] = 1.0
        return BodyContactSample(closest, normals)

    rollout = dynamics.rollout(
        rest,
        latent,
        frame_count=12,
        body_contact_query=plane_contact,
    )
    assert rollout.report.event_restarts >= 1
    assert rollout.report.contact_events
    assert rollout.report.minimum_contact_gap_m is not None
    assert rollout.positions[-1, :, 1].min() >= 0.003 - 1.0e-5
    assert rollout.report.generalized_flow["representation_count"]["so2_complex"] > 0
    assert rollout.report.generalized_flow["dissipative_sign_violations"] == 0
    assert "momentum fibre" in rollout.report.bundle_measure_port_hamiltonian[
        "contact_reset"
    ]


def test_dynamic_rollout_remains_differentiable_without_discrete_contact_events() -> None:
    rest = torch.tensor(
        [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, -0.1, 0.0], [0.1, -0.1, 0.0]]
    )
    dynamics, latent = _system(
        rest,
        rest[:, :2],
        torch.tensor([0, 1]),
        config=NeuralCloth4DDynamicsConfig(
            dt=0.001,
            substeps_per_frame=1,
            metric_projection_iterations=4,
        ),
    )
    initial = rest.clone().requires_grad_(True)
    rollout = dynamics.rollout(initial, latent, frame_count=3, differentiable=True)
    rollout.positions[-1].square().sum().backward()
    assert initial.grad is not None
    assert torch.isfinite(initial.grad).all()


def test_feasible_static_state_does_not_run_metric_recovery() -> None:
    rest = torch.tensor(
        [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, -0.1, 0.0], [0.1, -0.1, 0.0]]
    )
    dynamics, latent = _system(
        rest,
        rest[:, :2],
        torch.empty(0, dtype=torch.long),
        config=NeuralCloth4DDynamicsConfig(
            dt=0.002,
            substeps_per_frame=1,
            gravity_m_s2=(0.0, 0.0, 0.0),
            shell_force_scale=0.0,
            residual_force_scale=0.0,
            metric_projection_iterations=4,
        ),
    )
    rollout = dynamics.rollout(rest, latent, frame_count=3)
    assert torch.equal(rollout.positions, rest.expand(3, -1, -1))
    assert rollout.report.metric_projection_recovery_count == 0
    assert rollout.report.metric_projection_recovery_fraction == 0.0


def test_metric_projection_can_be_disabled_for_hamiltonian_rollout() -> None:
    rest = torch.tensor(
        [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, -0.1, 0.0], [0.1, -0.1, 0.0]]
    )
    dynamics, latent = _system(
        rest,
        rest[:, :2],
        torch.empty(0, dtype=torch.long),
        config=NeuralCloth4DDynamicsConfig(
            dt=0.002,
            substeps_per_frame=1,
            gravity_m_s2=(0.0, 0.0, 0.0),
            shell_force_scale=0.0,
            residual_force_scale=0.0,
            enable_metric_projection=False,
        ),
    )
    rollout = dynamics.rollout(rest, latent, frame_count=3)
    assert rollout.report.metric_projection_enabled is False
    assert rollout.report.metric_projection_recovery_count == 0
    assert torch.equal(rollout.positions, rest.expand(3, -1, -1))


def test_intrinsic_trim_band_contracts_without_world_anchors() -> None:
    rest = torch.tensor(
        [[-0.05, 0.05, 0.0], [0.05, 0.05, 0.0], [-0.05, -0.05, 0.0], [0.05, -0.05, 0.0]]
    )
    ring = torch.tensor([[0, 1], [1, 3], [3, 2], [2, 0]], dtype=torch.long)
    dynamics, latent = _system(
        rest,
        rest[:, :2],
        torch.empty(0, dtype=torch.long),
        retention_edges=ring,
        config=NeuralCloth4DDynamicsConfig(
            dt=0.001,
            substeps_per_frame=1,
            gravity_m_s2=(0.0, 0.0, 0.0),
            shell_force_scale=0.0,
            residual_force_scale=0.0,
            retention_band_axial_rigidity_n=10.0,
            metric_projection_iterations=8,
        ),
    )
    initial = 1.05 * rest
    rollout = dynamics.rollout(initial, latent, frame_count=3)

    def perimeter(points: torch.Tensor) -> torch.Tensor:
        return torch.linalg.vector_norm(
            points[ring[:, 1]] - points[ring[:, 0]], dim=-1
        ).sum()

    assert perimeter(rollout.positions[-1]) < perimeter(initial)
    assert rollout.report.retention_band_edges == 4
    assert rollout.report.metric_projection_recovery_count == 0
