import pytest
import torch
import trimesh
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aigc_fashion.data import MaterialSpec
from aigc_fashion.hypernetwork_kirchhoff_love import (
    HyperNetworkKirchhoffLoveShell,
    build_kirchhoff_love_shell_geometry,
)
from aigc_fashion.neuralclothsim_dynamics import (
    NeuralCloth4DDynamics,
    NeuralCloth4DDynamicsConfig,
    NeuralCloth4DState,
)
from aigc_fashion.phase_manifold_solver import (
    IncrementalHamiltonianConfig,
    PhasePotentialTerms,
    solve_incremental_hamiltonian_step,
)
from aigc_fashion.rigid_lie_message_passing import SE3JointMessagePassing
from aigc_fashion.unified_cloth4d_scene import (
    Body4DState,
    BodyContactSample,
    ContinuousGarment4D,
    Garment4DSystem,
    MovingBodyContactQuery,
    SinusoidalSE3BodyTrajectory,
    UnifiedCloth4DSceneConfig,
    UnifiedNeuralCloth4DScene,
    WalkingSE3BodyTrajectory,
    _resolve_body_contact_with_clearance,
)


def _cloth(name: str, z: float, velocity_z: float) -> Garment4DSystem:
    rest = torch.tensor(
        [[0.0, 0.0, z], [0.1, 0.0, z], [0.0, -0.1, z], [0.1, -0.1, z]],
        dtype=torch.float32,
    )
    faces = torch.tensor([[0, 1, 2], [1, 3, 2]], dtype=torch.long)
    edges = torch.tensor([[0, 1], [0, 2], [1, 2], [1, 3], [2, 3]], dtype=torch.long)
    lengths = torch.linalg.vector_norm(rest[edges[:, 1]] - rest[edges[:, 0]], dim=-1)
    geometry = build_kirchhoff_love_shell_geometry(
        rest[:, :2],
        faces,
        torch.zeros(4, dtype=torch.long),
        reference_positions=rest,
    )
    model = HyperNetworkKirchhoffLoveShell(
        1, latent_dim=4, hyper_hidden_dim=8, node_hidden_dim=3, node_steps=2
    )
    dynamics = NeuralCloth4DDynamics(
        model,
        geometry,
        rest_positions=rest,
        edge_index=edges.T,
        rest_lengths=lengths,
        cotangent_weights=torch.ones_like(lengths),
        pinned_vertices=torch.empty(0, dtype=torch.long),
        material=MaterialSpec(name="cotton jersey knit"),
        config=NeuralCloth4DDynamicsConfig(
            dt=0.005,
            substeps_per_frame=1,
            gravity_m_s2=(0.0, 0.0, 0.0),
            velocity_damping_per_second=0.0,
            shell_force_scale=0.0,
            residual_force_scale=0.0,
            metric_projection_iterations=12,
        ),
    )
    velocity = torch.zeros_like(rest)
    velocity[:, 2] = velocity_z
    return Garment4DSystem(
        name,
        dynamics,
        torch.zeros(4),
        rest,
        velocity,
        faces,
    )


def test_continuous_trajectory_queries_between_physical_knots() -> None:
    times = torch.tensor([0.0, 1.0])
    positions = torch.tensor([[[0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]]])
    velocities = torch.tensor([[[1.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]]])
    field = ContinuousGarment4D(
        times,
        positions,
        velocities,
        torch.tensor([[0, 0, 0]], dtype=torch.long),
    )
    assert torch.allclose(field.query(0.25), torch.tensor([[0.25, 0.0, 0.0]]))
    assert torch.allclose(
        field.query_material_points(
            torch.tensor([0]),
            torch.tensor([[0.2, 0.3, 0.5]]),
            0.25,
        ),
        torch.tensor([[0.25, 0.0, 0.0]]),
    )
    assert torch.equal(field.query(1.0), positions[-1])
    with pytest.raises(ValueError):
        field.query(1.1)


def test_se3_body_motion_has_consistent_inverse_and_velocity() -> None:
    trajectory = SinusoidalSE3BodyTrajectory(
        torch.tensor([0.0, 0.0, 0.0]),
        period_s=1.0,
    )
    points = torch.tensor([[0.2, 0.4, -0.1], [-0.3, 0.1, 0.2]])
    world = trajectory.transform_points(points, 0.17)
    assert torch.allclose(trajectory.inverse_points(world, 0.17), points, atol=1.0e-6)
    assert torch.linalg.vector_norm(trajectory.point_velocities(world, 0.17), dim=-1).max() > 0.0


def test_velocity_only_body_contact_does_not_reproject_positions() -> None:
    positions = torch.tensor(
        [[0.0, 0.001, 0.0], [0.1, 0.001, 0.0], [0.0, 0.001, 0.1]],
        dtype=torch.float32,
    )
    state = NeuralCloth4DState(
        positions,
        torch.tensor([[0.0, -1.0, 0.0]]).expand_as(positions).clone(),
        0.0,
    )

    def plane_contact(points: torch.Tensor, _time: float) -> BodyContactSample:
        closest = points.clone()
        closest[:, 1] = 0.0
        normals = torch.zeros_like(points)
        normals[:, 1] = 1.0
        return BodyContactSample(closest, normals)

    resolved, _gap = _resolve_body_contact_with_clearance(
        state,
        plane_contact,
        torch.empty((0, 2), dtype=torch.long),
        torch.empty(0, dtype=torch.long),
        torch.empty((0, 3), dtype=torch.float32),
        clearance=0.003,
        friction=0.3,
        friction_activation_clearance=0.004,
        apply_position_correction=False,
    )
    assert torch.equal(resolved.positions, state.positions)
    assert resolved.velocities[:, 1].min() >= 0.0


def test_moving_reference_hamiltonian_transports_without_relative_strain() -> None:
    current = torch.zeros((1, 3), dtype=torch.float32)
    masses = torch.ones(1)
    dt = 0.1
    body_displacement = torch.tensor([[0.01, 0.0, 0.0]])
    body_velocity = torch.zeros_like(current)
    next_body_velocity = torch.tensor([[0.1, 0.0, 0.0]])

    def zero_potential(_points: torch.Tensor) -> PhasePotentialTerms:
        zero = current.new_zeros(())
        return PhasePotentialTerms(zero, zero, zero, zero, zero, zero, zero)

    step = solve_incremental_hamiltonian_step(
        current,
        torch.zeros_like(current),
        masses,
        dt=dt,
        potential=zero_potential,
        moving_reference_velocity=body_velocity,
        moving_reference_displacement=body_displacement,
        moving_reference_next_velocity=next_body_velocity,
        config=IncrementalHamiltonianConfig(
            iterations=1,
            line_search_steps=1,
            gradient_tolerance_n=1.0e-6,
        ),
    )

    assert torch.allclose(step.positions, body_displacement, atol=1.0e-6)
    assert torch.allclose(step.velocities, next_body_velocity, atol=1.0e-6)
    assert torch.allclose(step.momenta, torch.zeros_like(current), atol=1.0e-6)


def test_walking_body_inverse_skinning_and_opposed_leg_swing() -> None:
    trajectory = WalkingSE3BodyTrajectory(
        torch.zeros(3),
        step_duration_s=0.2,
        hip_y=-0.2,
        shoulder_y=0.2,
        body_center_z=0.0,
    )
    local = torch.tensor(
        [[-0.08, -0.55, 0.0], [0.08, -0.55, 0.0], [0.0, 0.1, 0.0]],
        dtype=torch.float32,
    )
    world = trajectory.transform_body_points(local, 0.1)
    recovered = trajectory.inverse_body_points(world, 0.1)
    articulated_local = trajectory.inverse_points(world, 0.1)

    assert torch.allclose(recovered, local, atol=2.0e-3)
    assert articulated_local[0, 2] * articulated_local[1, 2] < 0.0
    assert torch.linalg.vector_norm(world - local, dim=-1).max() > 0.02


def test_walking_body_keeps_each_lower_leg_rigid() -> None:
    trajectory = WalkingSE3BodyTrajectory(
        torch.zeros(3),
        step_duration_s=0.3,
        steps=2,
        hip_y=-0.2,
        body_center_z=0.0,
    )
    # These pairs represent neighboring vertices in the two lower-leg
    # branches. Their relative lengths must not change when the hips swing.
    local = torch.tensor(
        [
            [0.08, -0.55, 0.00],
            [0.12, -0.70, 0.03],
            [-0.08, -0.55, 0.00],
            [-0.12, -0.70, 0.03],
        ],
        dtype=torch.float32,
    )
    rest_lengths = torch.linalg.vector_norm(local[1::2] - local[0::2], dim=-1)
    for time in (0.075, 0.15, 0.225, 0.45):
        posed = trajectory.transform_body_points(local, time)
        posed_lengths = torch.linalg.vector_norm(posed[1::2] - posed[0::2], dim=-1)
        assert torch.allclose(posed_lengths, rest_lengths, atol=1.0e-5)


def test_walking_body_keeps_each_arm_rigid_and_swings_oppositely() -> None:
    trajectory = WalkingSE3BodyTrajectory(
        torch.zeros(3),
        step_duration_s=0.3,
        steps=2,
        hip_y=-0.2,
        shoulder_y=0.2,
        shoulder_half_width_m=0.17,
        body_center_z=0.0,
        yaw_amplitude_rad=0.0,
    )
    local = torch.tensor(
        [
            [0.20, 0.10, 0.00],
            [0.30, 0.04, 0.03],
            [-0.20, 0.10, 0.00],
            [-0.30, 0.04, 0.03],
        ],
        dtype=torch.float32,
    )
    rest_lengths = torch.linalg.vector_norm(local[1::2] - local[0::2], dim=-1)
    posed = trajectory.transform_body_points(local, 0.15)
    posed_lengths = torch.linalg.vector_norm(posed[1::2] - posed[0::2], dim=-1)

    assert torch.allclose(posed_lengths, rest_lengths, atol=1.0e-5)
    assert posed[0, 2] > posed[2, 2]


def test_low_hanging_arm_is_not_double_transformed_as_a_leg() -> None:
    trajectory = WalkingSE3BodyTrajectory(
        torch.zeros(3),
        step_duration_s=0.3,
        steps=1,
        hip_y=-0.2,
        shoulder_y=0.2,
        shoulder_half_width_m=0.17,
        yaw_amplitude_rad=0.0,
    )
    local = torch.tensor(
        [[0.20, -0.45, 0.00], [0.30, -0.55, 0.03]], dtype=torch.float32
    )
    rest_length = torch.linalg.vector_norm(local[1] - local[0])
    for time in (0.075, 0.15, 0.225):
        posed = trajectory.transform_body_points(local, time)
        assert torch.allclose(
            torch.linalg.vector_norm(posed[1] - posed[0]), rest_length, atol=1.0e-5
        )


def test_walking_trajectory_has_no_terminal_extra_stride_or_velocity_spike() -> None:
    trajectory = WalkingSE3BodyTrajectory(
        torch.zeros(3),
        step_duration_s=0.3,
        steps=2,
        step_length_m=0.02,
    )
    reference = torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.2, 0.3]])
    at_end = trajectory.transform_points(reference, 0.6)
    before_end = trajectory.transform_points(reference, 0.6 - 1.0e-4)
    terminal_velocity = trajectory.point_velocities(at_end, 0.6)

    assert torch.allclose(
        at_end.mean(dim=0)[2] - reference.mean(dim=0)[2],
        torch.tensor(0.04),
        atol=2.0e-4,
    )
    assert torch.linalg.vector_norm(at_end - before_end, dim=-1).max() < 1.0e-3
    assert torch.linalg.vector_norm(terminal_velocity, dim=-1).max() < 1.0


def test_rigid_lie_message_passing_executes_on_every_physics_substep() -> None:
    trajectory = WalkingSE3BodyTrajectory(torch.zeros(3), step_duration_s=0.04)
    model = SE3JointMessagePassing(8, 4, hidden_dim=8, wrench_limit=1.0)
    scene = UnifiedNeuralCloth4DScene(
        [_cloth("shirt", 0.0, 0.0)],
        rigid_lie_message_model=model,
        rigid_link_graph_query=trajectory.lie_message_graph,
        config=UnifiedCloth4DSceneConfig(
            frame_count=3,
            frame_dt=0.01,
            internal_dt=0.005,
        ),
    )

    report = scene.rollout().report.lie_group_message_passing

    assert report["enabled"] is True
    assert report["substep_calls"] == 4
    assert report["expected_substep_calls"] == 4
    assert report["maximum_relative_twist_norm"] > 0.0
    assert report["maximum_action_reaction_balance_error"] < 1.0e-5


def test_laplace_beltrami_force_diffusion_is_e3_equivariant_and_force_balanced() -> None:
    dynamics = _cloth("shirt", 0.0, 0.0).dynamics
    dynamics.laplace_eigenvalues = torch.tensor([0.0, 2.0])
    dynamics.laplace_eigenvectors = torch.tensor(
        [[0.5, 0.5], [0.5, 0.5], [0.5, -0.5], [0.5, -0.5]]
    )
    dynamics.laplace_vertex_mass = torch.ones(4)
    force = torch.tensor(
        [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0]]
    )
    rotation = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )

    diffused = dynamics._laplace_beltrami_diffuse(force)
    rotated = dynamics._laplace_beltrami_diffuse(force @ rotation.T)

    assert torch.allclose(rotated, diffused @ rotation.T, atol=1.0e-6)
    assert torch.allclose(diffused.sum(dim=0), torch.zeros(3), atol=1.0e-6)


def test_moving_body_query_preserves_signed_sdf_inside_outside() -> None:
    mesh = trimesh.creation.icosphere(subdivisions=2, radius=1.0)

    class _Avatar:
        _collision_mesh = mesh
        _proximity = trimesh.proximity.ProximityQuery(mesh)

    trajectory = SinusoidalSE3BodyTrajectory(
        torch.zeros(3),
        translation_amplitude_m=(0.0, 0.0, 0.0),
        yaw_amplitude_rad=0.0,
    )
    query = MovingBodyContactQuery(_Avatar(), trajectory)
    points = torch.tensor([[1.1, 0.0, 0.0], [0.9, 0.0, 0.0]])
    sample = query(points, 0.0)
    gap = ((points - sample.closest_points) * sample.normals).sum(dim=-1)
    assert gap[0] > 0.0
    assert gap[1] < 0.0


def test_shared_scene_localizes_inter_garment_contact_and_is_queryable() -> None:
    scene = UnifiedNeuralCloth4DScene(
        [_cloth("shirt", 0.0, 1.0), _cloth("skirt", 0.02, -1.0)],
        config=UnifiedCloth4DSceneConfig(
            frame_count=5,
            frame_dt=0.01,
            internal_dt=0.005,
            inter_garment_clearance_m=0.005,
        ),
    )
    rollout = scene.rollout()
    assert rollout.report.physical_duration_s == pytest.approx(0.04)
    assert rollout.report.event_restarts >= 1
    assert any(event["kind"] == "garment_contact_enter" for event in rollout.report.events)
    assert set(rollout.query(0.015)) == {"shirt", "skirt"}
    assert rollout.fields["shirt"].positions.shape == (5, 4, 3)
    assert rollout.report.metric_projection["shirt"]["violating_edges"] == 0
    assert rollout.report.generalized_flow["shirt"]["representation_count"]["e3_vector"] > 0
    assert rollout.report.reference_map_fiber["skirt"]["material_map_immutable"] is True
    assert rollout.report.bundle_measure_port_hamiltonian["shirt"]["garment_id"] == "shirt"
    assert rollout.report.bundle_measure_port_hamiltonian["skirt"]["face_atoms"] > 0


def test_scene_hamiltonian_audits_absolute_body_kinetic_energy() -> None:
    body_positions = torch.zeros((2, 3), dtype=torch.float32)
    body_velocities = torch.full((2, 3), 0.2, dtype=torch.float32)
    body_momentum = 0.5 * body_velocities

    def body_state(time: float) -> Body4DState:
        return Body4DState(
            body_positions,
            body_velocities,
            body_momentum,
            float(time),
        )

    scene = UnifiedNeuralCloth4DScene(
        [_cloth("shirt", 0.0, 0.0)],
        body_phase_state_provider=body_state,
        config=UnifiedCloth4DSceneConfig(
            frame_count=2,
            frame_dt=0.01,
            internal_dt=0.005,
        ),
    )
    report = scene.rollout().report
    assert report.scene_hamiltonian["body"]["maximum_kinetic_energy_j"] > 0.0
    assert report.scene_hamiltonian["body"]["maximum_total_momentum_norm_kg_m_s"] > 0.0
    assert report.body_phase_state["maximum_kinetic_energy_j"] > 0.0


def test_real_garment_layer_names_use_render_guard_until_triangle_ccd() -> None:
    scene = UnifiedNeuralCloth4DScene(
        [_cloth("tshirt", 0.0, 0.0), _cloth("pleated_skirt", 0.003, 0.0)],
        config=UnifiedCloth4DSceneConfig(
            frame_count=2,
            frame_dt=0.01,
            internal_dt=0.005,
            inter_garment_clearance_m=0.005,
        ),
    )
    # The production pair is intentionally excluded from the unreliable
    # vertex-nearest dynamic projector; the exporter owns its visual guard.
    assert scene._pair_names() == ()


def test_joint_contact_acceptance_never_allows_negative_body_gap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scene = UnifiedNeuralCloth4DScene(
        [_cloth("skirt", 0.0, 0.0)],
        config=UnifiedCloth4DSceneConfig(
            frame_count=2,
            frame_dt=0.01,
            internal_dt=0.005,
            contact_constraint_tolerance_m=0.0028,
        ),
    )
    state = scene.systems[0]
    states = {
        "skirt": NeuralCloth4DState(
            state.initial_positions, state.initial_velocities, 0.0
        )
    }
    monkeypatch.setattr(
        scene,
        "_event_values",
        lambda _states: {"skirt:body:min_gap": 1.0e-8},
    )
    assert scene._joint_constraints_satisfied(states)
    monkeypatch.setattr(
        scene,
        "_event_values",
        lambda _states: {"skirt:body:min_gap": -1.0e-8},
    )
    assert not scene._joint_constraints_satisfied(states)
