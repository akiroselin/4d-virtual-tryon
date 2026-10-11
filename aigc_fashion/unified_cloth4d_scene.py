from __future__ import annotations

from dataclasses import asdict, dataclass
from math import cos, floor, pi, sin
import inspect
from typing import Callable, Mapping

import numpy as np
import torch
from torch import Tensor
from scipy.spatial import cKDTree

from .behavioral_phase import BehavioralPhaseTrajectory
from .bvh_motion import BVHMotion
from .dynamic_events import (
    ActiveSide,
    EventSurface,
    EventSurfaceKind,
    ZeroCrossingEventDetector,
)
from .generalized_flow import GeneralizedFlowRole
from .gauge_contact import GaugeEquivariantContactCNN
from .neuralclothsim_dynamics import (
    BodyContactQuery,
    BodyContactSample,
    InterGarmentContact,
    NeuralCloth4DDynamics,
    NeuralCloth4DState,
)
from .rigid_lie_message_passing import SE3JointMessagePassing
from .lie_group_diffusion import SE3GenerativeDynamicsHead


AttachmentTrajectory = Callable[[Tensor, float], Tensor]
ScenePhaseForceProvider = Callable[
    [str, NeuralCloth4DState, Mapping[str, NeuralCloth4DState], "Body4DState | None", float], Tensor
]
BodyRootTransportProvider = Callable[
    [Tensor, float, float], tuple[Tensor, Tensor, Tensor]
]


@dataclass(frozen=True)
class Body4DState:
    """The body mesh state carried by the shared scene-graph clock.

    The current Anny gait is prescribed, so this is a kinematic root state
    rather than a free rigid-body solve. Keeping q_b and p_b explicit still
    matters: contact and phase forces can be audited against the same state
    that drives the cloth, instead of treating the avatar as render-only
    geometry.
    """

    positions: Tensor
    velocities: Tensor
    generalized_momentum: Tensor
    time: float

    def __post_init__(self) -> None:
        if self.positions.ndim != 2 or self.positions.shape[1] != 3:
            raise ValueError("body positions must have shape (V,3)")
        if self.velocities.shape != self.positions.shape:
            raise ValueError("body velocities must match body positions")
        if self.generalized_momentum.shape != self.positions.shape:
            raise ValueError("body momentum must match body positions")


@dataclass(frozen=True)
class Garment4DSystem:
    """One intrinsic material manifold participating in a shared 4D scene."""

    name: str
    dynamics: NeuralCloth4DDynamics
    material_latent: Tensor
    initial_positions: Tensor
    initial_velocities: Tensor
    faces: Tensor
    attachment_trajectory: AttachmentTrajectory | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("garment name must not be empty")
        if self.initial_positions.shape != self.dynamics.rest_positions.shape:
            raise ValueError("initial garment positions must match its intrinsic rest shell")
        if self.initial_velocities.shape != self.initial_positions.shape:
            raise ValueError("initial velocities must match garment positions")
        if self.faces.ndim != 2 or self.faces.shape[1] != 3:
            raise ValueError("garment faces must have shape (F,3)")


@dataclass(frozen=True)
class UnifiedCloth4DSceneConfig:
    frame_count: int = 25
    frame_dt: float = 1.0 / 24.0
    internal_dt: float = 1.0 / 240.0
    inter_garment_clearance_m: float = 0.006
    # Contact force starts at the full 3.5 mm range. The hard shell is only a
    # 0.2 mm non-penetration guard; fabric thickness belongs to the smooth
    # Hamiltonian potential, not a local position projection.
    contact_constraint_tolerance_m: float = 0.0033
    inter_garment_friction: float = 0.22
    inter_garment_contact_stiffness_n_m: float = 20.0
    contact_projection_iterations: int = 32
    maximum_event_restarts_per_step: int = 1
    initial_contact_equilibration_steps: int = 2
    # A value greater than one enables the interactive multi-rate path. Full
    # Hyper-NODE Hamiltonian solves run at this cadence; intervening substeps
    # use the bounded E(3) preview flow and the same contact/event stage.
    phase_hamiltonian_stride: int = 1

    def __post_init__(self) -> None:
        if self.frame_count < 2:
            raise ValueError("frame_count must be at least two")
        if min(self.frame_dt, self.internal_dt, self.inter_garment_clearance_m) <= 0.0:
            raise ValueError("scene time steps and contact clearance must be positive")
        if not 0.0 <= self.contact_constraint_tolerance_m < self.inter_garment_clearance_m:
            raise ValueError("contact tolerance must be non-negative and below clearance")
        ratio = self.frame_dt / self.internal_dt
        if abs(ratio - round(ratio)) > 1.0e-6:
            raise ValueError("frame_dt must be an integer multiple of internal_dt")
        if not 0.0 <= self.inter_garment_friction <= 1.0:
            raise ValueError("inter-garment friction must be in [0,1]")
        if self.inter_garment_contact_stiffness_n_m <= 0.0:
            raise ValueError("inter-garment contact stiffness must be positive")
        if self.contact_projection_iterations <= 0:
            raise ValueError("contact_projection_iterations must be positive")
        if (
            self.maximum_event_restarts_per_step < 0
            or self.initial_contact_equilibration_steps < 0
        ):
            raise ValueError("event restarts and initial equilibration must be non-negative")
        if self.phase_hamiltonian_stride <= 0:
            raise ValueError("phase Hamiltonian stride must be positive")


@dataclass(frozen=True)
class ContinuousGarment4D:
    """C1 piecewise-cubic embedding X(material vertex, physical time)."""

    times: Tensor
    positions: Tensor
    velocities: Tensor
    faces: Tensor

    def query(self, physical_time: float | Tensor) -> Tensor:
        scalar = float(torch.as_tensor(physical_time).detach())
        if scalar < float(self.times[0]) - 1.0e-9 or scalar > float(self.times[-1]) + 1.0e-9:
            raise ValueError("physical_time lies outside the simulated interval")
        if scalar >= float(self.times[-1]):
            return self.positions[-1]
        index = int(torch.searchsorted(self.times, self.times.new_tensor(scalar), right=True)) - 1
        index = max(0, min(index, len(self.times) - 2))
        t0, t1 = self.times[index], self.times[index + 1]
        h = (t1 - t0).clamp_min(1.0e-12)
        u = ((self.times.new_tensor(scalar) - t0) / h).clamp(0.0, 1.0)
        h00 = 2.0 * u**3 - 3.0 * u**2 + 1.0
        h10 = u**3 - 2.0 * u**2 + u
        h01 = -2.0 * u**3 + 3.0 * u**2
        h11 = u**3 - u**2
        return (
            h00 * self.positions[index]
            + h10 * h * self.velocities[index]
            + h01 * self.positions[index + 1]
            + h11 * h * self.velocities[index + 1]
        )

    def query_material_points(
        self,
        face_ids: Tensor,
        barycentric_coordinates: Tensor,
        physical_time: float | Tensor,
    ) -> Tensor:
        """Evaluate X_g(u,v,t) inside intrinsic material triangles."""

        if face_ids.ndim != 1 or barycentric_coordinates.shape != (len(face_ids), 3):
            raise ValueError("face_ids and barycentric coordinates have incompatible shapes")
        if face_ids.numel() and (face_ids.min() < 0 or face_ids.max() >= len(self.faces)):
            raise ValueError("face id lies outside the intrinsic shell")
        barycentric = barycentric_coordinates.to(self.positions)
        if not torch.allclose(
            barycentric.sum(dim=-1),
            torch.ones(len(barycentric), device=barycentric.device),
            atol=1.0e-5,
        ):
            raise ValueError("barycentric coordinates must sum to one")
        if torch.any(barycentric < -1.0e-6):
            raise ValueError("barycentric coordinates must be non-negative")
        vertices = self.query(physical_time)
        triangles = vertices[self.faces.to(vertices.device)[face_ids.to(vertices.device)]]
        return (triangles * barycentric[:, :, None]).sum(dim=1)


@dataclass(frozen=True)
class UnifiedCloth4DReport:
    representation: str
    garments: tuple[str, ...]
    frames: int
    internal_steps: int
    physical_duration_s: float
    initial_contact_equilibration_steps: int
    event_restarts: int
    events: tuple[dict[str, object], ...]
    minimum_body_gap_m: Mapping[str, float | None]
    minimum_inter_garment_gap_m: Mapping[str, float]
    metric_projection: Mapping[str, Mapping[str, float | int]]
    hypernode_strain_limit: Mapping[str, Mapping[str, float | int]]
    metric_projection_recovery_count: Mapping[str, int]
    constitutive_force_scale: Mapping[str, float]
    lie_group_message_passing: Mapping[str, object]
    generalized_flow: Mapping[str, Mapping[str, object]]
    reference_map_fiber: Mapping[str, Mapping[str, object]]
    bundle_measure_port_hamiltonian: Mapping[str, Mapping[str, object]]
    phase_hamiltonian: Mapping[str, Mapping[str, object]]
    scene_hamiltonian: Mapping[str, object]
    noether_audit: Mapping[str, Mapping[str, object]]
    scene_graph_phase_port: Mapping[str, object]
    moving_reference_hamiltonian: Mapping[str, object]
    se3_generative_dynamics: Mapping[str, object]
    body_phase_state: Mapping[str, object]
    continuous_time_query: str
    limitations: tuple[str, ...]
    metric_projection_enabled: bool = True

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class UnifiedCloth4DRollout:
    fields: Mapping[str, ContinuousGarment4D]
    report: UnifiedCloth4DReport

    def query(self, physical_time: float | Tensor) -> dict[str, Tensor]:
        return {name: field.query(physical_time) for name, field in self.fields.items()}


class SinusoidalSE3BodyTrajectory:
    """Small analytic SE(3) body motion used by collision and rendering alike."""

    def __init__(
        self,
        origin: Tensor,
        *,
        period_s: float = 1.2,
        translation_amplitude_m: tuple[float, float, float] = (0.008, 0.002, 0.0),
        yaw_amplitude_rad: float = 2.5 * pi / 180.0,
    ) -> None:
        if period_s <= 0.0:
            raise ValueError("body motion period must be positive")
        self.origin = origin.detach().to(torch.float32)
        self.period_s = float(period_s)
        self.translation_amplitude_m = tuple(float(value) for value in translation_amplitude_m)
        self.yaw_amplitude_rad = float(yaw_amplitude_rad)

    def pose(self, time: float, reference: Tensor) -> tuple[Tensor, Tensor]:
        phase = 2.0 * pi * float(time) / self.period_s
        angle = self.yaw_amplitude_rad * sin(phase)
        c, s = cos(angle), sin(angle)
        rotation = reference.new_tensor([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
        amplitude = reference.new_tensor(self.translation_amplitude_m)
        translation = amplitude * reference.new_tensor(
            [sin(phase), 0.5 * (1.0 - cos(phase)), sin(phase)]
        )
        return rotation, translation

    def velocity_parameters(self, time: float, reference: Tensor) -> tuple[Tensor, Tensor]:
        phase = 2.0 * pi * float(time) / self.period_s
        omega = 2.0 * pi / self.period_s
        angle_rate = self.yaw_amplitude_rad * omega * cos(phase)
        amplitude = reference.new_tensor(self.translation_amplitude_m)
        translation_velocity = amplitude * reference.new_tensor(
            [omega * cos(phase), 0.5 * omega * sin(phase), omega * cos(phase)]
        )
        angular_velocity = reference.new_tensor([0.0, angle_rate, 0.0])
        return angular_velocity, translation_velocity

    def transform_points(self, points: Tensor, time: float) -> Tensor:
        rotation, translation = self.pose(time, points)
        origin = self.origin.to(points)
        return (points - origin) @ rotation.T + origin + translation

    def inverse_points(self, points: Tensor, time: float) -> Tensor:
        rotation, translation = self.pose(time, points)
        origin = self.origin.to(points)
        return (points - origin - translation) @ rotation + origin

    def transform_vectors(self, vectors: Tensor, time: float) -> Tensor:
        rotation, _translation = self.pose(time, vectors)
        return vectors @ rotation.T

    def point_velocities(self, world_points: Tensor, time: float) -> Tensor:
        _rotation, translation = self.pose(time, world_points)
        angular, linear = self.velocity_parameters(time, world_points)
        center = self.origin.to(world_points) + translation
        return linear + torch.linalg.cross(angular.expand_as(world_points), world_points - center)

    def root_transport(
        self, world_points: Tensor, time: float, next_time: float
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Transport world points with the shared, non-articulated body root.

        The cloth's canonical momentum is relative to this root motion.  It
        is important that this is a pointwise SE(3) map: a COM translation
        loses the rotational part and, more importantly, cannot preserve the
        garment's offset from the moving torso.
        """

        local_points = self.inverse_points(world_points, time)
        transported = self.transform_points(local_points, next_time)
        current_velocity = self.point_velocities(world_points, time)
        next_velocity = self.point_velocities(transported, next_time)
        return transported, current_velocity, next_velocity


class WalkingSE3BodyTrajectory(SinusoidalSE3BodyTrajectory):
    """One forward step with articulated limb skinning and a shared SDF clock."""

    def __init__(
        self,
        origin: Tensor,
        *,
        step_duration_s: float = 0.12,
        steps: int = 1,
        step_length_m: float = 0.02,
        hip_y: float = -0.265,
        shoulder_y: float = 0.05,
        hip_half_width_m: float = 0.075,
        shoulder_half_width_m: float = 0.17,
        body_center_z: float = 0.88,
        leg_swing_rad: float = 5.0 * pi / 180.0,
        knee_flex_rad: float = 34.0 * pi / 180.0,
        knee_y: float = -0.56,
        arm_swing_rad: float = 10.0 * pi / 180.0,
        yaw_amplitude_rad: float = 1.5 * pi / 180.0,
        vertical_bob_m: float = 0.004,
        lateral_sway_m: float = 0.003,
        behavior_profile: BehavioralPhaseTrajectory | None = None,
    ) -> None:
        if step_duration_s <= 0.0 or step_length_m < 0.0 or steps <= 0:
            raise ValueError("walking duration/steps must be positive and step length non-negative")
        super().__init__(
            origin,
            period_s=2.0 * step_duration_s,
            translation_amplitude_m=(lateral_sway_m, vertical_bob_m, step_length_m),
            yaw_amplitude_rad=yaw_amplitude_rad,
        )
        self.step_duration_s = float(step_duration_s)
        self.steps = int(steps)
        self.step_length_m = float(step_length_m)
        self.hip_y = float(hip_y)
        self.shoulder_y = float(shoulder_y)
        self.hip_half_width_m = float(hip_half_width_m)
        self.shoulder_half_width_m = float(shoulder_half_width_m)
        self.body_center_z = float(body_center_z)
        self.leg_swing_rad = float(leg_swing_rad)
        self.knee_flex_rad = float(knee_flex_rad)
        self.knee_y = float(knee_y)
        self.arm_swing_rad = float(arm_swing_rad)
        self.vertical_bob_m = float(vertical_bob_m)
        self.lateral_sway_m = float(lateral_sway_m)
        self.behavior_profile = behavior_profile or BehavioralPhaseTrajectory()

    def _progress(self, time: float) -> float:
        total_progress = self._total_progress(time)
        if total_progress >= self.steps:
            return 1.0
        return total_progress - floor(total_progress)

    def _total_progress(self, time: float) -> float:
        port = self.behavior_profile.body_control_port(time)
        effective_duration = self.step_duration_s / port.cadence_scale
        return min(max(float(time) / effective_duration, 0.0), float(self.steps))

    def _step_phase(self, time: float) -> float:
        # A complete gait cycle is continuous across step boundaries.  The
        # translation still advances one stride per local step, while the
        # articulation phase carries the opposite-leg exchange into the next
        # step instead of snapping back to zero after a quarter-cycle.
        return pi * self._total_progress(time)

    def _leg_motion_angles(self, time: float, reference: Tensor) -> tuple[Tensor, Tensor]:
        """Return (left, right) hinge angles in the project x/y/z frame."""
        phase = self._step_phase(time)
        gait = sin(phase)
        gesture_scale = self.behavior_profile.body_control_port(time).gesture_scale
        angle = reference.new_tensor(gesture_scale * self.leg_swing_rad * gait)
        return -angle, angle

    def _arm_motion_angles(self, time: float, reference: Tensor) -> tuple[Tensor, Tensor]:
        """Return (left, right) shoulder hinge angles in the project frame."""
        phase = self._step_phase(time)
        gait = sin(phase)
        gesture_scale = self.behavior_profile.body_control_port(time).gesture_scale
        angle = reference.new_tensor(gesture_scale * self.arm_swing_rad * gait)
        return angle, -angle

    def _knee_motion_angles(self, time: float, reference: Tensor) -> tuple[Tensor, Tensor]:
        """Return positive swing-leg flexion for the left and right knees."""

        phase = self._step_phase(time)
        gait = sin(phase)
        gesture_scale = self.behavior_profile.body_control_port(time).gesture_scale
        left_flex = max(gait, 0.0) ** 2
        right_flex = max(-gait, 0.0) ** 2
        scale = gesture_scale * self.knee_flex_rad
        return (
            reference.new_tensor(scale * left_flex),
            reference.new_tensor(scale * right_flex),
        )

    def pose(self, time: float, reference: Tensor) -> tuple[Tensor, Tensor]:
        phase = self._step_phase(time)
        port = self.behavior_profile.body_control_port(time)
        angle = port.gesture_scale * self.yaw_amplitude_rad * sin(phase)
        c, s = cos(angle), sin(angle)
        rotation = reference.new_tensor([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
        total_progress = self._total_progress(time)
        progress = self._progress(time)
        if total_progress >= self.steps:
            # At the terminal knot _progress() is 1.0. Do not add that local
            # endpoint to completed_steps again, or the body jumps by one
            # extra stride and finite-difference velocities explode.
            stride_progress = float(self.steps)
        else:
            smooth_progress = progress * progress * (3.0 - 2.0 * progress)
            completed_steps = floor(total_progress)
            stride_progress = completed_steps + smooth_progress
        translation = reference.new_tensor(
            [
                port.sway_scale * self.lateral_sway_m * sin(phase),
                port.sway_scale * self.vertical_bob_m * (1.0 - cos(2.0 * phase)) * 0.5,
                port.stride_scale * self.step_length_m * stride_progress,
            ]
        )
        return rotation, translation

    def velocity_parameters(self, time: float, reference: Tensor) -> tuple[Tensor, Tensor]:
        epsilon = 1.0e-4
        before = max(0.0, float(time) - epsilon)
        after = float(time) + epsilon
        denominator = max(after - before, epsilon)
        rotation_before, translation_before = self.pose(before, reference)
        rotation_after, translation_after = self.pose(after, reference)
        yaw_before = torch.atan2(rotation_before[0, 2], rotation_before[0, 0])
        yaw_after = torch.atan2(rotation_after[0, 2], rotation_after[0, 0])
        yaw_rate = (yaw_after - yaw_before) / denominator
        angular = torch.stack((yaw_rate.new_zeros(()), yaw_rate, yaw_rate.new_zeros(())))
        return angular, (translation_after - translation_before) / denominator

    def transform_body_points(self, points: Tensor, time: float) -> Tensor:
        return self.transform_points(self._articulate(points, time, inverse=False), time)

    def inverse_body_points(self, points: Tensor, time: float) -> Tensor:
        local = self.inverse_points(points, time)
        # Two fixed-point inverse-skinning iterations are sufficient for the
        # small gait angles used by the collision SDF.
        estimate = local
        for _ in range(2):
            estimate = self._articulate(local, time, inverse=True, weight_points=estimate)
        return estimate

    def transform_body_vectors(
        self, vectors: Tensor, local_points: Tensor, time: float
    ) -> Tensor:
        articulated = self._articulate_vectors(vectors, local_points, time)
        return self.transform_vectors(articulated, time)

    def body_point_velocities(self, local_points: Tensor, time: float) -> Tensor:
        epsilon = 1.0e-4
        before = max(0.0, float(time) - epsilon)
        after = float(time) + epsilon
        denominator = max(after - before, epsilon)
        return (
            self.transform_body_points(local_points, after)
            - self.transform_body_points(local_points, before)
        ) / denominator

    def lie_message_graph(
        self, time: float, reference: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return pelvis/torso/limb SE(3) nodes and typed joint edges."""

        phase = self._step_phase(time)
        left_arm_angle, right_arm_angle = self._arm_motion_angles(time, reference)
        left_leg_angle, right_leg_angle = self._leg_motion_angles(time, reference)
        global_rotation, global_translation = self.pose(time, reference)
        origin = self.origin.to(reference)
        global_transform = torch.eye(4, dtype=reference.dtype, device=reference.device)
        global_transform[:3, :3] = global_rotation
        global_transform[:3, 3] = origin + global_translation - global_rotation @ origin

        left_knee_angle, right_knee_angle = self._knee_motion_angles(time, reference)
        transforms = global_transform[None].repeat(8, 1, 1)
        pivots = reference.new_tensor(
            [
                [0.0, self.hip_y, self.body_center_z],
                [0.0, self.shoulder_y, self.body_center_z],
                [-self.shoulder_half_width_m, self.shoulder_y, self.body_center_z],
                [self.shoulder_half_width_m, self.shoulder_y, self.body_center_z],
                [-self.hip_half_width_m, self.hip_y, self.body_center_z],
                [self.hip_half_width_m, self.hip_y, self.body_center_z],
                [-self.hip_half_width_m, self.knee_y, self.body_center_z],
                [self.hip_half_width_m, self.knee_y, self.body_center_z],
            ]
        )
        angles = torch.stack(
            (
                reference.new_zeros(()),
                reference.new_zeros(()),
                left_arm_angle,
                right_arm_angle,
                left_leg_angle,
                right_leg_angle,
                left_knee_angle,
                right_knee_angle,
            )
        )
        for index in range(2, 6):
            c, s = torch.cos(angles[index]), torch.sin(angles[index])
            local = torch.eye(4, dtype=reference.dtype, device=reference.device)
            local[:3, :3] = reference.new_tensor(
                [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]]
            )
            local[:3, 3] = pivots[index] - local[:3, :3] @ pivots[index]
            transforms[index] = global_transform @ local

        def hinge_transform(pivot: Tensor, angle: Tensor) -> Tensor:
            c, s = torch.cos(angle), torch.sin(angle)
            transform = torch.eye(
                4, dtype=reference.dtype, device=reference.device
            )
            transform[:3, :3] = torch.stack(
                (
                    reference.new_tensor([1.0, 0.0, 0.0]),
                    torch.stack((angle.new_zeros(()), c, -s)),
                    torch.stack((angle.new_zeros(()), s, c)),
                )
            )
            transform[:3, 3] = pivot - transform[:3, :3] @ pivot
            return transform

        left_hip = hinge_transform(pivots[4], left_leg_angle)
        right_hip = hinge_transform(pivots[5], right_leg_angle)
        left_knee_pivot = left_hip[:3, :3] @ pivots[6] + left_hip[:3, 3]
        right_knee_pivot = right_hip[:3, :3] @ pivots[7] + right_hip[:3, 3]
        transforms[6] = (
            global_transform
            @ hinge_transform(left_knee_pivot, left_knee_angle)
            @ left_hip
        )
        transforms[7] = (
            global_transform
            @ hinge_transform(right_knee_pivot, right_knee_angle)
            @ right_hip
        )

        progress = self._progress(time)
        node_features = reference.new_tensor(
            [
                [1, 0, 0, 0, 0, progress, sin(phase), cos(phase)],
                [0, 1, 0, 0, 0, progress, sin(phase), cos(phase)],
                [0, 0, 1, 0, -1, progress, sin(phase), cos(phase)],
                [0, 0, 1, 0, 1, progress, sin(phase), cos(phase)],
                [0, 0, 0, 1, -1, progress, sin(phase), cos(phase)],
                [0, 0, 0, 1, 1, progress, sin(phase), cos(phase)],
                [0, 0, 0, 1, -1, progress, float(left_knee_angle), cos(phase)],
                [0, 0, 0, 1, 1, progress, float(right_knee_angle), cos(phase)],
            ]
        )
        edge_index = torch.tensor(
            [[0, 1, 1, 0, 0, 4, 5], [1, 2, 3, 4, 5, 6, 7]],
            dtype=torch.long,
            device=reference.device,
        )
        joint_features = reference.new_tensor(
            [
                [1, 0, 0, progress],
                [0, 1, -1, progress],
                [0, 1, 1, progress],
                [0, 0, -1, progress],
                [0, 0, 1, progress],
                [0, 0, -1, float(left_knee_angle)],
                [0, 0, 1, float(right_knee_angle)],
            ]
        )
        return transforms, node_features, edge_index, joint_features

    def _articulation_angles(self, points: Tensor, time: float) -> Tensor:
        left_leg_angle, right_leg_angle = self._leg_motion_angles(time, points)
        left_arm_angle, right_arm_angle = self._arm_motion_angles(time, points)
        x, y = points[:, 0], points[:, 1]
        side = torch.tanh(x / 0.018)
        leg_vertical = torch.clamp((self.hip_y + 0.035 - y) / 0.10, 0.0, 1.0)
        leg_center = torch.clamp((0.22 - torch.abs(x)) / 0.05, 0.0, 1.0)
        arm_lateral = torch.clamp(
            (torch.abs(x) - (self.shoulder_half_width_m - 0.025)) / 0.10,
            0.0,
            1.0,
        )
        arm_vertical = torch.clamp((y - (self.shoulder_y - 0.30)) / 0.08, 0.0, 1.0)
        leg_angle = torch.where(x < 0.0, left_leg_angle, right_leg_angle)
        arm_angle = torch.where(x < 0.0, left_arm_angle, right_arm_angle)
        return (
            leg_angle * leg_vertical * leg_center
            - arm_angle * arm_lateral * arm_vertical
        )

    def _leg_skinning_weights(
        self, reference: Tensor, time: float, *, inverse: bool
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Return hip-transition weights and one rigid angle per leg."""
        left_angle, right_angle = self._leg_motion_angles(time, reference)
        x, y = reference[:, 0], reference[:, 1]
        # Fully below the pelvis, each leg receives one rigid transform. A
        # short smooth transition band is kept only at the hip so the crotch
        # does not crack; do not attenuate weights along the calf/foot, which
        # was the source of visible longitudinal stretching.
        lower_body = torch.clamp(
            (self.hip_y - y - 0.02) / 0.04, 0.0, 1.0
        )
        lower_body = lower_body.square() * (3.0 - 2.0 * lower_body)
        # The Anny mesh has arms that descend below the hip line in its
        # relaxed A-pose. They must not receive the leg hinge first and the
        # arm hinge second; that double transform is the source of visible
        # rubbery limb deformation. Shoulder-outside vertices belong to the
        # arm branch even when their y coordinate is low.
        # The low-poly Anny legs are wider than the nominal hip and overlap
        # the old shoulder-width test in x.  That made lower-leg vertices
        # receive a second shoulder rotation, visibly stretching the feet.
        # Require both an outer-arm x position and a height above the lower
        # hip band; the lower arm may hang, but it must not be confused with
        # the outer calf.
        arm_lateral_threshold = max(self.shoulder_half_width_m + 0.01, 0.18)
        arm_region = torch.clamp(
            (torch.abs(x) - (arm_lateral_threshold - 0.01)) / 0.03,
            0.0,
            1.0,
        )
        arm_region = arm_region.square() * (3.0 - 2.0 * arm_region)
        arm_height = torch.clamp(
            (y - (self.hip_y - 0.50)) / 0.12,
            0.0,
            1.0,
        )
        arm_height = arm_height.square() * (3.0 - 2.0 * arm_height)
        # A few low-resolution body triangles straddle x=0 at the inner
        # thighs. Hardly assigning their two corners to opposite hips makes
        # those triangles stretch dramatically. Fade the hinge only in this
        # narrow medial seam; the actual leg branches remain rigid.
        medial_seam = torch.clamp((torch.abs(x) - 0.006) / 0.045, 0.0, 1.0)
        medial_seam = medial_seam.square() * (3.0 - 2.0 * medial_seam)
        arm_region = arm_region * arm_height
        lower_body = lower_body * (1.0 - arm_region) * medial_seam
        # Once below the transition band, a vertex belongs to exactly one
        # articulated leg.  A sigmoid here looks smooth but keeps blending
        # the two hip rotations through the calf, which changes edge lengths
        # and makes the avatar itself appear rubbery during a stride.
        right_branch = (x >= 0.0).to(dtype=reference.dtype)
        left_branch = 1.0 - right_branch
        right = lower_body * right_branch
        left = lower_body * left_branch
        pelvis = (1.0 - left - right).clamp(0.0, 1.0)
        if inverse:
            left_angle, right_angle = -left_angle, -right_angle
        return pelvis, left, right, left_angle, right_angle

    def _arm_skinning_weights(
        self, reference: Tensor, time: float, *, inverse: bool
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Return shoulder-transition weights and one rigid angle per arm."""
        left_angle, right_angle = self._arm_motion_angles(time, reference)
        x, y = reference[:, 0], reference[:, 1]
        arm_lateral_threshold = max(self.shoulder_half_width_m + 0.01, 0.18)
        arm_lateral = torch.clamp(
            (torch.abs(x) - (arm_lateral_threshold - 0.01)) / 0.03,
            0.0,
            1.0,
        )
        arm_lateral = arm_lateral.square() * (3.0 - 2.0 * arm_lateral)
        arm_height = torch.clamp(
            (y - (self.hip_y - 0.50)) / 0.12,
            0.0,
            1.0,
        )
        arm_height = arm_height.square() * (3.0 - 2.0 * arm_height)
        # Do not fade the swing along an actual arm's length. The height gate
        # only rejects low leg/foot vertices that happen to be wide in x; it
        # keeps the hand and forearm on one rigid arm branch.
        arm_body = arm_lateral * arm_height
        right_branch = (x >= 0.0).to(dtype=reference.dtype)
        left_branch = 1.0 - right_branch
        right = arm_body * right_branch
        left = arm_body * left_branch
        torso = (1.0 - left - right).clamp(0.0, 1.0)
        if inverse:
            left_angle, right_angle = -left_angle, -right_angle
        return torso, left, right, left_angle, right_angle

    def _knee_skinning_weights(self, reference: Tensor) -> tuple[Tensor, Tensor]:
        """Blend only across the knee ring; each shin remains a rigid branch."""

        x, y = reference[:, 0], reference[:, 1]
        lower_leg = torch.clamp(
            (self.knee_y + 0.06 - y) / 0.05,
            0.0,
            1.0,
        )
        lower_leg = lower_leg.square() * (3.0 - 2.0 * lower_leg)
        left = lower_leg * (x < 0.0).to(reference.dtype)
        right = lower_leg * (x >= 0.0).to(reference.dtype)
        return left, right

    def _joint_pivots(self, points: Tensor) -> Tensor:
        x = points[:, 0]
        is_arm = torch.abs(x) > self.shoulder_half_width_m - 0.025
        side = torch.where(x < 0.0, -torch.ones_like(x), torch.ones_like(x))
        pivot_x = torch.where(
            is_arm,
            side * self.shoulder_half_width_m,
            side * self.hip_half_width_m,
        )
        pivot_y = torch.where(
            is_arm,
            torch.full_like(x, self.shoulder_y),
            torch.full_like(x, self.hip_y),
        )
        return torch.stack(
            (pivot_x, pivot_y, torch.full_like(x, self.body_center_z)), dim=-1
        )

    def _articulate(
        self,
        points: Tensor,
        time: float,
        *,
        inverse: bool,
        weight_points: Tensor | None = None,
    ) -> Tensor:
        reference = points if weight_points is None else weight_points
        _pelvis, left, right, left_angle, right_angle = self._leg_skinning_weights(
            reference, time, inverse=inverse
        )
        left_knee_weight, right_knee_weight = self._knee_skinning_weights(reference)
        left_knee_angle, right_knee_angle = self._knee_motion_angles(time, reference)

        def rotate_about(value: Tensor, pivot: Tensor, angle: Tensor) -> Tensor:
            relative = value - pivot
            c, s = torch.cos(angle), torch.sin(angle)
            return torch.stack(
                (
                    relative[:, 0],
                    c * relative[:, 1] - s * relative[:, 2],
                    s * relative[:, 1] + c * relative[:, 2],
                ),
                dim=-1,
            ) + pivot

        side = torch.where(
            reference[:, 0] < 0.0,
            -torch.ones_like(reference[:, 0]),
            torch.ones_like(reference[:, 0]),
        )
        hip_pivot = torch.stack(
            (
                side * self.hip_half_width_m,
                torch.full_like(side, self.hip_y),
                torch.full_like(side, self.body_center_z),
            ),
            dim=-1,
        )
        knee_base = torch.stack(
            (
                side * self.hip_half_width_m,
                torch.full_like(side, self.knee_y),
                torch.full_like(side, self.body_center_z),
            ),
            dim=-1,
        )
        hip_angle = left * left_angle + right * right_angle
        knee_angle = (
            left_knee_weight * left_knee_angle
            + right_knee_weight * right_knee_angle
        )

        _torso, left_arm, right_arm, left_arm_angle, right_arm_angle = (
            self._arm_skinning_weights(reference, time, inverse=inverse)
        )
        arm_pivot = torch.stack(
            (
                side * self.shoulder_half_width_m,
                torch.full_like(side, self.shoulder_y),
                torch.full_like(side, self.body_center_z),
            ),
            dim=-1,
        )
        arm_angle = left_arm * left_arm_angle + right_arm * right_arm_angle

        # Interpolate on SO(2), not between Euclidean point positions. This is
        # the single-axis dual-quaternion/geodesic blend and avoids the volume
        # collapse of linear blend skinning at shoulders, hips, and knees.
        if inverse:
            articulated = rotate_about(points, arm_pivot, arm_angle)
            forward_hip_angle = -hip_angle
            forward_knee_pivot = rotate_about(
                knee_base, hip_pivot, forward_hip_angle
            )
            articulated = rotate_about(
                articulated, forward_knee_pivot, -knee_angle
            )
            return rotate_about(articulated, hip_pivot, hip_angle)

        articulated = rotate_about(points, hip_pivot, hip_angle)
        knee_pivot = rotate_about(knee_base, hip_pivot, hip_angle)
        articulated = rotate_about(articulated, knee_pivot, knee_angle)
        return rotate_about(articulated, arm_pivot, arm_angle)

    def _articulate_vectors(
        self, vectors: Tensor, local_points: Tensor, time: float
    ) -> Tensor:
        _pelvis, left, right, left_angle, right_angle = self._leg_skinning_weights(
            local_points, time, inverse=False
        )
        left_knee_weight, right_knee_weight = self._knee_skinning_weights(local_points)
        left_knee_angle, right_knee_angle = self._knee_motion_angles(
            time, local_points
        )

        def rotate_vector(value: Tensor, angle: Tensor) -> Tensor:
            c, s = torch.cos(angle), torch.sin(angle)
            return torch.stack(
                (
                    value[:, 0],
                    c * value[:, 1] - s * value[:, 2],
                    s * value[:, 1] + c * value[:, 2],
                ),
                dim=-1,
            )

        hip_angle = left * left_angle + right * right_angle
        knee_angle = (
            left_knee_weight * left_knee_angle
            + right_knee_weight * right_knee_angle
        )
        articulated = rotate_vector(vectors, hip_angle)
        articulated = rotate_vector(articulated, knee_angle)
        _torso, left_arm, right_arm, left_arm_angle, right_arm_angle = (
            self._arm_skinning_weights(local_points, time, inverse=False)
        )
        arm_angle = left_arm * left_arm_angle + right_arm * right_arm_angle
        return rotate_vector(articulated, arm_angle)


class CapturedWalkingSE3BodyTrajectory(WalkingSE3BodyTrajectory):
    """Retarget a captured BVH walk onto the compact Anny body interface.

    The low-poly avatar remains the geometric source of truth. This class
    imports the capture root path, yaw, thigh hinge and upper-arm swing, then
    retargets them to Anny's measured pivots. Cloth still sees the same moving
    body SDF, contact velocity, root transport and Lie-message graph.
    """

    def __init__(
        self,
        origin: Tensor,
        motion_path: str,
        *,
        step_duration_s: float = 2.0,
        steps: int = 2,
        step_length_m: float = 0.06,
        hip_y: float = -0.265,
        shoulder_y: float = 0.05,
        hip_half_width_m: float = 0.075,
        shoulder_half_width_m: float = 0.17,
        body_center_z: float = 0.88,
        leg_swing_rad: float = 5.0 * pi / 180.0,
        knee_flex_rad: float = 34.0 * pi / 180.0,
        knee_y: float = -0.56,
        arm_swing_rad: float = 10.0 * pi / 180.0,
        yaw_amplitude_rad: float = 1.5 * pi / 180.0,
        behavior_profile: BehavioralPhaseTrajectory | None = None,
        motion_limb_scale: float = 0.50,
        motion_arm_scale: float = 1.00,
        motion_yaw_scale: float = 0.35,
        vertical_bob_m: float = 0.003,
        lateral_sway_m: float = 0.002,
    ) -> None:
        if (
            motion_limb_scale <= 0.0
            or motion_arm_scale <= 0.0
            or motion_yaw_scale < 0.0
        ):
            raise ValueError(
                "motion limb/arm scales must be positive, with non-negative yaw scale"
            )
        self.motion_path = str(motion_path)
        self.motion = BVHMotion.from_file(motion_path)
        if self.motion.frame_count < 2:
            raise ValueError("captured walk must contain at least two frames")
        self.motion_limb_scale = float(motion_limb_scale)
        self.motion_arm_scale = float(motion_arm_scale)
        self.motion_yaw_scale = float(motion_yaw_scale)
        self.captured_vertical_bob_m = float(vertical_bob_m)
        self.captured_lateral_sway_m = float(lateral_sway_m)
        self._target_duration_s = max(float(step_duration_s) * int(steps), 1.0e-6)

        root_x = self.motion.channel("Hips", "Xposition")
        root_y = self.motion.channel("Hips", "Yposition")
        root_z = self.motion.channel("Hips", "Zposition")
        del root_y
        source_path_xz = np.stack((root_x - root_x[0], root_z - root_z[0]), axis=-1)
        forward = source_path_xz[-1].astype(np.float64)
        forward_distance = float(np.linalg.norm(forward))
        if forward_distance < 1.0e-5:
            forward = np.asarray([0.0, 1.0], dtype=np.float64)
            forward_distance = 1.0
        else:
            forward /= forward_distance
        lateral = np.asarray([forward[1], -forward[0]], dtype=np.float64)
        forward_component = source_path_xz @ forward
        if forward_component[-1] < 0.0:
            forward = -forward
            lateral = -lateral
            forward_component = source_path_xz @ forward
        lateral_component = source_path_xz @ lateral
        target_distance_m = max(float(step_length_m) * int(steps), 0.02)
        self._target_distance_m = target_distance_m
        root_scale = target_distance_m / max(float(forward_component[-1]) / 100.0, 1.0e-5)
        self._captured_root_translation = np.stack(
            (
                lateral_component * root_scale / 100.0,
                np.zeros_like(lateral_component),
                forward_component * root_scale / 100.0,
            ),
            axis=-1,
        ).astype(np.float32)
        root_yaw = np.deg2rad(self.motion.channel("Hips", "Yrotation"))
        self._captured_root_yaw = np.unwrap(root_yaw).astype(np.float32)
        self._captured_root_yaw -= self._captured_root_yaw[0]

        right_leg = self.motion.channel("RightUpLeg", "Xrotation")
        left_leg = self.motion.channel("LeftUpLeg", "Xrotation")
        right_knee = self.motion.channel("RightLeg", "Xrotation")
        left_knee = self.motion.channel("LeftLeg", "Xrotation")
        # The Anny branch rotates around its local X axis (the sagittal
        # swing axis), so retarget the capture's Xrotation rather than its
        # BVH lateral Zrotation channel.
        right_arm = self.motion.channel("RightArm", "Xrotation")
        left_arm = self.motion.channel("LeftArm", "Xrotation")
        # The capture has a few keyframe-scale thigh steps. Feeding those
        # raw channels into the moving SDF creates an artificial 1-2 m/s
        # contact velocity, so a skirt can receive a real but visually wrong
        # impulse at the posterior hip. Low-pass the captured leg signal while
        # preserving its phase and endpoint values before retargeting.
        self._captured_left_leg = self._smooth_leg_signal(
            np.deg2rad(left_leg - left_leg[0]).astype(np.float32)
        )
        self._captured_right_leg = self._smooth_leg_signal(
            np.deg2rad(right_leg - right_leg[0]).astype(np.float32)
        )
        self._captured_left_knee = self._smooth_leg_signal(
            np.deg2rad(left_knee - left_knee[0]).astype(np.float32)
        )
        self._captured_right_knee = self._smooth_leg_signal(
            np.deg2rad(right_knee - right_knee[0]).astype(np.float32)
        )
        self._captured_left_arm = self._smooth_arm_signal(
            np.deg2rad(left_arm - left_arm[0]).astype(np.float32)
        )
        self._captured_right_arm = self._smooth_arm_signal(
            np.deg2rad(right_arm - right_arm[0]).astype(np.float32)
        )

        super().__init__(
            origin,
            step_duration_s=step_duration_s,
            steps=steps,
            step_length_m=step_length_m,
            hip_y=hip_y,
            shoulder_y=shoulder_y,
            hip_half_width_m=hip_half_width_m,
            shoulder_half_width_m=shoulder_half_width_m,
            body_center_z=body_center_z,
            leg_swing_rad=leg_swing_rad,
            knee_flex_rad=knee_flex_rad,
            knee_y=knee_y,
            arm_swing_rad=arm_swing_rad,
            yaw_amplitude_rad=yaw_amplitude_rad,
            vertical_bob_m=0.0,
            lateral_sway_m=0.0,
            behavior_profile=behavior_profile,
        )

        self._sample_times = np.arange(self.motion.frame_count, dtype=np.float32) * self.motion.frame_time_s

    def _capture_fraction(self, time: float) -> float:
        return float(np.clip(float(time) / self._target_duration_s, 0.0, 1.0))

    def _capture_sample(self, values: np.ndarray, time: float) -> float:
        capture_time = self._capture_fraction(time) * self.motion.duration_s
        return float(np.interp(capture_time, self._sample_times, values))

    def _total_progress(self, time: float) -> float:
        return self.steps * self._capture_fraction(time)

    def _step_phase(self, time: float) -> float:
        # The phase channel remains periodic for the scene graph, while the
        # actual limb angles below come directly from the capture.
        return 2.0 * pi * self._capture_fraction(time)

    def _leg_motion_angles(self, time: float, reference: Tensor) -> tuple[Tensor, Tensor]:
        """Blend a continuous gait prior with the captured leg residual."""

        port = self.behavior_profile.body_control_port(time)
        captured_left = (
            port.gesture_scale
            * self.motion_limb_scale
            * self._capture_sample(self._captured_left_leg, time)
        )
        captured_right = (
            port.gesture_scale
            * self.motion_limb_scale
            * self._capture_sample(self._captured_right_leg, time)
        )
        phase = self._step_phase(time)
        gait = port.gesture_scale * self.leg_swing_rad * sin(phase)
        left = 0.85 * (-gait) + 0.15 * captured_left
        right = 0.85 * gait + 0.15 * captured_right
        return reference.new_tensor(left), reference.new_tensor(right)

    def _knee_motion_angles(self, time: float, reference: Tensor) -> tuple[Tensor, Tensor]:
        """Blend captured knee flexion with one clean swing-leg gait cycle."""

        port = self.behavior_profile.body_control_port(time)
        phase = self._step_phase(time)
        gait = sin(phase)
        prior_left = self.knee_flex_rad * max(gait, 0.0) ** 2
        prior_right = self.knee_flex_rad * max(-gait, 0.0) ** 2
        captured_left = max(
            0.0, self._capture_sample(self._captured_left_knee, time)
        )
        captured_right = max(
            0.0, self._capture_sample(self._captured_right_knee, time)
        )
        scale = port.gesture_scale * self.motion_limb_scale
        left = 0.75 * port.gesture_scale * prior_left + 0.25 * scale * captured_left
        right = 0.75 * port.gesture_scale * prior_right + 0.25 * scale * captured_right
        return reference.new_tensor(left), reference.new_tensor(right)

    def pose(self, time: float, reference: Tensor) -> tuple[Tensor, Tensor]:
        port = self.behavior_profile.body_control_port(time)
        yaw = (
            port.gesture_scale
            * self.motion_yaw_scale
            * self._capture_sample(self._captured_root_yaw, time)
        )
        c, s = cos(yaw), sin(yaw)
        rotation = reference.new_tensor([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
        translation_np = np.asarray(
            [self._capture_sample(self._captured_root_translation[:, axis], time) for axis in range(3)],
            dtype=np.float32,
        )
        # A long BVH clip can contain a slow lead-in and several footfalls.
        # Retarget its lateral character, but use one monotone minimum-jerk
        # forward stride for the requested simulation interval. Otherwise a
        # nominal 22 cm step may travel only a few millimetres during the
        # visible knee-lift phase and then rush forward near the final frame.
        fraction = self._capture_fraction(time)
        forward_progress = fraction * fraction * (3.0 - 2.0 * fraction)
        translation_np[2] = self._target_distance_m * forward_progress
        translation = reference.new_tensor(translation_np)
        translation = translation * reference.new_tensor(
            [port.sway_scale, 1.0, port.stride_scale]
        )
        # The capture root path carries forward travel, but its vertical and
        # lateral channels are discarded during retargeting. Restore a small
        # gait-phase pelvis excursion so cloth receives real inertial input
        # without inventing a wind force or changing body shape.
        phase = self._step_phase(time)
        translation = translation + reference.new_tensor(
            [
                port.sway_scale * self.captured_lateral_sway_m * sin(phase),
                port.sway_scale
                * self.captured_vertical_bob_m
                * 0.5
                * (1.0 - cos(2.0 * phase)),
                0.0,
            ]
        )
        return rotation, translation

    def _arm_motion_angles(self, time: float, reference: Tensor) -> tuple[Tensor, Tensor]:
        # The capture contains useful asymmetry, but its upper-arm channels
        # are noisy and are not guaranteed to be phase-locked to the retargeted
        # legs. Use a smooth anti-phase gait prior as the main signal, then
        # add a small symmetric projection of the captured asymmetry. This
        # keeps the walk readable without making the shoulder motion robotic.
        port = self.behavior_profile.body_control_port(time)
        scale = port.gesture_scale * self.motion_arm_scale
        captured_left = self._capture_sample(self._captured_left_arm, time) * scale
        captured_right = self._capture_sample(self._captured_right_arm, time) * scale
        captured_swing = 0.5 * (captured_left - captured_right)
        phase = self._step_phase(time)
        gait_swing = port.gesture_scale * self.arm_swing_rad * sin(phase)
        angle = 0.70 * gait_swing + 0.30 * captured_swing
        return reference.new_tensor(angle), reference.new_tensor(-angle)

    @staticmethod
    def _smooth_arm_signal(values: np.ndarray) -> np.ndarray:
        """Remove frame-level BVH jitter without changing the gait phase."""

        if values.size < 5:
            return values
        kernel = np.asarray([1.0, 2.0, 3.0, 2.0, 1.0], dtype=np.float32)
        kernel /= float(kernel.sum())
        padded = np.pad(values, (2, 2), mode="edge")
        return np.convolve(padded, kernel, mode="valid").astype(np.float32)

    @staticmethod
    def _smooth_leg_signal(values: np.ndarray) -> np.ndarray:
        """Remove capture keyframe steps from the body contact trajectory."""

        if values.size < 7:
            return values
        kernel = np.asarray([1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0], dtype=np.float32)
        kernel /= float(kernel.sum())
        padded = np.pad(values, (3, 3), mode="edge")
        smoothed = np.convolve(padded, kernel, mode="valid")
        # Keep the capture's neutral first frame and terminal pose exact;
        # only the interior channel is filtered.
        smoothed[0] = values[0]
        smoothed[-1] = values[-1]
        return smoothed.astype(np.float32)

    def capture_summary(self) -> dict[str, object]:
        return {
            "source_path": self.motion_path,
            "format": "BVH",
            "frames": self.motion.frame_count,
            "frame_time_s": self.motion.frame_time_s,
            "duration_s": self.motion.duration_s,
            "retarget_duration_s": self._target_duration_s,
            "motion_limb_scale": self.motion_limb_scale,
            "motion_arm_scale": self.motion_arm_scale,
            "motion_yaw_scale": self.motion_yaw_scale,
            "vertical_bob_m": self.captured_vertical_bob_m,
            "lateral_sway_m": self.captured_lateral_sway_m,
            "root_path_unit": "centimeter converted to project meters",
            "root_forward_axis": "capture dominant XZ travel projected to project +z",
            "root_forward_timing": (
                "monotone cubic stride over the requested interval; captured "
                "lateral path and yaw retained"
            ),
            "retargeted_channels": {
                "root": "Hips.{Xposition,Yposition,Zposition,Yrotation}",
                "legs": (
                    "{Left,Right}UpLeg.Xrotation + "
                    "{Left,Right}Leg.Xrotation knee flexion"
                ),
                "arms": "{Left,Right}Arm.Xrotation",
            },
            "arm_motion_policy": (
                "70% smooth anti-phase gait prior + 30% symmetric captured "
                "upper-arm signal; shoulder pivot remains rigid"
            ),
            "body_geometry": "Anny low-poly mesh and measured pivots remain unchanged",
        }


class MovingBodyContactQuery:
    """Closest-point query against a rigid or invertibly articulated body mesh."""

    def __init__(self, avatar: object, trajectory: SinusoidalSE3BodyTrajectory) -> None:
        if not hasattr(avatar, "_proximity") or not hasattr(avatar, "_collision_mesh"):
            raise TypeError("avatar must expose the BodyMeshAvatar proximity interface")
        self.avatar = avatar
        self.trajectory = trajectory
        collision_mesh = avatar._collision_mesh
        self._surface_points = np.asarray(collision_mesh.vertices, dtype=np.float32)
        self._surface_normals = np.asarray(collision_mesh.vertex_normals, dtype=np.float32)
        self._surface_normals /= np.maximum(
            np.linalg.norm(self._surface_normals, axis=1, keepdims=True), 1.0e-8
        )
        self._surface_faces = np.asarray(collision_mesh.faces, dtype=np.int64)
        self._surface_tree = cKDTree(self._surface_points)

    def _triangle_vertex_normals(
        self, points: np.ndarray, triangle_ids: np.ndarray
    ) -> np.ndarray:
        """Interpolate body normals instead of switching at face boundaries."""

        face_vertices = self._surface_faces[np.asarray(triangle_ids, dtype=np.int64)]
        triangles = self._surface_points[face_vertices]
        edge_0 = triangles[:, 1] - triangles[:, 0]
        edge_1 = triangles[:, 2] - triangles[:, 0]
        offset = np.asarray(points, dtype=np.float32) - triangles[:, 0]
        d00 = np.einsum("ij,ij->i", edge_0, edge_0)
        d01 = np.einsum("ij,ij->i", edge_0, edge_1)
        d11 = np.einsum("ij,ij->i", edge_1, edge_1)
        d20 = np.einsum("ij,ij->i", offset, edge_0)
        d21 = np.einsum("ij,ij->i", offset, edge_1)
        denominator = d00 * d11 - d01 * d01
        v = (d11 * d20 - d01 * d21) / np.maximum(denominator, 1.0e-12)
        w = (d00 * d21 - d01 * d20) / np.maximum(denominator, 1.0e-12)
        barycentric = np.stack((1.0 - v - w, v, w), axis=1)
        vertex_normals = self._surface_normals[face_vertices]
        normals = np.sum(vertex_normals * barycentric[..., None], axis=1)
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1.0e-8)
        return normals.astype(np.float32, copy=False)

    def _triangle_contact_normals(
        self,
        points: np.ndarray,
        closest: np.ndarray,
        distances: np.ndarray,
        signed_distance: np.ndarray,
        triangle_ids: np.ndarray,
    ) -> np.ndarray:
        """Use the closest-point direction as the smooth contact normal.

        On a coarse body mesh, interpolated vertex normals can point away from
        the actual closest triangle plane and make a cloth vertex jump outward
        at a hip or glute boundary. The signed closest-point direction is the
        local SDF normal where it is resolved; barycentric vertex normals are
        retained only for points numerically on the surface.
        """

        fallback = self._triangle_vertex_normals(closest, triangle_ids)
        delta = np.asarray(points, dtype=np.float32) - np.asarray(closest, dtype=np.float32)
        length = np.linalg.norm(delta, axis=1)
        radial = delta / np.maximum(length[:, None], 1.0e-8)
        # trimesh's signed distance is positive inside and negative outside.
        # Flip the displacement for inside samples to keep an outward normal.
        radial *= np.where(np.asarray(signed_distance)[:, None] > 0.0, -1.0, 1.0)
        use_radial = np.asarray(distances) > 1.0e-5
        normals = np.where(use_radial[:, None], radial, fallback)
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1.0e-8)
        return normals.astype(np.float32, copy=False)

    def __call__(self, positions: Tensor, physical_time: float) -> BodyContactSample:
        inverse_body = getattr(self.trajectory, "inverse_body_points", None)
        local = (
            inverse_body(positions, physical_time)
            if inverse_body is not None
            else self.trajectory.inverse_points(positions, physical_time)
        )
        local_np = local.detach().cpu().numpy()
        distances, surface_ids = self._surface_tree.query(local_np, k=8, workers=1)
        distances = np.asarray(distances, dtype=np.float32)
        surface_ids = np.asarray(surface_ids, dtype=np.int64)
        weights = 1.0 / np.maximum(distances, 1.0e-5)
        weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1.0e-8)
        closest = np.sum(
            self._surface_points[surface_ids] * weights[..., None], axis=1
        )
        normals = np.sum(
            self._surface_normals[surface_ids] * weights[..., None], axis=1
        )
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1.0e-8)
        # The closest-point face normal is the local SDF gradient. Using its
        # oriented projection preserves the cloth gap sign while avoiding a
        # full ray-cast ``signed_distance`` call for every vertex and every
        # contact iteration. This is the differentiable local-contact model;
        # topology preprocessing still uses the exact watertight SDF.
        outside_gap = np.sum((local_np - closest) * normals, axis=-1).astype(
            np.float32,
            copy=False,
        )
        # The KD-tree is a fast broad phase. Refine only the near-body band
        # with exact triangle closest points and the watertight signed
        # distance. A closest-face normal alone is not a reliable inside /
        # outside test on concave limbs, which can accept a skirt after it
        # has crossed the body.
        # The cloth contact clearance is 3.5 mm in the calibrated scenes.
        # Keep the exact watertight query in a narrow band around that shell;
        # using a 10 mm band made every boundary sample enter trimesh's
        # triangle search during long rollouts without improving contact.
        near_body = np.flatnonzero(outside_gap < 0.0038)
        if len(near_body):
            exact_closest, exact_distance, triangle_ids = (
                self.avatar._proximity.on_surface(local_np[near_body])
            )
            signed_distance = np.asarray(
                self.avatar._proximity.signed_distance(local_np[near_body]),
                dtype=np.float32,
            )
            exact_normals = self._triangle_contact_normals(
                local_np[near_body],
                exact_closest,
                np.asarray(exact_distance, dtype=np.float32),
                signed_distance,
                triangle_ids,
            )
            closest[near_body] = exact_closest
            normals[near_body] = exact_normals
            # trimesh uses positive signed distance inside. The contact gap
            # is positive outside, so the robust local gap is its negative.
            outside_gap[near_body] = -signed_distance
        # ``face_normals`` are the local outward geometry prior. Avoid a
        # per-query ray cast here: contact is resolved from the closest
        # triangle and its oriented normal, while topology preprocessing may
        # still use the exact watertight SDF offline.
        outside_gap = np.sum((local_np - closest) * normals, axis=-1).astype(
            np.float32, copy=False
        )
        if len(near_body):
            # Preserve the watertight SDF sign after the fast geometric
            # recomputation above.
            outside_gap[near_body] = -signed_distance
        closest = local_np - outside_gap[:, None] * normals
        closest_tensor = torch.as_tensor(closest, dtype=positions.dtype, device=positions.device)
        normal_tensor = torch.as_tensor(normals, dtype=positions.dtype, device=positions.device)
        transform_body = getattr(self.trajectory, "transform_body_points", None)
        transform_body_vectors = getattr(self.trajectory, "transform_body_vectors", None)
        body_point_velocities = getattr(self.trajectory, "body_point_velocities", None)
        world_closest = (
            transform_body(closest_tensor, physical_time)
            if transform_body is not None
            else self.trajectory.transform_points(closest_tensor, physical_time)
        )
        world_normals = (
            transform_body_vectors(normal_tensor, closest_tensor, physical_time)
            if transform_body_vectors is not None
            else self.trajectory.transform_vectors(normal_tensor, physical_time)
        )
        body_velocity = (
            body_point_velocities(closest_tensor, physical_time)
            if body_point_velocities is not None
            else self.trajectory.point_velocities(world_closest, physical_time)
        )
        return BodyContactSample(world_closest, world_normals, body_velocity)

    def project_surface(
        self,
        positions: Tensor,
        faces: Tensor,
        physical_time: float,
        clearance_m: float,
    ) -> Tensor:
        """Project cloth triangle interiors out of the moving body.

        Vertex-only SDF checks miss the common shoulder/axilla failure mode:
        all three cloth vertices can be outside while the triangle span cuts
        through the arm or torso.  Three barycentric samples per face provide
        a small, topology-preserving continuous contact pass without changing
        the intrinsic shell or adding hard garment anchors.
        """

        if clearance_m <= 0.0:
            raise ValueError("surface contact clearance must be positive")
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError("surface contact faces must have shape (F,3)")
        points = positions.detach().cpu().numpy().astype(np.float32, copy=False)
        face_indices = faces.detach().cpu().numpy().astype(np.int64, copy=False)
        triangles = points[face_indices]
        barycentric = np.asarray(
            (
                (2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0),
                (1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0),
                (1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0),
            ),
            dtype=np.float32,
        )
        samples = np.einsum("qk,fkj->fqj", barycentric, triangles).reshape(-1, 3)
        inverse_body = getattr(self.trajectory, "inverse_body_points", None)
        local = (
            inverse_body(torch.as_tensor(samples), physical_time).numpy()
            if inverse_body is not None
            else self.trajectory.inverse_points(torch.as_tensor(samples), physical_time).numpy()
        )
        closest, distances, triangle_ids = self.avatar._proximity.on_surface(local)
        signed_distance = np.asarray(
            self.avatar._proximity.signed_distance(local), dtype=np.float32
        )
        local_normals = self._triangle_contact_normals(
            local,
            closest,
            np.asarray(distances, dtype=np.float32),
            signed_distance,
            triangle_ids,
        )
        # With the radial contact normal this is the signed closest-point
        # distance, including the correct inside/outside orientation.
        local_gap = -signed_distance
        # Only the narrow band is corrected.  This prevents a free hem or a
        # loose sleeve from being pulled to a distant body surface.
        active = (local_gap < float(clearance_m)) & (distances < 0.025)
        if not np.any(active):
            return positions
        closest_tensor = torch.as_tensor(closest, dtype=positions.dtype, device=positions.device)
        normal_tensor = torch.as_tensor(local_normals, dtype=positions.dtype, device=positions.device)
        transform_body = getattr(self.trajectory, "transform_body_points", None)
        transform_body_vectors = getattr(self.trajectory, "transform_body_vectors", None)
        world_closest = (
            transform_body(closest_tensor, physical_time)
            if transform_body is not None
            else self.trajectory.transform_points(closest_tensor, physical_time)
        )
        world_normals = (
            transform_body_vectors(normal_tensor, closest_tensor, physical_time)
            if transform_body_vectors is not None
            else self.trajectory.transform_vectors(normal_tensor, physical_time)
        )
        world_normals = torch.nn.functional.normalize(world_normals, dim=-1).cpu().numpy()
        world_samples = torch.as_tensor(samples, dtype=positions.dtype, device=positions.device)
        world_closest_np = world_closest.detach().cpu().numpy()
        world_gaps = np.sum(
            (samples - world_closest_np) * world_normals, axis=-1
        )
        correction = np.maximum(float(clearance_m) - world_gaps, 0.0)[:, None] * world_normals
        correction = np.clip(correction, -0.008, 0.008)
        correction[~active] = 0.0
        updates = np.zeros_like(points)
        weights = np.zeros(len(points), dtype=np.float32)
        for sample_id, correction_value in enumerate(correction):
            if not np.any(correction_value):
                continue
            face_id, bary_id = divmod(sample_id, len(barycentric))
            for local_id, weight in enumerate(barycentric[bary_id]):
                vertex_id = int(face_indices[face_id, local_id])
                updates[vertex_id] += correction_value * float(weight)
                weights[vertex_id] += float(weight)
        corrected = points + updates / np.maximum(weights[:, None], 1.0e-8)
        return torch.as_tensor(corrected, dtype=positions.dtype, device=positions.device)


class UnifiedNeuralCloth4DScene:
    """Shared physical-time integrator for body and multiple cloth manifolds.

    Each garment keeps its own immutable rest metric and Hyper-NODE material
    latent. They share one clock, moving-body query, and explicit contact-event
    graph. The resulting knots are exposed as a continuous spacetime embedding.
    """

    def __init__(
        self,
        systems: list[Garment4DSystem],
        *,
        body_contact_query: BodyContactQuery | None = None,
        rigid_lie_message_model: SE3JointMessagePassing | None = None,
        rigid_link_graph_query: Callable[
            [float, Tensor], tuple[Tensor, Tensor, Tensor, Tensor]
        ]
        | None = None,
        scene_phase_force_provider: ScenePhaseForceProvider | None = None,
        body_phase_state_provider: Callable[[float], Body4DState] | None = None,
        body_root_transport_provider: BodyRootTransportProvider | None = None,
        se3_generative_dynamics: SE3GenerativeDynamicsHead | None = None,
        config: UnifiedCloth4DSceneConfig = UnifiedCloth4DSceneConfig(),
    ) -> None:
        if not systems:
            raise ValueError("at least one garment system is required")
        names = [system.name for system in systems]
        if len(set(names)) != len(names):
            raise ValueError("garment names must be unique")
        self.systems = systems
        self.body_contact_query = body_contact_query
        if (rigid_lie_message_model is None) != (rigid_link_graph_query is None):
            raise ValueError("rigid Lie model and link graph query must be supplied together")
        self.rigid_lie_message_model = rigid_lie_message_model
        self.rigid_link_graph_query = rigid_link_graph_query
        self.scene_phase_force_provider = scene_phase_force_provider
        self.body_phase_state_provider = body_phase_state_provider
        self.body_root_transport_provider = body_root_transport_provider
        self.se3_generative_dynamics = se3_generative_dynamics
        self._scene_phase_provider_accepts_body = False
        if scene_phase_force_provider is not None:
            try:
                self._scene_phase_provider_accepts_body = (
                    len(inspect.signature(scene_phase_force_provider).parameters) >= 5
                )
            except (TypeError, ValueError):
                self._scene_phase_provider_accepts_body = True
        self._lie_substep_calls = 0
        self._lie_max_relative_twist = 0.0
        self._lie_max_world_wrench = 0.0
        self._lie_max_force_balance_error = 0.0
        self._phase_port_max_force = 0.0
        self._body_phase_max_speed = 0.0
        self._body_phase_max_momentum = 0.0
        self._body_phase_max_reaction_force = 0.0
        self._body_phase_max_kinetic_energy = 0.0
        self._body_phase_max_total_momentum = 0.0
        self._body_phase_last_kinetic_energy = 0.0
        self._body_phase_samples = 0
        self._body_phase_vertex_count = 0
        self._se3_score_max_norm = 0.0
        self._se3_score_max_proposal_twist = 0.0
        self._se3_phase_twist = torch.zeros(
            6,
            dtype=self.systems[0].initial_positions.dtype,
            device=self.systems[0].initial_positions.device,
        )
        self._se3_score_max_norm = 0.0
        self._se3_score_max_proposal_twist = 0.0
        self._se3_phase_twist = torch.zeros(6, dtype=torch.float32)
        self._body_phase_max_speed = 0.0
        self._body_phase_max_momentum = 0.0
        self._body_phase_max_reaction_force = 0.0
        self._body_phase_max_kinetic_energy = 0.0
        self._body_phase_max_total_momentum = 0.0
        self._body_phase_last_kinetic_energy = 0.0
        self._body_phase_samples = 0
        self._body_phase_vertex_count = 0
        self._metric_projection_recovery_count = {
            system.name: 0 for system in systems
        }
        self.config = config
        self.body_boundary_edges = {
            system.name: _mesh_boundary_edges(system.faces) for system in systems
        }

    def rollout(self) -> UnifiedCloth4DRollout:
        for system in self.systems:
            system.dynamics.generalized_flow_ledger.reset()
        self._lie_substep_calls = 0
        self._lie_max_relative_twist = 0.0
        self._lie_max_world_wrench = 0.0
        self._lie_max_force_balance_error = 0.0
        self._phase_port_max_force = 0.0
        self._metric_projection_recovery_count = {
            system.name: 0 for system in self.systems
        }
        prepared_positions = {
            system.name: system.dynamics.prepare_initial_positions(
                system.initial_positions,
                system.material_latent,
                body_contact_query=self.body_contact_query,
            )[0]
            for system in self.systems
        }
        states = {}
        for system in self.systems:
            initial_velocity = system.initial_velocities
            if self.body_root_transport_provider is not None:
                _transported, root_velocity, _next_velocity = (
                    self.body_root_transport_provider(
                        prepared_positions[system.name], 0.0, self.config.internal_dt
                    )
                )
                if root_velocity.shape != prepared_positions[system.name].shape:
                    raise ValueError(
                        "body root transport provider returned an invalid velocity shape"
                    )
                # A body-conditioned shell with no prescribed launch velocity
                # starts in the co-moving frame. Preserve an explicitly
                # supplied non-zero initial relative velocity for experiments.
                if float(torch.linalg.vector_norm(initial_velocity.detach())) <= 1.0e-8:
                    initial_velocity = root_velocity.to(prepared_positions[system.name])
            states[system.name] = NeuralCloth4DState(
                prepared_positions[system.name],
                initial_velocity,
                0.0,
            )
        pinned_reference = {
            system.name: prepared_positions[system.name][system.dynamics.pinned_vertices].detach()
            for system in self.systems
        }
        # The first stored frame must already be a jointly feasible wearing
        # state. Previously body and layer contact started at the first
        # physical substep, so frame 0 could show a garment inside the avatar
        # or the waistband inside the shirt hem. Use the same coupled contact
        # solve here, but preserve the supplied initial velocities so this
        # initialization does not inject an artificial launch impulse.
        initial_velocities = {
            name: state.velocities for name, state in states.items()
        }
        states = self._project_and_resolve(
            states,
            pinned_reference,
            0.0,
            previous_states=states,
        )
        scales = {
            system.name: system.dynamics._calibrate_constitutive_force_scale(
                states[system.name].positions,
                system.material_latent,
                differentiable=False,
            )[0]
            for system in self.systems
        }
        flow_checkpoints = {
            system.name: system.dynamics.generalized_flow_ledger.checkpoint()
            for system in self.systems
        }
        reaction_checkpoint = self._body_phase_max_reaction_force

        def static_contact_query(points: Tensor, _time: float) -> BodyContactSample:
            if self.body_contact_query is None:
                raise RuntimeError("static contact query requires a body surface")
            sample = self.body_contact_query(points, 0.0)
            return BodyContactSample(
                sample.closest_points,
                sample.normals,
                torch.zeros_like(points),
            )

        for _ in range(self.config.initial_contact_equilibration_steps):
            equilibrated: dict[str, NeuralCloth4DState] = {}
            for system in self.systems:
                current = NeuralCloth4DState(
                    states[system.name].positions,
                    torch.zeros_like(states[system.name].velocities),
                    -self.config.internal_dt,
                )
                candidate, _acceleration, _maximum, _hits, _diagnostics = (
                    system.dynamics._advance_candidate(
                        current,
                        system.material_latent,
                        self.config.internal_dt,
                        static_contact_query if self.body_contact_query is not None else None,
                        False,
                        scales[system.name],
                        external_force=torch.zeros_like(current.positions),
                        se3_target_velocity=torch.zeros_like(current.positions),
                        moving_reference_velocity=torch.zeros_like(current.positions),
                        moving_reference_displacement=torch.zeros_like(current.positions),
                        moving_reference_next_velocity=torch.zeros_like(current.positions),
                        inter_garment_contacts=(),
                    )
                )
                equilibrated[system.name] = NeuralCloth4DState(
                    candidate.positions,
                    torch.zeros_like(candidate.velocities),
                    0.0,
                )
            states = self._project_and_resolve(
                equilibrated,
                pinned_reference,
                0.0,
                previous_states=states,
            )
        for system in self.systems:
            system.dynamics.generalized_flow_ledger.restore(
                flow_checkpoints[system.name]
            )
        self._body_phase_max_reaction_force = reaction_checkpoint
        states = {
            name: NeuralCloth4DState(
                state.positions,
                initial_velocities[name],
                0.0,
            )
            for name, state in states.items()
        }
        # Calibrate the physical rollout on the equilibrated wearing state,
        # not on the discarded contact-projection transient.
        scales = {
            system.name: system.dynamics._calibrate_constitutive_force_scale(
                states[system.name].positions,
                system.material_latent,
                differentiable=False,
            )[0]
            for system in self.systems
        }
        frame_positions = {name: [state.positions] for name, state in states.items()}
        frame_velocities = {name: [state.velocities] for name, state in states.items()}
        detector = self._event_detector(states)
        events: list[dict[str, object]] = []
        event_restarts = 0
        minimum_body = {system.name: None for system in self.systems}
        pair_names = self._pair_names()
        minimum_pair = {pair: float("inf") for pair in pair_names}
        steps_per_frame = round(self.config.frame_dt / self.config.internal_dt)
        total_steps = (self.config.frame_count - 1) * steps_per_frame

        for step in range(total_steps):
            current_time = step * self.config.internal_dt
            proposed_time = current_time + self.config.internal_dt
            body_state = self._body_state_at(current_time)
            flow_checkpoints = {
                system.name: system.dynamics.generalized_flow_ledger.checkpoint()
                for system in self.systems
            }
            lie_checkpoint = (
                self._lie_substep_calls,
                self._lie_max_relative_twist,
                self._lie_max_world_wrench,
                self._lie_max_force_balance_error,
            )
            use_phase_solver = step % self.config.phase_hamiltonian_stride == 0
            proposed = self._advance_all(
                states,
                scales,
                self.config.internal_dt,
                body_state=body_state,
                use_phase_solver=use_phase_solver,
            )
            current_values = self._event_values(states)
            proposed_values = self._event_values(proposed)
            crossing = self._earliest_crossing(current_values, proposed_values)
            if crossing is not None and self.config.maximum_event_restarts_per_step:
                surface_id, alpha = crossing
                root_time = current_time + alpha * self.config.internal_dt
                for system in self.systems:
                    system.dynamics.generalized_flow_ledger.restore(
                        flow_checkpoints[system.name]
                    )
                (
                    self._lie_substep_calls,
                    self._lie_max_relative_twist,
                    self._lie_max_world_wrench,
                    self._lie_max_force_balance_error,
                ) = lie_checkpoint
                root_duration = root_time - current_time
                root = (
                    self._advance_all(
                        states,
                        scales,
                        root_duration,
                        body_state=self._body_state_at(root_time),
                        use_phase_solver=use_phase_solver,
                    )
                    if root_duration > 1.0e-9
                    else {
                        name: NeuralCloth4DState(
                            state.positions, state.velocities, root_time
                        )
                        for name, state in states.items()
                    }
                )
                root = self._resolve_all_contacts(root, pinned_reference, root_time)
                batch = detector.record_external_root(
                    surface_id,
                    time=root_time,
                    active=True,
                    bracket=(current_time, proposed_time),
                    sample_values=(current_values[surface_id], proposed_values[surface_id]),
                    reason="shared_scene_ccd_root",
                )
                events.extend(_event_dict(event) for event in batch.events)
                event_restarts += int(batch.restart_required)
                remaining = proposed_time - root_time
                proposed = (
                    self._advance_all(
                        root,
                        scales,
                        remaining,
                        body_state=self._body_state_at(root_time),
                        use_phase_solver=use_phase_solver,
                    )
                    if remaining > 1.0e-9
                    else root
                )

            proposed = self._project_and_resolve(
                proposed,
                pinned_reference,
                proposed_time,
                previous_states=states,
            )
            states = proposed
            observed = detector.observe(proposed_time, self._event_values(states))
            events.extend(_event_dict(event) for event in observed.events)
            values = self._event_values(states)
            for system in self.systems:
                key = f"{system.name}:body:min_gap"
                if key in values:
                    gap = values[key]
                    previous = minimum_body[system.name]
                    minimum_body[system.name] = gap if previous is None else min(previous, gap)
            for pair in pair_names:
                minimum_pair[pair] = min(minimum_pair[pair], values[f"{pair}:min_gap"])
            if (step + 1) % steps_per_frame == 0:
                for name, state in states.items():
                    frame_positions[name].append(state.positions)
                    frame_velocities[name].append(state.velocities)

        times = torch.arange(self.config.frame_count, dtype=torch.float32) * self.config.frame_dt
        fields = {
            name: ContinuousGarment4D(
                times,
                torch.stack(frame_positions[name]),
                torch.stack(frame_velocities[name]),
                next(system.faces for system in self.systems if system.name == name),
            )
            for name in frame_positions
        }
        metric = {
            system.name: system.dynamics.metric_projector.diagnostics(
                fields[system.name].positions[-1]
            ).to_dict()
            for system in self.systems
        }
        hypernode_metric = {
            system.name: system.dynamics.hypernode_strain_limit_diagnostics(
                fields[system.name].positions[-1]
            )
            for system in self.systems
        }
        report = UnifiedCloth4DReport(
            representation="continuous X_g(u,v,t)->R3 over a shared E(3) scene clock",
            garments=tuple(system.name for system in self.systems),
            frames=self.config.frame_count,
            internal_steps=total_steps,
            physical_duration_s=(self.config.frame_count - 1) * self.config.frame_dt,
            initial_contact_equilibration_steps=(
                self.config.initial_contact_equilibration_steps
            ),
            event_restarts=event_restarts,
            events=tuple(events),
            minimum_body_gap_m=minimum_body,
            minimum_inter_garment_gap_m=minimum_pair,
            metric_projection=metric,
            hypernode_strain_limit=hypernode_metric,
            metric_projection_recovery_count=dict(
                self._metric_projection_recovery_count
            ),
            constitutive_force_scale=scales,
            lie_group_message_passing={
                "enabled": self.rigid_lie_message_model is not None,
                "module": "SE3JointMessagePassing (RigidLieMessagePassing)",
                "substep_calls": self._lie_substep_calls,
                "expected_substep_calls": total_steps + event_restarts,
                "maximum_relative_twist_norm": self._lie_max_relative_twist,
                "maximum_world_wrench_norm": self._lie_max_world_wrench,
                "maximum_action_reaction_balance_error": (
                    self._lie_max_force_balance_error
                ),
                "role": (
                    "per-substep SE(3) scene context and conservative wrench prior; "
                    "analytic joints/contact remain hard constraints"
                ),
            },
            generalized_flow={
                system.name: system.dynamics.generalized_flow_ledger.audit().to_dict()
                for system in self.systems
            },
            reference_map_fiber={
                system.name: system.dynamics.reference_map_bundle.report(
                    fields[system.name].positions[-1]
                ).to_dict()
                for system in self.systems
            },
            bundle_measure_port_hamiltonian={
                system.name: system.dynamics.bundle_measure_report(
                    fields[system.name].positions[-1],
                    fields[system.name].velocities[-1],
                    system.material_latent,
                    physical_time=(self.config.frame_count - 1)
                    * self.config.frame_dt,
                    constitutive_scale=scales[system.name],
                    garment_id=system.name,
                )
                for system in self.systems
            },
            phase_hamiltonian={
                system.name: dict(
                    system.dynamics._latest_phase_hamiltonian_diagnostics
                )
                for system in self.systems
            },
            scene_hamiltonian={
                "structure": (
                    "H_scene = H_body(q_b,p_b) + sum_g H_g(q_g-q_b,p_g) "
                    "+ prescribed body/cloth contact work"
                ),
                "body": {
                    "state": "q_b=body mesh positions, p_b=generalized momentum",
                    "maximum_kinetic_energy_j": self._body_phase_max_kinetic_energy,
                    "last_kinetic_energy_j": self._body_phase_last_kinetic_energy,
                    "maximum_total_momentum_norm_kg_m_s": self._body_phase_max_total_momentum,
                    "samples": self._body_phase_samples,
                },
                "garment_relative_kinetic_energy_j": {
                    name: float(
                        diagnostics.get("kinetic_energy_j", 0.0)
                    )
                    for name, diagnostics in (
                        (
                            system.name,
                            system.dynamics._latest_phase_hamiltonian_diagnostics,
                        )
                        for system in self.systems
                    )
                },
                "garment_relative_total_hamiltonian_j": {
                    name: float(
                        diagnostics.get("total_hamiltonian_j", 0.0)
                    )
                    for name, diagnostics in (
                        (
                            system.name,
                            system.dynamics._latest_phase_hamiltonian_diagnostics,
                        )
                        for system in self.systems
                    )
                },
                "last_total_snapshot_hamiltonian_j": (
                    self._body_phase_last_kinetic_energy
                    + sum(
                        float(
                            system.dynamics._latest_phase_hamiltonian_diagnostics.get(
                                "total_hamiltonian_j", 0.0
                            )
                        )
                        for system in self.systems
                    )
                ),
                "transport_accounting": (
                    "body kinetic energy is included in this scene audit; "
                    "garment local H remains body-relative so rigid co-motion "
                    "does not masquerade as cloth strain"
                ),
                "dynamic_status": (
                    "body gait is prescribed; H_body is audited but not varied. "
                    "A free-body solve is required before contact work can change "
                    "the body trajectory"
                ),
            },
            noether_audit={
                system.name: {
                    "structure_group": (
                        "diagonal E(3) on body-relative garment configurations; "
                        "SE(3) transport acts on the shared reference frame"
                    ),
                    "internal_force_balance_max_n": (
                        system.dynamics._maximum_internal_force_balance_n
                    ),
                    "internal_torque_balance_max_n_m": (
                        system.dynamics._maximum_internal_torque_balance_n_m
                    ),
                    "momentum_measure": (
                        "sum_v m_v(v_v - V_body_root) for relative cloth momentum; "
                        "body root momentum is audited separately"
                    ),
                    "energy_measure": (
                        "bundle-valued cloth Hamiltonian with gravity potential; "
                        "dissipation and prescribed-body work remain explicit ports"
                    ),
                    "strict_conservation_scope": (
                        "internal shell/edge interactions; not the open prescribed gait"
                    ),
                }
                for system in self.systems
            },
            scene_graph_phase_port={
                "enabled": self.scene_phase_force_provider is not None,
                "application": "local contact/friction phase port",
                "maximum_force_norm_n": self._phase_port_max_force,
                "source": (
                    "HierarchyAwareSpatiotemporalAttention/Lie phase residual"
                    if self.scene_phase_force_provider is not None
                    else "disabled"
                ),
            },
            moving_reference_hamiltonian={
                "enabled": self.body_phase_state_provider is not None,
                "application": (
                    "incremental action with pointwise prescribed body-root SE(3) "
                    "transport; COM fallback only when no root provider is supplied"
                ),
                "transport_provider": (
                    "pointwise root SE(3) map with current/next root velocity"
                    if self.body_root_transport_provider is not None
                    else "body-mesh COM translation fallback"
                ),
                "canonical_momentum": "p_g = M_g (v_g - V_body_root)",
                "discrete_inertial_term": (
                    "0.5 * ||q_next - q - Delta_q_body - dt*v_relative||_M^2 / dt^2"
                ),
                "body_reaction": (
                    "contact and projection impulses are recorded as the equal-and-opposite "
                    "reaction on the prescribed body root"
                ),
            },
            se3_generative_dynamics={
                "enabled": self.se3_generative_dynamics is not None,
                "module": "SE3RiemannianDDPM + SE3BodyScoreHead",
                "proposal_space": "right-trivialized se(3)",
                "application": (
                    "score proposal -> se(3) velocity field -> conservative "
                    "Hamiltonian potential"
                ),
                "maximum_score_norm": self._se3_score_max_norm,
                "maximum_proposal_twist_norm": self._se3_score_max_proposal_twist,
                "fresh_head_is_identity": (
                    self.se3_generative_dynamics is not None
                    and self._se3_score_max_proposal_twist <= 1.0e-8
                ),
            },
            body_phase_state={
                "enabled": self.body_phase_state_provider is not None,
                "representation": "body mesh q_b=positions, p_b=generalized_momentum",
                "role": (
                    "kinematic body root state coupled to cloth contact and phase momentum; "
                    "the prescribed gait remains the external behavioral drive"
                ),
                "mesh_vertex_count": self._body_phase_vertex_count,
                "samples": self._body_phase_samples,
                "maximum_speed_m_s": self._body_phase_max_speed,
                "maximum_momentum_norm": self._body_phase_max_momentum,
                "maximum_kinetic_energy_j": self._body_phase_max_kinetic_energy,
                "maximum_total_momentum_norm_kg_m_s": self._body_phase_max_total_momentum,
                "maximum_contact_reaction_force_n": self._body_phase_max_reaction_force,
                "contact_reaction_recorded": self._body_phase_max_reaction_force > 0.0,
            },
            continuous_time_query="piecewise cubic Hermite from physical positions and velocities",
            limitations=(
                "inter-garment narrow phase is sampled vertex contact, not full VF/EE IPC",
                "garment topology is fixed; contact graph topology changes are explicit events",
                "material constitutive latents remain scene-calibrated rather than lab-calibrated",
                "metric projection is opt-in; Hyper-NODE owns the normal material envelope",
            ),
            metric_projection_enabled=all(
                system.dynamics.config.enable_metric_projection
                for system in self.systems
            ),
        )
        return UnifiedCloth4DRollout(fields, report)

    def _advance_all(
        self,
        states: Mapping[str, NeuralCloth4DState],
        scales: Mapping[str, float],
        dt: float,
        *,
        body_state: Body4DState | None = None,
        use_phase_solver: bool = True,
    ) -> dict[str, NeuralCloth4DState]:
        self._run_rigid_lie_message_step(next(iter(states.values())).time)
        body_state = body_state or self._body_state_at(
            float(next(iter(states.values())).time)
        )
        output: dict[str, NeuralCloth4DState] = {}
        for system in self.systems:
            current = states[system.name]
            # Garment-layer contact is a nonsmooth complementarity event. Do
            # not preload the Hamiltonian with a nearest-vertex penalty: at a
            # waist interface its changing nearest id can create a fictitious
            # elastic force. The shared projector/event graph handles the
            # local separation only when layers nearly coincide.
            inter_garment_contacts = tuple(
                (
                    other.positions,
                    self.config.inter_garment_clearance_m,
                    self.config.inter_garment_contact_stiffness_n_m,
                    torch.cdist(
                        current.positions.detach(), other.positions.detach()
                    ).min(dim=1).indices,
                    _interface_contact_mask(current.positions, other.positions),
                )
                for other_system in self.systems
                if other_system.name != system.name
                and not _is_body_layer_interface(system.name, other_system.name)
                for other in (states[other_system.name],)
            )
            se3_target_velocity = self._se3_target_velocity(
                current.positions, body_state
            )
            moving_reference_velocity = None
            moving_reference_displacement = None
            moving_reference_next_velocity = None
            if body_state is not None and self.body_phase_state_provider is not None:
                next_body_state = self._body_state_at(float(current.time) + dt)
                if self.body_root_transport_provider is not None:
                    transported, moving_reference_velocity, moving_reference_next_velocity = (
                        self.body_root_transport_provider(
                            current.positions,
                            float(current.time),
                            float(current.time) + dt,
                        )
                    )
                    if transported.shape != current.positions.shape:
                        raise ValueError(
                            "body root transport provider returned an invalid position shape"
                        )
                    moving_reference_displacement = transported - current.positions
                else:
                    # Backward-compatible fallback for callers that only
                    # provide a body state. This is deliberately reported as
                    # COM transport because it cannot carry the full root SE(3).
                    body_com = body_state.positions.to(current.positions).mean(dim=0)
                    next_body_com = next_body_state.positions.to(current.positions).mean(dim=0)
                    body_com_velocity = body_state.velocities.to(current.positions).mean(dim=0)
                    next_body_com_velocity = (
                        next_body_state.velocities.to(current.positions).mean(dim=0)
                    )
                    moving_reference_velocity = body_com_velocity.expand_as(current.positions)
                    moving_reference_displacement = (
                        next_body_com - body_com
                    ).expand_as(current.positions)
                    moving_reference_next_velocity = (
                        next_body_com_velocity.expand_as(current.positions)
                    )
            phase_force = None
            if self.scene_phase_force_provider is not None:
                if self._scene_phase_provider_accepts_body:
                    phase_force = self.scene_phase_force_provider(
                        system.name,
                        current,
                        states,
                        body_state,
                        float(current.time),
                    )
                else:
                    # Keep the old four-argument callback contract usable for
                    # small external experiments while the built-in pipeline
                    # receives the explicit body phase state.
                    phase_force = self.scene_phase_force_provider(
                        system.name,
                        current,
                        states,
                        float(current.time),
                    )  # type: ignore[call-arg]
                if phase_force.shape != current.positions.shape:
                    raise ValueError(
                        "scene phase force provider must return (vertex_count, 3)"
                    )
                self._phase_port_max_force = max(
                    self._phase_port_max_force,
                    float(torch.linalg.vector_norm(phase_force.detach(), dim=-1).max()),
                )
            advance = (
                system.dynamics._advance_candidate
                if use_phase_solver
                else system.dynamics._advance_fast_preview_candidate
            )
            advance_kwargs = dict(
                external_force=phase_force,
                se3_target_velocity=se3_target_velocity,
                moving_reference_velocity=moving_reference_velocity,
                moving_reference_displacement=moving_reference_displacement,
                moving_reference_next_velocity=moving_reference_next_velocity,
            )
            if use_phase_solver:
                advance_kwargs["inter_garment_contacts"] = inter_garment_contacts
            (
                candidate,
                _acceleration,
                _maximum,
                _hits,
                _phase_hamiltonian,
            ) = advance(
                current,
                system.material_latent,
                dt,
                self.body_contact_query,
                False,
                scales[system.name],
                **advance_kwargs,
            )
            output[system.name] = candidate
        return output

    def _se3_target_velocity(
        self, positions: Tensor, body_state: Body4DState | None
    ) -> Tensor:
        if self.se3_generative_dynamics is None:
            return torch.zeros_like(positions)
        twist = self._se3_phase_twist.to(positions)
        center = (
            positions.mean(dim=0)
            if body_state is None
            else body_state.positions.to(positions).mean(dim=0)
        )
        angular = twist[:3].expand_as(positions)
        linear = twist[3:].expand_as(positions)
        return torch.linalg.cross(angular, positions - center) + linear

    def _body_state_at(self, time: float) -> Body4DState | None:
        if self.body_phase_state_provider is None:
            return None
        state = self.body_phase_state_provider(float(time))
        if not isinstance(state, Body4DState):
            raise TypeError("body phase state provider must return Body4DState")
        if abs(float(state.time) - float(time)) > 1.0e-5:
            raise ValueError("body phase state time does not match the scene clock")
        self._body_phase_samples += 1
        self._body_phase_vertex_count = int(len(state.positions))
        kinetic_energy = 0.5 * (
            state.generalized_momentum.detach() * state.velocities.detach()
        ).sum()
        total_momentum = state.generalized_momentum.detach().sum(dim=0)
        self._body_phase_last_kinetic_energy = float(kinetic_energy)
        self._body_phase_max_kinetic_energy = max(
            self._body_phase_max_kinetic_energy,
            self._body_phase_last_kinetic_energy,
        )
        self._body_phase_max_total_momentum = max(
            self._body_phase_max_total_momentum,
            float(torch.linalg.vector_norm(total_momentum)),
        )
        self._body_phase_max_speed = max(
            self._body_phase_max_speed,
            float(torch.linalg.vector_norm(state.velocities.detach(), dim=-1).max()),
        )
        self._body_phase_max_momentum = max(
            self._body_phase_max_momentum,
            float(
                torch.linalg.vector_norm(
                    state.generalized_momentum.detach(), dim=-1
                ).max()
            ),
        )
        return state

    def _run_rigid_lie_message_step(self, time: float) -> None:
        if (
            self.rigid_link_graph_query is None
            or (
                self.rigid_lie_message_model is None
                and self.se3_generative_dynamics is None
            )
        ):
            return
        reference = self.systems[0].initial_positions
        transforms, features, edges, joint_features = self.rigid_link_graph_query(
            time, reference
        )
        message = None
        if self.rigid_lie_message_model is not None:
            with torch.no_grad():
                message = self.rigid_lie_message_model(
                    transforms, features, edges, joint_features
                )
        if self.se3_generative_dynamics is not None:
            parent, child = edges
            relative = (
                torch.linalg.inv(transforms[parent]) @ transforms[child]
            )
            context = torch.cat(
                (
                    features[parent],
                    features[child],
                    joint_features,
                ),
                dim=-1,
            )
            diffusion_time = transforms.new_full(
                (len(relative), 1), min(max(float(time), 0.0), 1.0)
            )
            proposal = self.se3_generative_dynamics(
                relative,
                diffusion_time,
                context,
                step=0,
            )
            self._se3_score_max_norm = max(
                self._se3_score_max_norm,
                float(torch.linalg.vector_norm(proposal.score.detach(), dim=-1).max()),
            )
            self._se3_score_max_proposal_twist = max(
                self._se3_score_max_proposal_twist,
                float(
                    torch.linalg.vector_norm(
                        proposal.proposal_twist.detach(), dim=-1
                    ).max()
                ),
            )
            self._se3_phase_twist = proposal.proposal_twist.mean(dim=0).detach()
        self._lie_substep_calls += 1
        if message is not None:
            self._lie_max_relative_twist = max(
                self._lie_max_relative_twist,
                float(torch.linalg.vector_norm(message.relative_twist, dim=-1).max()),
            )
            self._lie_max_world_wrench = max(
                self._lie_max_world_wrench,
                float(torch.linalg.vector_norm(message.node_world_wrench, dim=-1).max()),
            )
            balance = torch.linalg.vector_norm(message.node_world_wrench.sum(dim=0))
            self._lie_max_force_balance_error = max(
                self._lie_max_force_balance_error, float(balance)
            )

    def _project_and_resolve(
        self,
        states: Mapping[str, NeuralCloth4DState],
        pinned_reference: Mapping[str, Tensor],
        time: float,
        *,
        previous_states: Mapping[str, NeuralCloth4DState],
    ) -> dict[str, NeuralCloth4DState]:
        unconstrained = dict(states)
        result = dict(states)
        converged = False
        for _ in range(self.config.contact_projection_iterations):
            result = self._resolve_all_contacts(result, pinned_reference, time)
            recovered: dict[str, NeuralCloth4DState] = {}
            for system in self.systems:
                state = result[system.name]
                targets = self._pinned_targets(
                    system, pinned_reference[system.name], time
                )
                material = system.dynamics.hypernode_strain_limit_diagnostics(
                    state.positions
                )
                # The Hyper-NODE barrier handles ordinary material strain. A
                # shared metric projection is reserved for a genuinely large
                # excursion; applying it to every small violation erases cloth
                # momentum and makes the garment behave like a rigid shell.
                severe_violation = (
                    float(material["maximum_limit_violation"])
                    > system.dynamics.config.metric_projection_severe_violation
                )
                if (
                    system.dynamics.config.enable_metric_projection
                    and material["emergency_components"]
                    and severe_violation
                ):
                    positions = system.dynamics.metric_projector(
                        state.positions,
                        pinned_positions=targets,
                    )
                    self._metric_projection_recovery_count[system.name] += 1
                else:
                    positions = state.positions
                recovered[system.name] = NeuralCloth4DState(
                    positions, state.velocities, time
                )
            # Metric recovery can move a panel across the body or garment
            # contact surface. Re-apply the coupled contact map before testing
            # convergence, including on the final outer iteration.
            result = self._resolve_all_contacts(recovered, pinned_reference, time)
            if self._joint_constraints_satisfied(result):
                converged = True
                break
        if not converged:
            diagnostics = {
                system.name: system.dynamics.metric_projector.diagnostics(
                    result[system.name].positions
                ).to_dict()
                for system in self.systems
            }
            gaps = self._event_values(result)
            raise RuntimeError(
                f"shared body/inter-garment contact resolution did not converge at t={time:.4f} s: "
                f"metric={diagnostics}, gaps={gaps}"
            )
        for system in self.systems:
            state = result[system.name]
            # Contact and metric projectors may move positions to satisfy a
            # hard constraint. Their displacement is not free-flight motion:
            # rebuilding velocity from delta-position here would inject a
            # fictitious normal impulse, commonly seen as an upward garment
            # bounce while the body walks. The contact solver already returns
            # the admissible velocity relative to the moving body.
            velocity = state.velocities
            velocity_state = NeuralCloth4DState(state.positions, velocity, time)
            system.dynamics._record_momentum_projection(
                "shared_scene_contact_resolution",
                unconstrained[system.name].velocities,
                velocity_state.velocities,
                system.dynamics._vertex_mass(velocity_state.positions),
                time,
                GeneralizedFlowRole.CONSTRAINT,
            )
            if self.body_contact_query is not None:
                targets = self._pinned_targets(
                    system, pinned_reference[system.name], time
                )
                # Position projection reconstructs velocity from the accepted
                # displacement. Reapply velocity-level friction afterward so
                # that reconstruction cannot erase static/Coulomb contact.
                contact_sample = self.body_contact_query(
                    velocity_state.positions.detach(), time
                )
                before_contact_velocity = velocity_state.velocities
                velocity_state, _gap = _resolve_body_contact_with_clearance(
                    velocity_state,
                    self.body_contact_query,
                    self.body_boundary_edges[system.name],
                    system.dynamics.pinned_vertices,
                    targets,
                    0.0,
                    system.dynamics.config.friction_coefficient,
                    system.dynamics.config.contact_clearance_m,
                    system.dynamics.contact_model,
                    system.dynamics.edge_index,
                    system.faces,
                    # The accepted positions already passed the shared
                    # variational/CCD contact solve above. This final pass is
                    # strictly velocity-level friction/impulse handling;
                    # projecting the posterior surface a second time creates
                    # a visible outward pop when hip contact changes state.
                    apply_position_correction=False,
                )
                contact_body_velocity = (
                    torch.zeros_like(before_contact_velocity)
                    if contact_sample.body_velocities is None
                    else contact_sample.body_velocities.to(before_contact_velocity)
                )
                system.dynamics._record_contact_impulse(
                    "shared_body_contact",
                    before_contact_velocity,
                    velocity_state.velocities,
                    contact_body_velocity,
                    contact_sample.normals,
                    system.dynamics._vertex_mass(velocity_state.positions),
                    time,
                )
                # The avatar is kinematically driven, but its reaction must
                # still be visible in the shared phase graph. This is the
                # equal-and-opposite impulse that would update p_b in a fully
                # dynamic body solve.
                body_mass = system.dynamics._vertex_mass(velocity_state.positions)
                reaction_force = -(
                    body_mass[:, None]
                    * (velocity_state.velocities - before_contact_velocity)
                ).sum(dim=0) / max(self.config.internal_dt, 1.0e-12)
                self._body_phase_max_reaction_force = max(
                    self._body_phase_max_reaction_force,
                    float(torch.linalg.vector_norm(reaction_force.detach())),
                )
            result[system.name] = velocity_state
        return result

    def _joint_constraints_satisfied(
        self, states: Mapping[str, NeuralCloth4DState]
    ) -> bool:
        if any(
            system.dynamics.config.enable_metric_projection
            and float(
                system.dynamics.hypernode_strain_limit_diagnostics(
                    states[system.name].positions
                )["maximum_limit_violation"]
            )
            > system.dynamics.config.metric_projection_severe_violation
            for system in self.systems
        ):
            return False
        values = self._event_values(states)
        contact_tolerance = self.config.contact_constraint_tolerance_m
        for surface_id, gap in values.items():
            if ":body:" in surface_id:
                garment = surface_id.split(":", 1)[0]
                threshold = next(
                    system.dynamics.config.contact_clearance_m
                    for system in self.systems
                    if system.name == garment
                )
            else:
                # The current layer narrow phase is vertex-sampled; keep its
                # gap as a diagnostic/event signal, but do not reject a state
                # on it until triangle-level CCD/IPC is available.
                threshold = 0.0
            # Keep a non-penetration floor even when the force-contact shell is
            # intentionally thicker than the accepted geometric shell.
            minimum_gap = max(threshold - contact_tolerance - 2.0e-4, 0.0)
            if gap < minimum_gap:
                return False
        return True

    def _resolve_all_contacts(
        self,
        states: Mapping[str, NeuralCloth4DState],
        pinned_reference: Mapping[str, Tensor],
        time: float,
    ) -> dict[str, NeuralCloth4DState]:
        result = dict(states)
        # Inter-garment vertex-nearest projection is intentionally disabled;
        # body SDF remains the authoritative geometric contact constraint.
        # Run one complete body contact map per outer fixed-point iteration.
        # A previous implementation first applied a reduced-clearance pass
        # and then immediately applied the full pass below, duplicating both
        # position correction and Coulomb impulse in the same substep.
        if self.body_contact_query is not None:
            for system in self.systems:
                targets = self._pinned_targets(system, pinned_reference[system.name], time)
                # The variational contact potential begins at the full cloth
                # thickness clearance. The non-smooth projection owns only
                # the thinner geometric non-penetration shell; projecting to
                # the full force range made garments float and over-stretched
                # locally refined edges.
                geometric_clearance = max(
                    system.dynamics.config.contact_clearance_m
                    - self.config.contact_constraint_tolerance_m,
                    2.0e-4,
                )
                result[system.name], _gap = _resolve_body_contact_with_clearance(
                    result[system.name],
                    self.body_contact_query,
                    self.body_boundary_edges[system.name],
                    system.dynamics.pinned_vertices,
                    targets,
                    geometric_clearance,
                    system.dynamics.config.friction_coefficient,
                    system.dynamics.config.contact_clearance_m,
                    system.dynamics.contact_model,
                    system.dynamics.edge_index,
                    system.faces,
                )
        for system in self.systems:
            if len(system.dynamics.pinned_vertices):
                targets = self._pinned_targets(system, pinned_reference[system.name], time)
                state = result[system.name]
                positions, velocities = state.positions.clone(), state.velocities.clone()
                positions[system.dynamics.pinned_vertices] = targets
                velocities[system.dynamics.pinned_vertices] = 0.0
                result[system.name] = NeuralCloth4DState(positions, velocities, time)
        return result

    def _pinned_targets(self, system: Garment4DSystem, reference: Tensor, time: float) -> Tensor:
        if system.attachment_trajectory is None:
            return reference
        targets = system.attachment_trajectory(reference, time)
        if targets.shape != reference.shape:
            raise ValueError("attachment trajectory returned an invalid shape")
        return targets

    def _event_detector(self, states: Mapping[str, NeuralCloth4DState]) -> ZeroCrossingEventDetector:
        detector = ZeroCrossingEventDetector(emit_initial_events=False)
        for system in self.systems:
            if self.body_contact_query is not None:
                detector.register(
                    EventSurface(
                        f"{system.name}:body:min_gap",
                        EventSurfaceKind.BODY_CONTACT,
                        ActiveSide.BELOW,
                        threshold=system.dynamics.config.contact_clearance_m,
                    )
                )
        for pair in self._pair_names():
            detector.register(
                EventSurface(
                    f"{pair}:min_gap",
                    EventSurfaceKind.GARMENT_CONTACT,
                    ActiveSide.BELOW,
                    threshold=self.config.inter_garment_clearance_m,
                )
            )
        detector.observe(0.0, self._event_values(states))
        return detector

    def _event_values(self, states: Mapping[str, NeuralCloth4DState]) -> dict[str, float]:
        values: dict[str, float] = {}
        if self.body_contact_query is not None:
            for system in self.systems:
                state = states[system.name]
                samples = torch.cat(
                    (
                        state.positions.detach(),
                        _boundary_sample_positions(
                            state.positions.detach(),
                            self.body_boundary_edges[system.name],
                        ),
                    ),
                    dim=0,
                )
                sample = self.body_contact_query(samples, state.time)
                normals = torch.nn.functional.normalize(sample.normals.to(samples), dim=-1)
                gap = ((samples - sample.closest_points.to(samples)) * normals).sum(-1)
                values[f"{system.name}:body:min_gap"] = float(gap.detach().min())
        for first_index, first in enumerate(self.systems):
            for second in self.systems[first_index + 1 :]:
                if _is_body_layer_interface(first.name, second.name):
                    continue
                values[f"{first.name}|{second.name}:min_gap"] = _minimum_sampled_gap(
                    states[first.name].positions, states[second.name].positions
                )
        return values

    def _pair_names(self) -> tuple[str, ...]:
        return tuple(
            f"{first.name}|{second.name}"
            for index, first in enumerate(self.systems)
            for second in self.systems[index + 1 :]
            if not _is_body_layer_interface(first.name, second.name)
        )

    def _earliest_crossing(
        self, current: Mapping[str, float], proposed: Mapping[str, float]
    ) -> tuple[str, float] | None:
        candidates: list[tuple[float, str]] = []
        for surface_id, start in current.items():
            # Vertex-sampled garment gaps are diagnostics until the planned
            # triangle-level CCD/IPC narrow phase is available. They must not
            # restart the Hamiltonian substep from a false nearest-vertex
            # crossing; body SDF remains the authoritative dynamic event.
            if "|" in surface_id:
                first_name, second_name = surface_id.split("|", 1)
                if _is_body_layer_interface(first_name, second_name):
                    continue
            if ":body:" in surface_id:
                garment = surface_id.split(":", 1)[0]
                threshold = next(
                    system.dynamics.config.contact_clearance_m
                    for system in self.systems
                    if system.name == garment
                )
            else:
                threshold = self.config.inter_garment_clearance_m
            end = proposed[surface_id]
            if start > threshold and end <= threshold:
                alpha = (start - threshold) / max(start - end, 1.0e-12)
                candidates.append((float(np.clip(alpha, 0.0, 1.0)), surface_id))
        if not candidates:
            return None
        alpha, surface_id = min(candidates)
        return surface_id, alpha


def _is_body_layer_interface(first_name: str, second_name: str) -> bool:
    """Return true for garment layers kept in the render-only guard path."""

    return {first_name, second_name} == {"tshirt", "pleated_skirt"}


def _minimum_sampled_gap(first: Tensor, second: Tensor) -> float:
    return float(torch.cdist(first.detach(), second.detach()).min())


def _interface_contact_mask(current: Tensor, other: Tensor) -> Tensor:
    """Select only the two garment boundary bands that can physically meet.

    The upper garment is active near its lower hem; the lower garment is
    active near its waistband. Without this mask a pleat valley can be the
    nearest vertex to a shirt vertex, incorrectly coupling the whole skirt
    interior to the shirt layer.
    """

    current_y = current[:, 1]
    other_y = other[:, 1]
    current_mean = current_y.mean()
    other_mean = other_y.mean()
    band_m = current.new_tensor(0.05)
    if current_mean >= other_mean:
        return current_y <= other_y.max() + band_m
    return current_y >= other_y.min() - band_m


def _mesh_boundary_edges(faces: Tensor) -> Tensor:
    counts: dict[tuple[int, int], int] = {}
    for face in faces.detach().cpu().tolist():
        for first, second in (
            (face[0], face[1]),
            (face[1], face[2]),
            (face[2], face[0]),
        ):
            edge = tuple(sorted((int(first), int(second))))
            counts[edge] = counts.get(edge, 0) + 1
    boundary = [edge for edge, count in counts.items() if count == 1]
    return torch.as_tensor(boundary, dtype=torch.long).reshape(-1, 2)


def _boundary_sample_positions(positions: Tensor, boundary_edges: Tensor) -> Tensor:
    if not len(boundary_edges):
        return positions.new_empty((0, 3))
    edges = boundary_edges.to(positions.device)
    first, second = positions[edges[:, 0]], positions[edges[:, 1]]
    alpha = positions.new_tensor([0.25, 0.5, 0.75])
    samples = (
        (1.0 - alpha[None, :, None]) * first[:, None, :]
        + alpha[None, :, None] * second[:, None, :]
    )
    return samples.reshape(-1, 3)


def _project_boundary_contact_samples(
    positions: Tensor,
    query: BodyContactQuery,
    boundary_edges: Tensor,
    clearance: float,
    physical_time: float,
    edge_index: Tensor | None = None,
) -> Tensor:
    """Distribute one boundary sample correction over the intrinsic graph."""

    if not len(boundary_edges):
        return positions
    edges = boundary_edges.to(positions.device)
    alpha = positions.new_tensor([0.25, 0.5, 0.75])
    first, second = edges[:, 0], edges[:, 1]
    samples = _boundary_sample_positions(positions, edges)
    sample = query(samples.detach(), physical_time)
    normals = torch.nn.functional.normalize(sample.normals.to(samples), dim=-1)
    gap = ((samples - sample.closest_points.to(samples)) * normals).sum(-1)
    correction = torch.relu(clearance - gap)[:, None] * normals
    correction = correction.reshape(len(edges), len(alpha), 3)
    updates = torch.zeros_like(positions)
    weights = positions.new_zeros((len(positions), 1))
    for index, weight_first in enumerate((0.75, 0.5, 0.25)):
        weight_second = 1.0 - weight_first
        update = correction[:, index]
        updates.index_add_(0, first, weight_first * update)
        updates.index_add_(0, second, weight_second * update)
        weights.index_add_(
            0, first, positions.new_full((len(first), 1), weight_first)
        )
        weights.index_add_(
            0, second, positions.new_full((len(second), 1), weight_second)
        )
    correction = updates / weights.clamp_min(1.0)
    if edge_index is not None:
        correction = _diffuse_intrinsic_contact_correction(
            correction, edge_index, strength=0.65
        )
    return positions + correction


def _resolve_body_contact_with_clearance(
    state: NeuralCloth4DState,
    query: BodyContactQuery,
    boundary_edges: Tensor,
    pinned_vertices: Tensor,
    pinned_positions: Tensor,
    clearance: float,
    friction: float,
    friction_activation_clearance: float,
    contact_model: GaugeEquivariantContactCNN | None = None,
    edge_index: Tensor | None = None,
    faces: Tensor | None = None,
    apply_position_correction: bool = True,
) -> tuple[NeuralCloth4DState, float]:
    sample = query(state.positions.detach(), state.time)
    normals = torch.nn.functional.normalize(sample.normals.to(state.positions), dim=-1)
    gap = ((state.positions - sample.closest_points.to(state.positions)) * normals).sum(-1)
    penetration = torch.relu(clearance - gap)
    if apply_position_correction:
        contact_correction = penetration[:, None] * normals
        if edge_index is not None:
            contact_correction = _diffuse_intrinsic_contact_correction(
                contact_correction, edge_index, strength=0.65
            )
        positions = state.positions + contact_correction
    else:
        # The accepted position already passed the variational/CCD contact
        # solve. The post-solve pass is velocity-only; applying a second
        # surface projection here would spend the same contact work twice.
        positions = state.positions
    body_velocity = (
        torch.zeros_like(state.velocities)
        if sample.body_velocities is None
        else sample.body_velocities.to(state.velocities)
    )
    relative = state.velocities - body_velocity
    normal_speed = (relative * normals).sum(dim=-1, keepdim=True)
    tangent = relative - normal_speed * normals
    tangent_speed = torch.linalg.vector_norm(tangent, dim=-1, keepdim=True)
    near_contact = gap[:, None] < friction_activation_clearance
    static_contact = near_contact & (tangent_speed < 0.04)
    effective_friction = torch.full_like(gap, friction)
    if contact_model is not None:
        if edge_index is None:
            raise ValueError("gauge contact requires the intrinsic cloth edge graph")
        response = contact_model(
            relative,
            normals,
            penetration,
            near_contact[:, 0],
            edge_index,
            clearance_m=friction_activation_clearance,
        )
        effective_friction = (friction * response.friction_scale).clamp(0.0, 1.0)
    sliding_tangent = (1.0 - effective_friction[:, None]) * tangent
    friction_tangent = torch.where(
        static_contact, torch.zeros_like(tangent), sliding_tangent
    )
    resolved_relative = friction_tangent + torch.clamp_min(normal_speed, 0.0) * normals
    velocities = torch.where(
        near_contact,
        body_velocity + resolved_relative,
        state.velocities,
    )

    surface_projector = getattr(query, "project_surface", None)
    if apply_position_correction and faces is not None and callable(surface_projector):
        # Continuous triangle contact is the authoritative narrow phase for
        # shoulder/axilla/waist crossings.  It runs after the broad-phase
        # vertex correction, so normal contact is not diluted by the learned
        # intrinsic diffusion above.
        projected_surface = surface_projector(
            positions,
            faces,
            state.time,
            clearance,
        )
        surface_correction = projected_surface - positions
        if edge_index is not None:
            surface_correction = _diffuse_intrinsic_contact_correction(
                surface_correction, edge_index, strength=0.65
            )
        positions = positions + surface_correction
        positions = _project_boundary_contact_samples(
            positions,
            query,
            boundary_edges,
            clearance,
            state.time,
            edge_index,
        )
        final_sample = query(positions.detach(), state.time)
        final_normals = torch.nn.functional.normalize(
            final_sample.normals.to(positions), dim=-1
        )
        final_gap = (
            (positions - final_sample.closest_points.to(positions)) * final_normals
        ).sum(-1)
        final_correction = torch.relu(clearance - final_gap)[:, None] * final_normals
        if edge_index is not None:
            final_correction = _diffuse_intrinsic_contact_correction(
                final_correction, edge_index, strength=0.65
            )
        positions = positions + final_correction
    if len(pinned_vertices):
        positions = positions.clone()
        velocities = velocities.clone()
        positions[pinned_vertices] = pinned_positions
        velocities[pinned_vertices] = 0.0
    resolved_points = torch.cat(
        (positions.detach(), _boundary_sample_positions(positions.detach(), boundary_edges)),
        dim=0,
    )
    resolved = query(resolved_points, state.time)
    resolved_normals = torch.nn.functional.normalize(resolved.normals.to(resolved_points), dim=-1)
    resolved_gap = (
        (resolved_points - resolved.closest_points.to(resolved_points)) * resolved_normals
    ).sum(-1)
    return (
        NeuralCloth4DState(positions, velocities, state.time),
        float(resolved_gap.detach().min()),
    )


def _diffuse_intrinsic_contact_correction(
    correction: Tensor, edge_index: Tensor, *, strength: float
) -> Tensor:
    """Smooth vertex contact impulses over the intrinsic cloth graph."""

    if correction.ndim != 2 or correction.shape[1] != 3:
        raise ValueError("contact correction must have shape (V, 3)")
    if not 0.0 <= strength <= 1.0:
        raise ValueError("contact correction diffusion strength must lie in [0,1]")
    edges = edge_index.to(correction.device)
    sender, receiver = edges[0], edges[1]
    aggregate = torch.zeros_like(correction)
    aggregate.index_add_(0, receiver, correction[sender])
    degree = correction.new_zeros((len(correction), 1))
    degree.index_add_(
        0,
        receiver,
        torch.ones((len(receiver), 1), device=correction.device, dtype=correction.dtype),
    )
    neighbor_mean = aggregate / degree.clamp_min(1.0)
    has_neighbor = degree[:, 0] > 0.0
    smoothed = torch.where(has_neighbor[:, None], neighbor_mean, correction)
    return (1.0 - strength) * correction + strength * smoothed


def _resolve_sampled_layer_contact(
    first: NeuralCloth4DState,
    second: NeuralCloth4DState,
    first_pins: Tensor,
    second_pins: Tensor,
    clearance: float,
    friction: float,
    secondary_mobility: float = 1.0,
    preferred_first_vertical_direction: float | None = None,
) -> tuple[NeuralCloth4DState, NeuralCloth4DState]:
    distance = torch.cdist(first.positions, second.positions)
    nearest_distance, nearest_second = distance.min(dim=1)
    active_first = nearest_distance < clearance
    if not torch.any(active_first):
        return first, second
    if not 0.0 <= secondary_mobility <= 1.0:
        raise ValueError("secondary layer mobility must lie in [0,1]")
    first_ids = torch.nonzero(active_first, as_tuple=False).flatten()
    second_ids = nearest_second[first_ids]
    delta = first.positions[first_ids] - second.positions[second_ids]
    raw_length = torch.linalg.vector_norm(delta, dim=-1)
    length = raw_length.clamp_min(1.0e-8)
    normal = delta / length[:, None]
    relative = first.velocities[first_ids] - second.velocities[second_ids]
    fallback = -torch.nn.functional.normalize(relative, dim=-1)
    still_undefined = torch.linalg.vector_norm(fallback, dim=-1) < 1.0e-6
    if torch.any(still_undefined):
        fallback = fallback.clone()
        fallback[still_undefined] = fallback.new_tensor([0.0, 0.0, 1.0])
    normal = torch.where((raw_length < 1.0e-7)[:, None], fallback, normal)
    if preferred_first_vertical_direction is not None:
        if preferred_first_vertical_direction == 0.0:
            raise ValueError("preferred layer direction cannot be zero")
        # At a waist interface, nearest-point normals can point through the
        # body because the two hems are locally interleaved. Orient the
        # separation normal toward the declared upper/lower layer ordering.
        vertical_component = normal[:, 1] * float(preferred_first_vertical_direction)
        normal = torch.where(
            (vertical_component < 0.0)[:, None],
            -normal,
            normal,
        )
    # Layer and body contact projectors are coupled at the waist. Relax the
    # layer correction so alternating projections converge instead of
    # repeatedly pushing the waistband back through the body.
    correction = 0.35 * (clearance - length)[:, None] * normal
    first_movable = torch.ones(len(first.positions), dtype=torch.bool, device=first.positions.device)
    second_movable = torch.ones(len(second.positions), dtype=torch.bool, device=second.positions.device)
    first_movable[first_pins] = False
    second_movable[second_pins] = False
    weight_first = first_movable[first_ids].to(first.positions.dtype)
    weight_second = (
        secondary_mobility
        * second_movable[second_ids].to(second.positions.dtype)
    )
    denominator = (weight_first + weight_second).clamp_min(1.0)
    first_share = weight_first / denominator
    second_share = weight_second / denominator
    first_positions = first.positions.clone()
    second_positions = second.positions.clone()
    first_positions.index_add_(0, first_ids, first_share[:, None] * correction)
    second_positions.index_add_(0, second_ids, -second_share[:, None] * correction)

    normal_speed = (relative * normal).sum(dim=-1)
    impulse = torch.minimum(normal_speed, torch.zeros_like(normal_speed))[:, None] * normal
    tangent = relative - normal_speed[:, None] * normal
    impulse = impulse + friction * tangent
    first_velocity = first.velocities.clone()
    second_velocity = second.velocities.clone()
    first_velocity.index_add_(0, first_ids, -first_share[:, None] * impulse)
    second_velocity.index_add_(0, second_ids, second_share[:, None] * impulse)
    return (
        NeuralCloth4DState(first_positions, first_velocity, first.time),
        NeuralCloth4DState(second_positions, second_velocity, second.time),
    )


def _event_dict(event: object) -> dict[str, object]:
    return {
        "kind": event.kind.value,
        "event_id": event.event_id,
        "time": event.time,
        "bracket": list(event.bracket),
        "localized": event.localized,
        "active": event.active,
        "payload": dict(event.payload),
    }
