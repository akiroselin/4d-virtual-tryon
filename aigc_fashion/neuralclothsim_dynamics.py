from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np
import torch
from torch import Tensor, nn

from .bundle_measure_port_hamiltonian import (
    BundleMeasureBuilder,
    assemble_bundle_measure_hamiltonian,
    symplectic_euler_measure_step,
    vertex_momentum_fibre_push_forward,
)
from .data import MaterialSpec
from .dynamic_events import (
    ActiveSide,
    EventSurface,
    EventSurfaceKind,
    ZeroCrossingEventDetector,
)
from .geometric_cloth_prior import IntrinsicEquivariantClothPrior
from .gauge_contact import GaugeEquivariantContactCNN
from .generalized_flow import (
    GeneralizedFlowLedger,
    GeneralizedFlowPort,
    GeneralizedFlowRepresentation,
    GeneralizedFlowRole,
    GeneralizedImpulseTransfer,
    kinetic_energy_from_velocity,
)
from .genesis_tactile_adapter import project_tangent_vector, tangent_frames_from_normals
from .geometric_phase_graph import (
    IntrinsicClothPhaseState,
    apply_sdf_contact_momentum_reset,
)
from .hypernetwork_kirchhoff_love import (
    HyperNetworkKirchhoffLoveShell,
    KirchhoffLoveShellGeometry,
)
from .metric_projection import (
    DifferentiableMetricProjector,
    MaterialStrainLimits,
    MetricProjectionDiagnostics,
    material_strain_limits,
)
from .phase_manifold_solver import (
    IncrementalHamiltonianConfig,
    PhasePotentialTerms,
    reference_surface_contact_potential,
    reference_surface_contact_quadrature,
    solve_incremental_hamiltonian_step,
)
from .reference_map_bundle import ReferenceMapFiberBundle


@dataclass(frozen=True)
class NeuralCloth4DDynamicsConfig:
    dt: float = 0.004
    substeps_per_frame: int = 3
    areal_density_kg_m2: float = 0.18
    gravity_m_s2: tuple[float, float, float] = (0.0, -9.81, 0.0)
    velocity_damping_per_second: float = 0.8
    bending_damping_per_second: float = 0.0
    maximum_acceleration_m_s2: float = 120.0
    maximum_velocity_m_s: float = 3.0
    shell_force_scale: float = 1.0
    minimum_constitutive_force_scale: float = 0.1
    bending_force_scale: float = 1.0
    initial_shell_acceleration_target_m_s2: float = 20.0
    residual_force_scale: float = 0.02
    residual_acceleration_limit_m_s2: float = 5.0
    laplace_beltrami_diffusion_time_m2: float = 0.0025
    seam_stiffness_n_m: float = 120.0
    seam_damping_n_s_m: float = 0.8
    retention_band_axial_rigidity_n: float = 0.0
    retention_band_damping_n_s_m: float = 0.04
    contact_clearance_m: float = 0.003
    contact_stiffness_n_m: float = 600.0
    contact_pressure_stiffness_n_m3: float = 2.0e5
    contact_damping_n_s_m: float = 2.0
    friction_coefficient: float = 0.35
    metric_projection_iterations: int = 16
    metric_projection_adaptive_iterations: int = 32
    metric_projection_tolerance: float = 1.5e-3
    metric_projection_emergency_margin: float = 0.05
    metric_projection_severe_violation: float = 0.15
    metric_projection_max_correction_m: float = 0.0015
    # Position projection is outside the variational flow. Keep it opt-in for
    # physical rollouts; Hyper-NODE remains the material guard.
    enable_metric_projection: bool = True
    strain_limit_onset_fraction: float = 0.8
    strain_limit_barrier_scale: float = 0.02
    implicit_material_iterations: int = 5
    implicit_material_relaxation: float = 0.5
    maximum_implicit_material_correction_m: float = 0.003
    phase_hamiltonian_iterations: int = 6
    phase_hamiltonian_line_search_steps: int = 5
    phase_hamiltonian_gradient_tolerance_n: float = 2.0e-4
    phase_hamiltonian_trust_radius_m: float = 0.003
    se3_generative_potential_weight: float = 0.0
    # In the moving-body frame, body acceleration appears as a generalized
    # inertial port. It transfers momentum without adding wind or a position
    # correction, and is solved together with the Hamiltonian increment.
    body_inertial_force_scale: float = 0.75
    initial_material_relaxation_iterations: int = 4
    initial_shell_relaxation_force_scale: float = 0.25
    maximum_event_restarts_per_substep: int = 2
    gauge_contact_channels: int = 4
    maximum_log_friction_scale: float = 0.35

    def __post_init__(self) -> None:
        positive = (
            self.dt,
            self.areal_density_kg_m2,
            self.maximum_acceleration_m_s2,
            self.maximum_velocity_m_s,
            self.initial_shell_acceleration_target_m_s2,
            self.residual_acceleration_limit_m_s2,
            self.laplace_beltrami_diffusion_time_m2,
            self.contact_clearance_m,
            self.contact_stiffness_n_m,
            self.contact_pressure_stiffness_n_m3,
            self.phase_hamiltonian_gradient_tolerance_n,
            self.phase_hamiltonian_trust_radius_m,
            self.metric_projection_max_correction_m,
        )
        if any(value <= 0.0 for value in positive):
            raise ValueError("dynamic time, mass, limits, and contact settings must be positive")
        if (
            self.substeps_per_frame <= 0
            or self.metric_projection_iterations <= 0
            or self.metric_projection_adaptive_iterations < 0
            or self.implicit_material_iterations <= 0
            or self.phase_hamiltonian_iterations <= 0
            or self.phase_hamiltonian_line_search_steps <= 0
        ):
            raise ValueError("dynamic substeps and projection iterations must be positive")
        if self.initial_material_relaxation_iterations < 0:
            raise ValueError("initial material relaxation iterations must be non-negative")
        if self.gauge_contact_channels <= 0:
            raise ValueError("gauge_contact_channels must be positive")
        if self.maximum_event_restarts_per_substep < 0:
            raise ValueError("maximum_event_restarts_per_substep must be non-negative")
        if not 0.0 < self.strain_limit_onset_fraction < 1.0:
            raise ValueError("strain_limit_onset_fraction must lie in (0, 1)")
        if not 0.0 < self.implicit_material_relaxation <= 1.0:
            raise ValueError("implicit_material_relaxation must lie in (0, 1]")
        nonnegative = (
            self.velocity_damping_per_second,
            self.bending_damping_per_second,
            self.shell_force_scale,
            self.minimum_constitutive_force_scale,
            self.bending_force_scale,
            self.residual_force_scale,
            self.seam_stiffness_n_m,
            self.seam_damping_n_s_m,
            self.retention_band_axial_rigidity_n,
            self.retention_band_damping_n_s_m,
            self.contact_damping_n_s_m,
            self.friction_coefficient,
            self.metric_projection_tolerance,
            self.metric_projection_emergency_margin,
            self.metric_projection_severe_violation,
            self.strain_limit_barrier_scale,
            self.maximum_implicit_material_correction_m,
            self.maximum_log_friction_scale,
            self.se3_generative_potential_weight,
            self.body_inertial_force_scale,
            self.initial_shell_relaxation_force_scale,
        )
        if any(value < 0.0 for value in nonnegative):
            raise ValueError("dynamic damping, force scales, and tolerance must be non-negative")


@dataclass(frozen=True)
class BodyContactSample:
    closest_points: Tensor
    normals: Tensor
    body_velocities: Tensor | None = None


BodyContactQuery = Callable[[Tensor, float], BodyContactSample]
InterGarmentContact = tuple[Tensor, float, float, Tensor, Tensor]


@dataclass(frozen=True)
class NeuralCloth4DState:
    positions: Tensor
    velocities: Tensor
    time: float


@dataclass(frozen=True)
class NeuralCloth4DRolloutReport:
    method: str
    frames: int
    substeps: int
    physical_duration_s: float
    material_profile: str
    event_restarts: int
    contact_events: tuple[dict[str, object], ...]
    maximum_speed_m_s: float
    maximum_acceleration_m_s2: float
    maximum_unclamped_acceleration_m_s2: float
    acceleration_cap_hits: int
    acceleration_cap_fraction: float
    constitutive_force_scale: float
    bending_force_scale: float
    minimum_constitutive_force_scale: float
    constitutive_calibration_status: str
    initial_shell_acceleration_p95_m_s2: float
    residual_acceleration_limit_m_s2: float
    minimum_contact_gap_m: float | None
    metric_projection: dict[str, float | int]
    hypernode_strain_limit: dict[str, float | int]
    initial_material_relaxation_steps: int
    metric_projection_recovery_count: int
    metric_projection_recovery_fraction: float
    seam_pairs: int
    retention_band_edges: int
    phase_space: dict[str, object]
    phase_hamiltonian: dict[str, object]
    noether_audit: dict[str, object]
    generalized_flow: dict[str, object]
    reference_map_fiber: dict[str, object]
    bundle_measure_port_hamiltonian: dict[str, object]
    limitations: tuple[str, ...]
    metric_projection_enabled: bool = True

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class NeuralCloth4DRollout:
    positions: Tensor
    velocities: Tensor
    report: NeuralCloth4DRolloutReport


class NeuralCloth4DDynamics(nn.Module):
    """Physical-time rollout for one scene-optimised NeuralClothSim field.

    Kirchhoff-Love Hyper-NODE forces are authoritative. The E(3)-equivariant
    edge network contributes only a bounded residual force. Manufacturing
    collision response remains an explicit solver stage. Material strain limits
    live in the Hyper-NODE energy; metric projection is emergency recovery only.
    """

    def __init__(
        self,
        shell_model: HyperNetworkKirchhoffLoveShell,
        shell_geometry: KirchhoffLoveShellGeometry,
        *,
        rest_positions: Tensor,
        edge_index: Tensor,
        rest_lengths: Tensor,
        cotangent_weights: Tensor,
        pinned_vertices: Tensor,
        laplace_eigenvalues: Tensor | None = None,
        laplace_eigenvectors: Tensor | None = None,
        vertex_mass: Tensor | None = None,
        seam_pairs: Tensor | None = None,
        retention_edges: Tensor | None = None,
        retention_rest_scale: float = 1.0,
        material: MaterialSpec = MaterialSpec(),
        strain_limits: MaterialStrainLimits | None = None,
        residual_model: IntrinsicEquivariantClothPrior | None = None,
        contact_model: GaugeEquivariantContactCNN | None = None,
        config: NeuralCloth4DDynamicsConfig = NeuralCloth4DDynamicsConfig(),
    ) -> None:
        super().__init__()
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape (2,E)")
        if rest_positions.ndim != 2 or rest_positions.shape[1] != 3:
            raise ValueError("rest_positions must have shape (V,3)")
        if rest_lengths.shape != (edge_index.shape[1],):
            raise ValueError("rest_lengths must align with edge_index")
        if cotangent_weights.shape != rest_lengths.shape:
            raise ValueError("cotangent_weights must align with rest_lengths")
        self.shell_model = shell_model
        self.shell_geometry = shell_geometry
        self.residual_model = residual_model
        self.config = config
        self.material = material
        self.contact_model = (
            GaugeEquivariantContactCNN(
                config.gauge_contact_channels,
                maximum_log_friction_scale=config.maximum_log_friction_scale,
            )
            if contact_model is None
            else contact_model
        )
        self.register_buffer("rest_positions", rest_positions.detach().to(torch.float32))
        self.register_buffer("edge_index", edge_index.detach().to(torch.long))
        self.register_buffer("rest_lengths", rest_lengths.detach().to(torch.float32))
        self.register_buffer("cotangent_weights", cotangent_weights.detach().to(torch.float32))
        spectrum = (laplace_eigenvalues, laplace_eigenvectors, vertex_mass)
        if any(value is None for value in spectrum) and not all(
            value is None for value in spectrum
        ):
            raise ValueError("Laplace-Beltrami eigenvalues, eigenvectors, and mass are all required")
        if laplace_eigenvalues is None:
            laplace_eigenvalues = torch.empty(0, dtype=torch.float32)
            laplace_eigenvectors = torch.empty((len(rest_positions), 0), dtype=torch.float32)
            vertex_mass = torch.empty(0, dtype=torch.float32)
        assert laplace_eigenvectors is not None and vertex_mass is not None
        if laplace_eigenvectors.shape != (len(rest_positions), len(laplace_eigenvalues)):
            raise ValueError("Laplace-Beltrami eigenvectors must have shape (V,K)")
        if vertex_mass.shape not in {(0,), (len(rest_positions),)}:
            raise ValueError("Laplace-Beltrami mass must have shape (V,)")
        if torch.any(laplace_eigenvalues < -1.0e-7) or (
            vertex_mass.numel() and torch.any(vertex_mass <= 0.0)
        ):
            raise ValueError("Laplace-Beltrami spectrum and mass must be non-negative/positive")
        self.register_buffer(
            "laplace_eigenvalues", laplace_eigenvalues.detach().to(torch.float32)
        )
        self.register_buffer(
            "laplace_eigenvectors", laplace_eigenvectors.detach().to(torch.float32)
        )
        self.register_buffer("laplace_vertex_mass", vertex_mass.detach().to(torch.float32))
        self.register_buffer("pinned_vertices", pinned_vertices.detach().to(torch.long))
        pairs = (
            torch.empty((0, 2), dtype=torch.long)
            if seam_pairs is None
            else seam_pairs.detach().to(torch.long).reshape(-1, 2)
        )
        self.register_buffer("seam_pairs", pairs)
        bands = (
            torch.empty((0, 2), dtype=torch.long)
            if retention_edges is None
            else retention_edges.detach().to(torch.long).reshape(-1, 2)
        )
        if bands.numel() and (
            bands.min() < 0 or bands.max() >= len(rest_positions)
        ):
            raise ValueError("retention band edge lies outside the garment mesh")
        if not 0.0 < retention_rest_scale <= 1.0:
            raise ValueError("retention_rest_scale must lie in (0, 1]")
        self.register_buffer("retention_edges", bands)
        self.register_buffer(
            "retention_rest_lengths",
            torch.linalg.vector_norm(
                self.rest_positions[bands[:, 1]] - self.rest_positions[bands[:, 0]],
                dim=-1,
            ) * retention_rest_scale
            if len(bands)
            else torch.empty(0, dtype=torch.float32),
        )
        self.strain_limits = (
            material_strain_limits(material) if strain_limits is None else strain_limits
        )
        self.metric_projector = DifferentiableMetricProjector(
            self.edge_index.T,
            self.rest_positions,
            self.pinned_vertices,
            self.strain_limits,
            iterations=config.metric_projection_iterations,
            maximum_adaptive_iterations=config.metric_projection_adaptive_iterations,
            tolerance=config.metric_projection_tolerance,
        )
        self.reference_map_bundle = ReferenceMapFiberBundle.from_shell_geometry(
            shell_geometry,
            rest_positions,
        )
        self.bundle_measure_builder = BundleMeasureBuilder(
            self.reference_map_bundle,
            shell_geometry,
            areal_density_kg_m2=config.areal_density_kg_m2,
        )
        self.generalized_flow_ledger = GeneralizedFlowLedger()
        self._latest_phase_hamiltonian_diagnostics: dict[str, object] = {}
        self._maximum_internal_force_balance_n = 0.0
        self._maximum_internal_torque_balance_n_m = 0.0

    def rollout(
        self,
        initial_positions: Tensor,
        material_latent: Tensor,
        *,
        frame_count: int,
        initial_velocities: Tensor | None = None,
        body_contact_query: BodyContactQuery | None = None,
        differentiable: bool = False,
    ) -> NeuralCloth4DRollout:
        if frame_count < 2:
            raise ValueError("frame_count must be at least two")
        if initial_positions.shape != self.rest_positions.shape:
            raise ValueError("initial_positions must match rest_positions")
        self.generalized_flow_ledger.reset()
        self._maximum_internal_force_balance_n = 0.0
        self._maximum_internal_torque_balance_n_m = 0.0
        velocity = (
            torch.zeros_like(initial_positions)
            if initial_velocities is None
            else initial_velocities.to(initial_positions)
        )
        if velocity.shape != initial_positions.shape:
            raise ValueError("initial_velocities must match initial_positions")
        pinned_positions = initial_positions[self.pinned_vertices].detach()
        initial_positions, initial_relaxation_steps = self.prepare_initial_positions(
            initial_positions,
            material_latent,
            body_contact_query=body_contact_query,
            pinned_positions=pinned_positions,
        )
        constitutive_force_scale, initial_shell_acceleration_p95 = (
            self._calibrate_constitutive_force_scale(
                initial_positions,
                material_latent,
                differentiable=differentiable,
            )
        )
        state = NeuralCloth4DState(initial_positions, velocity, 0.0)
        position_frames = [state.positions]
        velocity_frames = [state.velocities]
        detector = ZeroCrossingEventDetector()
        detector.register(
            EventSurface(
                "cloth:body:min_gap",
                EventSurfaceKind.BODY_CONTACT,
                ActiveSide.BELOW,
                threshold=self.config.contact_clearance_m,
                hysteresis=2.5e-4,
            )
        )
        minimum_gap: float | None = None
        if body_contact_query is not None:
            initial_contact = _validate_contact_sample(
                body_contact_query(state.positions.detach(), state.time), state.positions
            )
            initial_gap = _contact_gap(state.positions, initial_contact)
            minimum_gap = float(initial_gap.min().detach())
            detector.observe(state.time, {"cloth:body:min_gap": minimum_gap})

        contact_events: list[dict[str, object]] = []
        event_restarts = 0
        maximum_speed = 0.0
        maximum_acceleration = 0.0
        maximum_unclamped_acceleration = 0.0
        acceleration_cap_hits = 0
        acceleration_samples = 0
        metric_projection_recovery_count = 0
        phase_hamiltonian_steps: list[dict[str, object]] = []
        total_substeps = (frame_count - 1) * self.config.substeps_per_frame
        for substep in range(total_substeps):
            flow_checkpoint = self.generalized_flow_ledger.checkpoint()
            (
                proposed,
                acceleration,
                unclamped_maximum,
                cap_hits,
                phase_step,
            ) = self._advance_candidate(
                state,
                material_latent,
                self.config.dt,
                body_contact_query,
                differentiable,
                constitutive_force_scale,
            )
            phase_hamiltonian_steps.append(phase_step)
            maximum_unclamped_acceleration = max(
                maximum_unclamped_acceleration, unclamped_maximum
            )
            acceleration_cap_hits += cap_hits
            acceleration_samples += int(len(state.positions))
            maximum_acceleration = max(
                maximum_acceleration,
                float(torch.linalg.vector_norm(acceleration.detach(), dim=-1).max()),
            )
            if body_contact_query is not None:
                current_contact = _validate_contact_sample(
                    body_contact_query(state.positions.detach(), state.time), state.positions
                )
                current_minimum = float(_contact_gap(state.positions, current_contact).min().detach())
                raw_contact = _validate_contact_sample(
                    body_contact_query(proposed.positions.detach(), proposed.time), proposed.positions
                )
                raw_gap = _contact_gap(proposed.positions, raw_contact)
                raw_minimum = float(raw_gap.min().detach())
                threshold = self.config.contact_clearance_m
                if current_minimum > threshold and raw_minimum <= threshold:
                    alpha = (current_minimum - threshold) / max(
                        current_minimum - raw_minimum, 1.0e-12
                    )
                    root_time = state.time + float(np.clip(alpha, 0.0, 1.0)) * (
                        proposed.time - state.time
                    )
                    batch = detector.record_external_root(
                        "cloth:body:min_gap",
                        time=root_time,
                        active=True,
                        bracket=(state.time, proposed.time),
                        sample_values=(current_minimum, raw_minimum),
                    )
                else:
                    batch = detector.observe(
                        proposed.time,
                        {"cloth:body:min_gap": raw_minimum},
                    )
                if batch.restart_required and event_restarts < (
                    total_substeps * self.config.maximum_event_restarts_per_substep
                ):
                    event = batch.events[0]
                    alpha = float(
                        np.clip(
                            (event.time - state.time) / max(proposed.time - state.time, 1.0e-12),
                            0.0,
                            1.0,
                        )
                    )
                    self.generalized_flow_ledger.restore(flow_checkpoint)
                    root_duration = event.time - state.time
                    if root_duration > 1.0e-9:
                        (
                            root_state,
                            _root_acceleration,
                            _root_maximum,
                            _root_hits,
                            root_phase_step,
                        ) = (
                            self._advance_candidate(
                                state,
                                material_latent,
                                root_duration,
                                body_contact_query,
                                differentiable,
                                constitutive_force_scale,
                            )
                        )
                        phase_hamiltonian_steps.append(root_phase_step)
                    else:
                        root_state = NeuralCloth4DState(
                            state.positions,
                            state.velocities,
                            event.time,
                        )
                    root_state, root_gap = self._resolve_contact(
                        root_state,
                        body_contact_query,
                        pinned_positions,
                    )
                    remaining = proposed.time - event.time
                    if remaining > 1.0e-9:
                        (
                            proposed,
                            restart_acceleration,
                            restart_unclamped_maximum,
                            restart_cap_hits,
                            restart_phase_step,
                        ) = self._advance_candidate(
                            root_state,
                            material_latent,
                            remaining,
                            body_contact_query,
                            differentiable,
                            constitutive_force_scale,
                        )
                        phase_hamiltonian_steps.append(restart_phase_step)
                        maximum_acceleration = max(
                            maximum_acceleration,
                            float(
                                torch.linalg.vector_norm(
                                    restart_acceleration.detach(), dim=-1
                                ).max()
                            ),
                        )
                        maximum_unclamped_acceleration = max(
                            maximum_unclamped_acceleration,
                            restart_unclamped_maximum,
                        )
                        acceleration_cap_hits += restart_cap_hits
                        acceleration_samples += int(len(state.positions))
                    event_restarts += 1
                    contact_events.extend(_event_payload(item) for item in batch.events)
                    minimum_gap = root_gap if minimum_gap is None else min(minimum_gap, root_gap)
                proposed, resolved_gap = self._resolve_contact(
                    proposed,
                    body_contact_query,
                    pinned_positions,
                )
                minimum_gap = resolved_gap if minimum_gap is None else min(minimum_gap, resolved_gap)

            proposed_material = self.hypernode_strain_limit_diagnostics(
                proposed.positions
            )
            if (
                self.config.enable_metric_projection
                and proposed_material["emergency_components"]
                and float(proposed_material["maximum_limit_violation"])
                > self.config.metric_projection_severe_violation
            ):
                projected_positions = self.metric_projector(
                    proposed.positions,
                    pinned_positions=pinned_positions,
                )
                # A full metric projection is a valid constraint solve but a
                # bad time integrator: on a pleated shell it can move a whole
                # fold by several millimetres in one 20 ms substep. Limit the
                # correction as a continuous constraint impulse; Hyper-NODE's
                # barrier and the next Hamiltonian step finish the recovery.
                correction = projected_positions - proposed.positions
                correction_norm = torch.linalg.vector_norm(
                    correction, dim=-1, keepdim=True
                )
                correction = correction * torch.clamp(
                    self.config.metric_projection_max_correction_m
                    / correction_norm.clamp_min(1.0e-12),
                    max=1.0,
                )
                projected_positions = proposed.positions + correction
                projected_velocity = (
                    projected_positions - state.positions
                ) / self.config.dt
                projected_velocity = _clamp_vector_norm(
                    projected_velocity,
                    self.config.maximum_velocity_m_s,
                )
                self._record_momentum_projection(
                    "material_metric_emergency_recovery",
                    proposed.velocities,
                    projected_velocity,
                    self._vertex_mass(projected_positions),
                    proposed.time,
                    GeneralizedFlowRole.CONSTRAINT,
                )
                metric_projection_recovery_count += 1
            else:
                projected_positions = proposed.positions
                projected_velocity = proposed.velocities
            if len(self.pinned_vertices):
                projected_velocity = projected_velocity.clone()
                projected_velocity[self.pinned_vertices] = 0.0
                projected_positions = projected_positions.clone()
                projected_positions[self.pinned_vertices] = pinned_positions
            state = NeuralCloth4DState(
                projected_positions,
                projected_velocity,
                (substep + 1) * self.config.dt,
            )
            maximum_speed = max(
                maximum_speed,
                float(torch.linalg.vector_norm(state.velocities.detach(), dim=-1).max()),
            )
            if (substep + 1) % self.config.substeps_per_frame == 0:
                position_frames.append(state.positions)
                velocity_frames.append(state.velocities)

        position_tensor = torch.stack(position_frames)
        velocity_tensor = torch.stack(velocity_frames)
        diagnostics = self.metric_projector.diagnostics(position_tensor[-1])
        hypernode_diagnostics = self.hypernode_strain_limit_diagnostics(
            position_tensor[-1]
        )
        noether_audit = self._noether_audit(
            position_tensor,
            velocity_tensor,
            material_latent,
            constitutive_force_scale,
        )
        report = NeuralCloth4DRolloutReport(
            method=(
                "physical-time Hyper-NODE Kirchhoff-Love + bounded E(3) residual "
                "+ SO(2) gauge-equivariant contact"
            ),
            frames=frame_count,
            substeps=total_substeps,
            physical_duration_s=(frame_count - 1)
            * self.config.substeps_per_frame
            * self.config.dt,
            material_profile=self.strain_limits.profile,
            event_restarts=event_restarts,
            contact_events=tuple(contact_events),
            maximum_speed_m_s=maximum_speed,
            maximum_acceleration_m_s2=maximum_acceleration,
            maximum_unclamped_acceleration_m_s2=maximum_unclamped_acceleration,
            acceleration_cap_hits=acceleration_cap_hits,
            acceleration_cap_fraction=(
                acceleration_cap_hits / acceleration_samples
                if acceleration_samples
                else 0.0
            ),
            constitutive_force_scale=constitutive_force_scale,
            bending_force_scale=self.config.bending_force_scale,
            minimum_constitutive_force_scale=self.config.minimum_constitutive_force_scale,
            constitutive_calibration_status=(
                "material_floor"
                if constitutive_force_scale <= self.config.minimum_constitutive_force_scale
                * self.config.shell_force_scale + 1.0e-12
                else "near_calibrated"
                if constitutive_force_scale >= 0.1 * self.config.shell_force_scale
                else "scene_rescaled"
                if constitutive_force_scale >= 0.01 * self.config.shell_force_scale
                else "constitutive_recalibration_required"
            ),
            initial_shell_acceleration_p95_m_s2=initial_shell_acceleration_p95,
            residual_acceleration_limit_m_s2=self.config.residual_acceleration_limit_m_s2,
            minimum_contact_gap_m=minimum_gap,
            metric_projection=diagnostics.to_dict(),
            hypernode_strain_limit=hypernode_diagnostics,
            initial_material_relaxation_steps=initial_relaxation_steps,
            metric_projection_recovery_count=metric_projection_recovery_count,
            metric_projection_recovery_fraction=(
                metric_projection_recovery_count / total_substeps
                if total_substeps
                else 0.0
            ),
            seam_pairs=int(len(self.seam_pairs)),
            retention_band_edges=int(len(self.retention_edges)),
            phase_space={
                "state": "(xi=(sheet,u,v), rho, m, q_frame, U_shell, contact_fibre, z_m)",
                "manifold": "T*Emb(S_cloth,R3) x Spin(3) fibre x SO(2) contact fibres",
                "integrator": (
                    "implicit incremental Hamiltonian/action solve on the full "
                    "cloth cotangent bundle"
                ),
                "contact": (
                    "reference-area surface quadrature barrier inside H plus SDF "
                    "event momentum reset"
                ),
                "learned_hamiltonian": True,
                "hypernode_role": (
                    "material latent generates the membrane, bending, and hard-strain "
                    "Hamiltonian densities optimized for q_(n+1)"
                ),
            },
            phase_hamiltonian=_aggregate_phase_hamiltonian_steps(
                phase_hamiltonian_steps
            ),
            noether_audit=noether_audit,
            generalized_flow=self.generalized_flow_ledger.audit().to_dict(),
            reference_map_fiber=self.reference_map_bundle.report(
                position_tensor[-1]
            ).to_dict(),
            bundle_measure_port_hamiltonian=self.bundle_measure_report(
                position_tensor[-1],
                velocity_tensor[-1],
                material_latent,
                physical_time=state.time,
                constitutive_scale=constitutive_force_scale,
            ),
            limitations=(
                "no cloth self-collision or multi-layer contact yet",
                "contact closest-point queries are piecewise differentiable",
                "material-family strain limits require physical calibration",
                "metric projection is an emergency recovery, not a constitutive update",
            ),
            metric_projection_enabled=self.config.enable_metric_projection,
        )
        return NeuralCloth4DRollout(position_tensor, velocity_tensor, report)

    def _noether_audit(
        self,
        position_frames: Tensor,
        velocity_frames: Tensor,
        material_latent: Tensor,
        constitutive_force_scale: float,
    ) -> dict[str, object]:
        """Audit the diagonal E(3) symmetry and the cloth Hamiltonian trace.

        The internal shell/edge graph is translation and rotation invariant.
        Gravity, prescribed body motion, contact, and Rayleigh ports are
        intentionally reported as open-system work rather than hidden inside
        a false conservation claim.
        """

        mass = self._vertex_mass(position_frames[0]).to(position_frames)
        momentum = (mass[None, :, None] * velocity_frames).sum(dim=1)
        hamiltonian: list[float] = []
        with torch.no_grad():
            for positions, velocities in zip(position_frames, velocity_frames):
                report = self.bundle_measure_report(
                    positions,
                    velocities,
                    material_latent,
                    physical_time=0.0,
                    constitutive_scale=constitutive_force_scale,
                )
                hamiltonian.append(float(report["hamiltonian_j"]["total"]))
        momentum_drift = torch.linalg.vector_norm(momentum - momentum[0], dim=-1)
        energy = np.asarray(hamiltonian, dtype=np.float64)
        return {
            "structure_group": "diagonal E(3) on all garment vertices; SE(3) body transport is a relative reference",
            "internal_force_balance_max_n": self._maximum_internal_force_balance_n,
            "internal_torque_balance_max_n_m": self._maximum_internal_torque_balance_n_m,
            "internal_balance_target": "zero net force and torque for Hyper-NODE shell, seam, trim, and paired E(3) residual ports",
            "cloth_linear_momentum_kg_m_s": momentum.detach().cpu().tolist(),
            "cloth_linear_momentum_drift_max_kg_m_s": float(momentum_drift.max()),
            "cloth_hamiltonian_j": energy.tolist(),
            "cloth_hamiltonian_drift_max_j": float(np.max(np.abs(energy - energy[0]))),
            "open_system_ports": (
                "gravity potential, prescribed body-root transport, SDF contact, "
                "gauge friction, and Rayleigh material dissipation"
            ),
            "strict_conservation_scope": (
                "internal interaction balance only; total body+cloth energy requires "
                "a free body and ground reaction model"
            ),
        }

    def prepare_initial_positions(
        self,
        initial_positions: Tensor,
        material_latent: Tensor,
        *,
        body_contact_query: BodyContactQuery | None = None,
        pinned_positions: Tensor | None = None,
    ) -> tuple[Tensor, int]:
        """Relax the imported shell before the shared physical clock starts."""

        if initial_positions.shape != self.rest_positions.shape:
            raise ValueError("initial_positions must match rest_positions")
        fixed = (
            initial_positions[self.pinned_vertices].detach()
            if pinned_positions is None
            else pinned_positions.to(initial_positions)
        )
        relaxed = initial_positions
        steps = 0
        relaxation_dt = max(self.config.dt, 0.008)
        for _ in range(self.config.initial_material_relaxation_iterations):
            relaxed = self.implicit_material_proximal_step(
                relaxed,
                inertial_target=relaxed,
                material_latent=material_latent,
                dt=relaxation_dt,
                pinned_positions=fixed,
                iterations=1,
            )
            steps += 1
        # The learned shell checkpoint can carry a large metric mismatch
        # before physical time starts.  Project once at initialization so the
        # Hamiltonian feasibility constraint has a valid starting point.  It
        # is intentionally not repeated per ODE substep; subsequent recovery
        # follows Hyper-NODE's constrained action solve.
        if self.config.enable_metric_projection:
            relaxed = self.metric_projector(
                relaxed,
                pinned_positions=fixed,
            )
        # Do not run the triangle-surface contact projector here.  The
        # imported shell is already placed outside the avatar, and applying a
        # moving-body face correction before t=0 can turn a material
        # relaxation into a spurious shoulder-wide translation.  The first
        # physical substep performs the full vertex + face contact solve.
        return relaxed, steps

    def _advance_candidate(
        self,
        state: NeuralCloth4DState,
        material_latent: Tensor,
        dt: float,
        body_contact_query: BodyContactQuery | None,
        differentiable: bool,
        constitutive_force_scale: float,
        external_force: Tensor | None = None,
        se3_target_velocity: Tensor | None = None,
        moving_reference_velocity: Tensor | None = None,
        moving_reference_displacement: Tensor | None = None,
        moving_reference_next_velocity: Tensor | None = None,
        inter_garment_contacts: tuple[InterGarmentContact, ...] = (),
    ) -> tuple[NeuralCloth4DState, Tensor, float, int, dict[str, object]]:
        positions = state.positions
        if external_force is None:
            external_force = torch.zeros_like(positions)
        elif external_force.shape != positions.shape:
            raise ValueError("external_force must match cloth position shape")
        else:
            external_force = external_force.to(positions)
        if not differentiable:
            external_force = external_force.detach()
        if se3_target_velocity is None:
            se3_target_velocity = torch.zeros_like(positions)
        elif se3_target_velocity.shape != positions.shape:
            raise ValueError("se3_target_velocity must match cloth position shape")
        else:
            se3_target_velocity = se3_target_velocity.to(positions)
        if not differentiable:
            se3_target_velocity = se3_target_velocity.detach()
        if moving_reference_velocity is None:
            moving_reference_velocity = torch.zeros_like(positions)
        elif moving_reference_velocity.shape != positions.shape:
            raise ValueError("moving_reference_velocity must match cloth position shape")
        else:
            moving_reference_velocity = moving_reference_velocity.to(positions)
        if moving_reference_displacement is None:
            moving_reference_displacement = torch.zeros_like(positions)
        elif moving_reference_displacement.shape != positions.shape:
            raise ValueError("moving_reference_displacement must match cloth position shape")
        else:
            moving_reference_displacement = moving_reference_displacement.to(positions)
        if moving_reference_next_velocity is None:
            moving_reference_next_velocity = moving_reference_velocity
        elif moving_reference_next_velocity.shape != positions.shape:
            raise ValueError("moving_reference_next_velocity must match cloth position shape")
        else:
            moving_reference_next_velocity = moving_reference_next_velocity.to(positions)
        if not differentiable:
            moving_reference_velocity = moving_reference_velocity.detach()
            moving_reference_displacement = moving_reference_displacement.detach()
            moving_reference_next_velocity = moving_reference_next_velocity.detach()
        relative_state_velocities = state.velocities - moving_reference_velocity
        force_positions = positions if positions.requires_grad else positions.detach().requires_grad_(True)
        shell_output = self.shell_model(
            force_positions,
            self.shell_geometry,
            material_latent,
            compute_bending_forces=True,
            create_graph=differentiable,
            maximum_tensile_strain=self.strain_limits.maximum_tensile_strain,
            maximum_compressive_strain=self.strain_limits.maximum_compressive_strain,
            strain_limit_onset_fraction=self.config.strain_limit_onset_fraction,
            strain_limit_barrier_scale=self.config.strain_limit_barrier_scale,
        )
        mass = self._vertex_mass(positions)
        moving_reference_acceleration = (
            moving_reference_next_velocity - moving_reference_velocity
        ) / float(dt)
        body_inertial_force = (
            -self.config.body_inertial_force_scale
            * mass[:, None]
            * moving_reference_acceleration
        )
        self._record_force_flow(
            "moving_body_inertial_momentum_transfer",
            "body_se3_inertial_port",
            "cloth_momentum",
            GeneralizedFlowRole.CONTROL,
            body_inertial_force,
            moving_reference_acceleration,
            state.time,
            dt,
        )
        calibrated_membrane_force = (
            shell_output.membrane_internal_force
            - shell_output.strain_limit_internal_force
        )
        shell_force = constitutive_force_scale * (
            calibrated_membrane_force
            + self.config.bending_force_scale * shell_output.bending_internal_force
        )
        strain_limit_force = shell_output.strain_limit_internal_force
        if not differentiable:
            shell_force = shell_force.detach()
            strain_limit_force = strain_limit_force.detach()
        self._record_force_flow(
            "hypernode_kirchhoff_love",
            "shell_elastic_storage",
            "cloth_momentum",
            GeneralizedFlowRole.CONSERVATIVE,
            shell_force,
            state.velocities,
            state.time,
            dt,
        )
        self._record_force_flow(
            "hypernode_intrinsic_edge_strain_barrier",
            "material_feasible_set_energy",
            "cloth_momentum",
            GeneralizedFlowRole.CONSERVATIVE,
            strain_limit_force,
            state.velocities,
            state.time,
            dt,
        )
        residual_force = torch.zeros_like(positions)
        if self.residual_model is not None and self.config.residual_force_scale > 0.0:
            residual = self.residual_model(
                positions,
                relative_state_velocities,
                self.edge_index,
                self.rest_lengths,
                self.cotangent_weights,
                material_latent,
            ).internal_force
            residual = self._laplace_beltrami_diffuse(residual)
            residual_acceleration = self.config.residual_force_scale * residual / mass[:, None]
            residual_acceleration = _clamp_vector_norm(
                residual_acceleration,
                self.config.residual_acceleration_limit_m_s2,
            )
            residual_force = mass[:, None] * residual_acceleration
            self._record_force_flow(
                "e3_intrinsic_learned_residual",
                "learned_cloth_field",
                "cloth_momentum",
                GeneralizedFlowRole.LEARNED_RESIDUAL,
                residual_force,
                state.velocities,
                state.time,
                dt,
            )
        seam_elastic, seam_damping = self._seam_force_components(
            positions, relative_state_velocities
        )
        retention_elastic, retention_damping = self._retention_band_force_components(
            positions, relative_state_velocities
        )
        internal_force = (
            shell_force
            + strain_limit_force
            + residual_force
            + seam_elastic
            + retention_elastic
        )
        internal_net_force = internal_force.sum(dim=0)
        centered_positions = positions - positions.mean(dim=0, keepdim=True)
        internal_net_torque = torch.linalg.vector_norm(
            torch.cross(centered_positions, internal_force, dim=-1).sum(dim=0)
        )
        self._maximum_internal_force_balance_n = max(
            self._maximum_internal_force_balance_n,
            float(torch.linalg.vector_norm(internal_net_force.detach())),
        )
        self._maximum_internal_torque_balance_n_m = max(
            self._maximum_internal_torque_balance_n_m,
            float(internal_net_torque.detach()),
        )
        self._record_force_flow(
            "seam_elasticity",
            "seam_elastic_storage",
            "cloth_momentum",
            GeneralizedFlowRole.CONSERVATIVE,
            seam_elastic,
            state.velocities,
            state.time,
            dt,
        )
        self._record_force_flow(
            "seam_damping",
            "seam_dissipation",
            "cloth_momentum",
            GeneralizedFlowRole.DISSIPATIVE,
            seam_damping,
            state.velocities,
            state.time,
            dt,
        )
        self._record_force_flow(
            "intrinsic_trim_band_elasticity",
            "trim_band_elastic_storage",
            "cloth_momentum",
            GeneralizedFlowRole.CONSERVATIVE,
            retention_elastic,
            state.velocities,
            state.time,
            dt,
        )
        self._record_force_flow(
            "intrinsic_trim_band_damping",
            "trim_band_dissipation",
            "cloth_momentum",
            GeneralizedFlowRole.DISSIPATIVE,
            retention_damping,
            state.velocities,
            state.time,
            dt,
        )
        # Learned residuals and damping are d'Alembert ports. Hyper-NODE shell,
        # seam/trim elasticity, gravity, and contact are assembled below into
        # one real Hamiltonian potential and optimized at q_(n+1).
        nonconservative_force = (
            residual_force
            + seam_damping
            + retention_damping
            + external_force
            + body_inertial_force
        )
        if torch.any(external_force != 0):
            self._record_force_flow(
                "scene_graph_phase_port",
                "hierarchy_phase_storage",
                "cloth_momentum",
                GeneralizedFlowRole.CONTROL,
                external_force,
                state.velocities,
                state.time,
                dt,
            )
        velocity_damping_force = (
            -self.config.velocity_damping_per_second
            * mass[:, None]
            * relative_state_velocities
        )
        self._record_force_flow(
            "rayleigh_velocity_damping",
            "material_dissipation",
            "cloth_momentum",
            GeneralizedFlowRole.DISSIPATIVE,
            velocity_damping_force,
            state.velocities,
            state.time,
            dt,
        )
        nonconservative_force = nonconservative_force + velocity_damping_force
        if self.config.bending_damping_per_second > 0.0:
            # A K-proportional Rayleigh term damps spatially varying velocity
            # modes while leaving rigid transport untouched.  The spectral
            # low-pass is the intrinsic Laplace-Beltrami approximation of the
            # bending-rate component, so this remains remeshing-aware.
            smooth_velocity = self._laplace_beltrami_diffuse(
                relative_state_velocities
            ) + relative_state_velocities.mean(dim=0, keepdim=True)
            bending_rate_velocity = relative_state_velocities - smooth_velocity
            bending_damping_force = (
                -self.config.bending_damping_per_second
                * mass[:, None]
                * bending_rate_velocity
            )
            self._record_force_flow(
                "rayleigh_bending_mode_damping",
                "material_bending_dissipation",
                "cloth_momentum",
                GeneralizedFlowRole.DISSIPATIVE,
                bending_damping_force,
                state.velocities,
                state.time,
                dt,
            )
            nonconservative_force = nonconservative_force + bending_damping_force
        if body_contact_query is not None:
            contact = _validate_contact_sample(
                body_contact_query(positions.detach(), state.time), positions
            )
            gap = _contact_gap(positions, contact)
            normals = torch.nn.functional.normalize(contact.normals.to(positions), dim=-1)
            body_velocity = (
                torch.zeros_like(positions)
                if contact.body_velocities is None
                else contact.body_velocities.to(positions)
            )
            relative_normal_speed = ((state.velocities - body_velocity) * normals).sum(dim=-1)
            compression = torch.relu(self.config.contact_clearance_m - gap)
            contact_active = (gap < self.config.contact_clearance_m).to(positions.dtype)
            # The material cards store a modal contact damping rate. Applying
            # that scalar as an absolute nodal N*s/m coefficient made the
            # result depend on remeshing and produced four-digit accelerations
            # on light cloth vertices. Mass weighting gives F=-m*c*v, matching
            # the Rayleigh ports above and preserving the continuum limit.
            contact_damping_force = (
                self.config.contact_damping_n_s_m
                * mass
                * contact_active
                * torch.relu(-relative_normal_speed)
            )[:, None] * normals
            relative_velocity = state.velocities - body_velocity
            self._record_force_flow(
                "sdf_contact_normal_damping",
                "contact_dissipation",
                "cloth_momentum",
                GeneralizedFlowRole.DISSIPATIVE,
                contact_damping_force,
                relative_velocity,
                state.time,
                dt,
            )
            self._record_force_flow(
                "moving_body_contact_control",
                "body_se3_motion",
                "cloth_momentum",
                GeneralizedFlowRole.CONTROL,
                contact_damping_force,
                body_velocity,
                state.time,
                dt,
            )
            nonconservative_force = nonconservative_force + contact_damping_force

        raw_port_acceleration = nonconservative_force / mass[:, None]
        raw_acceleration_norm = torch.linalg.vector_norm(
            raw_port_acceleration, dim=-1
        )
        unclamped_maximum = float(raw_acceleration_norm.detach().max())
        regular_cap = (
            torch.linalg.vector_norm(raw_port_acceleration.detach(), dim=-1)
            > self.config.maximum_acceleration_m_s2
        )
        cap_hits = int(regular_cap.sum().item())
        port_acceleration = _clamp_vector_norm(
            raw_port_acceleration, self.config.maximum_acceleration_m_s2
        )
        self._record_force_flow(
            "acceleration_cap",
            "solver_feasible_set",
            "cloth_momentum",
            GeneralizedFlowRole.CONSTRAINT,
            mass[:, None] * (port_acceleration - raw_port_acceleration),
            state.velocities,
            state.time,
            dt,
        )

        gravity = positions.new_tensor(self.config.gravity_m_s2)
        shell = self.shell_geometry.to(positions)
        linearized_contact = None
        if body_contact_query is not None:
            contact_samples = reference_surface_contact_quadrature(
                positions, shell.faces
            )
            # Contact geometry is frozen during the local nonlinear solve. The
            # outer event/CCD stage re-queries the true SDF after q_(n+1), so
            # this is a differentiable local contact linearization, not a
            # substitute for final collision validation.
            linearized_contact = body_contact_query(
                contact_samples.detach(), state.time + dt
            )
        diagnostic_raw_acceleration = (
            shell_force
            + strain_limit_force
            + seam_elastic
            + retention_elastic
            + nonconservative_force
            + mass[:, None] * gravity
        ) / mass[:, None]
        diagnostic_norm = torch.linalg.vector_norm(
            diagnostic_raw_acceleration.detach(), dim=-1
        )
        unclamped_maximum = max(
            unclamped_maximum, float(diagnostic_norm.max())
        )
        cap_hits = max(
            cap_hits,
            int((diagnostic_norm > self.config.maximum_acceleration_m_s2).sum()),
        )

        def phase_potential(candidate: Tensor) -> PhasePotentialTerms:
            output = self.shell_model(
                candidate,
                self.shell_geometry,
                material_latent,
                compute_internal_forces=False,
                compute_bending_forces=False,
                create_graph=differentiable,
                maximum_tensile_strain=self.strain_limits.maximum_tensile_strain,
                maximum_compressive_strain=self.strain_limits.maximum_compressive_strain,
                strain_limit_onset_fraction=self.config.strain_limit_onset_fraction,
                strain_limit_barrier_scale=self.config.strain_limit_barrier_scale,
            )
            membrane = constitutive_force_scale * output.membrane_energy.sum()
            bending = (
                constitutive_force_scale
                * self.config.bending_force_scale
                * output.bending_energy.sum()
            )
            strain_limit = output.strain_limit_energy.sum()
            gravity_energy = -(mass[:, None] * gravity * candidate).sum()
            contact_energy = candidate.new_zeros(())
            contact_gap = None
            quadrature_points = 0
            if body_contact_query is not None:
                contact_energy, contact_gap, quadrature_points = (
                    reference_surface_contact_potential(
                        candidate,
                        shell.faces,
                        shell.rest_areas,
                        body_contact_query,
                        time=state.time + dt,
                        clearance_m=self.config.contact_clearance_m,
                        pressure_stiffness_n_m3=(
                            self.config.contact_pressure_stiffness_n_m3
                        ),
                        contact_sample=linearized_contact,
                    )
                )
            inter_garment_energy = candidate.new_zeros(())
            inter_garment_gap = None
            inter_garment_points = 0
            for (
                other_positions,
                clearance_m,
                stiffness_n_m,
                nearest_ids,
                active_mask,
            ) in inter_garment_contacts:
                layer_energy, layer_gap, layer_points = _sampled_layer_contact_potential(
                    candidate,
                    other_positions.to(candidate),
                    nearest_ids.to(candidate.device),
                    active_mask.to(candidate.device),
                    clearance_m=clearance_m,
                    stiffness_n_m=stiffness_n_m,
                )
                inter_garment_energy = inter_garment_energy + layer_energy
                inter_garment_gap = (
                    layer_gap
                    if inter_garment_gap is None
                    else torch.minimum(inter_garment_gap, layer_gap)
                )
                inter_garment_points += layer_points
            if inter_garment_gap is not None:
                contact_gap = (
                    inter_garment_gap
                    if contact_gap is None
                    else torch.minimum(contact_gap, inter_garment_gap)
                )
            quadrature_points += inter_garment_points
            se3_generative = candidate.new_zeros(())
            if self.config.se3_generative_potential_weight > 0.0:
                # A score proposal is a generalized SE(3) drive, not a
                # rest-position spring. The linear potential preserves the
                # identity behavior of a fresh (zero-score) head while its
                # gradient contributes the learned generalized force inside
                # the Hamiltonian solve.
                se3_generative = -self.config.se3_generative_potential_weight * (
                    mass[:, None] * se3_target_velocity * candidate
                ).sum() / float(dt)
            return PhasePotentialTerms(
                hypernode_membrane=membrane,
                hypernode_bending=bending,
                hypernode_strain_limit=strain_limit,
                gravity=gravity_energy,
                seam=self._seam_elastic_energy(candidate),
                trim=self._retention_band_energy(candidate),
                contact=contact_energy,
                contact_minimum_gap=contact_gap,
                contact_quadrature_points=quadrature_points,
                inter_garment_contact=inter_garment_energy,
                se3_generative=se3_generative,
            )

        # Imported image-to-shell states can carry a non-zero constitutive
        # envelope residual before the physical rollout starts.  A rigid
        # body-root SE(3) transport preserves every intrinsic edge/triangle
        # metric, so requiring an absolute zero violation here would reject
        # the transport itself and leave the garment behind the body.  The
        # emergency projector may use the measured baseline as a guard, but
        # the no-projection path must let the Hyper-NODE barrier enter the
        # variational objective instead of hard-rejecting its recovery step.
        baseline_limit_violation = float(
            self.hypernode_strain_limit_diagnostics(state.positions.detach())[
                "maximum_limit_violation"
            ]
        )
        allowed_limit_violation = baseline_limit_violation + 1.0e-3
        # The material envelope is the admissible domain of the Hyper-NODE
        # Hamiltonian, not a post-step metric projection. A line-search trial
        # may lower the finite penalty while creating one catastrophic short
        # edge, so reject any trial that worsens the hard constitutive domain.
        feasible_constraint = lambda candidate: float(
            self.hypernode_strain_limit_diagnostics(candidate)[
                "maximum_limit_violation"
            ]
        ) <= allowed_limit_violation

        hamiltonian_step = solve_incremental_hamiltonian_step(
            state.positions,
            state.velocities,
            mass,
            dt=dt,
            potential=phase_potential,
            nonconservative_force=mass[:, None] * port_acceleration,
            moving_reference_velocity=moving_reference_velocity,
            moving_reference_displacement=moving_reference_displacement,
            moving_reference_next_velocity=moving_reference_next_velocity,
            pinned_vertices=self.pinned_vertices,
            pinned_positions=state.positions[self.pinned_vertices],
            maximum_acceleration_m_s2=self.config.maximum_acceleration_m_s2,
            maximum_velocity_m_s=self.config.maximum_velocity_m_s,
            differentiable=differentiable,
            # This is a Hamiltonian-domain test, not a position correction.
            # Metric projection remains disabled in the physical rollout.
            feasible=feasible_constraint,
            # Do not let a force-augmented ballistic target seed a large
            # material excursion when the runtime projector is disabled.
            # Start from rigid body transport and solve gravity/contact work
            # through the same incremental action instead.
            prefer_transport_target=not self.config.enable_metric_projection,
            config=IncrementalHamiltonianConfig(
                iterations=self.config.phase_hamiltonian_iterations,
                line_search_steps=self.config.phase_hamiltonian_line_search_steps,
                gradient_tolerance_n=(
                    self.config.phase_hamiltonian_gradient_tolerance_n
                ),
                maximum_iteration_displacement_m=(
                    self.config.phase_hamiltonian_trust_radius_m
                ),
            ),
        )
        self._latest_phase_hamiltonian_diagnostics = (
            hamiltonian_step.diagnostics.to_dict()
        )
        positions = hamiltonian_step.positions
        velocity = hamiltonian_step.velocities
        acceleration = hamiltonian_step.acceleration
        if self.config.se3_generative_potential_weight > 0.0:
            generative_force = (
                self.config.se3_generative_potential_weight
                * mass[:, None]
                * se3_target_velocity
                / float(dt)
            )
            self._record_force_flow(
                "se3_riemannian_score_potential",
                "se3_configuration_manifold",
                "cloth_phase_state",
                GeneralizedFlowRole.CONSERVATIVE,
                generative_force,
                velocity,
                state.time + dt,
                dt,
            )
        unconstrained_velocity = state.velocities + dt * port_acceleration
        self._record_force_flow(
            "implicit_hypernode_hamiltonian_gradient",
            "bundle_measure_hamiltonian",
            "cloth_phase_state",
            GeneralizedFlowRole.CONSERVATIVE,
            mass[:, None] * (acceleration - port_acceleration),
            velocity,
            state.time + dt,
            dt,
        )
        self._record_momentum_projection(
            "phase_solver_numerical_trust_region",
            unconstrained_velocity,
            velocity,
            mass,
            state.time + dt,
            GeneralizedFlowRole.CONSTRAINT,
        )
        return (
            NeuralCloth4DState(positions, velocity, state.time + dt),
            acceleration,
            max(
                unclamped_maximum,
                float(
                    torch.linalg.vector_norm(
                        hamiltonian_step.acceleration.detach(), dim=-1
                    ).max()
                ),
            ),
            cap_hits,
            self._latest_phase_hamiltonian_diagnostics,
        )

    def _advance_fast_preview_candidate(
        self,
        state: NeuralCloth4DState,
        material_latent: Tensor,
        dt: float,
        body_contact_query: BodyContactQuery | None,
        differentiable: bool,
        constitutive_force_scale: float,
        external_force: Tensor | None = None,
        se3_target_velocity: Tensor | None = None,
        moving_reference_velocity: Tensor | None = None,
        moving_reference_displacement: Tensor | None = None,
        moving_reference_next_velocity: Tensor | None = None,
    ) -> tuple[NeuralCloth4DState, Tensor, float, int, dict[str, object]]:
        """Advance a cheap multi-rate preview substep between teacher solves.

        This path never projects the material metric. It integrates the learned
        E(3) residual and open-system ports in the moving body frame, then uses
        a scalar line search to reject a deformation that would worsen the
        Hyper-NODE hard strain domain. The shared SDF/contact stage remains
        authoritative after this proposal.
        """

        del differentiable, constitutive_force_scale, se3_target_velocity
        positions = state.positions
        mass = self._vertex_mass(positions)
        reference_velocity = (
            torch.zeros_like(positions)
            if moving_reference_velocity is None
            else moving_reference_velocity.to(positions)
        )
        reference_displacement = (
            torch.zeros_like(positions)
            if moving_reference_displacement is None
            else moving_reference_displacement.to(positions)
        )
        reference_next_velocity = (
            reference_velocity
            if moving_reference_next_velocity is None
            else moving_reference_next_velocity.to(positions)
        )
        force = (
            torch.zeros_like(positions)
            if external_force is None
            else external_force.to(positions)
        )
        relative_velocity = state.velocities - reference_velocity
        acceleration = positions.new_tensor(self.config.gravity_m_s2).expand_as(positions)
        acceleration = acceleration + force / mass[:, None]
        acceleration = acceleration - (
            self.config.body_inertial_force_scale
            * (reference_next_velocity - reference_velocity)
            / float(dt)
        )
        acceleration = acceleration - self.config.velocity_damping_per_second * relative_velocity

        if self.residual_model is not None and self.config.residual_force_scale > 0.0:
            with torch.no_grad():
                residual = self.residual_model(
                    positions,
                    relative_velocity,
                    self.edge_index,
                    self.rest_lengths,
                    self.cotangent_weights,
                    material_latent,
                ).internal_force
                residual = self._laplace_beltrami_diffuse(residual)
                residual_acceleration = self.config.residual_force_scale * residual / mass[:, None]
                acceleration = acceleration + _clamp_vector_norm(
                    residual_acceleration,
                    self.config.residual_acceleration_limit_m_s2,
                )

        if body_contact_query is not None:
            sample = _validate_contact_sample(
                body_contact_query(positions.detach(), state.time), positions
            )
            normals = torch.nn.functional.normalize(sample.normals.to(positions), dim=-1)
            gap = _contact_gap(positions, sample)
            body_velocity = (
                torch.zeros_like(positions)
                if sample.body_velocities is None
                else sample.body_velocities.to(positions)
            )
            relative_normal_speed = ((state.velocities - body_velocity) * normals).sum(-1)
            compression = torch.relu(self.config.contact_clearance_m - gap)
            contact_force = (
                self.config.contact_stiffness_n_m * compression
                + self.config.contact_damping_n_s_m
                * mass
                * torch.relu(-relative_normal_speed)
            )[:, None] * normals
            acceleration = acceleration + contact_force / mass[:, None]

        raw_maximum = float(torch.linalg.vector_norm(acceleration.detach(), dim=-1).max())
        cap_mask = (
            torch.linalg.vector_norm(acceleration.detach(), dim=-1)
            > self.config.maximum_acceleration_m_s2
        )
        cap_hits = int(cap_mask.sum())
        acceleration = _clamp_vector_norm(
            acceleration, self.config.maximum_acceleration_m_s2
        )
        next_relative_velocity = relative_velocity + float(dt) * acceleration
        next_relative_velocity = _clamp_vector_norm(
            next_relative_velocity, self.config.maximum_velocity_m_s
        )
        transport = positions + reference_displacement
        deformation = float(dt) * next_relative_velocity

        baseline = float(
            self.hypernode_strain_limit_diagnostics(positions)[
                "maximum_limit_violation"
            ]
        )
        accepted_scale = 0.0
        candidate_positions = transport
        for scale in (1.0, 0.5, 0.25, 0.125, 0.0625, 0.0):
            trial = transport + scale * deformation
            violation = float(
                self.hypernode_strain_limit_diagnostics(trial)[
                    "maximum_limit_violation"
                ]
            )
            if violation <= baseline + 1.0e-3:
                accepted_scale = scale
                candidate_positions = trial
                break
        candidate_velocity = (
            reference_next_velocity + accepted_scale * next_relative_velocity
        )
        if len(self.pinned_vertices):
            candidate_positions = candidate_positions.clone()
            candidate_velocity = candidate_velocity.clone()
            candidate_positions[self.pinned_vertices] = positions[self.pinned_vertices]
            candidate_velocity[self.pinned_vertices] = 0.0

        diagnostics: dict[str, object] = {
            "mode": "multirate_e3_preview",
            "iterations": 0,
            "accepted_deformation_scale": accepted_scale,
            "material_metric_projection": False,
            "full_hamiltonian_deferred": True,
        }
        self._latest_phase_hamiltonian_diagnostics = diagnostics
        return (
            NeuralCloth4DState(
                candidate_positions, candidate_velocity, state.time + dt
            ),
            acceleration,
            raw_maximum,
            cap_hits,
            diagnostics,
        )

    def _laplace_beltrami_diffuse(self, vector_field: Tensor) -> Tensor:
        """Propagate an E(3) vector field intrinsically with the surface heat kernel."""

        if self.laplace_eigenvalues.numel() == 0:
            return vector_field
        values = self.laplace_eigenvalues.to(vector_field)
        basis = self.laplace_eigenvectors.to(vector_field)
        mass = self.laplace_vertex_mass.to(vector_field)
        coefficients = basis.transpose(0, 1) @ (mass[:, None] * vector_field)
        heat = torch.exp(
            -values * self.config.laplace_beltrami_diffusion_time_m2
        )
        diffused = basis @ (heat[:, None] * coefficients)
        # Spectral truncation can leave a tiny net residual. Internal forces
        # must not accelerate the garment center of mass.
        return diffused - diffused.mean(dim=0, keepdim=True)

    def _laplace_beltrami_precondition_direction(
        self, vector_field: Tensor
    ) -> Tensor:
        """Apply a symmetric positive intrinsic heat-kernel preconditioner."""

        if self.laplace_eigenvalues.numel() == 0:
            return vector_field
        mean = vector_field.mean(dim=0, keepdim=True)
        centered = vector_field - mean
        values = self.laplace_eigenvalues.to(vector_field)
        basis = self.laplace_eigenvectors.to(vector_field)
        coefficients = basis.transpose(0, 1) @ centered
        heat = torch.exp(
            -values * self.config.laplace_beltrami_diffusion_time_m2
        )
        smooth = basis @ (heat[:, None] * coefficients)
        # C V exp(-tL) V^T C is Euclidean symmetric positive semidefinite,
        # where C removes rigid translation. Keep a small identity branch so
        # truncated spectra cannot erase a legitimate local contact mode.
        smooth = smooth - smooth.mean(dim=0, keepdim=True)
        return mean + 0.35 * centered + 0.65 * smooth

    def _calibrate_constitutive_force_scale(
        self,
        positions: Tensor,
        material_latent: Tensor,
        *,
        differentiable: bool,
    ) -> tuple[float, float]:
        """Fit one scene-wide modulus scale, then keep it fixed for the rollout."""

        if self.config.shell_force_scale == 0.0:
            return 0.0, 0.0
        force_positions = (
            positions if positions.requires_grad else positions.detach().requires_grad_(True)
        )
        output = self.shell_model(
            force_positions,
            self.shell_geometry,
            material_latent,
            compute_bending_forces=True,
            create_graph=differentiable,
            maximum_tensile_strain=self.strain_limits.maximum_tensile_strain,
            maximum_compressive_strain=self.strain_limits.maximum_compressive_strain,
            strain_limit_onset_fraction=self.config.strain_limit_onset_fraction,
            strain_limit_barrier_scale=self.config.strain_limit_barrier_scale,
        )
        mass = self._vertex_mass(positions)
        acceleration = (
            output.membrane_internal_force
            - output.strain_limit_internal_force
            + output.bending_internal_force
        ) / mass[:, None]
        norms = torch.linalg.vector_norm(acceleration.detach(), dim=-1)
        if len(self.pinned_vertices):
            free = torch.ones(len(norms), dtype=torch.bool, device=norms.device)
            free[self.pinned_vertices.to(norms.device)] = False
            if torch.any(free):
                norms = norms[free]
        p95 = float(torch.quantile(norms, 0.95)) if len(norms) else 0.0
        calibration = min(
            1.0,
            self.config.initial_shell_acceleration_target_m_s2 / max(p95, 1.0e-8),
        )
        calibrated_scale = self.config.shell_force_scale * calibration
        # A large force at the imported shell is an initialization defect, not
        # evidence that the measured fabric is one thousand times softer. Keep
        # a physical lower bound and let the bounded phase solver handle the
        # remaining transient after material relaxation.
        minimum_scale = (
            self.config.shell_force_scale
            * self.config.minimum_constitutive_force_scale
        )
        return max(calibrated_scale, minimum_scale), p95

    def _seam_force_components(
        self, positions: Tensor, velocities: Tensor
    ) -> tuple[Tensor, Tensor]:
        if not len(self.seam_pairs):
            zero = torch.zeros_like(positions)
            return zero, zero
        a, b = self.seam_pairs[:, 0], self.seam_pairs[:, 1]
        delta = positions[b] - positions[a]
        length = torch.linalg.vector_norm(delta, dim=-1).clamp_min(1.0e-8)
        direction = delta / length[:, None]
        rest_gap = torch.linalg.vector_norm(
            self.rest_positions[b] - self.rest_positions[a], dim=-1
        ).clamp(5.0e-4, 1.5e-3)
        relative_speed = ((velocities[b] - velocities[a]) * direction).sum(dim=-1)
        elastic_edge_force = (
            self.config.seam_stiffness_n_m * (length - rest_gap)
        )[:, None] * direction
        damping_edge_force = (
            self.config.seam_damping_n_s_m * relative_speed
        )[:, None] * direction
        elastic = torch.zeros_like(positions)
        damping = torch.zeros_like(positions)
        elastic.index_add_(0, a, elastic_edge_force)
        elastic.index_add_(0, b, -elastic_edge_force)
        damping.index_add_(0, a, damping_edge_force)
        damping.index_add_(0, b, -damping_edge_force)
        return elastic, damping

    def _retention_band_force_components(
        self, positions: Tensor, velocities: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Tensile-only intrinsic trim law for collars and waistbands."""

        if not len(self.retention_edges) or self.config.retention_band_axial_rigidity_n == 0.0:
            zero = torch.zeros_like(positions)
            return zero, zero
        edges = self.retention_edges.to(positions.device)
        a, b = edges[:, 0], edges[:, 1]
        delta = positions[b] - positions[a]
        length = torch.linalg.vector_norm(delta, dim=-1).clamp_min(1.0e-8)
        direction = delta / length[:, None]
        rest = self.retention_rest_lengths.to(positions).clamp_min(1.0e-8)
        strain = length / rest - 1.0
        active_strain = torch.relu(strain)
        tension = self.config.retention_band_axial_rigidity_n * active_strain
        relative_speed = ((velocities[b] - velocities[a]) * direction).sum(dim=-1)
        damping_tension = (
            self.config.retention_band_damping_n_s_m
            * (strain > 0.0).to(positions.dtype)
            * relative_speed
        )
        elastic_edge_force = tension[:, None] * direction
        damping_edge_force = damping_tension[:, None] * direction
        elastic = torch.zeros_like(positions)
        damping = torch.zeros_like(positions)
        elastic.index_add_(0, a, elastic_edge_force)
        elastic.index_add_(0, b, -elastic_edge_force)
        damping.index_add_(0, a, damping_edge_force)
        damping.index_add_(0, b, -damping_edge_force)
        return elastic, damping

    def _seam_elastic_energy(self, positions: Tensor) -> Tensor:
        if not len(self.seam_pairs):
            return positions.new_zeros(())
        a, b = self.seam_pairs[:, 0], self.seam_pairs[:, 1]
        length = torch.linalg.vector_norm(positions[b] - positions[a], dim=-1)
        rest_gap = torch.linalg.vector_norm(
            self.rest_positions[b] - self.rest_positions[a], dim=-1
        ).clamp(5.0e-4, 1.5e-3)
        extension = length - rest_gap
        return 0.5 * self.config.seam_stiffness_n_m * extension.square().sum()

    def _retention_band_energy(self, positions: Tensor) -> Tensor:
        if not len(self.retention_edges):
            return positions.new_zeros(())
        edges = self.retention_edges.to(positions.device)
        length = torch.linalg.vector_norm(
            positions[edges[:, 1]] - positions[edges[:, 0]], dim=-1
        )
        rest = self.retention_rest_lengths.to(positions).clamp_min(1.0e-8)
        strain = torch.relu(length / rest - 1.0)
        return (
            0.5
            * self.config.retention_band_axial_rigidity_n
            * rest
            * strain.square()
        ).sum()

    def bundle_measure_report(
        self,
        positions: Tensor,
        velocities: Tensor,
        material_latent: Tensor,
        *,
        physical_time: float,
        constitutive_scale: float,
        garment_id: str = "cloth",
        hierarchy_level: int = 2,
    ) -> dict[str, object]:
        """Audit the accepted mesh as a graded bundle-valued shell measure."""

        with torch.no_grad():
            evaluated_positions = positions.detach()
            evaluated_velocities = velocities.detach()
            evaluated_latent = material_latent.detach().to(evaluated_positions)
            shell_output = self.shell_model(
                evaluated_positions,
                self.shell_geometry,
                evaluated_latent,
                compute_bending_forces=False,
                create_graph=False,
                maximum_tensile_strain=self.strain_limits.maximum_tensile_strain,
                maximum_compressive_strain=self.strain_limits.maximum_compressive_strain,
                strain_limit_onset_fraction=self.config.strain_limit_onset_fraction,
                strain_limit_barrier_scale=self.config.strain_limit_barrier_scale,
            )
            state = self.bundle_measure_builder.lift(
                evaluated_positions,
                evaluated_velocities,
                evaluated_latent,
                time=physical_time,
                garment_id=garment_id,
                hierarchy_level=hierarchy_level,
                shell_output=shell_output,
            )
            reference_positions = self.rest_positions.to(evaluated_positions)
            reference = self.bundle_measure_builder.lift(
                reference_positions,
                torch.zeros_like(reference_positions),
                evaluated_latent,
                time=0.0,
                garment_id=garment_id,
                hierarchy_level=hierarchy_level,
            )
            hamiltonian = assemble_bundle_measure_hamiltonian(
                state,
                shell_output.membrane_energy + shell_output.strain_limit_energy,
                shell_output.bending_energy,
                constitutive_scale=constitutive_scale,
                gravity_m_s2=self.config.gravity_m_s2,
                seam_energy=(
                    self._seam_elastic_energy(evaluated_positions)
                    + self._retention_band_energy(evaluated_positions)
                ),
            )
            return self.bundle_measure_builder.report(
                state,
                hamiltonian,
                reference_state=reference,
            ).to_dict()

    def _vertex_mass(self, reference: Tensor) -> Tensor:
        shell = self.shell_geometry.to(reference)
        return self.config.areal_density_kg_m2 * shell.vertex_areas.clamp_min(1.0e-8)

    def _implicit_material_solve_required(self, positions: Tensor) -> bool:
        if len(self.retention_edges):
            return True
        diagnostics = self.hypernode_strain_limit_diagnostics(positions)
        return bool(
            diagnostics["maximum_tensile_strain"]
            > self.config.strain_limit_onset_fraction
            * self.strain_limits.maximum_tensile_strain
            or diagnostics["maximum_compressive_strain"]
            > self.config.strain_limit_onset_fraction
            * self.strain_limits.maximum_compressive_strain
        )

    def implicit_material_proximal_step(
        self,
        positions: Tensor,
        *,
        inertial_target: Tensor,
        material_latent: Tensor,
        dt: float,
        pinned_positions: Tensor,
        iterations: int | None = None,
    ) -> Tensor:
        """Minimize the incremental Hyper-NODE/trim potential without projection."""

        if dt <= 0.0 or inertial_target.shape != positions.shape:
            raise ValueError("implicit material step received an invalid target or dt")
        if pinned_positions.shape != (len(self.pinned_vertices), 3):
            raise ValueError("implicit material pinned positions have invalid shape")
        count = self.config.implicit_material_iterations if iterations is None else iterations
        if count <= 0:
            raise ValueError("implicit material iterations must be positive")
        candidate = positions
        target = inertial_target.detach()
        mass = self._vertex_mass(positions).detach()
        for _ in range(count):
            force_positions = candidate.detach().requires_grad_(True)
            output = self.shell_model(
                force_positions,
                self.shell_geometry,
                material_latent.detach(),
                compute_internal_forces=True,
                compute_bending_forces=True,
                create_graph=False,
                maximum_tensile_strain=self.strain_limits.maximum_tensile_strain,
                maximum_compressive_strain=self.strain_limits.maximum_compressive_strain,
                strain_limit_onset_fraction=self.config.strain_limit_onset_fraction,
                strain_limit_barrier_scale=self.config.strain_limit_barrier_scale,
            )
            band_force, _band_damping = self._retention_band_force_components(
                force_positions, torch.zeros_like(force_positions)
            )
            # Relax the imported shell against the same calibrated
            # Hyper-NODE/Kirchhoff-Love force that drives the physical clock.
            # Previously this warm-up only reacted to the hard strain barrier,
            # leaving large rest-state membrane forces to be mistaken for a
            # very soft fabric during constitutive calibration.
            constitutive_force = self.config.shell_force_scale * (
                output.membrane_internal_force
                - output.strain_limit_internal_force
                + self.config.bending_force_scale * output.bending_internal_force
            ) + output.strain_limit_internal_force
            material_force = (
                self.config.initial_shell_relaxation_force_scale
                * constitutive_force
                + band_force
            )
            proximal_target = target + (dt * dt) * material_force / mass[:, None]
            correction = _clamp_vector_norm(
                proximal_target.detach() - candidate,
                self.config.maximum_implicit_material_correction_m,
            )
            if float(
                torch.linalg.vector_norm(correction.detach(), dim=-1).max()
            ) < 1.0e-7:
                break
            candidate = candidate + self.config.implicit_material_relaxation * correction
            if len(self.pinned_vertices):
                candidate = candidate.clone()
                candidate[self.pinned_vertices] = pinned_positions
        return candidate

    def hypernode_strain_limit_diagnostics(
        self, positions: Tensor
    ) -> dict[str, float | int]:
        """Measure the same immutable edge metric used by the Hyper-NODE barrier."""

        evaluated = positions.detach()
        edges = self.metric_projector.edges.to(evaluated.device)
        rest = self.metric_projector.rest_lengths.to(evaluated).clamp_min(1.0e-10)
        stretches = torch.linalg.vector_norm(
            evaluated[edges[:, 1]] - evaluated[edges[:, 0]], dim=-1
        ) / rest
        tensile = torch.relu(stretches - 1.0)
        compressive = torch.relu(1.0 - stretches)
        violation = torch.maximum(
            tensile - self.strain_limits.maximum_tensile_strain,
            compressive - self.strain_limits.maximum_compressive_strain,
        ).clamp_min(0.0)
        tolerance = self.config.metric_projection_tolerance
        emergency_margin = max(
            tolerance, self.config.metric_projection_emergency_margin
        )
        return {
            "maximum_tensile_strain": float(tensile.max()) if tensile.numel() else 0.0,
            "maximum_compressive_strain": (
                float(compressive.max()) if compressive.numel() else 0.0
            ),
            "maximum_limit_violation": (
                float(violation.max()) if violation.numel() else 0.0
            ),
            "violating_components": int((violation > tolerance).sum().item()),
            "emergency_components": int(
                (violation > emergency_margin).sum().item()
            ),
            "component_count": int(violation.numel()),
        }

    def assert_within_hypernode_strain_limits(
        self, positions: Tensor
    ) -> dict[str, float | int]:
        diagnostics = self.hypernode_strain_limit_diagnostics(positions)
        if diagnostics["violating_components"]:
            raise RuntimeError(
                "Hyper-NODE intrinsic metric strain envelope remains violated: "
                f"{diagnostics}"
            )
        return diagnostics

    def _record_force_flow(
        self,
        relation_id: str,
        source: str,
        target: str,
        role: GeneralizedFlowRole,
        force: Tensor,
        velocity: Tensor,
        time: float,
        duration: float,
    ) -> None:
        self.generalized_flow_ledger.record_port(
            GeneralizedFlowPort(
                relation_id,
                source,
                target,
                role,
                GeneralizedFlowRepresentation.E3_VECTOR,
                force,
                velocity,
                time,
            ),
            duration_s=duration,
        )

    def _record_momentum_projection(
        self,
        relation_id: str,
        before_velocity: Tensor,
        after_velocity: Tensor,
        mass: Tensor,
        time: float,
        role: GeneralizedFlowRole,
    ) -> None:
        self.generalized_flow_ledger.record_impulse(
            GeneralizedImpulseTransfer(
                relation_id,
                "unconstrained_cloth_momentum",
                "cloth_momentum",
                role,
                GeneralizedFlowRepresentation.E3_VECTOR,
                mass[:, None] * (after_velocity - before_velocity),
                kinetic_energy_from_velocity(after_velocity, mass)
                - kinetic_energy_from_velocity(before_velocity, mass),
                time,
            )
        )

    def _record_contact_impulse(
        self,
        relation_prefix: str,
        before_velocity: Tensor,
        after_velocity: Tensor,
        body_velocity: Tensor,
        normals: Tensor,
        mass: Tensor,
        time: float,
    ) -> None:
        normal = torch.nn.functional.normalize(normals.to(before_velocity), dim=-1)
        relative_before = before_velocity - body_velocity.to(before_velocity)
        relative_after = after_velocity - body_velocity.to(after_velocity)
        impulse = mass[:, None] * (after_velocity - before_velocity)
        if not torch.any(impulse.detach().abs() > 1.0e-12):
            return
        before_normal_speed = (relative_before * normal).sum(dim=-1)
        after_normal_speed = (relative_after * normal).sum(dim=-1)
        before_tangent = relative_before - before_normal_speed[:, None] * normal
        after_tangent = relative_after - after_normal_speed[:, None] * normal
        normal_impulse = (impulse * normal).sum(dim=-1, keepdim=True) * normal
        tangent_impulse = impulse - normal_impulse
        normal_work = 0.5 * (
            mass * (after_normal_speed.square() - before_normal_speed.square())
        ).sum()
        tangent_work = 0.5 * (
            mass
            * (
                after_tangent.square().sum(dim=-1)
                - before_tangent.square().sum(dim=-1)
            )
        ).sum()
        tangent_u, tangent_v = tangent_frames_from_normals(normal)
        self.generalized_flow_ledger.record_impulse(
            GeneralizedImpulseTransfer(
                f"{relation_prefix}_normal_reset",
                "contact_constraint",
                "cloth_momentum",
                GeneralizedFlowRole.DISSIPATIVE,
                GeneralizedFlowRepresentation.E3_VECTOR,
                normal_impulse,
                normal_work,
                time,
            )
        )
        self.generalized_flow_ledger.record_impulse(
            GeneralizedImpulseTransfer(
                f"{relation_prefix}_so2_gauge_friction_reset",
                "contact_friction",
                "cloth_tangent_momentum",
                GeneralizedFlowRole.DISSIPATIVE,
                GeneralizedFlowRepresentation.SO2_COMPLEX,
                project_tangent_vector(tangent_impulse, tangent_u, tangent_v),
                tangent_work,
                time,
            )
        )

    def _resolve_contact(
        self,
        state: NeuralCloth4DState,
        query: BodyContactQuery,
        pinned_positions: Tensor,
    ) -> tuple[NeuralCloth4DState, float]:
        sample = _validate_contact_sample(query(state.positions.detach(), state.time), state.positions)
        normals = torch.nn.functional.normalize(sample.normals.to(state.positions), dim=-1)
        gap = _contact_gap(state.positions, sample)
        penetration = torch.relu(self.config.contact_clearance_m - gap)
        positions = state.positions + penetration[:, None] * normals
        body_velocity = (
            torch.zeros_like(state.velocities)
            if sample.body_velocities is None
            else sample.body_velocities.to(state.velocities)
        )
        relative = state.velocities - body_velocity
        active = penetration > 0.0
        contact_response = self.contact_model(
            relative,
            normals,
            penetration,
            active,
            self.edge_index,
            clearance_m=self.config.contact_clearance_m,
        )
        effective_friction = (
            self.config.friction_coefficient * contact_response.friction_scale
        ).clamp(0.0, 1.0)
        phase = IntrinsicClothPhaseState.from_velocity(
            self.shell_geometry.vertex_material_coordinates,
            positions,
            state.velocities,
            self._vertex_mass(positions),
            state.time,
        )
        reset = apply_sdf_contact_momentum_reset(
            phase,
            normals,
            body_velocity,
            active,
            effective_friction,
        )
        pre_reset_measure = self.bundle_measure_builder.vertex_measure(
            positions,
            state.velocities,
        )
        reset_measure = vertex_momentum_fibre_push_forward(
            pre_reset_measure,
            reset.state.velocities,
        )
        velocities = reset_measure.velocities
        self._record_contact_impulse(
            "sdf_contact",
            state.velocities,
            velocities,
            body_velocity,
            normals,
            reset.state.lumped_mass,
            state.time,
        )
        reset_velocity = velocities
        if len(self.pinned_vertices):
            positions = positions.clone()
            velocities = velocities.clone()
            positions[self.pinned_vertices] = pinned_positions
            velocities[self.pinned_vertices] = 0.0
            self._record_momentum_projection(
                "contact_attachment_projection",
                reset_velocity,
                velocities,
                self._vertex_mass(positions),
                state.time,
                GeneralizedFlowRole.CONSTRAINT,
            )
        # A vertex contact pass cannot see a limb passing through the middle
        # of a cloth triangle.  MovingBodyContactQuery exposes the optional
        # continuous surface pass; keep the generic BodyContactQuery contract
        # intact for lightweight tests and analytic callers.
        surface_projector = getattr(query, "project_surface", None)
        if callable(surface_projector):
            positions = surface_projector(
                positions,
                self.shell_geometry.faces,
                state.time,
                self.config.contact_clearance_m,
            )
            surface_sample = _validate_contact_sample(
                query(positions.detach(), state.time), positions
            )
            surface_normals = torch.nn.functional.normalize(
                surface_sample.normals.to(positions), dim=-1
            )
            surface_gap = _contact_gap(positions, surface_sample)
            surface_penetration = torch.relu(
                self.config.contact_clearance_m - surface_gap
            )
            positions = positions + surface_penetration[:, None] * surface_normals
            if len(self.pinned_vertices):
                positions = positions.clone()
                positions[self.pinned_vertices] = pinned_positions
        resolved_sample = _validate_contact_sample(query(positions.detach(), state.time), positions)
        resolved_gap = float(_contact_gap(positions, resolved_sample).min().detach())
        return NeuralCloth4DState(positions, velocities, state.time), resolved_gap


def _contact_gap(positions: Tensor, sample: BodyContactSample) -> Tensor:
    normals = torch.nn.functional.normalize(sample.normals.to(positions), dim=-1)
    return ((positions - sample.closest_points.to(positions)) * normals).sum(dim=-1)


def _validate_contact_sample(sample: BodyContactSample, reference: Tensor) -> BodyContactSample:
    if sample.closest_points.shape != reference.shape or sample.normals.shape != reference.shape:
        raise ValueError("contact closest points and normals must match cloth positions")
    if sample.body_velocities is not None and sample.body_velocities.shape != reference.shape:
        raise ValueError("body_velocities must match cloth positions")
    return sample


def _clamp_vector_norm(vectors: Tensor, maximum: float) -> Tensor:
    norm = torch.linalg.vector_norm(vectors, dim=-1, keepdim=True).clamp_min(1.0e-9)
    return vectors * torch.clamp(maximum / norm, max=1.0)


def _event_payload(event: object) -> dict[str, object]:
    return {
        "kind": event.kind.value,
        "event_id": event.event_id,
        "time": float(event.time),
        "bracket": [float(value) for value in event.bracket],
        "localized": bool(event.localized),
        "active": bool(event.active),
        "payload": dict(event.payload),
    }


def _aggregate_phase_hamiltonian_steps(
    steps: list[dict[str, object]],
) -> dict[str, object]:
    if not steps:
        return {
            "step_count": 0,
            "converged_fraction": 0.0,
            "maximum_gradient_n": 0.0,
            "contact_quadrature_points": 0,
            "last_step": {},
        }
    converged = sum(bool(step["converged"]) for step in steps)
    return {
        "solver": "implicit incremental action on T*Emb(S_cloth,R3)",
        "step_count": len(steps),
        "converged_fraction": converged / len(steps),
        "maximum_gradient_n": max(
            float(step["maximum_gradient_n"]) for step in steps
        ),
        "contact_quadrature_points": max(
            int(step["contact_quadrature_points"]) for step in steps
        ),
        "last_step": steps[-1],
    }


def _sampled_layer_contact_potential(
    positions: Tensor,
    other_positions: Tensor,
    nearest_ids: Tensor,
    active_mask: Tensor,
    *,
    clearance_m: float,
    stiffness_n_m: float,
) -> tuple[Tensor, Tensor, int]:
    """Penalty energy for one garment layer against the other layer.

    The scene-level contact projector remains responsible for the final
    non-penetration impulse. This nearest-sample penalty gives the incremental
    Hamiltonian a differentiable layer-interaction term before projection.
    """

    if positions.ndim != 2 or other_positions.ndim != 2:
        raise ValueError("layer contact positions must be rank-2 tensors")
    if positions.shape[1] != 3 or other_positions.shape[1] != 3:
        raise ValueError("layer contact positions must have shape (V,3)")
    if nearest_ids.shape != (len(positions),):
        raise ValueError("layer contact nearest ids must match the first layer")
    if active_mask.shape != (len(positions),):
        raise ValueError("layer contact active mask must match the first layer")
    if clearance_m <= 0.0 or stiffness_n_m <= 0.0:
        raise ValueError("layer contact clearance and stiffness must be positive")
    nearest_ids = nearest_ids.to(device=positions.device, dtype=torch.long)
    if torch.any(nearest_ids < 0) or torch.any(nearest_ids >= len(other_positions)):
        raise ValueError("layer contact nearest id lies outside the other garment")
    nearest = torch.linalg.vector_norm(
        positions - other_positions[nearest_ids].to(positions), dim=-1
    )
    compression = torch.relu(positions.new_tensor(clearance_m) - nearest)
    active_mask = active_mask.to(device=positions.device, dtype=torch.bool)
    compression = torch.where(active_mask, compression, torch.zeros_like(compression))
    # One nearest counterpart per vertex keeps the energy independent of AMR
    # sampling density when one garment has been refined more than the other.
    energy = 0.25 * float(stiffness_n_m) * compression.square().sum()
    active_nearest = nearest[active_mask]
    minimum_gap = (
        active_nearest.min()
        if active_nearest.numel()
        else positions.new_tensor(float("inf"))
    )
    return energy, minimum_gap, int((compression > 0.0).sum().item())
