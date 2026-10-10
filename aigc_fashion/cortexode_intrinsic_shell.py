from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, TYPE_CHECKING

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F

from .data import GarmentMesh, MaterialSpec
from .hypernetwork_kirchhoff_love import (
    HyperNetworkKirchhoffLoveShell,
    build_kirchhoff_love_shell_geometry,
)
from .physics_conditioned_flow import PhysicsConditionedCortexODE, SceneMaterialLatent
from .metric_projection import (
    DifferentiableMetricProjector,
    material_strain_limits,
)
from .topology_correction import (
    SurfaceDomain,
    TopologyCorrectionStage,
    TopologyDisposition,
    analyze_triangle_mesh,
)

if TYPE_CHECKING:
    from .body_mesh_avatar import BodyMeshAvatar


@dataclass(frozen=True)
class CortexODEIntrinsicShellConfig:
    optimization_steps: int = 120
    rk4_steps: int = 4
    learning_rate: float = 6.0e-3
    hidden_dim: int = 64
    hidden_layers: int = 3
    latent_dim: int = 16
    spectral_cap: float = 0.9
    velocity_limit_m: float = 0.12
    surface_weight: float = 1.0
    metric_weight: float = 0.12
    seam_weight: float = 0.18
    orientation_weight: float = 0.05
    latent_prior_weight: float = 1.0e-4
    shell_energy_weight: float = 0.025
    equilibrium_weight: float = 2.0e-4
    traction_weight: float = 2.0e-4
    contact_weight: float = 0.08
    genesis_teacher_weight: float = 0.12
    physics_interval: int = 4
    metric_projection_iterations: int = 12
    metric_projection_tolerance: float = 1.0e-3
    initial_smoothing_iterations: int = 18
    initial_smoothing_strength: float = 0.34
    seed: int = 97

    def __post_init__(self) -> None:
        integer_values = (
            self.optimization_steps,
            self.rk4_steps,
            self.hidden_dim,
            self.hidden_layers,
            self.latent_dim,
            self.physics_interval,
            self.metric_projection_iterations,
        )
        if any(value <= 0 for value in integer_values):
            raise ValueError("CortexODE dimensions and iteration counts must be positive")
        if self.initial_smoothing_iterations < 0:
            raise ValueError("initial_smoothing_iterations must be non-negative")
        if self.learning_rate <= 0.0 or self.velocity_limit_m <= 0.0:
            raise ValueError("learning_rate and velocity_limit_m must be positive")
        if self.metric_projection_tolerance < 0.0:
            raise ValueError("metric_projection_tolerance must be non-negative")
        if not 0.0 < self.initial_smoothing_strength < 1.0:
            raise ValueError("initial_smoothing_strength must lie in (0, 1)")
        weights = (
            self.surface_weight,
            self.metric_weight,
            self.seam_weight,
            self.orientation_weight,
            self.latent_prior_weight,
            self.shell_energy_weight,
            self.equilibrium_weight,
            self.traction_weight,
            self.contact_weight,
            self.genesis_teacher_weight,
        )
        if any(value < 0.0 for value in weights):
            raise ValueError("CortexODE loss weights must be non-negative")


@dataclass(frozen=True)
class CortexODEIntrinsicShellReport:
    method: str
    geometry_authority: str
    material_authority: str
    physics_supervision: tuple[str, ...]
    physics_limitations: tuple[str, ...]
    rest_metric_source: str
    seam_constraint_pairs: int
    material_strain_profile: str
    maximum_tensile_strain: float
    maximum_compressive_strain: float
    metric_projection: dict[str, float | int]
    initial_rmse_m: float
    final_rmse_m: float
    final_p95_error_m: float
    initial_to_final_displacement_p95_m: float
    edge_strain_p95: float
    edge_strain_max: float
    minimum_orientation_jacobian: float
    boundary_loops: int
    genus: int
    topology_preserved: bool
    lipschitz_upper_bound: float
    optimization_steps: int
    zero_crossing_events: tuple[dict[str, Any], ...]
    collision: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_coarse_intrinsic_initial_shell(
    target: GarmentMesh,
    *,
    iterations: int = 18,
    strength: float = 0.34,
) -> GarmentMesh:
    """Remove high-frequency shape detail while retaining the garment graph.

    This is the open-garment counterpart of CortexODE's genus-corrected initial
    cortical surface. Faces, boundary loops, UVs, and semantic labels are kept;
    only embedding coordinates are regularized before the neural flow.
    """

    if iterations < 0 or not 0.0 < strength < 1.0:
        raise ValueError("invalid intrinsic-shell smoothing configuration")
    vertices = np.asarray(target.vertices, dtype=np.float64).copy()
    neighbors = _vertex_neighbors(len(vertices), target.faces)
    fixed = np.zeros(len(vertices), dtype=bool)
    fixed[np.asarray(target.pinned_vertices, dtype=np.int64)] = True
    original_fixed = vertices[fixed].copy()
    for iteration in range(iterations):
        means = vertices.copy()
        for vertex_id, adjacent in enumerate(neighbors):
            if adjacent:
                means[vertex_id] = vertices[np.fromiter(adjacent, dtype=np.int64)].mean(axis=0)
        # Alternating Taubin-like passes attenuate folds without collapsing the shell.
        coefficient = strength if iteration % 2 == 0 else -0.36 * strength
        vertices += coefficient * (means - vertices)
        vertices[fixed] = original_fixed
    return _mesh_with_vertices(
        target,
        vertices.astype(np.float32),
        rest_vertices=target.material_vertices,
    )


def reconstruct_intrinsic_shell_cortexode(
    target: GarmentMesh,
    *,
    avatar: BodyMeshAvatar | None = None,
    initial: GarmentMesh | None = None,
    volume_evidence: Any | None = None,
    genesis_teacher_trajectory: np.ndarray | None = None,
    material: MaterialSpec | None = None,
    config: CortexODEIntrinsicShellConfig = CortexODEIntrinsicShellConfig(),
) -> tuple[GarmentMesh, CortexODEIntrinsicShellReport, np.ndarray]:
    """Optimize one fixed-topology CortexODE flow for one garment scene.

    ``target`` is surface evidence, not a manufacturing pattern. In the current
    prototype it comes from the image/category/body prior; a multiview diffusion
    encoder can replace that producer without changing this shell optimizer.
    """

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    initial = initial or build_coarse_intrinsic_initial_shell(
        target,
        iterations=config.initial_smoothing_iterations,
        strength=config.initial_smoothing_strength,
    )
    if not np.array_equal(initial.faces, target.faces):
        raise ValueError("CortexODE initial and target shells must share face connectivity")
    if initial.vertices.shape != target.vertices.shape:
        raise ValueError("CortexODE initial and target shells must share vertices")

    gate = TopologyCorrectionStage()
    initial_gate = gate.validate_extracted_mesh(
        initial.vertices,
        initial.faces,
        domain=SurfaceDomain.GARMENT_INTRINSIC_SHELL,
    )
    target_gate = gate.validate_extracted_mesh(
        target.vertices,
        target.faces,
        domain=SurfaceDomain.GARMENT_INTRINSIC_SHELL,
    )
    if initial_gate.disposition is not TopologyDisposition.PRESERVE_SHELL_TOPOLOGY:
        raise ValueError(initial_gate.reason)
    if target_gate.disposition is not TopologyDisposition.PRESERVE_SHELL_TOPOLOGY:
        raise ValueError(target_gate.reason)
    if initial_gate.report.boundary_loops != target_gate.report.boundary_loops:
        raise ValueError("CortexODE must preserve the garment boundary-loop count")

    x0 = torch.as_tensor(initial.vertices, dtype=torch.float32)
    target_x = torch.as_tensor(target.vertices, dtype=torch.float32)
    rest_x = torch.as_tensor(target.material_vertices, dtype=torch.float32)
    faces = torch.as_tensor(target.faces, dtype=torch.long)
    coordinates = torch.as_tensor(_material_coordinates(target), dtype=torch.float32)
    edges = torch.as_tensor(_unique_edges(target.faces), dtype=torch.long)
    seam_pairs = torch.as_tensor(target.seam_edges, dtype=torch.long).reshape(-1, 2)
    material = material or MaterialSpec()
    strain_limits = material_strain_limits(material)
    metric_projector = DifferentiableMetricProjector(
        edges,
        rest_x,
        torch.as_tensor(target.pinned_vertices, dtype=torch.long),
        strain_limits,
        iterations=config.metric_projection_iterations,
        tolerance=config.metric_projection_tolerance,
    )
    projection_pins = x0[torch.as_tensor(target.pinned_vertices, dtype=torch.long)].detach()
    volume_sampler = None
    if volume_evidence is not None:
        from .cortexode_volume_pipeline import MultiscaleSDFSampler

        volume_sampler = MultiscaleSDFSampler(volume_evidence)
        body_context = volume_sampler(x0).detach()
    else:
        body_context = _surface_evidence_context(x0, target_x, faces)
    panel_context = torch.as_tensor(_panel_context(target), dtype=torch.float32)

    flow = PhysicsConditionedCortexODE(
        latent_dim=config.latent_dim,
        hidden_dim=config.hidden_dim,
        hidden_layers=config.hidden_layers,
        spectral_cap=config.spectral_cap,
        velocity_limit=config.velocity_limit_m,
    )
    latent = SceneMaterialLatent(config.latent_dim)
    panel_ids = _panel_ids(target)
    shell_geometry = build_kirchhoff_love_shell_geometry(
        coordinates,
        faces,
        panel_ids,
        seam_pairs=seam_pairs,
        reference_positions=rest_x,
    )
    shell_model = HyperNetworkKirchhoffLoveShell(
        int(panel_ids.max().item()) + 1,
        latent_dim=config.latent_dim,
        hyper_hidden_dim=24,
        node_hidden_dim=4,
        node_steps=3,
        spectral_cap=config.spectral_cap,
    )
    optimizer = torch.optim.Adam(
        (*flow.parameters(), *latent.parameters(), *shell_model.parameters()),
        lr=config.learning_rate,
    )
    rest_edge_lengths = torch.linalg.vector_norm(
        rest_x[edges[:, 0]] - rest_x[edges[:, 1]], dim=-1
    ).clamp_min(1.0e-6)
    target_cross = torch.cross(
        target_x[faces[:, 1]] - target_x[faces[:, 0]],
        target_x[faces[:, 2]] - target_x[faces[:, 0]],
        dim=-1,
    )
    target_area2 = target_cross.square().sum(dim=-1).clamp_min(1.0e-12)

    contact_closest = contact_normals = None
    teacher = None
    if genesis_teacher_trajectory is not None:
        teacher = torch.as_tensor(genesis_teacher_trajectory, dtype=torch.float32)
        if teacher.ndim != 3 or teacher.shape[1:] != x0.shape:
            raise ValueError("Genesis teacher trajectory must have shape (T, V, 3)")

    for optimization_step in range(config.optimization_steps):
        optimizer.zero_grad(set_to_none=True)
        reconstructed, ode_trajectory = integrate_cortexode_rk4(
            flow,
            x0,
            material_coordinates=coordinates,
            material_latent=latent(),
            body_context=body_context,
            panel_context=panel_context,
            steps=config.rk4_steps,
            volume_sampler=volume_sampler,
            return_trajectory=True,
        )
        # Work in centimetre-normalized coordinates so the metric and surface
        # terms have comparable gradient scales while geometry remains in SI.
        surface_loss = F.mse_loss(reconstructed, target_x) / (0.01**2)
        current_lengths = torch.linalg.vector_norm(
            reconstructed[edges[:, 0]] - reconstructed[edges[:, 1]], dim=-1
        )
        metric_loss = ((current_lengths / rest_edge_lengths - 1.0) ** 2).mean()
        seam_loss = reconstructed.new_zeros(())
        if len(seam_pairs):
            seam_gap = reconstructed[seam_pairs[:, 0]] - reconstructed[seam_pairs[:, 1]]
            seam_loss = seam_gap.square().sum(dim=-1).mean() / (0.002**2)
        current_cross = torch.cross(
            reconstructed[faces[:, 1]] - reconstructed[faces[:, 0]],
            reconstructed[faces[:, 2]] - reconstructed[faces[:, 0]],
            dim=-1,
        )
        orientation = (current_cross * target_cross).sum(dim=-1) / target_area2
        orientation_loss = F.relu(0.12 - orientation).square().mean()
        shell_output = shell_model(
            reconstructed,
            shell_geometry,
            latent(),
            compute_bending_forces=(optimization_step % config.physics_interval == 0),
            create_graph=True,
            maximum_tensile_strain=strain_limits.maximum_tensile_strain,
            maximum_compressive_strain=strain_limits.maximum_compressive_strain,
        )
        reference_area = shell_geometry.rest_areas.to(reconstructed).sum().clamp_min(1.0e-8)
        shell_energy = torch.log1p(
            (
                shell_output.membrane_energy.sum()
                + shell_output.strain_limit_energy.sum()
                + shell_output.bending_energy.sum()
            )
            / reference_area
            / 100.0
        )
        traction_loss = (
            shell_output.free_boundary_traction.square().mean()
            if len(shell_output.free_boundary_traction)
            else reconstructed.new_zeros(())
        )
        equilibrium_loss = reconstructed.new_zeros(())
        if optimization_step % config.physics_interval == 0:
            shell = shell_geometry.to(reconstructed)
            gravity_force = (
                0.34
                * shell.vertex_areas[:, None]
                * reconstructed.new_tensor((0.0, -9.81, 0.0))
            )
            residual = shell_output.membrane_internal_force + shell_output.bending_internal_force + gravity_force
            active = torch.ones(len(reconstructed), dtype=torch.bool)
            active[torch.as_tensor(target.pinned_vertices, dtype=torch.long)] = False
            equilibrium_loss = torch.log1p((residual[active] / 2.0e-3).square()).mean()
        contact_loss = reconstructed.new_zeros(())
        if avatar is not None and (
            contact_closest is None or optimization_step % config.physics_interval == 0
        ):
            closest_np, _distance, triangle_ids = avatar._proximity.on_surface(
                reconstructed.detach().cpu().numpy()
            )
            contact_closest = torch.as_tensor(closest_np, dtype=reconstructed.dtype)
            contact_normals = torch.as_tensor(
                avatar._collision_mesh.face_normals[triangle_ids],
                dtype=reconstructed.dtype,
            )
        if contact_closest is not None and contact_normals is not None:
            gap = ((reconstructed - contact_closest) * contact_normals).sum(dim=-1)
            contact_loss = F.relu(0.003 - gap).square().mean() / (0.003**2)
        teacher_loss = reconstructed.new_zeros(())
        if teacher is not None:
            indices = torch.linspace(0, len(teacher) - 1, len(ode_trajectory)).round().long()
            teacher_samples = teacher[indices]
            teacher_loss = F.mse_loss(torch.stack(ode_trajectory), teacher_samples) / (0.01**2)
        loss = (
            config.surface_weight * surface_loss
            + config.metric_weight * metric_loss
            + config.seam_weight * seam_loss
            + config.orientation_weight * orientation_loss
            + config.latent_prior_weight * latent.prior_loss()
            + config.shell_energy_weight * shell_energy
            + config.equilibrium_weight * equilibrium_loss
            + config.traction_weight * traction_loss
            + config.contact_weight * contact_loss
            + config.genesis_teacher_weight * teacher_loss
        )
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        final, trajectory = integrate_cortexode_rk4(
            flow,
            x0,
            material_coordinates=coordinates,
            material_latent=latent(),
            body_context=body_context,
            panel_context=panel_context,
            steps=max(8, config.rk4_steps),
            volume_sampler=volume_sampler,
            return_trajectory=True,
        )
        recovery_required = bool(
            metric_projector.diagnostics(final).violating_edges
        )
        if recovery_required:
            final = metric_projector(
                final,
                pinned_positions=projection_pins,
            )
            trajectory[-1] = final
    projection_diagnostics = metric_projector.assert_within_limits(final)
    final_np = final.cpu().numpy().astype(np.float32)
    trajectory_np = np.stack([state.cpu().numpy() for state in trajectory]).astype(np.float32)
    # CortexODE owns only the current embedding. Manufacturing rest metric,
    # panel topology, seams and UV coordinates remain construction authority.
    output = _mesh_with_vertices(
        target,
        final_np,
        rest_vertices=target.material_vertices,
    )
    from .construction_hypotheses import validate_cortexode_refinement_authority

    authority = validate_cortexode_refinement_authority(target, output)
    if not authority.passed:
        raise RuntimeError("CortexODE refinement violated construction authority: " + "; ".join(authority.reasons))
    final_topology = analyze_triangle_mesh(output.vertices, output.faces)
    errors = np.linalg.norm(final_np - target.vertices, axis=1)
    initial_errors = np.linalg.norm(initial.vertices - target.vertices, axis=1)
    rest_lengths_np = np.linalg.norm(
        target.material_vertices[edges[:, 0].numpy()]
        - target.material_vertices[edges[:, 1].numpy()],
        axis=1,
    )
    final_lengths_np = np.linalg.norm(
        final_np[edges[:, 0].numpy()] - final_np[edges[:, 1].numpy()], axis=1
    )
    edge_strain = np.abs(final_lengths_np / np.maximum(rest_lengths_np, 1.0e-8) - 1.0)
    minimum_orientation = _minimum_orientation_jacobian(final_np, target.vertices, target.faces)
    collision = (
        avatar.collision_diagnostics_np(final_np, target.faces)
        if avatar is not None
        else {}
    )
    events = _shell_zero_crossing_events(trajectory_np, target, avatar)
    topology_preserved = (
        final_topology.connected_components == initial_gate.report.connected_components
        and final_topology.boundary_loops == initial_gate.report.boundary_loops
        and final_topology.genus == initial_gate.report.genus
        and final_topology.nonmanifold_edges == 0
    )
    report = CortexODEIntrinsicShellReport(
        method="per-scene latent-conditioned CortexODE intrinsic-shell reconstruction",
        geometry_authority="image/category/body surface evidence on a fixed garment topology",
        material_authority="immutable construction rest metric and panel chart",
        physics_supervision=(
            "Kirchhoff-Love membrane and rest-curvature energy",
            "explicit seam-pair closure and stitched-boundary classification",
            "quasi-static nodal equilibrium under gravity",
            "free-boundary Piola traction",
            "periodically refreshed body signed-gap contact",
            "Hyper-NODE intrinsic rest-edge barrier in every physics loss",
            "one final metric recovery only if the learned flow exits the feasible envelope",
            "Genesis FEM/MPM trajectory distillation" if teacher is not None else "Genesis teacher hook present but no trajectory supplied",
        ),
        physics_limitations=(
            "no cloth self-collision or multi-layer collision loss in this optimizer",
            "no continuous collision detection inside an ODE step",
            "zero-crossing events are diagnosed after rollout; solver restart is not integrated",
            "material parameters remain latent priors unless calibrated or teacher-supervised",
        ),
        rest_metric_source="GarmentMesh.material_vertices (construction authority)",
        seam_constraint_pairs=int(len(seam_pairs)),
        material_strain_profile=strain_limits.profile,
        maximum_tensile_strain=strain_limits.maximum_tensile_strain,
        maximum_compressive_strain=strain_limits.maximum_compressive_strain,
        metric_projection=projection_diagnostics.to_dict(),
        initial_rmse_m=float(np.sqrt(np.mean(initial_errors**2))),
        final_rmse_m=float(np.sqrt(np.mean(errors**2))),
        final_p95_error_m=float(np.percentile(errors, 95.0)),
        initial_to_final_displacement_p95_m=float(
            np.percentile(np.linalg.norm(final_np - initial.vertices, axis=1), 95.0)
        ),
        edge_strain_p95=float(np.percentile(edge_strain, 95.0)),
        edge_strain_max=float(edge_strain.max(initial=0.0)),
        minimum_orientation_jacobian=minimum_orientation,
        boundary_loops=int(final_topology.boundary_loops or 0),
        genus=int(final_topology.genus if final_topology.genus is not None else -1),
        topology_preserved=topology_preserved,
        lipschitz_upper_bound=flow.state_lipschitz_upper_bound(),
        optimization_steps=config.optimization_steps,
        zero_crossing_events=tuple(events),
        collision={key: float(value) for key, value in collision.items()},
    )
    return output, report, trajectory_np


def integrate_cortexode_rk4(
    flow: PhysicsConditionedCortexODE,
    initial_state: Tensor,
    *,
    material_coordinates: Tensor,
    material_latent: Tensor,
    body_context: Tensor | None,
    panel_context: Tensor,
    steps: int,
    volume_sampler: torch.nn.Module | None = None,
    metric_projector: DifferentiableMetricProjector | None = None,
    projection_pinned_positions: Tensor | None = None,
    return_trajectory: bool = False,
) -> Tensor | tuple[Tensor, list[Tensor]]:
    """Differentiable fixed-step RK4 for canonical dressing pseudo-time [0, 1]."""

    if steps <= 0:
        raise ValueError("RK4 steps must be positive")
    state = initial_state
    trajectory = [state]
    dt = 1.0 / steps

    def velocity(value: Tensor, tau: float) -> Tensor:
        local_context = volume_sampler(value) if volume_sampler is not None else body_context
        return flow(
            value,
            tau,
            material_coordinates=material_coordinates,
            material_latent=material_latent,
            body_context=local_context,
            panel_context=panel_context,
        )

    for step in range(steps):
        tau = step * dt
        k1 = velocity(state, tau)
        k2 = velocity(state + 0.5 * dt * k1, tau + 0.5 * dt)
        k3 = velocity(state + 0.5 * dt * k2, tau + 0.5 * dt)
        k4 = velocity(state + dt * k3, tau + dt)
        state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        if metric_projector is not None:
            state = metric_projector(
                state,
                pinned_positions=projection_pinned_positions,
            )
            metric_projector.assert_within_limits(state)
        trajectory.append(state)
    if return_trajectory:
        return state, trajectory
    return state


def _surface_evidence_context(initial: Tensor, target: Tensor, faces: Tensor) -> Tensor:
    normals = torch.zeros_like(target)
    face_normals = torch.cross(
        target[faces[:, 1]] - target[faces[:, 0]],
        target[faces[:, 2]] - target[faces[:, 0]],
        dim=-1,
    )
    for corner in range(3):
        normals.index_add_(0, faces[:, corner], face_normals)
    normals = F.normalize(normals, dim=-1, eps=1.0e-8)
    displacement = target - initial
    magnitude = torch.linalg.vector_norm(displacement, dim=-1, keepdim=True)
    displacement_direction = F.normalize(displacement, dim=-1, eps=1.0e-8)
    # Match the typed CortexODE context contract: two (scalar, polar-vector)
    # fields, analogous to the fine/coarse (SDF, grad SDF) volume samples.
    return torch.cat(
        (
            magnitude,
            normals,
            magnitude,
            displacement_direction,
        ),
        dim=-1,
    )


def _material_coordinates(mesh: GarmentMesh) -> np.ndarray:
    if mesh.uvs is not None and np.asarray(mesh.uvs).shape == (len(mesh.vertices), 2):
        return np.asarray(mesh.uvs, dtype=np.float32)
    centered = np.asarray(mesh.vertices, dtype=np.float64) - np.mean(mesh.vertices, axis=0)
    _, _, basis = np.linalg.svd(centered, full_matrices=False)
    coordinates = centered @ basis[:2].T
    minimum = coordinates.min(axis=0)
    span = np.maximum(coordinates.max(axis=0) - minimum, 1.0e-8)
    return ((coordinates - minimum) / span).astype(np.float32)


def _panel_context(mesh: GarmentMesh) -> np.ndarray:
    boundary = np.zeros(len(mesh.vertices), dtype=np.float32)
    edge_counts: dict[tuple[int, int], int] = {}
    for face in mesh.faces:
        for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            edge = tuple(sorted((int(a), int(b))))
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
    for edge, count in edge_counts.items():
        if count == 1:
            boundary[list(edge)] = 1.0
    context = np.zeros((len(mesh.vertices), 4), dtype=np.float32)
    context[:, 0] = np.asarray([label.startswith("front_") for label in mesh.panel_labels])
    context[:, 1] = np.asarray(["sleeve" in label for label in mesh.panel_labels])
    context[:, 2] = boundary
    context[:, 3] = 1.0
    return context


def _panel_ids(mesh: GarmentMesh) -> Tensor:
    # A welded intrinsic shell has already compiled sewing links into its face
    # graph. Keep one physical chart so cross-link faces and hinges are not
    # discarded merely because their semantic labels differ.
    if not mesh.seam_edges:
        return torch.zeros(len(mesh.vertices), dtype=torch.long)
    names = {name: index for index, name in enumerate(sorted(set(mesh.panel_labels)))}
    return torch.as_tensor([names[name] for name in mesh.panel_labels], dtype=torch.long)


def _unique_edges(faces: np.ndarray) -> np.ndarray:
    edges: set[tuple[int, int]] = set()
    for face in np.asarray(faces):
        for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            edges.add(tuple(sorted((int(a), int(b)))))
    return np.asarray(sorted(edges), dtype=np.int64)


def _vertex_neighbors(count: int, faces: np.ndarray) -> list[set[int]]:
    neighbors = [set() for _ in range(count)]
    for a, b, c in np.asarray(faces, dtype=np.int64):
        neighbors[int(a)].update((int(b), int(c)))
        neighbors[int(b)].update((int(a), int(c)))
        neighbors[int(c)].update((int(a), int(b)))
    return neighbors


def _minimum_orientation_jacobian(
    vertices: np.ndarray, target: np.ndarray, faces: np.ndarray
) -> float:
    current = np.cross(
        vertices[faces[:, 1]] - vertices[faces[:, 0]],
        vertices[faces[:, 2]] - vertices[faces[:, 0]],
    )
    reference = np.cross(
        target[faces[:, 1]] - target[faces[:, 0]],
        target[faces[:, 2]] - target[faces[:, 0]],
    )
    denominator = np.maximum(np.sum(reference * reference, axis=1), 1.0e-12)
    return float(np.min(np.sum(current * reference, axis=1) / denominator))


def _shell_zero_crossing_events(
    trajectory: np.ndarray,
    target: GarmentMesh,
    avatar: BodyMeshAvatar | None,
) -> list[dict[str, Any]]:
    signals: dict[str, list[float]] = {
        "orientation_jacobian": [
            _minimum_orientation_jacobian(frame, target.vertices, target.faces)
            for frame in trajectory
        ]
    }
    if avatar is not None:
        # Trimesh uses positive signed distance inside this watertight avatar.
        signals["body_sdf_clearance"] = [
            float(np.min(-(avatar._proximity.signed_distance(frame) + avatar.clearance)))
            for frame in trajectory
        ]
    events: list[dict[str, Any]] = []
    times = np.linspace(0.0, 1.0, len(trajectory))
    for signal_name, values in signals.items():
        for index in range(1, len(values)):
            first, second = values[index - 1], values[index]
            if (first > 0.0) == (second > 0.0):
                continue
            fraction = abs(first) / max(abs(first) + abs(second), 1.0e-12)
            root = times[index - 1] + fraction * (times[index] - times[index - 1])
            events.append(
                {
                    "signal": signal_name,
                    "tau": float(root),
                    "bracket": [float(times[index - 1]), float(times[index])],
                    "values": [float(first), float(second)],
                    "requires_solver_restart": True,
                }
            )
    return events


def _mesh_with_vertices(
    mesh: GarmentMesh,
    vertices: np.ndarray,
    *,
    rest_vertices: np.ndarray,
) -> GarmentMesh:
    return GarmentMesh(
        vertices=np.asarray(vertices, dtype=np.float32).copy(),
        faces=mesh.faces.copy(),
        pinned_vertices=mesh.pinned_vertices.copy(),
        panel_labels=list(mesh.panel_labels),
        seam_edges=list(mesh.seam_edges),
        rest_vertices=np.asarray(rest_vertices, dtype=np.float32).copy(),
        uvs=None if mesh.uvs is None else mesh.uvs.copy(),
        seam_interfaces=list(mesh.seam_interfaces),
    )
