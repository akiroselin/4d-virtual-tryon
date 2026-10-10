from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import torch

OUTPUTS_ROOT = Path(__file__).resolve().parents[2]
GENESIS_GEOMETRIC = OUTPUTS_ROOT / "genesis_geometric"
if GENESIS_GEOMETRIC.exists():
    sys.path.insert(0, str(GENESIS_GEOMETRIC))

try:
    from genesis_geometric.soft_body import cotangent_laplacian, lumped_vertex_areas, normalized_graph_laplacian  # noqa: E402
except ImportError:
    from .mesh_operators import cotangent_laplacian, lumped_vertex_areas, normalized_graph_laplacian  # noqa: E402

from .avatar import AvatarCollisionReport, SmplAvatarProxy
from .data import GarmentMesh, MaterialSpec


_SEAM_COLLISION_ACTIVATION_M = 0.025


@dataclass(frozen=True)
class SimulationConfig:
    steps: int = 96
    dt: float = 0.008
    frame_interval: int = 2
    learned_scale: float = 0.0
    hidden_dim: int = 64
    edge_projection_iterations: int = 96
    stretch_tolerance: float = 0.012
    bend_smoothing: float = 0.018
    rest_shape_projection: float = 0.0
    gravity_y: float = -9.81
    max_velocity: float = 1.0
    attachment_mode: str = "fixed"
    attachment_release_step: int | None = None
    attachment_halfspace_y: float | None = None


class IntrinsicClothSimulator(torch.nn.Module):
    """Metric-aware cloth predictor; constraints are resolved after prediction."""

    def __init__(
        self,
        material: MaterialSpec,
        material_rest: torch.Tensor,
        faces: torch.Tensor,
        learned_scale: float = 0.0,
    ):
        super().__init__()
        if learned_scale != 0.0:
            raise ValueError("learned_scale requires a trained checkpoint; random residuals are disabled")
        self.material = material
        self.register_buffer("material_rest", material_rest.clone())
        self.register_buffer("faces", faces.clone())
        self.register_buffer(
            "rest_curvature",
            cotangent_laplacian(material_rest, material_rest, faces).detach(),
        )

    def forward(
        self,
        positions: torch.Tensor,
        velocity: torch.Tensor,
        pinned: torch.Tensor,
        pin_positions: torch.Tensor,
        dt: float,
        gravity_y: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # A curved garment rest surface (pressed pleats, molded cups, collars)
        # must be force-free. Acting on absolute curvature incorrectly irons it
        # flat; only curvature change is a bending strain in this approximation.
        curvature = (
            cotangent_laplacian(positions, self.material_rest, self.faces)
            - self.rest_curvature
        )
        gravity = torch.tensor([0.0, gravity_y, 0.0], dtype=positions.dtype, device=positions.device)
        density_scale = max(float(self.material.density), 0.05)
        acceleration = gravity + (7.5 * self.material.bending_stiffness / density_scale) * curvature
        damping = float(self.material.damping) ** (dt / 0.008)
        next_velocity = damping * velocity + dt * acceleration
        next_positions = positions + dt * next_velocity
        next_velocity[pinned] = 0.0
        next_positions[pinned] = pin_positions
        return next_positions, next_velocity


def _attachment_is_active(config: SimulationConfig, step: int) -> bool:
    return config.attachment_release_step is None or step < config.attachment_release_step


def _apply_attachment_constraints(
    positions: torch.Tensor,
    vertex_ids: torch.Tensor,
    fixed_positions: torch.Tensor,
    config: SimulationConfig,
    *,
    step: int,
) -> torch.Tensor:
    if not len(vertex_ids) or not _attachment_is_active(config, step):
        return positions
    result = positions.clone()
    if config.attachment_mode == "fixed":
        result[vertex_ids] = fixed_positions
    else:
        result[vertex_ids, 1] = torch.clamp_min(
            result[vertex_ids, 1],
            float(config.attachment_halfspace_y),
        )
    return result


def simulate_intrinsic_cloth(
    mesh: GarmentMesh,
    material: MaterialSpec,
    config: SimulationConfig,
    avatar: SmplAvatarProxy | None = None,
) -> tuple[list[np.ndarray], np.ndarray, dict[str, float]]:
    if config.steps <= 0 or config.frame_interval <= 0 or config.dt <= 0.0:
        raise ValueError("steps, frame_interval and dt must be positive")
    if config.attachment_mode not in {"fixed", "waist_halfspace"}:
        raise ValueError("attachment_mode must be fixed or waist_halfspace")
    if config.attachment_mode == "waist_halfspace" and config.attachment_halfspace_y is None:
        raise ValueError("waist_halfspace requires attachment_halfspace_y")
    if mesh.vertices.shape != mesh.material_vertices.shape:
        raise ValueError("vertices and rest_vertices must have the same shape")

    faces = torch.as_tensor(mesh.faces, dtype=torch.long)
    pinned = torch.as_tensor(mesh.pinned_vertices, dtype=torch.long)
    material_rest = torch.as_tensor(mesh.material_vertices, dtype=torch.float32)
    positions = torch.as_tensor(mesh.vertices, dtype=torch.float32).clone()
    pin_positions = positions[pinned].clone()
    body_attachment_positions = pin_positions.clone()
    if config.attachment_mode == "waist_halfspace" and len(pinned):
        body_attachment_positions[:, 1] = float(config.attachment_halfspace_y)

    structural_np = _unique_edges(mesh.faces)
    structural_edges = torch.as_tensor(structural_np, dtype=torch.long)
    metric_reference = positions.clone()
    structural_rest = torch.linalg.norm(
        material_rest[structural_edges[:, 0]] - material_rest[structural_edges[:, 1]],
        dim=-1,
    )
    valid = structural_rest > 1e-6
    structural_edges = structural_edges[valid]
    structural_rest = structural_rest[valid]
    initial_structural_rest = structural_rest.clone()
    prefit_metric_growth = torch.ones_like(structural_rest)
    bending_np = _hinge_opposite_pairs(mesh.faces)
    bending_edges = torch.as_tensor(bending_np, dtype=torch.long)
    bending_rest = torch.linalg.norm(
        material_rest[bending_edges[:, 0]] - material_rest[bending_edges[:, 1]],
        dim=-1,
    )
    bending_valid = bending_rest > 1.0e-6
    bending_edges = bending_edges[bending_valid]
    bending_rest = bending_rest[bending_valid]

    if mesh.seam_edges:
        seam_edges = torch.as_tensor(np.asarray(mesh.seam_edges, dtype=np.int64), dtype=torch.long)
        seam_edges = seam_edges[seam_edges[:, 0] != seam_edges[:, 1]]
    else:
        seam_edges = torch.empty((0, 2), dtype=torch.long)
    seam_final_targets = torch.full(
        (seam_edges.shape[0],), max(float(material.thickness) * 0.5, 2e-4), dtype=torch.float32
    )
    seam_initial_targets = (
        torch.linalg.norm(positions[seam_edges[:, 0]] - positions[seam_edges[:, 1]], dim=-1)
        if seam_edges.numel()
        else seam_final_targets
    )

    vertex_area = lumped_vertex_areas(material_rest, faces)
    mass = vertex_area * max(float(material.density), 0.05)
    inverse_mass = mass.reciprocal()
    if config.attachment_mode == "fixed" or config.attachment_release_step is None:
        inverse_mass[pinned] = 0.0
    movable = inverse_mass > 0.0
    support_z = float(positions[:, 2].min() - 1.0)
    model = IntrinsicClothSimulator(material, material_rest, faces, learned_scale=config.learned_scale)
    model.eval()
    velocity = torch.zeros_like(positions)
    collision_reports = []
    structural_compliance = 2.0e-7 / max(float(material.stretch_stiffness), 1e-4)
    bending_compliance = 4.0e-5 / max(float(material.bending_stiffness), 1.0e-4)

    prefit_cycles = 0
    if avatar is not None and hasattr(avatar, "project_surface_torch"):
        diagnostics_fn = getattr(avatar, "collision_diagnostics_np", None)
        if diagnostics_fn is None:
            prefit_cycles = 8
        else:
            initial_collision = diagnostics_fn(
                positions.detach().cpu().numpy(),
                mesh.faces,
            )
            if (
                initial_collision.get("avatar_inside_vertices_final", 0.0) > 0.0
                or initial_collision.get("avatar_surface_intersecting_faces_final", 0.0)
                > 0.0
            ):
                prefit_cycles = 8

    if prefit_cycles:
        # Resolve the dressed shell against triangle interiors before recording
        # frame zero. This prevents the rollout from starting with an arm
        # already threaded through a coarse sleeve triangle.
        for _ in range(prefit_cycles):
            positions = project_distance_constraints(
                positions,
                structural_edges,
                structural_rest,
                inverse_mass,
                max(12, config.edge_projection_iterations // 3),
                compliance=structural_compliance,
                dt=config.dt,
                min_ratio=1.0 - config.stretch_tolerance,
                max_ratio=1.0 + config.stretch_tolerance,
            )
            if bending_edges.numel():
                positions = project_distance_constraints(
                    positions,
                    bending_edges,
                    bending_rest,
                    inverse_mass,
                    max(6, config.edge_projection_iterations // 4),
                    compliance=bending_compliance,
                    dt=config.dt,
                    min_ratio=1.0,
                    max_ratio=1.0,
                )
            if seam_edges.numel():
                positions = project_distance_constraints(
                    positions,
                    seam_edges,
                    seam_initial_targets,
                    inverse_mass,
                    16,
                    compliance=1e-6,
                    dt=config.dt,
                )
            positions, reports = _project_avatar_constraints(
                avatar,
                positions,
                faces,
                mesh.panel_labels,
                seam_edges,
                collision_mask=movable,
            )
            collision_reports.extend(reports)
        positions = _apply_attachment_constraints(
            positions, pinned, pin_positions, config, step=0
        )
        raw_prefit_rest = torch.linalg.norm(
            positions[structural_edges[:, 0]] - positions[structural_edges[:, 1]],
            dim=-1,
        )
        prefit_metric_growth = raw_prefit_rest / torch.clamp(initial_structural_rest, min=1e-8)

    frames: list[np.ndarray] = [positions.detach().cpu().numpy().copy()]

    with torch.no_grad():
        for step in range(config.steps):
            sewing_t = min(1.0, (step + 1) / max(config.steps * 0.65, 1.0))
            sewing_t = sewing_t * sewing_t * (3.0 - 2.0 * sewing_t)
            seam_targets = (1.0 - sewing_t) * seam_initial_targets + sewing_t * seam_final_targets
            previous = positions.clone()
            active_fixed_pins = (
                pinned
                if (
                    config.attachment_mode == "fixed"
                    or (
                        config.attachment_mode == "waist_halfspace"
                        and config.attachment_release_step is None
                    )
                )
                and _attachment_is_active(config, step)
                else torch.empty(0, dtype=torch.long, device=positions.device)
            )
            positions, velocity = model(
                positions,
                velocity,
                active_fixed_pins,
                body_attachment_positions if len(active_fixed_pins) else positions.new_empty((0, 3)),
                config.dt,
                config.gravity_y,
            )
            positions = project_distance_constraints(
                positions,
                structural_edges,
                structural_rest,
                inverse_mass,
                config.edge_projection_iterations,
                compliance=structural_compliance,
                dt=config.dt,
                min_ratio=1.0 - config.stretch_tolerance,
                max_ratio=1.0 + config.stretch_tolerance,
            )
            if bending_edges.numel():
                positions = project_distance_constraints(
                    positions,
                    bending_edges,
                    bending_rest,
                    inverse_mass,
                    max(6, config.edge_projection_iterations // 4),
                    compliance=bending_compliance,
                    dt=config.dt,
                    min_ratio=1.0,
                    max_ratio=1.0,
                )
            if seam_edges.numel():
                positions = project_distance_constraints(
                    positions,
                    seam_edges,
                    seam_targets,
                    inverse_mass,
                    max(12, config.edge_projection_iterations),
                    compliance=1e-6,
                    dt=config.dt,
                    min_ratio=1.0,
                    max_ratio=1.0,
                )
            if config.bend_smoothing > 0.0:
                bend_step = min(0.12, config.bend_smoothing * material.bending_stiffness)
                rest_curvature = cotangent_laplacian(material_rest, material_rest, faces)
                curvature_change = (
                    cotangent_laplacian(positions, material_rest, faces)
                    - rest_curvature
                )
                bent = positions + bend_step * curvature_change
                if len(active_fixed_pins):
                    bent[active_fixed_pins] = body_attachment_positions
                positions = bent
            if avatar is not None:
                positions, reports = _project_avatar_constraints(
                    avatar,
                    positions,
                    faces,
                    mesh.panel_labels,
                    seam_edges,
                    collision_mask=movable,
                )
                collision_reports.extend(reports)
                # A short correction restores cloth strain, then a final collision
                # pass guarantees that constraints do not push cloth into the body.
                for _ in range(3):
                    positions = project_distance_constraints(
                        positions,
                        structural_edges,
                        structural_rest,
                        inverse_mass,
                        max(4, config.edge_projection_iterations // 4),
                        compliance=structural_compliance,
                        dt=config.dt,
                        min_ratio=1.0 - config.stretch_tolerance,
                        max_ratio=1.0 + config.stretch_tolerance,
                    )
                    if bending_edges.numel():
                        positions = project_distance_constraints(
                            positions,
                            bending_edges,
                            bending_rest,
                            inverse_mass,
                            max(4, config.edge_projection_iterations // 6),
                            compliance=bending_compliance,
                            dt=config.dt,
                            min_ratio=1.0,
                            max_ratio=1.0,
                        )
                    if seam_edges.numel():
                        positions = project_distance_constraints(
                            positions,
                            seam_edges,
                            seam_targets,
                            inverse_mass,
                            12,
                            compliance=1e-6,
                            dt=config.dt,
                        )
                    positions, reports = _project_avatar_constraints(
                        avatar,
                        positions,
                        faces,
                        mesh.panel_labels,
                        seam_edges,
                        collision_mask=movable,
                    )
                    collision_reports.extend(reports)

            positions[:, 2] = torch.clamp(positions[:, 2], min=support_z)
            positions = _apply_attachment_constraints(
                positions, pinned, pin_positions, config, step=step
            )
            velocity = torch.clamp((positions - previous) / config.dt, -config.max_velocity, config.max_velocity)
            velocity *= float(material.damping) ** (config.dt / 0.008)
            if len(active_fixed_pins):
                velocity[active_fixed_pins] = 0.0
            elif config.attachment_mode == "waist_halfspace" and _attachment_is_active(config, step):
                on_plane = positions[pinned, 1] <= float(config.attachment_halfspace_y) + 1.0e-6
                velocity[pinned[on_plane], 1] = torch.clamp_min(
                    velocity[pinned[on_plane], 1], 0.0
                )
            if (step + 1) % config.frame_interval == 0 or step == config.steps - 1:
                frames.append(positions.detach().cpu().numpy().copy())

        if avatar is not None:
            # Finish with a coupled relaxation. Collision-only cleanup can pull
            # nearby cloth edges far beyond their material metric, especially
            # around shoulders and the hem. Alternating all three constraints
            # converges to a dressed state instead of trading penetration for
            # visible strain in the final frame.
            for _ in range(30):
                positions = project_distance_constraints(
                    positions,
                    structural_edges,
                    structural_rest,
                    inverse_mass,
                    max(12, config.edge_projection_iterations // 3),
                    compliance=structural_compliance,
                    dt=config.dt,
                    min_ratio=1.0 - config.stretch_tolerance,
                    max_ratio=1.0 + config.stretch_tolerance,
                )
                if bending_edges.numel():
                    positions = project_distance_constraints(
                        positions,
                        bending_edges,
                        bending_rest,
                        inverse_mass,
                        max(6, config.edge_projection_iterations // 4),
                        compliance=bending_compliance,
                        dt=config.dt,
                        min_ratio=1.0,
                        max_ratio=1.0,
                    )
                if seam_edges.numel():
                    positions = project_distance_constraints(
                        positions,
                        seam_edges,
                        seam_final_targets,
                        inverse_mass,
                        16,
                        compliance=1e-6,
                        dt=config.dt,
                    )
                positions, reports = _project_avatar_constraints(
                    avatar,
                    positions,
                    faces,
                    mesh.panel_labels,
                    seam_edges,
                    collision_mask=movable,
                )
                collision_reports.extend(reports)
            for _ in range(4):
                positions, reports = _project_avatar_constraints(
                    avatar,
                    positions,
                    faces,
                    mesh.panel_labels,
                    seam_edges,
                    collision_mask=movable,
                )
                collision_reports.extend(reports)
            positions = _apply_attachment_constraints(
                positions, pinned, pin_positions, config, step=config.steps
            )
            frames[-1] = positions.detach().cpu().numpy().copy()

    final = positions.detach().cpu().numpy()
    rest_np = mesh.material_vertices
    fitted_reference_np = metric_reference.detach().cpu().numpy()
    fitted_edges_np = structural_edges.detach().cpu().numpy()
    fitted_rest_np = structural_rest.detach().cpu().numpy()
    final_lengths = np.linalg.norm(final[fitted_edges_np[:, 0]] - final[fitted_edges_np[:, 1]], axis=1)
    stretch = final_lengths / np.maximum(fitted_rest_np, 1e-8)
    material_rest_lengths = np.linalg.norm(
        rest_np[structural_np[:, 0]] - rest_np[structural_np[:, 1]], axis=1
    )
    material_final_lengths = np.linalg.norm(final[structural_np[:, 0]] - final[structural_np[:, 1]], axis=1)
    valid_material = material_rest_lengths > 1e-6
    material_embedding_ratio = material_final_lengths[valid_material] / material_rest_lengths[valid_material]
    seam_gap = (
        np.linalg.norm(final[np.asarray(mesh.seam_edges)[:, 0]] - final[np.asarray(mesh.seam_edges)[:, 1]], axis=1)
        if mesh.seam_edges
        else np.asarray([0.0], dtype=np.float32)
    )
    area_rest = _surface_area(rest_np, mesh.faces)
    area_fitted = _surface_area(fitted_reference_np, mesh.faces)
    area_final = _surface_area(final, mesh.faces)
    pinned_drift = (
        np.linalg.norm(final[mesh.pinned_vertices] - mesh.vertices[mesh.pinned_vertices], axis=1)
        if mesh.pinned_vertices.size and config.attachment_mode == "fixed"
        else np.asarray([0.0], dtype=np.float32)
    )
    metrics = {
        "min_height_z": float(final[:, 2].min()),
        "max_height_z": float(final[:, 2].max()),
        "min_vertical_y": float(final[:, 1].min()),
        "max_vertical_y": float(final[:, 1].max()),
        "mean_edge_stretch": float(stretch.mean()),
        "p95_edge_stretch": float(np.percentile(stretch, 95)),
        "p99_edge_stretch": float(np.percentile(stretch, 99)),
        "max_edge_stretch": float(stretch.max()),
        "surface_area_ratio": float(area_final / max(area_fitted, 1e-10)),
        "material_embedding_mean_edge_ratio": float(material_embedding_ratio.mean()),
        "material_embedding_p95_edge_ratio": float(np.percentile(material_embedding_ratio, 95)),
        "material_embedding_p99_edge_ratio": float(np.percentile(material_embedding_ratio, 99)),
        "material_embedding_max_edge_ratio": float(material_embedding_ratio.max()),
        "material_embedding_surface_area_ratio": float(area_final / max(area_rest, 1e-10)),
        "prefit_metric_growth_p95": float(torch.quantile(prefit_metric_growth, 0.95).item()),
        "prefit_metric_growth_max": float(prefit_metric_growth.max().item()),
        "mean_seam_gap_m": float(seam_gap.mean()),
        "max_seam_gap_m": float(seam_gap.max()),
        "pinned_drift": float(pinned_drift.max()),
        "waist_halfspace_attachment": float(config.attachment_mode == "waist_halfspace"),
        "attachment_release_step": float(
            config.attachment_release_step
            if config.attachment_release_step is not None
            else config.steps
        ),
    }
    if collision_reports:
        metrics.update(
            {
                "avatar_collision_corrected_vertices_last": float(collision_reports[-1].corrected_vertices),
                "avatar_collision_max_penetration_last": collision_reports[-1].max_penetration,
                "avatar_collision_mean_penetration_last": collision_reports[-1].mean_penetration,
                "avatar_collision_total_corrections": float(sum(r.corrected_vertices for r in collision_reports)),
            }
        )
    if avatar is not None and hasattr(avatar, "collision_diagnostics_np"):
        metrics.update(avatar.collision_diagnostics_np(final, mesh.faces))
    return frames, final, metrics


def _project_avatar_constraints(
    avatar: SmplAvatarProxy,
    positions: torch.Tensor,
    faces: torch.Tensor,
    panel_labels: list[str],
    seam_edges: torch.Tensor,
    collision_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, list[AvatarCollisionReport]]:
    before = positions
    continuous_shell = seam_edges.numel() == 0
    active_seam_edges = _active_seam_edges(
        positions,
        seam_edges,
        max_gap=_SEAM_COLLISION_ACTIVATION_M,
    )
    vertex_collision_mask = collision_mask
    seam_group_projector = getattr(avatar, "project_seam_groups_torch", None)
    if seam_group_projector is not None and active_seam_edges.numel():
        seam_vertices = torch.unique(active_seam_edges.reshape(-1))
        vertex_collision_mask = (
            torch.ones(positions.shape[0], dtype=torch.bool, device=positions.device)
            if collision_mask is None
            else collision_mask.clone()
        )
        vertex_collision_mask[seam_vertices] = False
    projected, vertex_report = avatar.project_torch(
        positions,
        None if continuous_shell else panel_labels,
        collision_mask=vertex_collision_mask,
    )
    reports = [vertex_report]
    surface_projector = getattr(avatar, "project_surface_torch", None)
    if surface_projector is not None:
        projected, surface_report = surface_projector(
            projected,
            faces,
            None if continuous_shell else panel_labels,
            # Sample corrections may touch seam vertices; the displacement is
            # coupled across each stitch group immediately below.
            collision_mask=collision_mask,
        )
        reports.append(surface_report)
        projected = _couple_seam_collision_displacements(
            before,
            projected,
            active_seam_edges,
            collision_mask=collision_mask,
        )
    if seam_group_projector is not None and active_seam_edges.numel():
        projected, seam_report = seam_group_projector(
            projected,
            active_seam_edges,
            panel_labels=panel_labels,
            collision_mask=collision_mask,
        )
        reports.append(seam_report)
    return projected, reports


def _active_seam_edges(
    positions: torch.Tensor,
    seam_edges: torch.Tensor,
    *,
    max_gap: float,
) -> torch.Tensor:
    """Return stitches close enough to behave as one collision component."""

    if seam_edges.numel() == 0:
        return seam_edges
    gap = torch.linalg.norm(
        positions[seam_edges[:, 0]] - positions[seam_edges[:, 1]],
        dim=-1,
    )
    return seam_edges[gap <= float(max_gap)]


def _couple_seam_collision_displacements(
    before: torch.Tensor,
    after: torch.Tensor,
    seam_edges: torch.Tensor,
    collision_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Give vertices sewn to the same point one shared collision displacement."""

    if seam_edges.numel() == 0:
        return after
    pairs = seam_edges.detach().cpu().numpy().astype(np.int32)
    parent: dict[int, int] = {}

    def find(index: int) -> int:
        parent.setdefault(index, index)
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(a: int, b: int) -> None:
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    for a, b in pairs:
        union(int(a), int(b))
    groups: dict[int, list[int]] = {}
    for index in parent:
        groups.setdefault(find(index), []).append(index)

    result = after.clone()
    displacement = after - before
    movable = (
        torch.ones(before.shape[0], dtype=torch.bool, device=before.device)
        if collision_mask is None
        else collision_mask
    )
    for group in groups.values():
        ids = torch.as_tensor(group, dtype=torch.long, device=before.device)
        if not torch.all(movable[ids]):
            result[ids[movable[ids]]] = before[ids[movable[ids]]]
            continue
        shared = displacement[ids].mean(dim=0)
        result[ids] = before[ids] + shared
    return result


def project_distance_constraints(
    positions: torch.Tensor,
    edges: torch.Tensor,
    target_lengths: torch.Tensor,
    inverse_mass: torch.Tensor,
    iterations: int,
    compliance: float,
    dt: float,
    min_ratio: float = 1.0,
    max_ratio: float = 1.0,
) -> torch.Tensor:
    if edges.numel() == 0 or iterations <= 0:
        return positions
    projected = positions.clone()
    a, b = edges[:, 0], edges[:, 1]
    wa = inverse_mass[a]
    wb = inverse_mass[b]
    alpha = float(compliance) / max(dt * dt, 1e-12)
    for _ in range(iterations):
        delta = projected[b] - projected[a]
        length = torch.linalg.norm(delta, dim=-1).clamp_min(1e-8)
        desired = torch.clamp(length, target_lengths * min_ratio, target_lengths * max_ratio)
        correction = delta * ((length - desired) / length).unsqueeze(-1)
        scale = 1.0 / (wa + wb + alpha).clamp_min(1e-8)
        updates = torch.zeros_like(projected)
        weights = torch.zeros(projected.shape[0], dtype=projected.dtype, device=projected.device)
        updates.index_add_(0, a, correction * (wa * scale).unsqueeze(-1))
        updates.index_add_(0, b, -correction * (wb * scale).unsqueeze(-1))
        weights.index_add_(0, a, (wa > 0).to(projected.dtype))
        weights.index_add_(0, b, (wb > 0).to(projected.dtype))
        projected += updates / weights.clamp_min(1.0).unsqueeze(-1)
        projected = torch.nan_to_num(projected, nan=0.0, posinf=3.0, neginf=-3.0)
    return projected


def project_edge_lengths(
    positions: torch.Tensor,
    edges: torch.Tensor,
    rest_lengths: torch.Tensor,
    movable: torch.Tensor,
    iterations: int,
    min_ratio: float,
    max_ratio: float,
    update_scale: float = 1.0,
) -> torch.Tensor:
    inverse_mass = movable.to(dtype=positions.dtype) * max(float(update_scale), 0.0)
    return project_distance_constraints(
        positions,
        edges,
        rest_lengths,
        inverse_mass,
        iterations,
        compliance=0.0,
        dt=1.0,
        min_ratio=min_ratio,
        max_ratio=max_ratio,
    )


def damp_bending_noise(
    positions: torch.Tensor,
    faces: torch.Tensor,
    pinned: torch.Tensor,
    strength: float,
) -> torch.Tensor:
    if strength <= 0.0:
        return positions
    lap = normalized_graph_laplacian(positions.unsqueeze(0), faces).squeeze(0)
    result = positions + strength * lap
    result[pinned] = positions[pinned]
    return result


def project_rest_shape(
    positions: torch.Tensor,
    rest_positions: torch.Tensor,
    movable: torch.Tensor,
    strength: float,
) -> torch.Tensor:
    projected = positions.clone()
    blend = float(np.clip(strength, 0.0, 1.0))
    projected[movable] += blend * (rest_positions[movable] - projected[movable])
    return projected


def _surface_area(vertices: np.ndarray, faces: np.ndarray) -> float:
    tri = vertices[faces]
    return float(0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1).sum())


def _unique_edges(faces: np.ndarray) -> np.ndarray:
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0)
    return np.unique(np.sort(edges, axis=1), axis=0)


def _hinge_opposite_pairs(faces: np.ndarray) -> np.ndarray:
    """Return opposite-vertex pairs across manifold interior edges."""

    incidence: dict[tuple[int, int], list[int]] = {}
    for a_raw, b_raw, c_raw in np.asarray(faces, dtype=np.int64):
        a, b, c = int(a_raw), int(b_raw), int(c_raw)
        for first, second, opposite in ((a, b, c), (b, c, a), (c, a, b)):
            edge = tuple(sorted((first, second)))
            incidence.setdefault(edge, []).append(opposite)
    pairs = {
        tuple(sorted((opposites[0], opposites[1])))
        for opposites in incidence.values()
        if len(opposites) == 2 and opposites[0] != opposites[1]
    }
    return np.asarray(sorted(pairs), dtype=np.int64).reshape(-1, 2)


def _simulation_edges(faces: np.ndarray, seam_edges: list[tuple[int, int]]) -> np.ndarray:
    face_edges = _unique_edges(faces)
    if not seam_edges:
        return face_edges
    stitched = np.asarray(seam_edges, dtype=np.int64)
    stitched = stitched[stitched[:, 0] != stitched[:, 1]]
    return np.unique(np.concatenate([face_edges, np.sort(stitched, axis=1)], axis=0), axis=0)
