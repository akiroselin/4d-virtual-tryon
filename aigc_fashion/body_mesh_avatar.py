from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .avatar import SmplAvatarProxy
from .arrangement import (
    ArrangementBoundingVolume,
    ArrangementPoint,
    AvatarArrangement,
    unit_vector,
)
from .data import AvatarSpec, GarmentMesh


_SURFACE_COLLISION_BARYCENTRICS = np.asarray(
    [
        [0.5, 0.5, 0.0],
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5],
        [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
    ],
    dtype=np.float32,
)


class BodyMeshAvatar(SmplAvatarProxy):
    """Avatar support surface backed by a generic human body mesh export.

    The class expects a legal OBJ mesh from any commercial-safe body model or
    scanner. It maps the source axes into the prototype convention:
    x = horizontal, y = vertical, z = front/back depth.
    """

    def __init__(
        self,
        spec: AvatarSpec,
        vertices: np.ndarray,
        faces: np.ndarray,
        mesh_path: str | Path,
        clearance: float = 0.004,
    ):
        super().__init__(spec, clearance=clearance)
        self.mesh_path = str(mesh_path)
        self.mesh_vertices = vertices.astype(np.float32)
        self.mesh_faces = faces.astype(np.int32)
        self._build_spatial_index()
        self._build_width_profile()
        self._build_garment_landmarks()
        self._build_arm_profiles()
        self._build_arrangement_model()
        self._last_sleeve_grading: dict[str, float] = {}
        self._last_arrangement_assignments: dict[str, tuple[str, ...]] = {}
        self._last_arrangement_diagnostics: dict[str, float] = {}
        self._last_placement_mode = "not_dressed"

    @classmethod
    def from_obj(
        cls,
        mesh_path: str | Path,
        spec: AvatarSpec,
        source_x_axis: str = "x",
        source_up_axis: str = "auto",
        source_front_axis: str = "auto",
        scale: float | None = None,
    ) -> "BodyMeshAvatar":
        mesh_path = Path(mesh_path)
        vertices, faces = _load_obj_mesh(mesh_path)
        resolved_axes = _resolve_axis_mapping(
            vertices,
            source_x_axis,
            source_up_axis,
            source_front_axis,
        )
        vertices = _remap_axes(vertices, *resolved_axes)
        _validate_axis_mapping(vertices, resolved_axes)

        height = float(vertices[:, 1].max() - vertices[:, 1].min())
        target_height = max(spec.height_cm / 100.0, 1e-6)
        if scale is None and height > 0.25:
            scale = target_height / height
        vertices = vertices * float(scale if scale is not None else 1.0)
        vertices = _morph_body_measurements(vertices, spec)

        proxy = SmplAvatarProxy(spec)
        shoulder_target_y = proxy.center_y + 0.78 * proxy.torso_half_height
        vertices[:, 0] -= float(np.median(vertices[:, 0]))
        vertices[:, 1] -= float(np.percentile(vertices[:, 1], 82.0) - shoulder_target_y)
        torso_mask = (vertices[:, 1] > proxy.center_y - proxy.torso_half_height) & (
            vertices[:, 1] < proxy.center_y + proxy.torso_half_height
        )
        z_center = float(np.median(vertices[torso_mask, 2])) if torso_mask.any() else float(np.median(vertices[:, 2]))
        vertices[:, 2] += proxy.base_z - z_center
        avatar = cls(spec=spec, vertices=vertices, faces=faces, mesh_path=mesh_path)
        avatar.source_axes = {
            "x": resolved_axes[0],
            "up": resolved_axes[1],
            "front": resolved_axes[2],
        }
        return avatar

    def summary(self) -> dict[str, float | str | int]:
        data = super().summary()
        data.update(
            {
                "type": "body_mesh",
                "mesh_path": self.mesh_path,
                "source_axes": dict(self.source_axes),
                "vertices": int(self.mesh_vertices.shape[0]),
                "faces": int(self.mesh_faces.shape[0]),
                "collision": "seam-component SDF plus panel-aware sampled-surface projection",
                "shirt_landmarks": {
                    key: float(value)
                    for key, value in self._shirt_landmarks.items()
                    if isinstance(value, (float, np.floating))
                },
                "arm_profile": {
                    "samples": int(self._arm_profile_y.shape[0]),
                    "left_center_x_mean": float(np.mean(self._arm_profile_abs_x[-1.0])),
                    "right_center_x_mean": float(np.mean(self._arm_profile_abs_x[1.0])),
                },
                "arrangement": {
                    "standard": "oriented points attached to avatar bounding volumes",
                    "placement_mode": self._last_placement_mode,
                    "volumes": len(self._arrangement.volumes),
                    "points": len(self._arrangement.points),
                    "point_names": [point.name for point in self._arrangement.points],
                    "last_panel_assignments": {
                        role: list(points)
                        for role, points in self._last_arrangement_assignments.items()
                    },
                    "initial_collision_diagnostics": dict(
                        self._last_arrangement_diagnostics
                    ),
                },
                "sleeve_grading_scale": dict(self._last_sleeve_grading),
                "shoulder_wearing_ease_m": float(np.clip(self.spec.height_cm / 100.0 * 0.006, 0.006, 0.012)),
            }
        )
        return data

    @property
    def arrangement(self) -> AvatarArrangement:
        return self._arrangement

    def tshirt_fit_constraints(self) -> dict[str, float]:
        """Return avatar-derived minimum sleeve and armhole circumferences."""

        landmarks = self._shirt_landmarks
        cuff_y = float(landmarks["sleeve_cuff_y"])
        armhole_y = float(landmarks["armhole_y"])
        profile_y = np.linspace(cuff_y, armhole_y, 32, dtype=np.float32)
        cuff_profile_y = np.linspace(cuff_y, min(cuff_y + 0.022, armhole_y), 6, dtype=np.float32)

        circumferences: list[np.ndarray] = []
        cuff_circumferences: list[np.ndarray] = []
        for side in (-1.0, 1.0):
            _arm_x, front, back = self._arm_profile_np(side, profile_y)
            _cuff_x, cuff_front, cuff_back = self._arm_profile_np(side, cuff_profile_y)
            circumferences.append(np.pi * np.maximum(front - back, 0.0) * 100.0)
            cuff_circumferences.append(np.pi * np.maximum(cuff_front - cuff_back, 0.0) * 100.0)

        upper_arm = np.concatenate(circumferences)
        cuff_arm = np.concatenate(cuff_circumferences)
        upper_arm = upper_arm[np.isfinite(upper_arm) & (upper_arm > 1.0)]
        cuff_arm = cuff_arm[np.isfinite(cuff_arm) & (cuff_arm > 1.0)]
        if upper_arm.size == 0 or cuff_arm.size == 0:
            fallback = max(self.spec.chest_cm * 0.30, 16.0)
            upper_arm_cm = fallback
            cuff_arm_cm = fallback * 0.88
        else:
            upper_arm_cm = float(np.percentile(upper_arm, 90.0))
            cuff_arm_cm = float(np.percentile(cuff_arm, 80.0))

        allowed_wearing_stretch = 0.08
        wearing_ease_cm = 1.0
        return {
            "upper_arm_circumference_cm": upper_arm_cm,
            "cuff_arm_circumference_cm": cuff_arm_cm,
            "minimum_armhole_circumference_cm": upper_arm_cm / (1.0 + allowed_wearing_stretch) + wearing_ease_cm,
            "minimum_sleeve_opening_cm": cuff_arm_cm / (1.0 + allowed_wearing_stretch) + wearing_ease_cm,
            "wearing_ease_cm": wearing_ease_cm,
            "allowed_wearing_stretch": allowed_wearing_stretch,
        }

    def support_height_np(self, xy: np.ndarray) -> np.ndarray:
        return self._surface_height_np(xy, side="front")

    def back_height_np(self, xy: np.ndarray) -> np.ndarray:
        return self._surface_height_np(xy, side="back")

    def support_height_torch(self, positions: torch.Tensor) -> torch.Tensor:
        xy = positions[:, :2].detach().cpu().numpy()
        support = self.support_height_np(xy)
        return torch.as_tensor(support, dtype=positions.dtype, device=positions.device)

    def back_height_torch(self, positions: torch.Tensor) -> torch.Tensor:
        xy = positions[:, :2].detach().cpu().numpy()
        support = self.back_height_np(xy)
        return torch.as_tensor(support, dtype=positions.dtype, device=positions.device)

    def project_torch(
        self,
        positions: torch.Tensor,
        panel_labels: list[str] | None = None,
        collision_mask: torch.Tensor | None = None,
    ):
        points = positions.detach().cpu().numpy()
        closest, distances, triangle_ids = self._proximity.on_surface(points)
        normals = self._collision_mesh.face_normals[triangle_ids]
        # trimesh uses positive signed distance inside a watertight mesh and
        # negative distance outside. This is robust at triangle edges where a
        # nearest-face normal dot product can select the wrong side.
        signed_distance = self._proximity.signed_distance(points)
        penetration = self.clearance + signed_distance
        active = (penetration > 0.0) & ((distances < 0.14) | (signed_distance >= 0.0))
        if collision_mask is not None:
            active &= collision_mask.detach().cpu().numpy().astype(bool)
        projected_np = points.copy()
        target = closest[active] + self.clearance * normals[active]
        correction = target - points[active]
        correction_length = np.linalg.norm(correction, axis=1, keepdims=True)
        correction *= np.minimum(1.0, 0.010 / np.maximum(correction_length, 1e-9))
        projected_np[active] = points[active] + correction

        sleeve_barrier_depth = np.empty(0, dtype=np.float32)
        if panel_labels is not None:
            labels = np.asarray(panel_labels)
            allowed = (
                np.ones(points.shape[0], dtype=bool)
                if collision_mask is None
                else collision_mask.detach().cpu().numpy().astype(bool)
            )
            barrier_depths: list[np.ndarray] = []
            for label_name, side in (("left_sleeve", -1.0), ("right_sleeve", 1.0)):
                sleeve = np.asarray([label_name in label for label in labels], dtype=bool) & allowed
                if not np.any(sleeve):
                    continue
                sleeve_ids = np.flatnonzero(sleeve)
                _arm_x, arm_front, arm_back = self._arm_profile_np(side, projected_np[sleeve_ids, 1])
                midpoint = 0.5 * (arm_front + arm_back)
                front_shell = np.asarray([labels[index].startswith("front_") for index in sleeve_ids])
                desired_front = midpoint + 0.0005
                desired_back = midpoint - 0.0005
                front_depth = np.maximum(desired_front - projected_np[sleeve_ids, 2], 0.0)
                back_depth = np.maximum(projected_np[sleeve_ids, 2] - desired_back, 0.0)
                projected_np[sleeve_ids[front_shell], 2] += np.minimum(front_depth[front_shell], 0.006)
                projected_np[sleeve_ids[~front_shell], 2] -= np.minimum(back_depth[~front_shell], 0.006)
                barrier_depths.extend([front_depth[front_shell], back_depth[~front_shell]])

            radial_body = np.asarray(["front_body" in label for label in labels], dtype=bool) & allowed
            radial_body &= np.abs(projected_np[:, 0]) > 0.85 * self._shirt_landmarks["shoulder_half"]
            radial_body &= projected_np[:, 1] > self._shirt_landmarks["sleeve_cuff_y"] - 0.02
            for side in (-1.0, 1.0):
                side_body = radial_body & (side * projected_np[:, 0] > 0.0)
                body_ids = np.flatnonzero(side_body)
                if body_ids.size == 0:
                    continue
                _arm_x, arm_front, arm_back = self._arm_profile_np(side, projected_np[body_ids, 1])
                midpoint = 0.5 * (arm_front + arm_back)
                front_shell = np.asarray([labels[index].startswith("front_") for index in body_ids])
                front_depth = np.maximum(midpoint + 0.0005 - projected_np[body_ids, 2], 0.0)
                back_depth = np.maximum(projected_np[body_ids, 2] - (midpoint - 0.0005), 0.0)
                projected_np[body_ids[front_shell], 2] += np.minimum(front_depth[front_shell], 0.006)
                projected_np[body_ids[~front_shell], 2] -= np.minimum(back_depth[~front_shell], 0.006)
                barrier_depths.extend([front_depth[front_shell], back_depth[~front_shell]])
            nonempty_depths = [depth for depth in barrier_depths if depth.size]
            if nonempty_depths:
                sleeve_barrier_depth = np.concatenate(nonempty_depths)

        projected = torch.as_tensor(projected_np, dtype=positions.dtype, device=positions.device)
        from .avatar import AvatarCollisionReport

        active_depth = penetration[active]
        corrected_barrier = int(np.sum(sleeve_barrier_depth > 0.0))
        combined_depth = (
            np.concatenate([active_depth, sleeve_barrier_depth])
            if sleeve_barrier_depth.size
            else active_depth
        )
        return projected, AvatarCollisionReport(
            corrected_vertices=int(active.sum()) + corrected_barrier,
            max_penetration=float(combined_depth.max()) if combined_depth.size else 0.0,
            mean_penetration=float(combined_depth.mean()) if combined_depth.size else 0.0,
        )

    def project_surface_torch(
        self,
        positions: torch.Tensor,
        faces: torch.Tensor,
        panel_labels: list[str] | None = None,
        collision_mask: torch.Tensor | None = None,
    ):
        """Project sampled cloth triangle interiors outside the body mesh.

        Vertex-only collision misses intersections where a body limb passes
        through the interior of a cloth triangle. Edge midpoints and centroids
        provide a compact continuous-surface constraint without replacing the
        intrinsic cloth state with a volumetric soft body.
        """

        points = positions.detach().cpu().numpy()
        face_indices = faces.detach().cpu().numpy().astype(np.int32)
        triangles = points[face_indices]
        barycentrics = _SURFACE_COLLISION_BARYCENTRICS
        samples = np.einsum("sv,fvc->fsc", barycentrics, triangles)
        flat_samples = samples.reshape(-1, 3)

        closest, distances, triangle_ids = self._proximity.on_surface(flat_samples)
        signed_distance = self._proximity.signed_distance(flat_samples)
        # Use one body-wide SDF rule for every garment. The front/back height
        # fields remain placement priors, but cannot detect a hip or leg
        # passing through the interior of a skirt triangle.
        penetration = self.clearance + signed_distance
        active = (penetration > 0.0) & (distances < 0.035)

        if collision_mask is None:
            movable = np.ones(points.shape[0], dtype=bool)
        else:
            movable = collision_mask.detach().cpu().numpy().astype(bool)
        sample_faces = np.repeat(face_indices, barycentrics.shape[0], axis=0)
        active &= np.any(movable[sample_faces], axis=1)
        active &= np.any(movable[sample_faces], axis=1)

        result = points.copy()
        updates = np.zeros_like(points)
        weights = np.zeros(points.shape[0], dtype=np.float32)
        if np.any(active):
            normals = np.asarray(
                self._collision_mesh.face_normals[triangle_ids[active]],
                dtype=np.float32,
            )
            normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1.0e-8)
            target = closest[active] + self.clearance * normals
            correction = target - flat_samples[active]
            correction_length = np.linalg.norm(correction, axis=1, keepdims=True)
            correction *= np.minimum(1.0, 0.008 / np.maximum(correction_length, 1.0e-9))

            active_ids = np.flatnonzero(active)
            for correction_value, sample_id in zip(correction, active_ids):
                face_id = sample_id // barycentrics.shape[0]
                bary_id = sample_id % barycentrics.shape[0]
                for local_id, bary_weight in enumerate(barycentrics[bary_id]):
                    vertex_id = int(face_indices[face_id, local_id])
                    if bary_weight <= 0.0 or not movable[vertex_id]:
                        continue
                    updates[vertex_id] += correction_value * float(bary_weight)
                    weights[vertex_id] += float(bary_weight)

        corrected = weights > 0.0
        result[corrected] += updates[corrected] / weights[corrected, None]

        from .avatar import AvatarCollisionReport

        active_depth = penetration[active]
        return torch.as_tensor(result, dtype=positions.dtype, device=positions.device), AvatarCollisionReport(
            corrected_vertices=int(np.sum(corrected)),
            max_penetration=float(active_depth.max()) if active_depth.size else 0.0,
            mean_penetration=float(active_depth.mean()) if active_depth.size else 0.0,
        )

    def project_seam_groups_torch(
        self,
        positions: torch.Tensor,
        seam_edges: torch.Tensor,
        panel_labels: list[str] | None = None,
        collision_mask: torch.Tensor | None = None,
    ):
        points = positions.detach().cpu().numpy()
        pairs = seam_edges.detach().cpu().numpy().astype(np.int32)
        groups = _seam_components(pairs)
        movable = (
            np.ones(points.shape[0], dtype=bool)
            if collision_mask is None
            else collision_mask.detach().cpu().numpy().astype(bool)
        )
        result = points.copy()
        active_depths: list[float] = []
        corrected = 0
        for group in groups:
            group_ids = np.asarray(group, dtype=np.int32)
            movable_ids = group_ids[movable[group_ids]]
            if movable_ids.size == 0:
                continue
            center = np.mean(result[group_ids], axis=0, keepdims=True)
            closest, distance, triangle_ids = self._proximity.on_surface(center)
            signed_distance = self._proximity.signed_distance(center)
            group_has_sleeve = panel_labels is not None and any("sleeve" in panel_labels[index] for index in group)
            group_clearance = self.clearance + (0.008 if group_has_sleeve else 0.0)
            penetration = float(group_clearance + signed_distance[0])
            if penetration <= 0.0 or (distance[0] >= 0.14 and signed_distance[0] < 0.0):
                continue
            normal = self._collision_mesh.face_normals[triangle_ids[0]]
            correction = closest[0] + group_clearance * normal - center[0]
            correction_length = float(np.linalg.norm(correction))
            if correction_length > 0.010:
                correction *= 0.010 / correction_length
            result[movable_ids] += correction
            corrected += int(movable_ids.size)
            active_depths.append(penetration)

        from .avatar import AvatarCollisionReport

        return torch.as_tensor(result, dtype=positions.dtype, device=positions.device), AvatarCollisionReport(
            corrected_vertices=corrected,
            max_penetration=max(active_depths, default=0.0),
            mean_penetration=float(np.mean(active_depths)) if active_depths else 0.0,
        )

    def collision_diagnostics_np(
        self,
        vertices: np.ndarray,
        faces: np.ndarray | None = None,
    ) -> dict[str, float]:
        signed_distance = self._proximity.signed_distance(np.asarray(vertices, dtype=np.float32))
        penetration = np.maximum(self.clearance + signed_distance, 0.0)
        active = penetration > 0.0
        diagnostics = {
            "avatar_inside_vertices_final": float(np.sum(signed_distance > 0.0)),
            "avatar_clearance_violations_final": float(np.sum(active)),
            "avatar_collision_max_penetration_final": float(penetration.max(initial=0.0)),
            "avatar_collision_mean_penetration_final": float(penetration[active].mean()) if np.any(active) else 0.0,
        }
        if faces is not None:
            face_indices = np.asarray(faces, dtype=np.int32)
            triangles = np.asarray(vertices, dtype=np.float32)[face_indices]
            samples = np.einsum("sv,fvc->fsc", _SURFACE_COLLISION_BARYCENTRICS, triangles)
            surface_distance = self._proximity.signed_distance(samples.reshape(-1, 3)).reshape(
                face_indices.shape[0],
                _SURFACE_COLLISION_BARYCENTRICS.shape[0],
            )
            inside_samples = surface_distance > 0.0
            intersecting_faces = np.any(inside_samples, axis=1)
            diagnostics.update(
                {
                    "avatar_surface_inside_samples_final": float(np.sum(inside_samples)),
                    "avatar_surface_inside_sample_fraction_final": float(np.mean(inside_samples)),
                    "avatar_surface_intersecting_faces_final": float(np.sum(intersecting_faces)),
                    "avatar_surface_intersecting_face_fraction_final": float(np.mean(intersecting_faces)),
                    "avatar_surface_max_penetration_final": float(surface_distance.max(initial=0.0)),
                }
            )
        return diagnostics

    def dress_mesh(
        self,
        mesh: GarmentMesh,
        garment_offset: float = 0.006,
        placement_mode: str = "arrangement_points",
    ) -> GarmentMesh:
        graded_mesh = self._grade_sleeve_openings(mesh)
        self._last_arrangement_assignments = {
            "front_body": (
                "neck_front",
                "chest_front",
                "chest_side_L",
                "chest_side_R",
                "waist_front",
                "waist_side_L",
                "waist_side_R",
            ),
            "back_body": (
                "neck_back",
                "chest_back",
                "chest_side_L",
                "chest_side_R",
                "waist_back",
                "waist_side_L",
                "waist_side_R",
            ),
            "left_sleeve": ("shoulder_joint_L", "upper_arm_L", "sleeve_cuff_L"),
            "right_sleeve": ("shoulder_joint_R", "upper_arm_R", "sleeve_cuff_R"),
        }
        if placement_mode == "arrangement_points":
            vertices = self._arrange_tshirt_panels_isometric(
                graded_mesh,
                garment_offset=garment_offset,
            )
        elif placement_mode == "legacy_predrape":
            vertices = self._initial_tshirt_fit(
                graded_mesh,
                garment_offset=garment_offset,
            )
            vertices = self._preclose_shoulder_seams(vertices, graded_mesh)
        else:
            raise ValueError(
                "placement_mode must be arrangement_points or legacy_predrape"
            )
        self._last_placement_mode = placement_mode
        self._last_arrangement_diagnostics = self.collision_diagnostics_np(
            vertices,
            graded_mesh.faces,
        )
        dressed_labels = self._classify_sleeve_shell_labels(vertices, graded_mesh.panel_labels)
        if placement_mode == "arrangement_points":
            # Arrangement points initialize each free panel. They are not hard
            # Dirichlet pins: freezing front/back planes and offset sleeve tubes
            # would make the requested seam closure mechanically impossible.
            pinned_vertices = np.empty(0, dtype=np.int32)
        else:
            pinned_vertices = self._select_wearing_pins(
                vertices,
                dressed_labels,
                graded_mesh.material_vertices,
                graded_mesh.seam_edges,
            )
            vertices = self._project_pins_outside(vertices, pinned_vertices)
        return GarmentMesh(
            vertices=vertices.astype(np.float32),
            faces=graded_mesh.faces.copy(),
            pinned_vertices=pinned_vertices,
            panel_labels=dressed_labels,
            seam_edges=list(graded_mesh.seam_edges),
            rest_vertices=None if graded_mesh.rest_vertices is None else graded_mesh.rest_vertices.copy(),
            uvs=None if graded_mesh.uvs is None else graded_mesh.uvs.copy(),
            seam_interfaces=list(graded_mesh.seam_interfaces),
        )

    def _arrange_tshirt_panels_isometric(
        self,
        mesh: GarmentMesh,
        garment_offset: float,
    ) -> np.ndarray:
        """Place flat panels on avatar volumes without changing their metric.

        Front and back body panels remain rigid planes attached to the neck,
        chest, back, and waist frames. Sleeve charts are rolled isometrically
        around an upper-arm cylinder, with a short outward stand-off that keeps
        the open sleeve cap clear of the shoulder until sewing starts.
        """

        material = np.asarray(mesh.material_vertices, dtype=np.float32)
        labels = np.asarray(mesh.panel_labels)
        body = np.asarray(["front_body" in label for label in labels], dtype=bool)
        front_body = body & np.asarray(
            [label.startswith("front_") for label in labels],
            dtype=bool,
        )
        back_body = body & ~front_body
        if not np.any(body):
            return self._initial_tshirt_fit(mesh, garment_offset=garment_offset)

        result = material.copy()
        body_top = float(np.max(material[body, 1]))
        target_top = float(self._shirt_landmarks["neck_y"] - 0.010)
        result[body, 0] = material[body, 0]
        result[body, 1] = material[body, 1] + target_top - body_top

        front_names = ("neck_front", "chest_front", "waist_front", "hip_front")
        back_names = ("neck_back", "chest_back", "waist_back", "hip_back")
        panel_standoff = max(float(garment_offset), 0.012)
        front_plane = max(
            float(self.arrangement.point(name).position[2]) for name in front_names
        ) + panel_standoff
        back_plane = min(
            float(self.arrangement.point(name).position[2]) for name in back_names
        ) - panel_standoff
        result[front_body, 2] = front_plane
        result[back_body, 2] = back_plane

        sleeve_mask = np.asarray(["sleeve" in label for label in labels], dtype=bool)
        result[~body & ~sleeve_mask, 1] += target_top - body_top
        result[~body & ~sleeve_mask, 2] = front_plane

        for side, side_name, suffix in (
            (-1.0, "left_sleeve", "L"),
            (1.0, "right_sleeve", "R"),
        ):
            indices = np.flatnonzero(
                np.asarray([side_name in label for label in labels], dtype=bool)
            )
            if indices.size == 0:
                continue

            chart = material[indices, :2].astype(np.float64)
            centered = chart - np.mean(chart, axis=0, keepdims=True)
            _u, _s, axes = np.linalg.svd(centered, full_matrices=False)
            longitudinal_2d = axes[0]
            if side * longitudinal_2d[0] < 0.0:
                longitudinal_2d = -longitudinal_2d
            circumferential_2d = np.asarray(
                [-longitudinal_2d[1], longitudinal_2d[0]],
                dtype=np.float64,
            )
            along = centered @ longitudinal_2d
            cross = centered @ circumferential_2d
            along -= float(np.min(along))
            cross -= 0.5 * float(np.min(cross) + np.max(cross))

            volume = next(
                volume
                for volume in self.arrangement.volumes
                if volume.name == f"upper_arm_{suffix}"
            )
            start = np.asarray(volume.start, dtype=np.float64)
            end = np.asarray(volume.end, dtype=np.float64)
            axis = np.asarray(unit_vector(end - start), dtype=np.float64)
            front = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
            front -= axis * float(np.dot(front, axis))
            if np.linalg.norm(front) <= 1.0e-8:
                front = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
                front -= axis * float(np.dot(front, axis))
            front /= np.linalg.norm(front)
            around = np.cross(axis, front)
            around /= np.linalg.norm(around)

            arrangement_offset = self.arrangement.point(
                f"upper_arm_{suffix}"
            ).offset
            chart_radius = float(np.ptp(cross) / (2.0 * np.pi))
            radius = max(
                chart_radius,
                float(volume.radius_x) + arrangement_offset,
                float(volume.radius_y) + arrangement_offset,
            )
            cap_standoff = max(0.040, 2.5 * arrangement_offset)
            center = start + axis * cap_standoff
            angle = cross / max(radius, 1.0e-8)
            result[indices] = (
                center[None, :]
                + along[:, None] * axis[None, :]
                + radius * np.sin(angle)[:, None] * around[None, :]
                + radius * np.cos(angle)[:, None] * front[None, :]
            ).astype(np.float32)

        return result

    def _attach_sleeve_caps_to_body(self, vertices: np.ndarray, mesh: GarmentMesh) -> np.ndarray:
        labels = np.asarray(mesh.panel_labels)
        result = vertices.copy()
        targets: dict[int, list[np.ndarray]] = {}
        for a, b in mesh.seam_edges:
            a_is_sleeve = "sleeve" in labels[int(a)]
            b_is_sleeve = "sleeve" in labels[int(b)]
            if a_is_sleeve == b_is_sleeve:
                continue
            sleeve_id = int(a) if a_is_sleeve else int(b)
            body_id = int(b) if a_is_sleeve else int(a)
            if "front_body" not in labels[body_id]:
                continue
            targets.setdefault(sleeve_id, []).append(vertices[body_id])
        for sleeve_id, body_targets in targets.items():
            result[sleeve_id] = np.mean(body_targets, axis=0)
        return result

    def _grade_sleeve_openings(self, mesh: GarmentMesh) -> GarmentMesh:
        vertices = mesh.vertices.astype(np.float32).copy()
        rest_vertices = mesh.material_vertices.astype(np.float32).copy()
        labels = np.asarray(mesh.panel_labels)
        self._last_sleeve_grading = {}

        sample_y = np.linspace(
            self._shirt_landmarks["sleeve_cuff_y"],
            self._shirt_landmarks["armhole_y"],
            24,
            dtype=np.float32,
        )
        for label_name, side in (("left_sleeve", -1.0), ("right_sleeve", 1.0)):
            mask = np.asarray([label_name in label for label in labels], dtype=bool)
            if not np.any(mask):
                continue
            indices = np.flatnonzero(mask)
            flat_outward = side * vertices[indices, 0]
            fit = np.column_stack([flat_outward, np.ones_like(flat_outward)])
            slope, intercept = np.linalg.lstsq(fit, vertices[indices, 1], rcond=None)[0]
            centerline = slope * flat_outward + intercept
            cross_offset = vertices[indices, 1] - centerline

            sleeve_to_body: list[tuple[int, int]] = []
            for a, b in mesh.seam_edges:
                if mask[int(a)] and "front_body" in labels[int(b)]:
                    sleeve_to_body.append((int(a), int(b)))
                elif mask[int(b)] and "front_body" in labels[int(a)]:
                    sleeve_to_body.append((int(b), int(a)))
            if not sleeve_to_body:
                continue
            sleeve_seam = np.asarray([pair[0] for pair in sleeve_to_body], dtype=np.int32)
            body_seam = np.asarray([pair[1] for pair in sleeve_to_body], dtype=np.int32)
            sleeve_seam_outward = side * vertices[sleeve_seam, 0]
            body_seam_outward = side * vertices[body_seam, 0]

            sleeve_underarm = float(np.max(sleeve_seam_outward))
            sleeve_apex = float(np.min(sleeve_seam_outward))
            target_underarm = float(np.max(body_seam_outward))
            target_apex = float(np.min(body_seam_outward))
            cap_scale = float(
                np.clip(
                    (target_underarm - target_apex) / max(sleeve_underarm - sleeve_apex, 1e-6),
                    1.0,
                    1.35,
                )
            )
            graded_outward = np.where(
                flat_outward < sleeve_underarm,
                target_underarm - (sleeve_underarm - flat_outward) * cap_scale,
                target_underarm + (flat_outward - sleeve_underarm),
            )

            _arm_x, arm_front, arm_back = self._arm_profile_np(side, sample_y)
            cuff_sample_y = np.linspace(
                self._shirt_landmarks["sleeve_cuff_y"],
                min(self._shirt_landmarks["sleeve_cuff_y"] + 0.022, self._shirt_landmarks["armhole_y"]),
                6,
                dtype=np.float32,
            )
            _cuff_x, cuff_front, cuff_back = self._arm_profile_np(side, cuff_sample_y)
            required_circumference = float(
                np.percentile(np.pi * (cuff_front - cuff_back), 80.0) / 1.08 + 0.010
            )
            cuff = flat_outward >= float(np.max(flat_outward)) - max(0.004, 0.04 * float(np.ptp(flat_outward)))
            cuff_width = max(float(np.ptp(cross_offset[cuff])), 1e-6) if np.any(cuff) else max(float(np.ptp(cross_offset)), 1e-6)
            # The depth-derived circular estimate is conservative near the
            # armpit and overestimates a relaxed jersey cuff. Limit the cuff
            # ease while leaving the independently matched sleeve-cap depth.
            cuff_scale = float(np.clip(required_circumference / cuff_width, 1.0, 1.10))

            along = np.clip(
                (flat_outward - sleeve_underarm) / max(float(np.max(flat_outward)) - sleeve_underarm, 1e-6),
                0.0,
                1.0,
            )
            along = along * along * (3.0 - 2.0 * along)
            cap_center_shift = float(np.mean(vertices[body_seam, 1]) - np.mean(vertices[sleeve_seam, 1]))
            graded_y = centerline + cap_center_shift + cross_offset * (1.0 + along * (cuff_scale - 1.0))

            vertices[indices, 0] = side * graded_outward
            vertices[indices, 1] = graded_y
            rest_vertices[indices, 0] = side * graded_outward
            rest_vertices[indices, 1] = graded_y
            self._last_sleeve_grading[f"{label_name}_cap"] = cap_scale
            self._last_sleeve_grading[f"{label_name}_cuff"] = cuff_scale

        return GarmentMesh(
            vertices=vertices,
            faces=mesh.faces.copy(),
            pinned_vertices=mesh.pinned_vertices.copy(),
            panel_labels=list(mesh.panel_labels),
            seam_edges=list(mesh.seam_edges),
            rest_vertices=rest_vertices,
            uvs=None if mesh.uvs is None else mesh.uvs.copy(),
            seam_interfaces=list(mesh.seam_interfaces),
        )

    def _classify_sleeve_shell_labels(self, vertices: np.ndarray, panel_labels: list[str]) -> list[str]:
        result = list(panel_labels)
        for index, label in enumerate(panel_labels):
            if "sleeve" not in label:
                continue
            side = -1.0 if "left_sleeve" in label else 1.0
            _arm_x, arm_front, arm_back = self._arm_profile_np(side, np.asarray([vertices[index, 1]]))
            midpoint = 0.5 * float(arm_front[0] + arm_back[0])
            shell = label.removeprefix("front_").removeprefix("back_")
            result[index] = f"{'front' if vertices[index, 2] >= midpoint else 'back'}_{shell}"
        return result

    def _project_pins_outside(self, vertices: np.ndarray, pinned: np.ndarray) -> np.ndarray:
        if pinned.size == 0:
            return vertices
        result = vertices.copy()
        points = result[pinned]
        closest, _distances, triangle_ids = self._proximity.on_surface(points)
        signed_distance = self._proximity.signed_distance(points)
        active = signed_distance > -self.clearance
        if np.any(active):
            normals = self._collision_mesh.face_normals[triangle_ids[active]]
            result[pinned[active]] = closest[active] + self.clearance * normals
        return result

    def _preclose_shoulder_seams(self, vertices: np.ndarray, mesh: GarmentMesh) -> np.ndarray:
        if mesh.seam_interfaces:
            return self._preclose_ordered_seam_interfaces(vertices, mesh)
        if not mesh.seam_edges:
            return vertices
        labels = np.asarray(mesh.panel_labels)
        rest_y = mesh.material_vertices[:, 1]
        body = np.asarray(["front_body" in label for label in labels], dtype=bool)
        shoulder_cutoff = float(np.percentile(rest_y[body], 84.0)) if np.any(body) else float(np.percentile(rest_y, 90.0))
        pairs: list[tuple[int, int]] = []
        for a, b in mesh.seam_edges:
            label_a = labels[int(a)]
            label_b = labels[int(b)]
            front_back = label_a.startswith("front_") != label_b.startswith("front_")
            both_body = "front_body" in label_a and "front_body" in label_b
            sleeve_body = ("sleeve" in label_a) != ("sleeve" in label_b) and (
                "front_body" in label_a or "front_body" in label_b
            )
            shoulder = front_back and both_body and min(rest_y[int(a)], rest_y[int(b)]) >= shoulder_cutoff
            if shoulder or sleeve_body:
                pairs.append((int(a), int(b)))
        if not pairs:
            return vertices

        result = vertices.copy()
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
            union(a, b)
        groups: dict[int, list[int]] = {}
        for index in parent:
            groups.setdefault(find(index), []).append(index)

        components = list(groups.values())
        centers = np.asarray([np.mean(result[group], axis=0) for group in components], dtype=np.float32)
        closest, _distances, triangle_ids = self._proximity.on_surface(centers)
        normals = self._collision_mesh.face_normals[triangle_ids]
        targets = closest + (self.clearance + 0.5e-3) * normals

        # Closing only boundary vertices creates needle-like triangles at the
        # sleeve cap. Spread the body-conditioned sewing displacement through
        # a local material-space neighborhood before fixing the seam exactly.
        seam_displacement: dict[int, np.ndarray] = {}
        for group, target in zip(components, targets):
            for index in group:
                seam_displacement[index] = target - result[index]

        material_xy = mesh.material_vertices[:, :2]
        influence_radius = 0.035
        labels_array = np.asarray(mesh.panel_labels)
        for panel_label in np.unique(labels_array):
            panel_ids = np.flatnonzero(labels_array == panel_label)
            anchors = np.asarray(
                [index for index in seam_displacement if labels_array[index] == panel_label],
                dtype=np.int32,
            )
            if panel_ids.size == 0 or anchors.size == 0:
                continue
            distances = np.linalg.norm(
                material_xy[panel_ids, None, :] - material_xy[anchors][None, :, :],
                axis=2,
            )
            weights = np.exp(-np.square(distances / influence_radius))
            weights[distances > 2.5 * influence_radius] = 0.0
            weight_sum = np.sum(weights, axis=1)
            active = weight_sum > 1e-6
            anchor_displacements = np.asarray([seam_displacement[int(index)] for index in anchors])
            propagated = weights[active] @ anchor_displacements
            propagated /= weight_sum[active, None]
            falloff = np.clip(np.max(weights[active], axis=1), 0.0, 1.0)
            result[panel_ids[active]] += propagated * falloff[:, None]

        for group, target in zip(components, targets):
            result[group] = target
        return result

    def _preclose_ordered_seam_interfaces(
        self,
        vertices: np.ndarray,
        mesh: GarmentMesh,
    ) -> np.ndarray:
        """Close sewn curves without collapsing segmented seam junctions."""

        labels = np.asarray(mesh.panel_labels)
        result = vertices.copy()
        rest_y = mesh.material_vertices[:, 1]
        body_mask = np.asarray(["front_body" in label for label in labels], dtype=bool)
        shoulder_cutoff = (
            float(np.percentile(rest_y[body_mask], 84.0))
            if np.any(body_mask)
            else float(np.percentile(rest_y, 90.0))
        )
        body_interfaces = []
        sleeve_body_interfaces = []
        sleeve_self_interfaces = []
        for interface in mesh.seam_interfaces:
            ids_a = np.asarray(interface.vertices_a, dtype=np.int32)
            ids_b = np.asarray(interface.vertices_b, dtype=np.int32)
            sleeve_a = bool(np.mean(["sleeve" in labels[index] for index in ids_a]) >= 0.5)
            sleeve_b = bool(np.mean(["sleeve" in labels[index] for index in ids_b]) >= 0.5)
            if not sleeve_a and not sleeve_b:
                if min(
                    float(np.min(rest_y[ids_a])),
                    float(np.min(rest_y[ids_b])),
                ) >= shoulder_cutoff:
                    body_interfaces.append(interface)
            elif sleeve_a != sleeve_b:
                sleeve_body_interfaces.append(interface)
            else:
                sleeve_self_interfaces.append(interface)

        body_targets: dict[int, list[np.ndarray]] = {}
        for interface in body_interfaces:
            ids_a = np.asarray(interface.vertices_a, dtype=np.int32)
            ids_b = np.asarray(interface.vertices_b, dtype=np.int32)
            parameter_a = _normalized_curve_parameter(mesh.material_vertices[ids_a, :2])
            parameter_b = _normalized_curve_parameter(mesh.material_vertices[ids_b, :2])
            samples = np.unique(np.concatenate((parameter_a, parameter_b)))
            curve_a = _interpolate_curve(result[ids_a], parameter_a, samples)
            curve_b = _interpolate_curve(result[ids_b], parameter_b, samples)
            centers = 0.5 * (curve_a + curve_b)
            closest, _distance, triangle_ids = self._proximity.on_surface(centers)
            normals = self._collision_mesh.face_normals[triangle_ids]
            target_curve = closest + (self.clearance + 0.5e-3) * normals
            target_a = _interpolate_curve(target_curve, samples, parameter_a)
            target_b = _interpolate_curve(target_curve, samples, parameter_b)
            _append_vertex_targets(body_targets, ids_a, target_a)
            _append_vertex_targets(body_targets, ids_b, target_b)
        result = self._propagate_seam_targets(result, mesh, body_targets, influence_radius=0.040)

        sleeve_targets: dict[int, list[np.ndarray]] = {}
        for interface in sleeve_body_interfaces:
            ids_a = np.asarray(interface.vertices_a, dtype=np.int32)
            ids_b = np.asarray(interface.vertices_b, dtype=np.int32)
            sleeve_a = bool(np.mean(["sleeve" in labels[index] for index in ids_a]) >= 0.5)
            sleeve_ids = ids_a if sleeve_a else ids_b
            body_ids = ids_b if sleeve_a else ids_a
            sleeve_parameter = _normalized_curve_parameter(
                mesh.material_vertices[sleeve_ids, :2]
            )
            body_parameter = _normalized_curve_parameter(mesh.material_vertices[body_ids, :2])
            target = _interpolate_curve(result[body_ids], body_parameter, sleeve_parameter)
            _append_vertex_targets(sleeve_targets, sleeve_ids, target)
        result = self._propagate_sleeve_cap_targets(result, mesh, sleeve_targets)

        underarm_targets: dict[int, list[np.ndarray]] = {}
        for interface in sleeve_self_interfaces:
            ids_a = np.asarray(interface.vertices_a, dtype=np.int32)
            ids_b = np.asarray(interface.vertices_b, dtype=np.int32)
            parameter_a = _normalized_curve_parameter(mesh.material_vertices[ids_a, :2])
            parameter_b = _normalized_curve_parameter(mesh.material_vertices[ids_b, :2])
            samples = np.unique(np.concatenate((parameter_a, parameter_b)))
            curve_a = _interpolate_curve(result[ids_a], parameter_a, samples)
            curve_b = _interpolate_curve(result[ids_b], parameter_b, samples)
            target_curve = 0.5 * (curve_a + curve_b)
            _append_vertex_targets(
                underarm_targets,
                ids_a,
                _interpolate_curve(target_curve, samples, parameter_a),
            )
            _append_vertex_targets(
                underarm_targets,
                ids_b,
                _interpolate_curve(target_curve, samples, parameter_b),
            )
        return self._propagate_seam_targets(
            result,
            mesh,
            underarm_targets,
            influence_radius=0.038,
        )

    def _propagate_sleeve_cap_targets(
        self,
        vertices: np.ndarray,
        mesh: GarmentMesh,
        target_lists: dict[int, list[np.ndarray]],
    ) -> np.ndarray:
        if not target_lists:
            return vertices
        if mesh.uvs is None or mesh.uvs.shape[0] != len(vertices):
            return self._propagate_seam_targets(
                vertices,
                mesh,
                target_lists,
                influence_radius=0.050,
            )
        result = vertices.copy()
        labels = np.asarray(mesh.panel_labels)
        targets = {
            index: np.mean(np.asarray(values, dtype=np.float32), axis=0)
            for index, values in target_lists.items()
        }
        for side_name in ("left_sleeve", "right_sleeve"):
            panel_ids = np.flatnonzero(
                np.asarray([side_name in label for label in labels], dtype=bool)
            )
            anchors = np.asarray(
                [index for index in targets if side_name in labels[index]],
                dtype=np.int32,
            )
            if panel_ids.size == 0 or anchors.size < 2:
                continue
            anchor_cross = np.asarray(mesh.uvs[anchors, 1], dtype=np.float32)
            displacement = np.asarray(
                [targets[int(index)] - result[int(index)] for index in anchors],
                dtype=np.float32,
            )
            cross_key, inverse = np.unique(
                np.round(anchor_cross, decimals=6),
                return_inverse=True,
            )
            cross_displacement = np.zeros((len(cross_key), 3), dtype=np.float32)
            counts = np.zeros(len(cross_key), dtype=np.float32)
            np.add.at(cross_displacement, inverse, displacement)
            np.add.at(counts, inverse, 1.0)
            cross_displacement /= np.maximum(counts[:, None], 1.0)
            query_cross = np.asarray(mesh.uvs[panel_ids, 1], dtype=np.float32)
            distributed = _periodic_interpolate_curve(
                cross_key.astype(np.float32),
                cross_displacement,
                query_cross,
            )
            along = np.clip(np.asarray(mesh.uvs[panel_ids, 0], dtype=np.float32), 0.0, 1.0)
            smooth = along * along * (3.0 - 2.0 * along)
            result[panel_ids] += distributed * (1.0 - smooth[:, None])
        for index, target in targets.items():
            result[index] = target
        return result

    def _propagate_seam_targets(
        self,
        vertices: np.ndarray,
        mesh: GarmentMesh,
        target_lists: dict[int, list[np.ndarray]],
        *,
        influence_radius: float,
    ) -> np.ndarray:
        if not target_lists:
            return vertices
        result = vertices.copy()
        targets = {
            index: np.mean(np.asarray(values, dtype=np.float32), axis=0)
            for index, values in target_lists.items()
        }
        seam_displacement = {
            index: target - result[index] for index, target in targets.items()
        }
        material_xy = mesh.material_vertices[:, :2]
        labels = np.asarray(mesh.panel_labels)
        for panel_label in np.unique(labels):
            panel_ids = np.flatnonzero(labels == panel_label)
            anchors = np.asarray(
                [index for index in seam_displacement if labels[index] == panel_label],
                dtype=np.int32,
            )
            if panel_ids.size == 0 or anchors.size == 0:
                continue
            distances = np.linalg.norm(
                material_xy[panel_ids, None, :] - material_xy[anchors][None, :, :],
                axis=2,
            )
            weights = np.exp(-np.square(distances / influence_radius))
            weights[distances > 2.5 * influence_radius] = 0.0
            weight_sum = np.sum(weights, axis=1)
            active = weight_sum > 1.0e-6
            anchor_displacements = np.asarray(
                [seam_displacement[int(index)] for index in anchors],
                dtype=np.float32,
            )
            propagated = weights[active] @ anchor_displacements
            propagated /= weight_sum[active, None]
            falloff = np.clip(np.max(weights[active], axis=1), 0.0, 1.0)
            result[panel_ids[active]] += propagated * falloff[:, None]
        for index, target in targets.items():
            result[index] = target
        return result

    def _initial_tshirt_fit(self, mesh: GarmentMesh, garment_offset: float) -> np.ndarray:
        original = mesh.vertices.astype(np.float32)
        x0 = original[:, 0]
        y0 = original[:, 1]
        is_back = np.asarray([label.startswith("back_") for label in mesh.panel_labels], dtype=bool)
        is_left_sleeve = np.asarray(["left_sleeve" in label for label in mesh.panel_labels], dtype=bool)
        is_right_sleeve = np.asarray(["right_sleeve" in label for label in mesh.panel_labels], dtype=bool)
        is_sleeve = is_left_sleeve | is_right_sleeve
        explicit_shoulder = np.asarray(["shoulder" in label for label in mesh.panel_labels], dtype=bool)

        body_source = np.asarray(["front_body" in label for label in mesh.panel_labels], dtype=bool)
        if np.any(body_source):
            body_half_flat = max(float(np.percentile(np.abs(x0[body_source]), 98.0)), 1e-4)
        else:
            body_half_flat = max(float(np.percentile(np.abs(x0), 72.0)), 1e-4)
        if np.any(body_source):
            body_y_min = float(np.min(y0[body_source]))
            body_y_max = float(np.max(y0[body_source]))
        else:
            body_y_min = float(np.min(y0[~is_sleeve])) if np.any(~is_sleeve) else float(np.min(y0))
            body_y_max = float(np.max(y0[~is_sleeve])) if np.any(~is_sleeve) else float(np.max(y0))
        body_y_span = max(body_y_max - body_y_min, 1e-6)
        if np.any(explicit_shoulder):
            is_shoulder = explicit_shoulder
        else:
            is_shoulder = body_source & (y0 >= body_y_min + 0.68 * body_y_span)

        shoulder_source = body_source | is_shoulder
        shoulder_half_flat = max(float(np.percentile(np.abs(x0[shoulder_source]), 98.0)), body_half_flat)
        y_min = body_y_min
        y_max = body_y_max
        y_span = body_y_span

        lm = self._shirt_landmarks
        v = np.empty_like(original)
        y_norm = np.clip((y0 - y_min) / y_span, 0.0, 1.0)
        collar_y = lm["neck_y"] - 0.010
        upper_chest_y = collar_y
        fitted_hem_y = upper_chest_y - min(y_span * 0.98, self.spec.height_cm / 100.0 * 0.43)
        base_y = fitted_hem_y + y_norm * (upper_chest_y - fitted_hem_y)
        garment_circumference = 4.0 * body_half_flat
        ease_ratio = np.clip(garment_circumference / max(self.spec.chest_cm / 100.0, 1e-4), 0.92, 1.35)
        width = self._torso_garment_half_width(base_y) * ease_ratio + garment_offset
        body_u = np.clip(x0 / body_half_flat, -1.08, 1.08)
        shoulder_u = np.clip(x0 / shoulder_half_flat, -1.08, 1.08)
        body_angle = np.clip(body_u, -1.0, 1.0) * (0.5 * np.pi)
        shoulder_angle = np.clip(shoulder_u, -1.0, 1.0) * (0.5 * np.pi)
        shoulder_ease = float(np.clip(self.spec.height_cm / 100.0 * 0.006, 0.006, 0.012))
        v[:, 0] = np.where(
            is_shoulder,
            np.sin(shoulder_angle) * (lm["shoulder_half"] + shoulder_ease),
            np.sin(body_angle) * width,
        )
        # Preserve the learned panel's shoulder and neckline profile. Raising
        # center vertices by x-position inverts a concave crew neck into a
        # straight or convex top edge after sewing.
        v[:, 1] = base_y

        if np.any(is_shoulder):
            shoulder_y0 = y0[is_shoulder]
            shoulder_min = float(np.min(shoulder_y0))
            shoulder_span = max(float(np.max(shoulder_y0) - shoulder_min), 1e-6)
            shoulder_top = np.clip((y0[is_shoulder] - shoulder_min) / shoulder_span, 0.0, 1.0)
            abs_u = np.abs(shoulder_u[is_shoulder])
            neckline_bias = np.clip((0.58 - abs_u) / 0.28, 0.0, 1.0)
            shoulder_base = lm["armhole_y"] + shoulder_top * (lm["shoulder_y"] - lm["armhole_y"])
            v[is_shoulder, 1] = shoulder_base + neckline_bias * shoulder_top * (collar_y - lm["shoulder_y"])

        sleeve_along = np.zeros(v.shape[0], dtype=np.float32)
        sleeve_cross = np.zeros(v.shape[0], dtype=np.float32)
        sleeve_axis_y_all = np.zeros(v.shape[0], dtype=np.float32)
        sleeve_axis_x_all = np.zeros(v.shape[0], dtype=np.float32)
        sleeve_normal_x = np.zeros(v.shape[0], dtype=np.float32)
        sleeve_normal_y = np.zeros(v.shape[0], dtype=np.float32)
        sleeve_material_radius = np.zeros(v.shape[0], dtype=np.float32)
        for mask, side in ((is_left_sleeve, -1.0), (is_right_sleeve, 1.0)):
            if not np.any(mask):
                continue
            flat_outward = side * x0[mask]
            stitched_to_body: list[int] = []
            for a, b in mesh.seam_edges:
                if mask[int(a)] and "front_body" in mesh.panel_labels[int(b)]:
                    stitched_to_body.append(int(a))
                elif mask[int(b)] and "front_body" in mesh.panel_labels[int(a)]:
                    stitched_to_body.append(int(b))
            sleeve_inner = (
                float(np.max(side * x0[np.asarray(stitched_to_body, dtype=np.int32)]))
                if stitched_to_body
                else float(np.min(flat_outward))
            )
            sleeve_outer = max(float(np.max(flat_outward)), sleeve_inner + 1e-4)
            geometric_along = np.clip((flat_outward - sleeve_inner) / (sleeve_outer - sleeve_inner), 0.0, 1.0)

            # Panel arrangement slopes the sleeve centerline downward. Remove
            # that slope before interpreting the second material coordinate as
            # circumference; otherwise a cuff maps to only part of a ring.
            centerline_fit = np.column_stack([flat_outward, np.ones_like(flat_outward)])
            slope, intercept = np.linalg.lstsq(centerline_fit, y0[mask], rcond=None)[0]
            cross_offset = y0[mask] - (slope * flat_outward + intercept)
            cross_span = max(float(np.ptp(cross_offset)), 1e-6)
            if mesh.uvs is not None and mesh.uvs.shape[0] == mesh.vertices.shape[0]:
                along = np.clip(mesh.uvs[mask, 0], 0.0, 1.0)
                cross = np.clip(mesh.uvs[mask, 1], 0.0, 1.0)
            else:
                along = geometric_along
                cross = np.clip((cross_offset - float(np.min(cross_offset))) / cross_span, 0.0, 1.0)
            sleeve_along[mask] = along
            sleeve_cross[mask] = cross
            fit_constraints = self.tshirt_fit_constraints()
            cap_circumference = max(
                cross_span,
                fit_constraints["minimum_armhole_circumference_cm"] / 100.0,
            )
            material_circumference = (1.0 - along) * cap_circumference + along * cross_span
            sleeve_material_radius[mask] = material_circumference / (2.0 * np.pi)

            sleeve_axis_y = (1.0 - along) * lm["armhole_y"] + along * lm["sleeve_cuff_y"]
            arm_abs_x, _arm_front, _arm_back = self._arm_profile_np(side, sleeve_axis_y)
            axis_x = side * arm_abs_x
            derivative_step = 0.002
            arm_x_low, _front_low, _back_low = self._arm_profile_np(side, sleeve_axis_y - derivative_step)
            arm_x_high, _front_high, _back_high = self._arm_profile_np(side, sleeve_axis_y + derivative_step)
            d_abs_x_dy = (arm_x_high - arm_x_low) / (2.0 * derivative_step)
            dy_dt = lm["sleeve_cuff_y"] - lm["armhole_y"]
            dx_dt = side * d_abs_x_dy * dy_dt
            tangent_length = np.sqrt(dx_dt * dx_dt + dy_dt * dy_dt)
            tangent_x = dx_dt / np.maximum(tangent_length, 1e-6)
            tangent_y = dy_dt / np.maximum(tangent_length, 1e-6)
            normal_x = side * -tangent_y
            normal_y = side * tangent_x

            sleeve_axis_y_all[mask] = sleeve_axis_y
            sleeve_axis_x_all[mask] = axis_x
            sleeve_normal_x[mask] = normal_x
            sleeve_normal_y[mask] = normal_y
            v[mask, 0] = axis_x
            v[mask, 1] = sleeve_axis_y

        centerline = np.column_stack([np.zeros(v.shape[0], dtype=np.float32), v[:, 1]])
        front_center = self.support_height_np(centerline)
        back_center = self.back_height_np(centerline)
        fallback_front, fallback_back = self._fallback_shell_depth(np.zeros(v.shape[0]), v[:, 1])
        front_center = np.where(np.isfinite(front_center), front_center, fallback_front)
        back_center = np.where(np.isfinite(back_center), back_center, fallback_back)
        shell_angle = np.where(is_shoulder, shoulder_angle, body_angle)
        side_blend = np.cos(np.clip(np.abs(shell_angle), 0.0, 0.5 * np.pi))
        radial_offset = self.clearance + garment_offset
        front_z = self.base_z + (front_center - self.base_z + radial_offset) * side_blend
        back_z = self.base_z + (back_center - self.base_z - radial_offset) * side_blend
        for mask, side in ((is_left_sleeve, -1.0), (is_right_sleeve, 1.0)):
            if not np.any(mask):
                continue
            along = sleeve_along[mask]
            cross = sleeve_cross[mask]
            side_labels = np.asarray(mesh.panel_labels)[mask]
            has_separate_back = np.any([label.startswith("back_") for label in side_labels])
            back_part = is_back[mask]
            if has_separate_back:
                theta = np.where(back_part, np.pi + np.pi * cross, np.pi * cross)
            else:
                theta = 2.0 * np.pi * cross
            axis_y = sleeve_axis_y_all[mask]
            _arm_abs_x, arm_front, arm_back = self._arm_profile_np(side, axis_y)
            suffix = "L" if side < 0.0 else "R"
            arm_volume = next(
                volume
                for volume in self.arrangement.volumes
                if volume.name == f"upper_arm_{suffix}"
            )
            center_z = (
                (1.0 - along) * float(arm_volume.start[2])
                + along * float(arm_volume.end[2])
            )
            material_radius = sleeve_material_radius[mask]
            body_radius = 0.5 * (arm_front - arm_back)
            radius_z = np.maximum(material_radius * 0.92, body_radius + self.clearance)
            radius_xy = np.maximum(material_radius, 0.025)
            radial = radius_xy * np.cos(theta)
            v[mask, 0] = sleeve_axis_x_all[mask] + sleeve_normal_x[mask] * radial
            v[mask, 1] = axis_y + sleeve_normal_y[mask] * radial
            front_z[mask] = center_z + radius_z * np.sin(theta)
            back_z[mask] = front_z[mask]
        v[:, 2] = np.where(is_back, back_z, front_z)

        # Keep the collar/shoulder bridge close to the upper torso instead of letting
        # sparse body mesh samples pull it up toward the face.
        neckline = is_shoulder & (np.abs(x0) < body_half_flat * 0.55) & (y_norm > 0.68)
        v[neckline & ~is_back, 2] = np.minimum(v[neckline & ~is_back, 2], self.base_z + 0.23)
        v[neckline & is_back, 2] = np.maximum(v[neckline & is_back, 2], self.base_z - 0.07)
        return v

    def _select_wearing_pins(
        self,
        vertices: np.ndarray,
        panel_labels: list[str],
        material_vertices: np.ndarray | None = None,
        seam_edges: list[tuple[int, int]] | None = None,
    ) -> np.ndarray:
        labels = np.asarray(panel_labels)
        is_back = np.asarray([label.startswith("back_") for label in labels], dtype=bool)
        is_sleeve = np.asarray(["sleeve" in label for label in labels], dtype=bool)
        lm = self._shirt_landmarks
        # Stay on the open neckline edge, away from the neckline/shoulder-seam
        # corner. Pinning that shared corner holds the back shoulder on the
        # chest-facing surface and creates an impossible through-body stretch.
        x_collar = min(lm["shoulder_half"] * 0.18, 0.032)
        collar_support_y = lm["neck_y"] - 0.018
        targets = [
            (-x_collar, collar_support_y, False),
            (x_collar, collar_support_y, False),
            (-x_collar, collar_support_y, True),
            (x_collar, collar_support_y, True),
        ]
        pins: list[int] = []
        seam_vertices = set(np.asarray(seam_edges or [], dtype=np.int32).reshape(-1).tolist())
        for x, y, back in targets:
            mask = (is_back == back) & (~is_sleeve)
            if not np.any(mask):
                mask = ~is_sleeve
            indices = np.flatnonzero(mask)
            distances = np.linalg.norm(vertices[indices, :2] - np.asarray([x, y], dtype=np.float32), axis=1)
            pins.append(int(indices[int(np.argmin(distances))]))

        # A worn sleeve has static friction against the upper arm. Keep opposing
        # front/back cuff samples as quasi-static contact anchors. The pair keeps
        # the cuff open around the arm instead of allowing a frictionless solve
        # to flatten the tube or slide it onto the chest.
        for label_name, side in (("left_sleeve", -1.0), ("right_sleeve", 1.0)):
            sleeve = np.asarray([label_name in label for label in labels], dtype=bool)
            indices = np.flatnonzero(sleeve)
            if indices.size == 0:
                continue
            source = vertices if material_vertices is None else material_vertices
            outward = side * source[indices, 0]
            outward_span = max(float(np.ptp(outward)), 1e-6)
            cuff = indices[
                outward
                >= float(np.max(outward)) - max(0.012, 0.18 * outward_span)
            ]
            cuff = np.asarray([index for index in cuff if int(index) not in seam_vertices], dtype=np.int32)
            if cuff.size and float(np.ptp(vertices[cuff, 2])) < 0.020:
                sleeve_interior = np.asarray(
                    [index for index in indices if int(index) not in seam_vertices],
                    dtype=np.int32,
                )
                if sleeve_interior.size:
                    cuff = sleeve_interior
            if cuff.size:
                # Arrangement cylinders may be offset from the final arm shell,
                # so classify the opposing contact pair geometrically here.
                pins.append(int(cuff[np.argmax(vertices[cuff, 2])]))
                pins.append(int(cuff[np.argmin(vertices[cuff, 2])]))

        return np.unique(np.asarray(pins, dtype=np.int32))

    def _build_spatial_index(self) -> None:
        import trimesh
        from scipy.spatial import cKDTree

        self._collision_mesh = trimesh.Trimesh(
            vertices=self.mesh_vertices,
            faces=self.mesh_faces,
            process=False,
        )
        self._collision_mesh.fix_normals()
        self._proximity = trimesh.proximity.ProximityQuery(self._collision_mesh)

        torso_pad = 0.28
        y_min = self.center_y - self.torso_half_height - torso_pad
        y_max = self.center_y + self.torso_half_height + torso_pad
        self._collision_vertices = self.mesh_vertices[(self.mesh_vertices[:, 1] >= y_min) & (self.mesh_vertices[:, 1] <= y_max)]
        if self._collision_vertices.shape[0] < 16:
            self._collision_vertices = self.mesh_vertices
        self._xy_tree = cKDTree(self._collision_vertices[:, :2])
        self._support_radius = max(0.09, 0.85 * self.chest_radius)

    def _surface_height_np(self, xy: np.ndarray, side: str) -> np.ndarray:
        xy = np.asarray(xy, dtype=np.float32)
        if xy.ndim == 1:
            xy = xy[None, :]
        k = min(36, self._collision_vertices.shape[0])
        distances, indices = self._xy_tree.query(xy, k=k, workers=-1)
        if k == 1:
            distances = distances[:, None]
            indices = indices[:, None]
        result = np.full(xy.shape[0], -np.inf if side == "front" else np.inf, dtype=np.float32)
        for row in range(xy.shape[0]):
            near = distances[row] <= self._support_radius
            if not np.any(near):
                continue
            z_values = self._collision_vertices[indices[row, near], 2]
            result[row] = float(np.max(z_values) if side == "front" else np.min(z_values))
        return result

    def _build_width_profile(self) -> None:
        y_min = self.center_y - self.torso_half_height - 0.10
        y_max = self.center_y + self.torso_half_height + 0.10
        ys = np.linspace(y_min, y_max, 96)
        torso_x_limit = max(self.shoulder_half_width * 1.18, self.chest_radius * 1.65)
        widths = []
        for y in ys:
            band = (np.abs(self.mesh_vertices[:, 1] - y) < 0.04) & (
                np.abs(self.mesh_vertices[:, 0]) < torso_x_limit
            )
            if np.any(band):
                widths.append(float(np.percentile(np.abs(self.mesh_vertices[band, 0]), 92.0)))
            else:
                y_norm = np.asarray([(y - self.center_y) / max(self.torso_half_height, 1e-6)])
                widths.append(float(super()._half_width_np(y_norm)[0]))
        self._profile_y = ys.astype(np.float32)
        self._profile_width = np.maximum(np.asarray(widths, dtype=np.float32), 0.03)
        front = self._surface_height_np(np.column_stack([np.zeros_like(ys), ys]), side="front")
        back = self._surface_height_np(np.column_stack([np.zeros_like(ys), ys]), side="back")
        finite = np.isfinite(front) & np.isfinite(back)
        if np.any(finite):
            self.chest_bulge = max(0.08, float(np.percentile(front[finite] - back[finite], 75.0) * 0.5))

    def _build_garment_landmarks(self) -> None:
        v = self.mesh_vertices
        upper_profile = (self._profile_y > self.center_y - 0.10) & (self._profile_y < self.center_y + 0.12)
        if np.any(upper_profile):
            target_half = self.shoulder_half_width
            upper_y = self._profile_y[upper_profile]
            upper_width = self._profile_width[upper_profile]
            shoulder_candidates = upper_y[upper_width >= target_half * 0.96]
            if shoulder_candidates.size:
                shoulder_y = float(np.max(shoulder_candidates))
            else:
                shoulder_y = float(upper_y[int(np.argmin(np.abs(upper_width - target_half)))])
            shoulder_half = float(np.interp(shoulder_y, self._profile_y, self._profile_width))
            shoulder_half = min(max(shoulder_half * 1.06, self.shoulder_half_width), self.shoulder_half_width * 1.12)
        else:
            shoulder_y = self.center_y + 0.06
            shoulder_half = self.shoulder_half_width

        side_mask = (np.abs(v[:, 0]) > self.shoulder_half_width * 0.85) & (v[:, 1] > shoulder_y - 0.22) & (v[:, 1] < shoulder_y + 0.02)
        if np.any(side_mask):
            arm_cuff_x = float(np.percentile(np.abs(v[side_mask, 0]), 88.0))
        else:
            arm_cuff_x = self.shoulder_half_width + 0.16
        neck_y = shoulder_y + 0.068
        hem_y = shoulder_y - 0.60
        self._shirt_landmarks = {
            "shoulder_y": shoulder_y,
            "neck_y": neck_y,
            "hem_y": hem_y,
            "armhole_y": shoulder_y - 0.055,
            "sleeve_cuff_y": shoulder_y - 0.145,
            "shoulder_half": shoulder_half,
            "arm_cuff_x": max(arm_cuff_x, self.shoulder_half_width + 0.13),
        }

    def _build_arm_profiles(self) -> None:
        lm = self._shirt_landmarks
        y_min = lm["sleeve_cuff_y"] - 0.04
        y_max = lm["shoulder_y"] + 0.035
        ys = np.linspace(y_min, y_max, 48, dtype=np.float32)
        self._arm_profile_y = ys
        self._arm_profile_abs_x: dict[float, np.ndarray] = {}
        self._arm_profile_front_z: dict[float, np.ndarray] = {}
        self._arm_profile_back_z: dict[float, np.ndarray] = {}

        for side in (-1.0, 1.0):
            upper_arm = (
                (side * self.mesh_vertices[:, 0] > self.shoulder_half_width * 0.78)
                & (side * self.mesh_vertices[:, 0] < lm["arm_cuff_x"] + 0.04)
                & (self.mesh_vertices[:, 1] > y_min - 0.04)
                & (self.mesh_vertices[:, 1] < y_max + 0.02)
            )
            arm_xy = np.column_stack(
                [
                    side * self.mesh_vertices[upper_arm, 0],
                    self.mesh_vertices[upper_arm, 1],
                ]
            )
            if arm_xy.shape[0] >= 12:
                arm_mean = np.mean(arm_xy, axis=0)
                _u, _s, axes = np.linalg.svd(arm_xy - arm_mean, full_matrices=False)
                arm_axis = axes[0]
                if arm_axis[0] < 0.0:
                    arm_axis = -arm_axis
                if arm_axis[1] > 0.0:
                    arm_axis = -arm_axis
                axis_y = arm_axis[1] if abs(float(arm_axis[1])) > 1e-4 else -1.0
                pca_centers = arm_mean[0] + arm_axis[0] * (ys - arm_mean[1]) / axis_y
            else:
                pca_centers = np.linspace(lm["shoulder_half"], lm["arm_cuff_x"], ys.shape[0])

            centers: list[float] = []
            fronts: list[float] = []
            backs: list[float] = []
            for sample_index, y in enumerate(ys):
                torso_width = float(np.interp(y, self._profile_y, self._profile_width))
                arm_threshold = max(self.shoulder_half_width * 0.78, torso_width * 1.02)
                band = (np.abs(self.mesh_vertices[:, 1] - y) < 0.032) & (side * self.mesh_vertices[:, 0] > arm_threshold)
                if np.any(band):
                    pts = self.mesh_vertices[band]
                    abs_x = side * pts[:, 0]
                    local_center = float(np.median(abs_x))
                    centers.append(float(0.72 * pca_centers[sample_index] + 0.28 * local_center))
                    fronts.append(float(np.percentile(pts[:, 2], 92.0)))
                    backs.append(float(np.percentile(pts[:, 2], 8.0)))
                    continue

                t = np.clip((lm["armhole_y"] - y) / max(lm["armhole_y"] - lm["sleeve_cuff_y"], 1e-6), 0.0, 1.0)
                fallback_abs_x = (1.0 - t) * lm["shoulder_half"] * 1.05 + t * lm["arm_cuff_x"]
                fallback_depth = max(0.040, 0.55 * self.chest_bulge * (1.0 - 0.35 * t))
                centers.append(float(fallback_abs_x))
                fronts.append(float(self.base_z + fallback_depth))
                backs.append(float(self.base_z - fallback_depth * 0.75))

            self._arm_profile_abs_x[side] = np.asarray(centers, dtype=np.float32)
            self._arm_profile_front_z[side] = np.asarray(fronts, dtype=np.float32)
            self._arm_profile_back_z[side] = np.asarray(backs, dtype=np.float32)

    def _build_arrangement_model(self) -> None:
        """Create reusable avatar placement volumes and oriented garment points."""

        vertices = self.mesh_vertices
        lm = self._shirt_landmarks
        height = max(float(np.ptp(vertices[:, 1])), 1.0e-4)
        y_min = float(np.min(vertices[:, 1]))
        offset = max(self.clearance + 0.008, 0.010)

        def torso_shell(y: float, front: bool) -> float:
            query = np.asarray([[0.0, y]], dtype=np.float32)
            value = self.support_height_np(query)[0] if front else self.back_height_np(query)[0]
            if np.isfinite(value):
                return float(value)
            return float(self.base_z + (self.chest_bulge if front else -0.65 * self.chest_bulge))

        def section_center(y: float, side: float | None = None) -> np.ndarray:
            band = np.abs(vertices[:, 1] - y) < max(0.025, 0.025 * height)
            if side is not None:
                band &= side * vertices[:, 0] > 0.015
            points = vertices[band]
            if points.shape[0] < 4:
                x = 0.0 if side is None else side * max(self.waist_radius * 0.45, 0.035)
                return np.asarray([x, y, self.base_z], dtype=np.float32)
            return np.asarray(
                [np.median(points[:, 0]), y, np.median(points[:, 2])],
                dtype=np.float32,
            )

        def orthogonal_normal(axis: np.ndarray, preferred: np.ndarray) -> tuple[float, float, float]:
            axis = np.asarray(unit_vector(axis), dtype=np.float64)
            normal = preferred.astype(np.float64) - axis * float(np.dot(axis, preferred))
            if np.linalg.norm(normal) <= 1.0e-8:
                alternate = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
                normal = alternate - axis * float(np.dot(axis, alternate))
            return unit_vector(normal)

        neck_y = float(lm["neck_y"])
        chest_y = float(lm["shoulder_y"] - 0.13 * height)
        waist_y = float(lm["shoulder_y"] - 0.27 * height)
        hip_y = float(max(y_min + 0.47 * height, lm["shoulder_y"] - 0.43 * height))
        torso_mid_z = 0.5 * (torso_shell(chest_y, True) + torso_shell(chest_y, False))
        chest_depth = max(
            0.5 * (torso_shell(chest_y, True) - torso_shell(chest_y, False)),
            0.035,
        )
        chest_width = max(float(np.interp(chest_y, self._profile_y, self._profile_width)), 0.08)

        volumes: list[ArrangementBoundingVolume] = [
            ArrangementBoundingVolume(
                "torso",
                "torso",
                (0.0, hip_y, torso_mid_z),
                (0.0, neck_y, torso_mid_z),
                chest_width,
                chest_depth,
            ),
            ArrangementBoundingVolume(
                "neck",
                "neck",
                (0.0, neck_y - 0.035, torso_mid_z),
                (0.0, neck_y + 0.035, torso_mid_z),
                max(0.035, 0.20 * chest_width),
                max(0.030, 0.42 * chest_depth),
            ),
            ArrangementBoundingVolume(
                "pelvis",
                "pelvis",
                (0.0, hip_y - 0.08 * height, torso_mid_z),
                (0.0, waist_y, torso_mid_z),
                max(0.78 * chest_width, 0.07),
                max(0.78 * chest_depth, 0.035),
            ),
        ]
        points: list[ArrangementPoint] = []

        def add_point(
            name: str,
            region: str,
            side: str,
            volume: str,
            position: np.ndarray,
            normal: tuple[float, float, float],
            axis: tuple[float, float, float],
            wrap_direction: str,
            roles: tuple[str, ...],
        ) -> None:
            points.append(
                ArrangementPoint(
                    name=name,
                    region=region,
                    side=side,
                    bounding_volume=volume,
                    position=tuple(float(value) for value in position),
                    normal=normal,
                    wrap_axis=axis,
                    offset=offset,
                    wrap_direction=wrap_direction,
                    panel_roles=roles,
                )
            )

        vertical = (0.0, 1.0, 0.0)
        for name, y, roles in (
            ("neck", neck_y, ("neckline", "front_body", "back_body")),
            ("chest", chest_y, ("front_body", "back_body")),
            ("waist", waist_y, ("front_body", "back_body", "waistband")),
            ("hip", hip_y, ("front_body", "back_body", "pelvis", "trouser_body")),
        ):
            add_point(
                f"{name}_front",
                name,
                "center",
                "torso" if name != "hip" else "pelvis",
                np.asarray([0.0, y, torso_shell(y, True) + offset]),
                (0.0, 0.0, 1.0),
                vertical,
                "down",
                roles,
            )
            add_point(
                f"{name}_back",
                name,
                "center",
                "torso" if name != "hip" else "pelvis",
                np.asarray([0.0, y, torso_shell(y, False) - offset]),
                (0.0, 0.0, -1.0),
                vertical,
                "down",
                roles,
            )

        for name, y, roles in (
            ("chest", chest_y, ("front_body", "back_body")),
            ("waist", waist_y, ("front_body", "back_body", "waistband")),
            ("hip", hip_y, ("front_body", "back_body", "trouser_body")),
        ):
            half_width = max(float(np.interp(y, self._profile_y, self._profile_width)), 0.06)
            middle_z = 0.5 * (torso_shell(y, True) + torso_shell(y, False))
            for side, suffix, side_name in ((-1.0, "L", "left"), (1.0, "R", "right")):
                add_point(
                    f"{name}_side_{suffix}",
                    name,
                    side_name,
                    "torso" if name != "hip" else "pelvis",
                    np.asarray([side * (half_width + offset), y, middle_z]),
                    (side, 0.0, 0.0),
                    vertical,
                    "down",
                    roles,
                )

        for side, suffix, side_name in ((-1.0, "L", "left"), (1.0, "R", "right")):
            armhole_y = float(lm["armhole_y"])
            cuff_y = float(lm["sleeve_cuff_y"])
            shoulder_center = np.asarray(
                [side * lm["shoulder_half"], armhole_y, self.base_z],
                dtype=np.float32,
            )
            arm_x, arm_front, arm_back = self._arm_profile_np(
                side,
                np.asarray([armhole_y, cuff_y], dtype=np.float32),
            )
            upper_center = np.asarray(
                [side * arm_x[0], armhole_y, 0.5 * (arm_front[0] + arm_back[0])],
                dtype=np.float32,
            )
            cuff_center = np.asarray(
                [side * arm_x[1], cuff_y, 0.5 * (arm_front[1] + arm_back[1])],
                dtype=np.float32,
            )
            arm_axis = np.asarray(unit_vector(cuff_center - upper_center), dtype=np.float32)
            shoulder_axis = np.asarray(unit_vector(upper_center - shoulder_center), dtype=np.float32)
            radial = orthogonal_normal(arm_axis, np.asarray([0.0, 0.0, 1.0]))
            shoulder_radial = orthogonal_normal(
                shoulder_axis,
                np.asarray([0.0, 0.0, 1.0]),
            )
            arm_radius = max(
                0.5 * float(np.max(arm_front - arm_back)) + self.clearance,
                0.026,
            )
            volumes.extend(
                [
                    ArrangementBoundingVolume(
                        f"shoulder_bridge_{suffix}",
                        "shoulder",
                        tuple(float(value) for value in shoulder_center),
                        tuple(float(value) for value in upper_center),
                        arm_radius,
                        arm_radius,
                    ),
                    ArrangementBoundingVolume(
                        f"upper_arm_{suffix}",
                        "upper_arm",
                        tuple(float(value) for value in upper_center),
                        tuple(float(value) for value in cuff_center),
                        arm_radius,
                        arm_radius,
                    ),
                ]
            )
            add_point(
                f"shoulder_joint_{suffix}",
                "shoulder",
                side_name,
                f"shoulder_bridge_{suffix}",
                shoulder_center + np.asarray(shoulder_radial) * offset,
                shoulder_radial,
                tuple(float(value) for value in shoulder_axis),
                "around",
                (f"{side_name}_sleeve", "armhole", "shoulder"),
            )
            add_point(
                f"upper_arm_{suffix}",
                "upper_arm",
                side_name,
                f"upper_arm_{suffix}",
                upper_center + np.asarray(radial) * (arm_radius + offset),
                radial,
                tuple(float(value) for value in arm_axis),
                "around",
                (f"{side_name}_sleeve", "long_sleeve"),
            )
            add_point(
                f"sleeve_cuff_{suffix}",
                "sleeve_cuff",
                side_name,
                f"upper_arm_{suffix}",
                cuff_center + np.asarray(radial) * (arm_radius + offset),
                radial,
                tuple(float(value) for value in arm_axis),
                "around",
                (f"{side_name}_sleeve", "sleeve_cuff"),
            )

            lateral = side * vertices[:, 0]
            hand_candidates = vertices[lateral >= np.percentile(lateral, 98.0)]
            hand_center = (
                np.median(hand_candidates, axis=0).astype(np.float32)
                if hand_candidates.shape[0] >= 4
                else cuff_center + np.asarray([side * 0.18, -0.12, 0.0], dtype=np.float32)
            )
            wrist_center = 0.78 * hand_center + 0.22 * cuff_center
            forearm_axis = np.asarray(unit_vector(wrist_center - cuff_center), dtype=np.float32)
            forearm_normal = orthogonal_normal(
                forearm_axis,
                np.asarray([0.0, 0.0, 1.0]),
            )
            volumes.append(
                ArrangementBoundingVolume(
                    f"forearm_{suffix}",
                    "forearm",
                    tuple(float(value) for value in cuff_center),
                    tuple(float(value) for value in wrist_center),
                    max(0.72 * arm_radius, 0.018),
                    max(0.72 * arm_radius, 0.018),
                )
            )
            add_point(
                f"wrist_{suffix}",
                "wrist",
                side_name,
                f"forearm_{suffix}",
                wrist_center + np.asarray(forearm_normal) * (0.72 * arm_radius + offset),
                forearm_normal,
                tuple(float(value) for value in forearm_axis),
                "around",
                ("long_sleeve", "glove"),
            )

            knee_y = y_min + 0.34 * height
            ankle_y = y_min + 0.08 * height
            thigh_center = section_center(hip_y - 0.12 * height, side)
            knee_center = section_center(knee_y, side)
            ankle_center = section_center(ankle_y, side)
            thigh_axis = np.asarray(unit_vector(knee_center - thigh_center), dtype=np.float32)
            shin_axis = np.asarray(unit_vector(ankle_center - knee_center), dtype=np.float32)
            thigh_normal = orthogonal_normal(thigh_axis, np.asarray([0.0, 0.0, 1.0]))
            shin_normal = orthogonal_normal(shin_axis, np.asarray([0.0, 0.0, 1.0]))
            leg_radius = max(0.12 * chest_width, 0.025)
            volumes.extend(
                [
                    ArrangementBoundingVolume(
                        f"thigh_{suffix}",
                        "thigh",
                        tuple(float(value) for value in thigh_center),
                        tuple(float(value) for value in knee_center),
                        leg_radius,
                        0.82 * leg_radius,
                    ),
                    ArrangementBoundingVolume(
                        f"lower_leg_{suffix}",
                        "lower_leg",
                        tuple(float(value) for value in knee_center),
                        tuple(float(value) for value in ankle_center),
                        0.82 * leg_radius,
                        0.68 * leg_radius,
                    ),
                ]
            )
            add_point(
                f"thigh_{suffix}",
                "thigh",
                side_name,
                f"thigh_{suffix}",
                thigh_center + np.asarray(thigh_normal) * (leg_radius + offset),
                thigh_normal,
                tuple(float(value) for value in thigh_axis),
                "down",
                ("trouser_leg", "skirt"),
            )
            add_point(
                f"knee_{suffix}",
                "knee",
                side_name,
                f"lower_leg_{suffix}",
                knee_center + np.asarray(shin_normal) * (0.82 * leg_radius + offset),
                shin_normal,
                tuple(float(value) for value in shin_axis),
                "down",
                ("trouser_leg",),
            )
            add_point(
                f"lower_leg_{suffix}",
                "lower_leg",
                side_name,
                f"lower_leg_{suffix}",
                ankle_center + np.asarray(shin_normal) * (0.68 * leg_radius + offset),
                shin_normal,
                tuple(float(value) for value in shin_axis),
                "down",
                ("trouser_leg", "sock"),
            )

        self._arrangement = AvatarArrangement(tuple(volumes), tuple(points))

    def _arm_profile_np(self, side: float, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        side_key = -1.0 if side < 0.0 else 1.0
        y = np.asarray(y, dtype=np.float32)
        abs_x = np.interp(y, self._arm_profile_y, self._arm_profile_abs_x[side_key])
        front = np.interp(y, self._arm_profile_y, self._arm_profile_front_z[side_key])
        back = np.interp(y, self._arm_profile_y, self._arm_profile_back_z[side_key])
        return abs_x.astype(np.float32), front.astype(np.float32), back.astype(np.float32)

    def _torso_garment_half_width(self, y: np.ndarray) -> np.ndarray:
        profiled = np.interp(y, self._profile_y, self._profile_width)
        capped = np.minimum(profiled, self.shoulder_half_width * 1.05)
        return np.maximum(capped, self.waist_radius * 1.20)

    def _fallback_shell_depth(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        width = np.maximum(self._torso_garment_half_width(y), 1e-6)
        u = np.clip(np.abs(x) / width, 0.0, 1.0)
        side_blend = np.sqrt(np.maximum(0.0, 1.0 - u * u))
        return self.base_z + self.chest_bulge * side_blend + self.clearance, self.base_z - 0.62 * self.chest_bulge * side_blend - self.clearance

    def _half_width_np(self, y_norm: np.ndarray) -> np.ndarray:
        y = self.center_y + y_norm * max(self.torso_half_height, 1e-6)
        return np.interp(y, self._profile_y, self._profile_width).astype(np.float32)

    def _half_width_torch(self, y_norm: torch.Tensor) -> torch.Tensor:
        y_norm_np = y_norm.detach().cpu().numpy()
        width = self._half_width_np(y_norm_np)
        return torch.as_tensor(width, dtype=y_norm.dtype, device=y_norm.device)


def _panel_local_cross_coordinates(
    along: np.ndarray,
    cross_offset: np.ndarray,
    faces: np.ndarray,
    global_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize panel width against its local boundary at each axial sample."""

    along = np.asarray(along, dtype=np.float32)
    cross_offset = np.asarray(cross_offset, dtype=np.float32)
    global_indices = np.asarray(global_indices, dtype=np.int32)
    global_span = max(float(np.ptp(cross_offset)), 1e-4)
    fallback_cross = np.clip((cross_offset - float(np.min(cross_offset))) / global_span, 0.0, 1.0)
    if global_indices.size < 3 or faces.size == 0:
        return fallback_cross.astype(np.float32), np.full_like(along, global_span)

    lookup = np.full(int(np.max(global_indices)) + 1, -1, dtype=np.int32)
    lookup[global_indices] = np.arange(global_indices.size, dtype=np.int32)
    valid_faces = np.asarray(faces, dtype=np.int32)
    valid_faces = valid_faces[np.all(valid_faces < lookup.shape[0], axis=1)]
    local_faces = lookup[valid_faces]
    local_faces = local_faces[np.all(local_faces >= 0, axis=1)]
    edge_counts: dict[tuple[int, int], int] = {}
    for a, b, c in local_faces:
        for edge in ((int(a), int(b)), (int(b), int(c)), (int(c), int(a))):
            key = tuple(sorted(edge))
            edge_counts[key] = edge_counts.get(key, 0) + 1
    boundary_edges = [edge for edge, count in edge_counts.items() if count == 1]
    if not boundary_edges:
        return fallback_cross.astype(np.float32), np.full_like(along, global_span)

    normalized = np.empty_like(along)
    local_spans = np.empty_like(along)
    tolerance = max(float(np.ptp(along)) * 1e-5, 1e-6)
    for index, sample_along in enumerate(along):
        intersections: list[float] = []
        for a, b in boundary_edges:
            u0, u1 = float(along[a]), float(along[b])
            w0, w1 = float(cross_offset[a]), float(cross_offset[b])
            if sample_along < min(u0, u1) - tolerance or sample_along > max(u0, u1) + tolerance:
                continue
            if abs(u1 - u0) <= tolerance:
                if abs(float(sample_along) - u0) <= tolerance:
                    intersections.extend([w0, w1])
                continue
            t = float(np.clip((sample_along - u0) / (u1 - u0), 0.0, 1.0))
            intersections.append((1.0 - t) * w0 + t * w1)
        if len(intersections) < 2:
            normalized[index] = fallback_cross[index]
            local_spans[index] = global_span
            continue
        lower, upper = min(intersections), max(intersections)
        span = max(upper - lower, 1e-4)
        normalized[index] = float(np.clip((cross_offset[index] - lower) / span, 0.0, 1.0))
        local_spans[index] = span
    return normalized.astype(np.float32), local_spans.astype(np.float32)


def _seam_components(pairs: np.ndarray) -> list[list[int]]:
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
    return list(groups.values())


def _normalized_curve_parameter(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if len(points) <= 1:
        return np.zeros(len(points), dtype=np.float32)
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths, dtype=np.float64)))
    if cumulative[-1] <= 1.0e-10:
        return np.linspace(0.0, 1.0, len(points), dtype=np.float32)
    return (cumulative / cumulative[-1]).astype(np.float32)


def _interpolate_curve(
    values: np.ndarray,
    parameter: np.ndarray,
    query: np.ndarray,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    parameter = np.asarray(parameter, dtype=np.float32)
    query = np.asarray(query, dtype=np.float32)
    if len(values) != len(parameter):
        raise ValueError("curve values and parameters must have matching lengths")
    if len(values) == 1:
        return np.repeat(values, len(query), axis=0)
    return np.column_stack(
        [np.interp(query, parameter, values[:, axis]) for axis in range(values.shape[1])]
    ).astype(np.float32)


def _periodic_interpolate_curve(
    parameter: np.ndarray,
    values: np.ndarray,
    query: np.ndarray,
) -> np.ndarray:
    parameter = np.asarray(parameter, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    order = np.argsort(parameter)
    parameter = parameter[order]
    values = values[order]
    if len(parameter) == 1:
        return np.repeat(values, len(query), axis=0)
    extended_parameter = np.concatenate((parameter - 1.0, parameter, parameter + 1.0))
    extended_values = np.concatenate((values, values, values), axis=0)
    return np.column_stack(
        [
            np.interp(np.mod(query, 1.0), extended_parameter, extended_values[:, axis])
            for axis in range(values.shape[1])
        ]
    ).astype(np.float32)


def _append_vertex_targets(
    target_lists: dict[int, list[np.ndarray]],
    vertex_ids: np.ndarray,
    targets: np.ndarray,
) -> None:
    for vertex, target in zip(vertex_ids, targets):
        target_lists.setdefault(int(vertex), []).append(np.asarray(target, dtype=np.float32))


def _morph_body_measurements(vertices: np.ndarray, spec: AvatarSpec) -> np.ndarray:
    """Apply a smooth measurement morph after the source mesh is height-scaled."""

    result = vertices.astype(np.float32).copy()
    y_min = float(result[:, 1].min())
    y_span = max(float(result[:, 1].max() - y_min), 1e-6)
    height_fraction = (result[:, 1] - y_min) / y_span
    center_x = float(np.median(result[:, 0]))
    center_z = float(np.median(result[:, 2]))
    torso_x_limit = float(np.clip(y_span * 0.11, 0.11, 0.18))

    def section_circumference(anchor: float) -> float:
        band = (np.abs(height_fraction - anchor) < 0.035) & (
            np.abs(result[:, 0] - center_x) < torso_x_limit
        )
        points = result[band][:, [0, 2]]
        if points.shape[0] < 6:
            return 0.0
        try:
            from scipy.spatial import ConvexHull

            return float(ConvexHull(points).area)
        except Exception:
            radii = np.percentile(np.abs(points - np.median(points, axis=0)), 90, axis=0)
            return float(2.0 * np.pi * np.sqrt(np.mean(radii * radii)))

    waist_current = section_circumference(0.56)
    chest_current = section_circumference(0.69)
    waist_scale = np.clip((spec.waist_cm / 100.0) / max(waist_current, 1e-4), 0.65, 1.45)
    chest_scale = np.clip((spec.chest_cm / 100.0) / max(chest_current, 1e-4), 0.65, 1.45)

    shoulder_band = (np.abs(height_fraction - 0.78) < 0.035) & (
        np.abs(result[:, 0] - center_x) < torso_x_limit
    )
    shoulder_width = 2.0 * float(np.percentile(np.abs(result[shoulder_band, 0] - center_x), 98)) if np.any(shoulder_band) else 0.0
    shoulder_scale = np.clip((spec.shoulder_cm / 100.0) / max(shoulder_width, 1e-4), 0.70, 1.35)

    anchors = np.asarray([0.0, 0.45, 0.56, 0.69, 0.78, 0.86, 1.0], dtype=np.float32)
    x_scales = np.asarray([1.0, 1.0, waist_scale, chest_scale, shoulder_scale, 1.0, 1.0], dtype=np.float32)
    z_scales = np.asarray([1.0, 1.0, waist_scale, chest_scale, chest_scale, 1.0, 1.0], dtype=np.float32)
    x_scale = np.interp(height_fraction, anchors, x_scales)
    z_scale = np.interp(height_fraction, anchors, z_scales)
    dx = result[:, 0] - center_x
    abs_dx = np.abs(dx)
    torso_blend = np.clip((torso_x_limit - abs_dx) / max(torso_x_limit * 0.28, 1e-6), 0.0, 1.0)
    torso_blend = torso_blend * torso_blend * (3.0 - 2.0 * torso_blend)

    arm_enter = np.clip((height_fraction - 0.46) / 0.08, 0.0, 1.0)
    arm_exit = np.clip((0.88 - height_fraction) / 0.08, 0.0, 1.0)
    arm_weight = arm_enter * arm_exit
    limb_shift = np.sign(dx) * torso_x_limit * (shoulder_scale - 1.0) * arm_weight
    torso_x = dx * x_scale
    limb_x = dx + limb_shift
    result[:, 0] = center_x + torso_blend * torso_x + (1.0 - torso_blend) * limb_x

    torso_z = center_z + (result[:, 2] - center_z) * z_scale
    result[:, 2] = torso_blend * torso_z + (1.0 - torso_blend) * result[:, 2]
    return result.astype(np.float32)


def _load_obj_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Body mesh not found: {path}")
    import trimesh

    loaded = trimesh.load(path, force="mesh", process=False)
    if isinstance(loaded, trimesh.Scene):
        loaded = trimesh.util.concatenate(tuple(loaded.geometry.values()))
    if loaded.vertices is None or loaded.faces is None:
        raise ValueError(f"Body mesh has no vertices/faces: {path}")
    components = loaded.split(only_watertight=False)
    if components:
        loaded = max(components, key=lambda component: len(component.faces))
    return np.asarray(loaded.vertices, dtype=np.float32), np.asarray(loaded.faces, dtype=np.int32)


def _remap_axes(vertices: np.ndarray, x_axis: str, up_axis: str, front_axis: str) -> np.ndarray:
    axes = [_axis_vector(x_axis), _axis_vector(up_axis), _axis_vector(front_axis)]
    used = [axis for axis, _sign in axes]
    if len(set(used)) != 3:
        raise ValueError("Body mesh axis mapping must use three distinct source axes.")
    mapped = np.stack([vertices[:, axis] * sign for axis, sign in axes], axis=1)
    return mapped.astype(np.float32)


def _resolve_axis_mapping(
    vertices: np.ndarray,
    x_axis: str,
    up_axis: str,
    front_axis: str,
) -> tuple[str, str, str]:
    """Resolve optional auto axes from a human mesh's bounding-box extents."""

    x_index, _x_sign = _axis_vector(x_axis)
    extents = np.ptp(vertices, axis=0)
    remaining = [axis for axis in range(3) if axis != x_index]
    names = np.asarray(["x", "y", "z"])

    if up_axis == "auto":
        up_index = max(remaining, key=lambda axis: float(extents[axis]))
        up_axis = str(names[up_index])
    else:
        up_index, _up_sign = _axis_vector(up_axis)

    if front_axis == "auto":
        candidates = [axis for axis in remaining if axis != up_index]
        if len(candidates) != 1:
            raise ValueError("Cannot infer body mesh front axis from the selected x/up axes.")
        front_axis = str(names[candidates[0]])

    # Reuse the strict distinct-axis validation in the remapper.
    _remap_axes(vertices[:1], x_axis, up_axis, front_axis)
    return x_axis, up_axis, front_axis


def _validate_axis_mapping(vertices: np.ndarray, axes: tuple[str, str, str]) -> None:
    extents = np.ptp(vertices, axis=0)
    if float(extents[1]) < 1.25 * max(float(extents[2]), 1e-8):
        raise ValueError(
            "Body mesh axis mapping is implausible for an upright human: "
            f"mapped up extent {extents[1]:.3f} is not larger than front-depth "
            f"extent {extents[2]:.3f}. Resolved axes were x={axes[0]}, "
            f"up={axes[1]}, front={axes[2]}. Pass explicit source axes if needed."
        )


def _axis_vector(name: str) -> tuple[int, float]:
    name = name.strip().lower()
    sign = -1.0 if name.startswith("-") else 1.0
    axis = name[1:] if name.startswith("-") else name
    if axis not in {"x", "y", "z"}:
        raise ValueError(f"Unsupported axis {name!r}; use x, y, z, -x, -y, or -z.")
    return {"x": 0, "y": 1, "z": 2}[axis], sign
