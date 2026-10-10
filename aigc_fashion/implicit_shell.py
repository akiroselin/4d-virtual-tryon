from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from .data import GarmentMesh
from .topology_correction import MeshTopologyReport, analyze_triangle_mesh

if TYPE_CHECKING:
    from .body_mesh_avatar import BodyMeshAvatar


@dataclass(frozen=True)
class ImplicitShellBuildReport:
    method: str
    field: str
    source_faces: int
    extracted_faces: int
    extracted_vertices: int
    level_set_offset_m: float
    cuff_level_set_offset_m: float
    learned_cuff_opening_m: float | None
    learned_neck_width_m: float | None
    front_neck_drop_m: float
    garment_top_y_m: float
    garment_hem_y_m: float
    topology: MeshTopologyReport
    collision: dict[str, float]
    runtime_seam_constraints: bool = False

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["topology"] = asdict(self.topology)
        return data


def _crotch_height(body_vertices: np.ndarray, low_y: float, high_y: float) -> float | None:
    """Top of the leg gap: scanning up from ``low_y``, the last height whose
    horizontal slice has no body surface on the midline. ``None`` if the slice
    at ``low_y`` already reaches the midline (the hem is above the crotch)."""

    crotch = None
    for y in np.arange(low_y, high_y, 0.005):
        band = body_vertices[np.abs(body_vertices[:, 1] - y) < 0.006]
        if len(band) == 0:
            continue
        if float(np.abs(band[:, 0]).min()) <= 0.012:
            break
        crotch = float(y)
    return crotch


def build_body_sdf_tshirt_shell(
    avatar: BodyMeshAvatar,
    panel_atlas: GarmentMesh,
    *,
    target_faces: int = 1800,
    level_set_offset_m: float | None = None,
) -> tuple[GarmentMesh, ImplicitShellBuildReport]:
    """Extract one continuous T-shirt midsurface from the avatar SDF.

    The learned panel atlas determines garment length and semantic provenance.
    Geometry comes from an outward body-distance level set restricted to torso
    and upper-arm placement volumes. The resulting genus-0 surface has four
    physical boundaries: neck, hem, and two cuffs. It contains no sewing edges.
    """

    if target_faces < 200:
        raise ValueError("target_faces must be at least 200")
    import trimesh

    body_vertices = np.asarray(avatar.mesh_vertices, dtype=np.float64)
    body_faces = np.asarray(avatar.mesh_faces, dtype=np.int32)
    centroids = body_vertices[body_faces].mean(axis=1)
    labels = np.asarray(panel_atlas.panel_labels)
    body_chart = np.asarray(["front_body" in label for label in labels], dtype=bool)
    if not np.any(body_chart):
        raise ValueError("the learned atlas needs front/back body panels")

    body_length = float(np.ptp(panel_atlas.material_vertices[body_chart, 1]))
    learned_neck_width = _learned_neck_width(panel_atlas)
    neck_half_width = max(
        0.5 * float(learned_neck_width or 0.0),
        0.21 * float(avatar.spec.shoulder_cm) / 100.0,
    )
    front_neck_drop = 0.026
    top_y = float(avatar._shirt_landmarks["neck_y"] - 0.010)
    hem_y = top_y - min(
        0.98 * body_length,
        0.46 * float(avatar.spec.height_cm) / 100.0,
    )
    # A hem below the crotch would wrap each leg separately and split the hem
    # into two loops, so the shell could not be a T-shirt. Keep it above.
    crotch_y = _crotch_height(body_vertices, hem_y, top_y)
    if crotch_y is not None:
        hem_y = max(hem_y, crotch_y + 0.03)
    torso_width = np.interp(
        centroids[:, 1],
        avatar._profile_y,
        avatar._profile_width,
    )
    front_surface = avatar.support_height_np(centroids[:, :2])
    back_surface = avatar.back_height_np(centroids[:, :2])
    valid_depth = np.isfinite(front_surface) & np.isfinite(back_surface)
    depth_midpoint = np.full(len(centroids), float(avatar.base_z), dtype=np.float64)
    depth_midpoint[valid_depth] = 0.5 * (
        front_surface[valid_depth] + back_surface[valid_depth]
    )
    front_side = centroids[:, 2] >= depth_midpoint
    neck_parameter = np.clip(
        np.abs(centroids[:, 0]) / max(neck_half_width, 1.0e-8),
        0.0,
        1.0,
    )
    neck_profile = 1.0 - np.square(neck_parameter)
    neck_side_y = float(avatar._shirt_landmarks["shoulder_y"] + 0.032)
    local_top_y = np.full(len(centroids), top_y, dtype=np.float64)
    neck_zone = np.abs(centroids[:, 0]) <= 1.04 * neck_half_width
    local_top_y[neck_zone & front_side] = (
        neck_side_y - front_neck_drop * neck_profile[neck_zone & front_side]
    )
    local_top_y[neck_zone & ~front_side] = (
        neck_side_y + 0.009 * neck_profile[neck_zone & ~front_side]
    )
    selected = (
        (centroids[:, 1] >= hem_y)
        & (centroids[:, 1] <= local_top_y)
        & (np.abs(centroids[:, 0]) <= 1.10 * torso_width)
    )

    for suffix in ("L", "R"):
        volume = next(
            volume
            for volume in avatar.arrangement.volumes
            if volume.name == f"upper_arm_{suffix}"
        )
        start = np.asarray(volume.start, dtype=np.float64)
        end = np.asarray(volume.end, dtype=np.float64)
        direction = end - start
        length = float(np.linalg.norm(direction))
        direction /= max(length, 1.0e-10)
        relative = centroids - start
        axial = relative @ direction
        radial = np.linalg.norm(
            relative - axial[:, None] * direction[None, :],
            axis=1,
        )
        selected |= (
            (axial >= -0.030)
            & (axial <= 1.05 * length)
            & (radial <= 1.65 * max(volume.radius_x, volume.radius_y))
            & (
                centroids[:, 1]
                <= float(avatar._shirt_landmarks["shoulder_y"] + 0.035)
            )
        )

    extracted = trimesh.Trimesh(
        vertices=body_vertices,
        faces=body_faces[selected],
        process=False,
    )
    extracted.remove_unreferenced_vertices()
    components = extracted.split(only_watertight=False)
    if not components:
        raise RuntimeError("the garment SDF region did not produce a surface")
    extracted = max(components, key=lambda component: len(component.faces))
    extracted.fix_normals()

    offset = max(
        float(level_set_offset_m or 0.0),
        float(avatar.clearance) + 0.012,
    )
    learned_cuff_opening = _learned_cuff_opening(panel_atlas)
    cuff_offset = max(float(avatar.clearance) + 0.004, 0.5 * offset)
    surface_normals = np.array(extracted.vertex_normals, dtype=np.float64, copy=True)
    probe_distance = max(0.25 * offset, 1.0e-3)
    probe_signed_distance = avatar._proximity.signed_distance(
        np.asarray(extracted.vertices, dtype=np.float64)
        + probe_distance * surface_normals
    )
    # Trimesh reports positive signed distance inside this source body. Open
    # patches do not have a reliable outward orientation after ``fix_normals``;
    # use the closed avatar SDF to choose the side that moves into free space.
    reverse_winding = float(np.median(probe_signed_distance)) > 0.0
    if reverse_winding:
        surface_normals *= -1.0
    extracted_vertices = np.asarray(extracted.vertices, dtype=np.float64)
    surface_offsets = np.full(len(extracted_vertices), offset, dtype=np.float64)
    for side, suffix in ((-1.0, "L"), (1.0, "R")):
        volume = next(
            volume
            for volume in avatar.arrangement.volumes
            if volume.name == f"upper_arm_{suffix}"
        )
        start = np.asarray(volume.start, dtype=np.float64)
        direction = np.asarray(volume.end, dtype=np.float64) - start
        length = float(np.linalg.norm(direction))
        direction /= max(length, 1.0e-10)
        relative = extracted_vertices - start
        axial = relative @ direction
        radial = np.linalg.norm(
            relative - axial[:, None] * direction[None, :],
            axis=1,
        )
        sleeve = (
            (axial >= -0.030)
            & (axial <= 1.12 * length)
            & (radial <= 1.90 * max(volume.radius_x, volume.radius_y))
            & (side * extracted_vertices[:, 0] > 0.68 * avatar.shoulder_half_width)
        )
        parameter = np.clip(
            (axial - 0.15 * length) / max(0.85 * length, 1.0e-10),
            0.0,
            1.0,
        )
        parameter = parameter * parameter * (3.0 - 2.0 * parameter)
        sleeve_offset = (1.0 - parameter) * offset + parameter * cuff_offset
        surface_offsets[sleeve] = np.minimum(
            surface_offsets[sleeve],
            sleeve_offset[sleeve],
        )
    shell_vertices = extracted_vertices + surface_offsets[:, None] * surface_normals
    shell_faces = np.asarray(extracted.faces, dtype=np.int32)
    if reverse_winding:
        shell_faces = shell_faces[:, [0, 2, 1]]
    shell_surface = trimesh.Trimesh(shell_vertices, shell_faces, process=False)
    if len(shell_faces) > target_faces:
        try:
            candidate = shell_surface.simplify_quadric_decimation(
                face_count=target_faces
            )
            candidate_vertices = np.asarray(candidate.vertices, dtype=np.float64)
            candidate_faces = np.asarray(candidate.faces, dtype=np.int32)
            candidate_topology = analyze_triangle_mesh(
                candidate_vertices,
                candidate_faces,
            )
            if (
                candidate_topology.connected_components == 1
                and candidate_topology.nonmanifold_edges == 0
                and candidate_topology.boundary_loops == 4
                and candidate_topology.genus == 0
            ):
                shell_vertices = candidate_vertices
                shell_faces = candidate_faces
        except Exception:
            pass

    shell_vertices = _regularize_tshirt_boundaries(
        avatar,
        shell_vertices,
        shell_faces,
    )

    topology = analyze_triangle_mesh(shell_vertices, shell_faces)
    if (
        topology.connected_components != 1
        or topology.nonmanifold_edges != 0
        or topology.boundary_loops != 4
        or topology.genus != 0
    ):
        raise RuntimeError(
            "implicit T-shirt shell must be one genus-0 manifold with neck, hem, and two cuff boundaries"
        )

    semantic_labels = _semantic_shell_labels(avatar, shell_vertices)
    uv = shell_vertices[:, :2].copy()
    uv -= uv.min(axis=0, keepdims=True)
    uv /= np.maximum(uv.max(axis=0, keepdims=True), 1.0e-8)
    mesh = GarmentMesh(
        vertices=shell_vertices.astype(np.float32),
        faces=shell_faces.astype(np.int32),
        pinned_vertices=np.empty(0, dtype=np.int32),
        panel_labels=semantic_labels,
        seam_edges=[],
        rest_vertices=shell_vertices.astype(np.float32),
        uvs=uv.astype(np.float32),
        seam_interfaces=[],
    )
    collision = avatar.collision_diagnostics_np(mesh.vertices, mesh.faces)
    report = ImplicitShellBuildReport(
        method="implicit body-SDF garment field to explicit thin shell",
        field="phi_g(x)=body_signed_distance(x)+placement-conditioned garment mask",
        source_faces=int(np.sum(selected)),
        extracted_faces=int(len(mesh.faces)),
        extracted_vertices=int(len(mesh.vertices)),
        level_set_offset_m=offset,
        cuff_level_set_offset_m=cuff_offset,
        learned_cuff_opening_m=learned_cuff_opening,
        learned_neck_width_m=learned_neck_width,
        front_neck_drop_m=front_neck_drop,
        garment_top_y_m=top_y,
        garment_hem_y_m=hem_y,
        topology=topology,
        collision={key: float(value) for key, value in collision.items()},
    )
    return mesh, report


def _learned_cuff_opening(panel_atlas: GarmentMesh) -> float | None:
    labels = np.asarray(panel_atlas.panel_labels)
    estimates: list[float] = []
    for side, name in ((-1.0, "left_sleeve"), (1.0, "right_sleeve")):
        indices = np.flatnonzero(np.asarray([name in label for label in labels]))
        if indices.size < 3:
            continue
        chart = np.asarray(panel_atlas.material_vertices[indices, :2], dtype=np.float64)
        outward = side * chart[:, 0]
        cutoff = float(np.max(outward) - 0.025 * max(np.ptp(outward), 1.0e-8))
        cuff = chart[outward >= cutoff]
        if len(cuff) >= 2:
            estimates.append(float(np.ptp(cuff[:, 1])))
    return float(np.mean(estimates)) if estimates else None


def _learned_neck_width(panel_atlas: GarmentMesh) -> float | None:
    labels = np.asarray(panel_atlas.panel_labels)
    edge_counts: dict[tuple[int, int], int] = {}
    for face in np.asarray(panel_atlas.faces, dtype=np.int64):
        for first, second in (
            (int(face[0]), int(face[1])),
            (int(face[1]), int(face[2])),
            (int(face[2]), int(face[0])),
        ):
            edge = tuple(sorted((first, second)))
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
    boundary = {
        vertex
        for edge, count in edge_counts.items()
        if count == 1
        for vertex in edge
    }
    indices = np.asarray(
        [
            index
            for index, label in enumerate(labels)
            if label == "front_front_body" and index in boundary
        ],
        dtype=np.int64,
    )
    if indices.size < 3:
        return None
    chart = np.asarray(panel_atlas.material_vertices[indices, :2], dtype=np.float64)
    width = max(float(np.ptp(chart[:, 0])), 1.0e-8)
    height = max(float(np.ptp(chart[:, 1])), 1.0e-8)
    neck = (
        (np.abs(chart[:, 0]) <= 0.25 * width)
        & (chart[:, 1] >= float(np.max(chart[:, 1]) - 0.055 * height))
    )
    if int(np.sum(neck)) < 2:
        return None
    return float(np.ptp(chart[neck, 0]))


def _regularize_tshirt_boundaries(
    avatar: BodyMeshAvatar,
    vertices: np.ndarray,
    faces: np.ndarray,
) -> np.ndarray:
    """Turn triangle-selection cuts into smooth neck, hem, and cuff curves."""

    result = np.asarray(vertices, dtype=np.float64).copy()
    loops = _ordered_boundary_loops(faces)
    if len(loops) != 4:
        return result
    loops = sorted(loops, key=lambda loop: float(np.mean(result[loop, 1])))
    hem, cuff_a, cuff_b, neck = loops

    for loop in loops:
        points = result[loop]
        for _ in range(3):
            laplacian = 0.5 * (np.roll(points, 1, axis=0) + np.roll(points, -1, axis=0)) - points
            points = points + 0.32 * laplacian
            laplacian = 0.5 * (np.roll(points, 1, axis=0) + np.roll(points, -1, axis=0)) - points
            points = points - 0.34 * laplacian
        result[loop] = points

    result[hem, 1] = float(np.mean(result[hem, 1]))
    for loop in (cuff_a, cuff_b):
        suffix = "L" if float(np.mean(result[loop, 0])) < 0.0 else "R"
        volume = next(
            volume
            for volume in avatar.arrangement.volumes
            if volume.name == f"upper_arm_{suffix}"
        )
        start = np.asarray(volume.start, dtype=np.float64)
        direction = np.asarray(volume.end, dtype=np.float64) - start
        direction /= max(float(np.linalg.norm(direction)), 1.0e-10)
        axial = (result[loop] - start) @ direction
        result[loop] += (float(np.mean(axial)) - axial)[:, None] * direction[None, :]

    # Keep the learned crew-neck depth while removing high-frequency mesh cuts.
    neck_center = np.mean(result[neck], axis=0)
    neck_radial = result[neck] - neck_center
    neck_scale = np.linalg.norm(neck_radial[:, (0, 2)], axis=1)
    if float(np.max(neck_scale)) > 1.0e-8:
        result[neck, 1] = np.convolve(
            np.r_[result[neck, 1][-2:], result[neck, 1], result[neck, 1][:2]],
            np.asarray([1.0, 2.0, 3.0, 2.0, 1.0]) / 9.0,
            mode="valid",
        )
    signed_distance = avatar._proximity.signed_distance(result)
    minimum_clearance = float(avatar.clearance) + 0.0030
    active = signed_distance > -minimum_clearance
    if np.any(active):
        closest, _distance, triangle_ids = avatar._proximity.on_surface(result[active])
        normals = np.asarray(
            avatar._collision_mesh.face_normals[triangle_ids],
            dtype=np.float64,
        )
        result[active] = closest + minimum_clearance * normals
    return result


def _ordered_boundary_loops(faces: np.ndarray) -> list[np.ndarray]:
    counts: dict[tuple[int, int], int] = {}
    for face in np.asarray(faces, dtype=np.int64):
        for first, second in (
            (int(face[0]), int(face[1])),
            (int(face[1]), int(face[2])),
            (int(face[2]), int(face[0])),
        ):
            edge = tuple(sorted((first, second)))
            counts[edge] = counts.get(edge, 0) + 1
    adjacency: dict[int, list[int]] = {}
    for (first, second), count in counts.items():
        if count != 1:
            continue
        adjacency.setdefault(first, []).append(second)
        adjacency.setdefault(second, []).append(first)
    if any(len(neighbours) != 2 for neighbours in adjacency.values()):
        return []

    loops: list[np.ndarray] = []
    visited: set[int] = set()
    for start in adjacency:
        if start in visited:
            continue
        loop: list[int] = []
        previous = -1
        current = start
        while current not in visited:
            visited.add(current)
            loop.append(current)
            neighbours = adjacency[current]
            following = neighbours[0] if neighbours[0] != previous else neighbours[1]
            previous, current = current, following
        loops.append(np.asarray(loop, dtype=np.int64))
    return loops


def _semantic_shell_labels(
    avatar: BodyMeshAvatar,
    vertices: np.ndarray,
) -> list[str]:
    vertices = np.asarray(vertices, dtype=np.float64)
    sleeve_side = np.zeros(len(vertices), dtype=np.int8)
    for side, suffix in ((-1, "L"), (1, "R")):
        volume = next(
            volume
            for volume in avatar.arrangement.volumes
            if volume.name == f"upper_arm_{suffix}"
        )
        start = np.asarray(volume.start, dtype=np.float64)
        end = np.asarray(volume.end, dtype=np.float64)
        direction = end - start
        length = float(np.linalg.norm(direction))
        direction /= max(length, 1.0e-10)
        relative = vertices - start
        axial = relative @ direction
        radial = np.linalg.norm(
            relative - axial[:, None] * direction[None, :],
            axis=1,
        )
        active = (
            (axial >= -0.035)
            & (axial <= 1.10 * length)
            & (radial <= 1.90 * max(volume.radius_x, volume.radius_y))
            & (side * vertices[:, 0] > 0.72 * avatar.shoulder_half_width)
        )
        sleeve_side[active] = side

    front_surface = avatar.support_height_np(vertices[:, :2])
    back_surface = avatar.back_height_np(vertices[:, :2])
    valid = np.isfinite(front_surface) & np.isfinite(back_surface)
    midpoint = np.full(len(vertices), float(avatar.base_z), dtype=np.float64)
    midpoint[valid] = 0.5 * (front_surface[valid] + back_surface[valid])
    front = vertices[:, 2] >= midpoint
    result: list[str] = []
    for index in range(len(vertices)):
        shell = "front" if front[index] else "back"
        if sleeve_side[index] < 0:
            result.append(f"{shell}_left_sleeve")
        elif sleeve_side[index] > 0:
            result.append(f"{shell}_right_sleeve")
        else:
            result.append(f"{shell}_front_body")
    return result
