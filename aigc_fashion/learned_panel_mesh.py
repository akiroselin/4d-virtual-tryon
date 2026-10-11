from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import json
from pathlib import Path
import re
from typing import Any

import numpy as np

from .data import GarmentMesh, SeamInterface
from .vto_preflight import infer_panel_role


@dataclass
class _PanelBuild:
    name: str
    label: str
    vertices: np.ndarray
    faces: np.ndarray
    edge_vertices: dict[int, list[int]]
    parameter_uvs: np.ndarray | None = None


@dataclass(frozen=True)
class _StitchSegment:
    source_index: int
    panel_a: str
    panel_b: str
    edge_a: int
    edge_b: int
    path_a: tuple[int, ...]
    path_b: tuple[int, ...]


@dataclass(frozen=True)
class _SeamDraft:
    source_indices: tuple[int, ...]
    panel_a: str
    panel_b: str
    edge_ids_a: tuple[int, ...]
    edge_ids_b: tuple[int, ...]
    path_a: tuple[int, ...]
    path_b: tuple[int, ...]
    endpoint_cost_same: float
    endpoint_cost_reversed: float


def garment_mesh_from_sewformer_spec(spec_path: str | Path, resolution: int = 24) -> GarmentMesh:
    """Convert a SewFormer/GarmentCode sewing-pattern JSON into our cloth mesh.

    SewFormer serializes learned panels as 2D vertices plus ordered edges and a
    stitch list. This adapter turns those panels into triangulated cloth patches,
    arranges T-shirt-like panels in the flat coordinate convention used by the
    avatar dressing code, and preserves stitches as seam constraints.
    """

    spec_path = Path(spec_path)
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    pattern = spec.get("pattern", spec)
    panels = pattern.get("panels", {})
    if not isinstance(panels, dict) or not panels:
        raise ValueError(f"learned pattern spec has no panels: {spec_path}")

    properties = spec.get("properties", pattern.get("properties", {}))
    units_in_meter = float(properties.get("units_in_meter", 100.0))
    if units_in_meter <= 0.0:
        raise ValueError("properties.units_in_meter must be positive")

    raw_outlines = {
        name: _panel_outline(np.asarray(panel.get("vertices", []), dtype=np.float32), units_in_meter)
        for name, panel in panels.items()
    }
    raw_outlines = {name: outline for name, outline in raw_outlines.items() if outline.shape[0] >= 3}
    if not raw_outlines:
        raise ValueError(f"learned pattern spec panels have no usable vertices: {spec_path}")

    labels = _classify_panel_labels(raw_outlines)
    arranged = _arrange_tshirt_panels(raw_outlines, labels)
    target_step = _target_step(arranged, resolution)

    builds: list[_PanelBuild] = []
    vertex_offset = 0
    all_vertices: list[np.ndarray] = []
    all_rest_vertices: list[np.ndarray] = []
    all_uvs: list[np.ndarray] = []
    all_faces: list[np.ndarray] = []
    panel_edge_vertices: dict[tuple[str, int], list[int]] = {}
    panel_vertex_labels: list[str] = []

    for name, panel in panels.items():
        if name not in arranged:
            continue
        build = _build_panel_mesh(name, labels[name], panel, arranged[name], target_step, units_in_meter)
        if build.vertices.shape[0] < 3 or build.faces.shape[0] == 0:
            continue
        builds.append(build)
        all_vertices.append(np.column_stack([build.vertices, np.ones(build.vertices.shape[0], dtype=np.float32)]))
        all_rest_vertices.append(np.column_stack([build.vertices, np.zeros(build.vertices.shape[0], dtype=np.float32)]))
        if build.parameter_uvs is not None:
            all_uvs.append(build.parameter_uvs.astype(np.float32))
        else:
            uv_min = build.vertices.min(axis=0)
            uv_span = np.maximum(build.vertices.max(axis=0) - uv_min, 1e-6)
            all_uvs.append(((build.vertices - uv_min) / uv_span).astype(np.float32))
        all_faces.append(build.faces + vertex_offset)
        panel_vertex_labels.extend([build.label] * build.vertices.shape[0])
        for edge_idx, ids in build.edge_vertices.items():
            panel_edge_vertices[(name, edge_idx)] = [vertex_offset + idx for idx in ids]
        vertex_offset += build.vertices.shape[0]

    if not builds:
        raise ValueError(f"learned pattern spec produced no triangulated panels: {spec_path}")

    vertices = np.concatenate(all_vertices, axis=0).astype(np.float32)
    rest_vertices = np.concatenate(all_rest_vertices, axis=0).astype(np.float32)
    uvs = np.concatenate(all_uvs, axis=0).astype(np.float32)
    faces = np.concatenate(all_faces, axis=0).astype(np.int32)
    seam_interfaces = _build_seam_interfaces(
        pattern.get("stitches", []),
        panel_edge_vertices,
        vertices[:, :2],
    )
    seam_edges = _seam_edges_from_interfaces(seam_interfaces, vertices[:, :2])
    pinned = _select_learned_pattern_pins(vertices, panel_vertex_labels)
    return GarmentMesh(
        vertices=vertices,
        faces=faces,
        pinned_vertices=pinned,
        panel_labels=panel_vertex_labels,
        seam_edges=sorted(seam_edges),
        rest_vertices=rest_vertices,
        uvs=uvs,
        seam_interfaces=seam_interfaces,
    )


def _panel_outline(vertices: np.ndarray, units_in_meter: float) -> np.ndarray:
    if vertices.ndim != 2 or vertices.shape[0] < 3 or vertices.shape[1] < 2:
        return np.empty((0, 2), dtype=np.float32)
    outline = vertices[:, :2].astype(np.float32) / units_in_meter
    return _drop_duplicate_tail(outline)


def _classify_panel_labels(outlines: dict[str, np.ndarray]) -> dict[str, str]:
    labels: dict[str, str] = {}
    areas = {name: abs(_polygon_area(points)) for name, points in outlines.items()}
    body_names = sorted(outlines, key=lambda name: areas[name], reverse=True)[:2]

    sleeve_candidates: list[str] = []
    for name in outlines:
        lowered = name.lower()
        if "sleeve" in lowered or "slv" in lowered:
            sleeve_candidates.append(name)
        elif name not in body_names:
            sleeve_candidates.append(name)

    for name in outlines:
        lowered = name.lower()
        if "sleeve" in lowered or name in sleeve_candidates:
            semantic_role = infer_panel_role(name)
            if semantic_role in {
                "front_left_sleeve",
                "back_left_sleeve",
                "front_right_sleeve",
                "back_right_sleeve",
            }:
                labels[name] = semantic_role
            else:
                side = _side_from_name(name)
                labels[name] = f"front_{side}_sleeve"
        elif infer_panel_role(name) == "back_body" or "back" in lowered:
            labels[name] = "back_front_body"
        elif infer_panel_role(name) == "front_body" or "front" in lowered:
            labels[name] = "front_front_body"
        elif name == (body_names[1] if len(body_names) > 1 else ""):
            labels[name] = "back_front_body"
        else:
            labels[name] = "front_front_body"

    ambiguous_sleeves = [name for name in sleeve_candidates if _sleeve_side_is_unmarked(name)]
    if len(sleeve_candidates) == 2 and len(ambiguous_sleeves) == 2:
        for name in sorted(ambiguous_sleeves)[1:]:
            labels[name] = "front_right_sleeve"
    return labels


def _side_from_name(name: str) -> str:
    lowered = name.lower()
    compact = re.sub(r"[^a-z0-9]+", "", lowered)
    if "right" in lowered or lowered.startswith("r_") or lowered.startswith("r-") or compact.startswith(("rfsleeve", "rbsleeve", "rslv")):
        return "right"
    return "left"


def _sleeve_side_is_unmarked(name: str) -> bool:
    lowered = name.lower()
    compact = re.sub(r"[^a-z0-9]+", "", lowered)
    explicit_prefix = compact.startswith(("lfsleeve", "lbsleeve", "rfsleeve", "rbsleeve", "lslv", "rslv"))
    return "left" not in lowered and "right" not in lowered and not explicit_prefix


def _arrange_tshirt_panels(outlines: dict[str, np.ndarray], labels: dict[str, str]) -> dict[str, np.ndarray]:
    body_outlines = [points for name, points in outlines.items() if "front_body" in labels[name]]
    if body_outlines:
        body_width = max(float(points[:, 0].max() - points[:, 0].min()) for points in body_outlines)
        body_length = max(float(points[:, 1].max() - points[:, 1].min()) for points in body_outlines)
    else:
        all_points = np.concatenate(list(outlines.values()), axis=0)
        body_width = float(all_points[:, 0].max() - all_points[:, 0].min())
        body_length = float(all_points[:, 1].max() - all_points[:, 1].min())
    body_width = max(body_width, 0.18)
    body_length = max(body_length, 0.24)
    body_half = body_width * 0.5
    y_min = -body_length * 0.58
    y_max = body_length * 0.42

    arranged: dict[str, np.ndarray] = {}
    for name, points in outlines.items():
        label = labels[name]
        mins = points.min(axis=0)
        maxs = points.max(axis=0)
        span = np.maximum(maxs - mins, 1e-6)
        normalized = (points - mins) / span
        if "sleeve" in label:
            side = -1.0 if "left_sleeve" in label else 1.0
            if span[0] >= span[1]:
                u = normalized[:, 0]
                cross = normalized[:, 1] - 0.5
                sleeve_length = span[0]
                sleeve_opening = span[1]
            else:
                u = normalized[:, 1]
                cross = normalized[:, 0] - 0.5
                sleeve_length = span[1]
                sleeve_opening = span[0]
            sleeve_length = max(float(sleeve_length), body_width * 0.28)
            sleeve_opening = max(float(sleeve_opening), body_width * 0.20)
            sleeve_center_y = y_max * 0.56 - u * max(0.035, sleeve_opening * 0.36)
            arranged[name] = np.column_stack(
                [
                    side * (body_half * 0.82 + u * sleeve_length),
                    sleeve_center_y + cross * sleeve_opening,
                ]
            ).astype(np.float32)
        else:
            x = (normalized[:, 0] - 0.5) * body_width
            y = y_min + normalized[:, 1] * (y_max - y_min)
            arranged[name] = np.column_stack([x, y]).astype(np.float32)
    return arranged


def _target_step(outlines: dict[str, np.ndarray], resolution: int) -> float:
    all_points = np.concatenate(list(outlines.values()), axis=0)
    span = np.maximum(all_points.max(axis=0) - all_points.min(axis=0), 1e-6)
    return float(max(span.max() / max(resolution, 8), 0.012))


def _build_panel_mesh(
    name: str,
    label: str,
    panel: dict[str, Any],
    outline: np.ndarray,
    target_step: float,
    units_in_meter: float,
) -> _PanelBuild:
    source_vertices = _panel_outline(np.asarray(panel.get("vertices", []), dtype=np.float32), units_in_meter)
    if source_vertices.shape[0] != outline.shape[0]:
        source_vertices = outline
    edges = panel.get("edges") or [
        {"endpoints": [idx, (idx + 1) % outline.shape[0]]}
        for idx in range(outline.shape[0])
    ]
    if _is_quad_panel(source_vertices, edges):
        return _build_quad_panel_mesh(name, label, arranged_vertices=outline, edges=edges, target_step=target_step)

    boundary, edge_vertices = _sample_panel_edges(source_vertices, outline, edges, target_step)
    points = _add_interior_points(boundary, target_step)
    parameter_points = _sleeve_parameter_coordinates(points, boundary) if "sleeve" in label else None
    faces = _triangulate_panel(points, boundary, parameter_points=parameter_points)
    points, faces, edge_vertices = _compact_panel_mesh(points, faces, edge_vertices)
    if "sleeve" in label:
        points, faces, edge_vertices = _subdivide_panel_once(points, faces, edge_vertices)
        parameter_uvs = _sleeve_mesh_uvs(points, faces, edge_vertices)
    else:
        parameter_uvs = None
    return _PanelBuild(
        name=name,
        label=label,
        vertices=points,
        faces=faces,
        edge_vertices=edge_vertices,
        parameter_uvs=parameter_uvs,
    )


def _is_quad_panel(vertices: np.ndarray, edges: list[dict[str, Any]]) -> bool:
    if vertices.shape[0] != 4 or len(edges) != 4:
        return False
    if any("curvature" in edge for edge in edges):
        return False
    endpoints = [edge.get("endpoints") for edge in edges]
    return all(isinstance(pair, list) and len(pair) == 2 for pair in endpoints)


def _build_quad_panel_mesh(
    name: str,
    label: str,
    arranged_vertices: np.ndarray,
    edges: list[dict[str, Any]],
    target_step: float,
) -> _PanelBuild:
    v0, v1, v2, v3 = arranged_vertices[:4]
    width = max(float(np.linalg.norm(v1 - v0)), float(np.linalg.norm(v2 - v3)))
    height = max(float(np.linalg.norm(v3 - v0)), float(np.linalg.norm(v2 - v1)))
    cols = max(2, int(np.ceil(width / max(target_step, 1e-6))))
    rows = max(2, int(np.ceil(height / max(target_step, 1e-6))))

    vertices: list[np.ndarray] = []
    for iy in range(rows + 1):
        t = iy / rows
        left = (1.0 - t) * v0 + t * v3
        right = (1.0 - t) * v1 + t * v2
        for ix in range(cols + 1):
            s = ix / cols
            vertices.append(((1.0 - s) * left + s * right).astype(np.float32))

    def grid_id(ix: int, iy: int) -> int:
        return iy * (cols + 1) + ix

    faces: list[list[int]] = []
    for iy in range(rows):
        for ix in range(cols):
            a = grid_id(ix, iy)
            b = grid_id(ix + 1, iy)
            c = grid_id(ix, iy + 1)
            d = grid_id(ix + 1, iy + 1)
            faces.append([a, b, c])
            faces.append([b, d, c])

    canonical_paths = {
        (0, 1): [grid_id(ix, 0) for ix in range(cols + 1)],
        (1, 2): [grid_id(cols, iy) for iy in range(rows + 1)],
        (2, 3): [grid_id(ix, rows) for ix in range(cols, -1, -1)],
        (3, 0): [grid_id(0, iy) for iy in range(rows, -1, -1)],
    }
    edge_vertices: dict[int, list[int]] = {}
    for edge_idx, edge in enumerate(edges):
        start, end = [int(value) for value in edge.get("endpoints", [])]
        path = canonical_paths.get((start, end))
        if path is None:
            reverse = canonical_paths.get((end, start))
            path = list(reversed(reverse)) if reverse is not None else []
        if len(path) >= 2:
            edge_vertices[edge_idx] = path

    return _PanelBuild(
        name=name,
        label=label,
        vertices=np.asarray(vertices, dtype=np.float32),
        faces=np.asarray(faces, dtype=np.int32),
        edge_vertices=edge_vertices,
    )


def _sample_panel_edges(
    source_vertices: np.ndarray,
    arranged_vertices: np.ndarray,
    edges: list[dict[str, Any]],
    target_step: float,
) -> tuple[np.ndarray, dict[int, list[int]]]:
    points: list[np.ndarray] = []
    edge_vertices: dict[int, list[int]] = {}

    def add_point(point: np.ndarray) -> int:
        if points and np.linalg.norm(points[-1] - point) < 1e-7:
            return len(points) - 1
        points.append(point.astype(np.float32))
        return len(points) - 1

    for edge_idx, edge in enumerate(edges):
        endpoints = edge.get("endpoints", [edge_idx, (edge_idx + 1) % arranged_vertices.shape[0]])
        start_idx, end_idx = int(endpoints[0]), int(endpoints[1])
        if start_idx >= arranged_vertices.shape[0] or end_idx >= arranged_vertices.shape[0]:
            continue
        start = arranged_vertices[start_idx]
        end = arranged_vertices[end_idx]
        source_start = source_vertices[start_idx]
        source_end = source_vertices[end_idx]
        curvature = edge.get("curvature")
        source_control = _relative_control_point(source_start, source_end, curvature)
        arranged_control = _relative_control_point(start, end, curvature)
        sample_t = np.linspace(0.0, 1.0, 17, dtype=np.float32)
        source_curve = _quadratic_bezier(source_start, source_control, source_end, sample_t)
        source_len = float(np.linalg.norm(np.diff(source_curve, axis=0), axis=1).sum())
        segments = max(1, int(np.ceil(source_len / max(target_step, 1e-6))))
        ids: list[int] = []
        for segment in range(segments + 1):
            t = segment / segments
            point = _quadratic_bezier(start, arranged_control, end, np.asarray([t], dtype=np.float32))[0]
            idx = add_point(point)
            ids.append(idx)
        edge_vertices[edge_idx] = ids

    if len(points) >= 2 and np.linalg.norm(points[0] - points[-1]) < 1e-7:
        points.pop()
        last_idx = len(points)
        for ids in edge_vertices.values():
            for idx, value in enumerate(ids):
                if value == last_idx:
                    ids[idx] = 0
    return np.asarray(points, dtype=np.float32), edge_vertices


def _relative_control_point(start: np.ndarray, end: np.ndarray, curvature: Any) -> np.ndarray:
    if not isinstance(curvature, (list, tuple, np.ndarray)) or len(curvature) < 2:
        return (start + end) * 0.5
    edge = end - start
    edge_perp = np.asarray([-edge[1], edge[0]], dtype=np.float32)
    return start + float(curvature[0]) * edge + float(curvature[1]) * edge_perp


def _quadratic_bezier(
    start: np.ndarray,
    control: np.ndarray,
    end: np.ndarray,
    t: np.ndarray,
) -> np.ndarray:
    t = t[:, None]
    return (1.0 - t) ** 2 * start + 2.0 * (1.0 - t) * t * control + t**2 * end


def _add_interior_points(boundary: np.ndarray, target_step: float) -> np.ndarray:
    mins = boundary.min(axis=0)
    maxs = boundary.max(axis=0)
    xs = np.arange(mins[0] + target_step, maxs[0], target_step, dtype=np.float32)
    ys = np.arange(mins[1] + target_step, maxs[1], target_step, dtype=np.float32)
    interior: list[list[float]] = []
    for x in xs:
        for y in ys:
            point = np.asarray([x, y], dtype=np.float32)
            if _point_in_polygon(point, boundary) and _distance_to_polygon_boundary(point, boundary) > 0.35 * target_step:
                interior.append([float(x), float(y)])
    if not interior:
        centroid = boundary.mean(axis=0, keepdims=True)
        return np.concatenate([boundary, centroid.astype(np.float32)], axis=0)
    return np.concatenate([boundary, np.asarray(interior, dtype=np.float32)], axis=0)


def _triangulate_panel(
    points: np.ndarray,
    boundary: np.ndarray,
    parameter_points: np.ndarray | None = None,
) -> np.ndarray:
    try:
        from scipy.spatial import Delaunay

        triangulation_points = points[:, :2] if parameter_points is None else parameter_points[:, :2]
        simplices = Delaunay(triangulation_points).simplices
        faces = []
        for tri in simplices:
            centroid = points[tri, :2].mean(axis=0)
            if _point_in_polygon(centroid, boundary):
                if _triangle_area(points[tri]) < 0.0:
                    tri = tri[[0, 2, 1]]
                faces.append(tri.astype(np.int32))
        if faces:
            return np.asarray(faces, dtype=np.int32)
    except Exception:
        pass

    centroid_idx = points.shape[0] - 1
    faces = []
    boundary_count = boundary.shape[0]
    for idx in range(boundary_count):
        faces.append([idx, (idx + 1) % boundary_count, centroid_idx])
    return np.asarray(faces, dtype=np.int32)


def _sleeve_parameter_coordinates(points: np.ndarray, boundary: np.ndarray) -> np.ndarray:
    """Map a flat sleeve panel to axial/circumference coordinates for meshing."""

    axis = points[:, 0].astype(np.float64)
    fit = np.column_stack([axis, np.ones_like(axis)])
    slope, intercept = np.linalg.lstsq(fit, points[:, 1], rcond=None)[0]
    cross_offset = points[:, 1] - (slope * axis + intercept)

    boundary_axis = boundary[:, 0].astype(np.float64)
    boundary_cross = boundary[:, 1] - (slope * boundary_axis + intercept)
    cross = np.empty(points.shape[0], dtype=np.float64)
    tolerance = max(float(np.ptp(axis)) * 1e-6, 1e-8)
    global_lower = float(np.min(boundary_cross))
    global_span = max(float(np.ptp(boundary_cross)), 1e-6)
    for index, sample_axis in enumerate(axis):
        intersections: list[float] = []
        for edge_index in range(boundary.shape[0]):
            next_index = (edge_index + 1) % boundary.shape[0]
            u0, u1 = float(boundary_axis[edge_index]), float(boundary_axis[next_index])
            w0, w1 = float(boundary_cross[edge_index]), float(boundary_cross[next_index])
            if sample_axis < min(u0, u1) - tolerance or sample_axis > max(u0, u1) + tolerance:
                continue
            if abs(u1 - u0) <= tolerance:
                if abs(float(sample_axis) - u0) <= tolerance:
                    intersections.extend([w0, w1])
                continue
            t = float(np.clip((sample_axis - u0) / (u1 - u0), 0.0, 1.0))
            intersections.append((1.0 - t) * w0 + t * w1)
        if len(intersections) >= 2:
            lower, upper = min(intersections), max(intersections)
            span = max(upper - lower, 1e-6)
        else:
            lower, span = global_lower, global_span
        cross[index] = np.clip((cross_offset[index] - lower) / span, 0.0, 1.0)

    axis_normalized = (axis - float(np.min(axis))) / max(float(np.ptp(axis)), 1e-6)
    return np.column_stack([axis_normalized, cross]).astype(np.float32)


def _sleeve_mesh_uvs(
    points: np.ndarray,
    faces: np.ndarray,
    edge_vertices: dict[int, list[int]],
) -> np.ndarray | None:
    """Build topology-aware axial/circumference coordinates for a sleeve panel."""

    required_edges = {0, 1, 2, 3, 4, 5, 6}
    if points.shape[0] < 3 or faces.size == 0 or not required_edges.issubset(edge_vertices):
        return None
    mesh_edges = np.unique(
        np.sort(
            np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0),
            axis=1,
        ),
        axis=0,
    )
    edge_lengths = np.linalg.norm(points[mesh_edges[:, 0]] - points[mesh_edges[:, 1]], axis=1)
    try:
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import dijkstra

        rows = np.concatenate([mesh_edges[:, 0], mesh_edges[:, 1]])
        cols = np.concatenate([mesh_edges[:, 1], mesh_edges[:, 0]])
        weights = np.concatenate([edge_lengths, edge_lengths])
        graph = coo_matrix((weights, (rows, cols)), shape=(points.shape[0], points.shape[0])).tocsr()

        def distance_to(indices: list[int]) -> np.ndarray:
            unique = np.unique(np.asarray(indices, dtype=np.int32))
            distances = dijkstra(graph, directed=False, indices=unique)
            return np.min(np.atleast_2d(distances), axis=0)

        underarm_a = edge_vertices[0]
        cuff = edge_vertices[1]
        underarm_b = edge_vertices[2]
        cap = sorted({index for edge_id in (3, 4, 5, 6) for index in edge_vertices[edge_id]})
        distance_a = distance_to(underarm_a)
        distance_b = distance_to(underarm_b)
        distance_cap = distance_to(cap)
        distance_cuff = distance_to(cuff)
        cross = distance_a / np.maximum(distance_a + distance_b, 1e-8)
        along = distance_cap / np.maximum(distance_cap + distance_cuff, 1e-8)
        cross[np.asarray(underarm_a, dtype=np.int32)] = 0.0
        cross[np.asarray(underarm_b, dtype=np.int32)] = 1.0
        along[np.asarray(cap, dtype=np.int32)] = 0.0
        along[np.asarray(cuff, dtype=np.int32)] = 1.0
        return np.column_stack([along, cross]).astype(np.float32)
    except Exception:
        return None


def _compact_panel_mesh(
    points: np.ndarray,
    faces: np.ndarray,
    edge_vertices: dict[int, list[int]],
) -> tuple[np.ndarray, np.ndarray, dict[int, list[int]]]:
    if faces.size == 0:
        return points, faces, edge_vertices
    used = np.unique(faces.reshape(-1))
    remap = np.full(points.shape[0], -1, dtype=np.int32)
    remap[used] = np.arange(used.shape[0], dtype=np.int32)
    compact_edges: dict[int, list[int]] = {}
    for edge_idx, ids in edge_vertices.items():
        mapped = [int(remap[idx]) for idx in ids if 0 <= idx < remap.shape[0] and remap[idx] >= 0]
        if len(mapped) >= 2:
            compact_edges[edge_idx] = mapped
    return points[used].astype(np.float32), remap[faces].astype(np.int32), compact_edges


def _subdivide_panel_once(
    points: np.ndarray,
    faces: np.ndarray,
    edge_vertices: dict[int, list[int]],
) -> tuple[np.ndarray, np.ndarray, dict[int, list[int]]]:
    """Split every triangle into four while preserving named boundary paths."""

    if faces.size == 0:
        return points, faces, edge_vertices

    refined_points = [point.astype(np.float32) for point in points]
    midpoint_ids: dict[tuple[int, int], int] = {}

    def midpoint(a: int, b: int) -> int:
        key = tuple(sorted((int(a), int(b))))
        if key not in midpoint_ids:
            midpoint_ids[key] = len(refined_points)
            refined_points.append(((points[key[0]] + points[key[1]]) * 0.5).astype(np.float32))
        return midpoint_ids[key]

    refined_faces: list[list[int]] = []
    for a, b, c in faces.astype(np.int32):
        ab = midpoint(int(a), int(b))
        bc = midpoint(int(b), int(c))
        ca = midpoint(int(c), int(a))
        refined_faces.extend(
            [
                [int(a), ab, ca],
                [ab, int(b), bc],
                [ca, bc, int(c)],
                [ab, bc, ca],
            ]
        )

    refined_edges: dict[int, list[int]] = {}
    for edge_id, path in edge_vertices.items():
        if len(path) < 2:
            continue
        refined_path: list[int] = [int(path[0])]
        for a, b in zip(path[:-1], path[1:]):
            refined_path.extend([midpoint(int(a), int(b)), int(b)])
        refined_edges[edge_id] = refined_path

    return (
        np.asarray(refined_points, dtype=np.float32),
        np.asarray(refined_faces, dtype=np.int32),
        refined_edges,
    )


def _stitch_edges(
    stitches: Any,
    panel_edge_vertices: dict[tuple[str, int], list[int]],
    xy: np.ndarray,
) -> set[tuple[int, int]]:
    interfaces = _build_seam_interfaces(stitches, panel_edge_vertices, xy)
    return _seam_edges_from_interfaces(interfaces, xy)


def _build_seam_interfaces(
    stitches: Any,
    panel_edge_vertices: dict[tuple[str, int], list[int]],
    xy: np.ndarray,
) -> list[SeamInterface]:
    """Build GarmentCode-style oriented interfaces from segmented stitches.

    SewFormer commonly splits a single armhole into several stitched edge
    pairs.  Orienting those pieces independently can map the sleeve shoulder to
    the underarm.  We first concatenate connected pieces, then solve all
    interface orientations together so sewing junctions remain topologically
    consistent.
    """

    if not isinstance(stitches, list):
        return []
    segments: list[_StitchSegment] = []
    for source_index, stitch in enumerate(stitches):
        if not isinstance(stitch, list) or len(stitch) != 2:
            continue
        side_a, side_b = stitch
        panel_a = str(side_a.get("panel"))
        panel_b = str(side_b.get("panel"))
        edge_a = int(side_a.get("edge", -1))
        edge_b = int(side_b.get("edge", -1))
        key_a = (panel_a, edge_a)
        key_b = (panel_b, edge_b)
        ids_a = panel_edge_vertices.get(key_a)
        ids_b = panel_edge_vertices.get(key_b)
        if not ids_a or not ids_b:
            continue
        if panel_a != panel_b and panel_a > panel_b:
            panel_a, panel_b = panel_b, panel_a
            edge_a, edge_b = edge_b, edge_a
            ids_a, ids_b = ids_b, ids_a
        segments.append(
            _StitchSegment(
                source_index=source_index,
                panel_a=panel_a,
                panel_b=panel_b,
                edge_a=edge_a,
                edge_b=edge_b,
                path_a=tuple(int(index) for index in ids_a),
                path_b=tuple(int(index) for index in ids_b),
            )
        )

    grouped: dict[tuple[str, str], list[_StitchSegment]] = {}
    for segment in segments:
        grouped.setdefault((segment.panel_a, segment.panel_b), []).append(segment)

    drafts: list[_SeamDraft] = []
    points = np.asarray(xy, dtype=np.float32)
    for panel_pair, group in sorted(grouped.items()):
        for component in _connected_stitch_components(group):
            path_a, edge_ids_a = _assemble_connected_paths(
                [(segment.edge_a, segment.path_a) for segment in component]
            )
            path_b, edge_ids_b = _assemble_connected_paths(
                [(segment.edge_b, segment.path_b) for segment in component]
            )
            same = float(
                np.linalg.norm(points[path_a[0]] - points[path_b[0]])
                + np.linalg.norm(points[path_a[-1]] - points[path_b[-1]])
            )
            reversed_cost = float(
                np.linalg.norm(points[path_a[0]] - points[path_b[-1]])
                + np.linalg.norm(points[path_a[-1]] - points[path_b[0]])
            )
            drafts.append(
                _SeamDraft(
                    source_indices=tuple(sorted(segment.source_index for segment in component)),
                    panel_a=panel_pair[0],
                    panel_b=panel_pair[1],
                    edge_ids_a=edge_ids_a,
                    edge_ids_b=edge_ids_b,
                    path_a=path_a,
                    path_b=path_b,
                    endpoint_cost_same=same,
                    endpoint_cost_reversed=reversed_cost,
                )
            )

    orientation = _solve_interface_orientations(drafts)
    interfaces: list[SeamInterface] = []
    for index, (draft, reverse_b) in enumerate(zip(drafts, orientation)):
        path_b = tuple(reversed(draft.path_b)) if reverse_b else draft.path_b
        edge_ids_b = tuple(reversed(draft.edge_ids_b)) if reverse_b else draft.edge_ids_b
        source = "_".join(str(value) for value in draft.source_indices)
        interfaces.append(
            SeamInterface(
                name=f"seam_{index:03d}_stitches_{source}",
                panel_a=draft.panel_a,
                panel_b=draft.panel_b,
                edge_ids_a=draft.edge_ids_a,
                edge_ids_b=edge_ids_b,
                vertices_a=draft.path_a,
                vertices_b=path_b,
                side_b_reversed=bool(reverse_b),
            )
        )
    return interfaces


def _connected_stitch_components(segments: list[_StitchSegment]) -> list[list[_StitchSegment]]:
    remaining = set(range(len(segments)))
    components: list[list[_StitchSegment]] = []
    while remaining:
        first = min(remaining)
        remaining.remove(first)
        component_ids = [first]
        frontier = [first]
        while frontier:
            current = frontier.pop()
            for candidate in sorted(remaining):
                if _paths_share_endpoint(
                    segments[current].path_a,
                    segments[candidate].path_a,
                ) and _paths_share_endpoint(
                    segments[current].path_b,
                    segments[candidate].path_b,
                ):
                    remaining.remove(candidate)
                    component_ids.append(candidate)
                    frontier.append(candidate)
        components.append([segments[index] for index in sorted(component_ids)])
    return components


def _paths_share_endpoint(path_a: tuple[int, ...], path_b: tuple[int, ...]) -> bool:
    return bool({path_a[0], path_a[-1]} & {path_b[0], path_b[-1]})


def _assemble_connected_paths(
    segments: list[tuple[int, tuple[int, ...]]],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if not segments:
        raise ValueError("cannot assemble an empty seam interface")
    degree: dict[int, int] = {}
    for _edge_id, path in segments:
        degree[path[0]] = degree.get(path[0], 0) + 1
        degree[path[-1]] = degree.get(path[-1], 0) + 1
    starts = sorted(vertex for vertex, value in degree.items() if value == 1)
    current = starts[0] if starts else min(degree)
    remaining = list(segments)
    path_out: list[int] = []
    edges_out: list[int] = []
    while remaining:
        candidates = [
            (index, edge_id, path)
            for index, (edge_id, path) in enumerate(remaining)
            if path[0] == current or path[-1] == current
        ]
        if not candidates:
            raise ValueError("stitch segments do not form a continuous boundary interface")
        segment_index, edge_id, segment_path = min(candidates, key=lambda item: item[1])
        remaining.pop(segment_index)
        if segment_path[-1] == current:
            segment_path = tuple(reversed(segment_path))
        if path_out:
            path_out.extend(segment_path[1:])
        else:
            path_out.extend(segment_path)
        edges_out.append(edge_id)
        current = segment_path[-1]
    return tuple(path_out), tuple(edges_out)


def _solve_interface_orientations(drafts: list[_SeamDraft]) -> tuple[bool, ...]:
    if not drafts:
        return ()
    geometric_seed = tuple(
        draft.endpoint_cost_reversed < draft.endpoint_cost_same for draft in drafts
    )
    if len(drafts) <= 16:
        candidates = product((False, True), repeat=len(drafts))
    else:
        candidates = _orientation_local_search_seeds(drafts, geometric_seed)
    best_score: tuple[int, float, tuple[bool, ...]] | None = None
    best: tuple[bool, ...] | None = None
    for candidate in candidates:
        bits = tuple(candidate)
        conflicts = _interface_junction_conflicts(drafts, bits)
        geometric_cost = sum(
            draft.endpoint_cost_reversed if reverse else draft.endpoint_cost_same
            for draft, reverse in zip(drafts, bits)
        )
        score = (conflicts, float(geometric_cost), bits)
        if best_score is None or score < best_score:
            best_score = score
            best = bits
    assert best is not None
    return best


def _orientation_local_search_seeds(
    drafts: list[_SeamDraft],
    geometric_seed: tuple[bool, ...],
) -> list[tuple[bool, ...]]:
    seeds = [geometric_seed, (False,) * len(drafts), (True,) * len(drafts)]
    results: list[tuple[bool, ...]] = []
    for seed in seeds:
        current = seed
        while True:
            current_score = (
                _interface_junction_conflicts(drafts, current),
                sum(
                    draft.endpoint_cost_reversed if reverse else draft.endpoint_cost_same
                    for draft, reverse in zip(drafts, current)
                ),
            )
            options: list[tuple[tuple[int, float], tuple[bool, ...]]] = []
            for index in range(len(drafts)):
                candidate = list(current)
                candidate[index] = not candidate[index]
                bits = tuple(candidate)
                score = (
                    _interface_junction_conflicts(drafts, bits),
                    sum(
                        draft.endpoint_cost_reversed if reverse else draft.endpoint_cost_same
                        for draft, reverse in zip(drafts, bits)
                    ),
                )
                options.append((score, bits))
            next_score, next_bits = min(options)
            if next_score >= current_score:
                break
            current = next_bits
        results.append(current)
    return sorted(set(results))


def _interface_junction_conflicts(
    drafts: list[_SeamDraft],
    orientation: tuple[bool, ...],
) -> int:
    parent: dict[int, int] = {}
    self_seam_parent: dict[int, int] = {}
    panel_by_vertex: dict[int, str] = {}

    def find(nodes: dict[int, int], vertex: int) -> int:
        nodes.setdefault(vertex, vertex)
        if nodes[vertex] != vertex:
            nodes[vertex] = find(nodes, nodes[vertex])
        return nodes[vertex]

    def union(nodes: dict[int, int], a: int, b: int) -> None:
        root_a = find(nodes, a)
        root_b = find(nodes, b)
        if root_a != root_b:
            nodes[root_b] = root_a

    for draft, reverse_b in zip(drafts, orientation):
        a0, a1 = draft.path_a[0], draft.path_a[-1]
        b0, b1 = draft.path_b[0], draft.path_b[-1]
        if reverse_b:
            b0, b1 = b1, b0
        panel_by_vertex.update({a0: draft.panel_a, a1: draft.panel_a})
        panel_by_vertex.update({b0: draft.panel_b, b1: draft.panel_b})
        union(parent, a0, b0)
        union(parent, a1, b1)
        if draft.panel_a == draft.panel_b:
            union(self_seam_parent, a0, b0)
            union(self_seam_parent, a1, b1)

    components: dict[int, list[int]] = {}
    for vertex in parent:
        components.setdefault(find(parent, vertex), []).append(vertex)
    conflicts = 0
    for vertices in components.values():
        by_panel: dict[str, list[int]] = {}
        for vertex in vertices:
            by_panel.setdefault(panel_by_vertex[vertex], []).append(vertex)
        for panel_vertices in by_panel.values():
            self_groups = {find(self_seam_parent, vertex) for vertex in panel_vertices}
            conflicts += max(0, len(self_groups) - 1)
    return conflicts


def _seam_edges_from_interfaces(
    interfaces: list[SeamInterface],
    xy: np.ndarray,
) -> set[tuple[int, int]]:
    seam_edges: set[tuple[int, int]] = set()
    points = np.asarray(xy, dtype=np.float32)
    for interface in interfaces:
        path_a = np.asarray(interface.vertices_a, dtype=np.int32)
        path_b = np.asarray(interface.vertices_b, dtype=np.int32)
        parameter_a = _normalized_path_arclength(points[path_a])
        parameter_b = _normalized_path_arclength(points[path_b])
        if len(path_a) <= len(path_b):
            for index_a, value in enumerate(parameter_a):
                index_b = int(np.argmin(np.abs(parameter_b - value)))
                seam_edges.add(tuple(sorted((int(path_a[index_a]), int(path_b[index_b])))))
        else:
            for index_b, value in enumerate(parameter_b):
                index_a = int(np.argmin(np.abs(parameter_a - value)))
                seam_edges.add(tuple(sorted((int(path_a[index_a]), int(path_b[index_b])))))
    return seam_edges


def _normalized_path_arclength(points: np.ndarray) -> np.ndarray:
    if len(points) <= 1:
        return np.zeros(len(points), dtype=np.float32)
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths, dtype=np.float64)))
    if cumulative[-1] <= 1.0e-10:
        return np.linspace(0.0, 1.0, len(points), dtype=np.float32)
    return (cumulative / cumulative[-1]).astype(np.float32)


def _orient_stitch(ids_a: list[int], ids_b: list[int], xy: np.ndarray) -> list[int]:
    same = np.linalg.norm(xy[ids_a[0]] - xy[ids_b[0]]) + np.linalg.norm(xy[ids_a[-1]] - xy[ids_b[-1]])
    flipped = np.linalg.norm(xy[ids_a[0]] - xy[ids_b[-1]]) + np.linalg.norm(xy[ids_a[-1]] - xy[ids_b[0]])
    return list(reversed(ids_b)) if flipped < same else ids_b


def _select_learned_pattern_pins(vertices: np.ndarray, labels: list[str]) -> np.ndarray:
    label_arr = np.asarray(labels)
    mask = np.asarray(["sleeve" not in label for label in label_arr], dtype=bool)
    if not np.any(mask):
        mask = np.ones(vertices.shape[0], dtype=bool)
    candidates = np.flatnonzero(mask)
    y = vertices[candidates, 1]
    top = candidates[y >= np.percentile(y, 88.0)]
    if top.size == 0:
        top = candidates
    targets = [
        np.asarray([-0.08, float(vertices[top, 1].max())], dtype=np.float32),
        np.asarray([0.08, float(vertices[top, 1].max())], dtype=np.float32),
    ]
    pins = []
    for target in targets:
        distances = np.linalg.norm(vertices[top, :2] - target, axis=1)
        pins.append(int(top[int(np.argmin(distances))]))
    return np.unique(np.asarray(pins, dtype=np.int32))


def _drop_duplicate_tail(points: np.ndarray) -> np.ndarray:
    if points.shape[0] > 1 and np.linalg.norm(points[0] - points[-1]) < 1e-7:
        return points[:-1]
    return points


def _point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    x, y = float(point[0]), float(point[1])
    inside = False
    count = polygon.shape[0]
    j = count - 1
    for i in range(count):
        xi, yi = float(polygon[i, 0]), float(polygon[i, 1])
        xj, yj = float(polygon[j, 0]), float(polygon[j, 1])
        intersects = (yi > y) != (yj > y)
        if intersects:
            x_cross = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < x_cross:
                inside = not inside
        j = i
    return inside


def _distance_to_polygon_boundary(point: np.ndarray, polygon: np.ndarray) -> float:
    starts = polygon
    ends = np.roll(polygon, -1, axis=0)
    segments = ends - starts
    denominator = np.maximum(np.sum(segments * segments, axis=1), 1e-12)
    t = np.clip(np.sum((point - starts) * segments, axis=1) / denominator, 0.0, 1.0)
    closest = starts + t[:, None] * segments
    return float(np.linalg.norm(closest - point, axis=1).min())


def _polygon_area(points: np.ndarray) -> float:
    x = points[:, 0]
    y = points[:, 1]
    return float(0.5 * np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))


def _triangle_area(points: np.ndarray) -> float:
    a, b, c = points[:, :2]
    ab = b - a
    ac = c - a
    return float(0.5 * (ab[0] * ac[1] - ab[1] * ac[0]))
