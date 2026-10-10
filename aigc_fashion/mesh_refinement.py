from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .data import GarmentMesh
from .topology_correction import analyze_triangle_mesh


@dataclass(frozen=True)
class MeshRefinementReport:
    levels: int
    vertices_before: int
    vertices_after: int
    faces_before: int
    faces_after: int
    edges_before: int
    edges_after: int
    boundary_loops: int
    genus: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def midpoint_subdivide_garment(
    mesh: GarmentMesh,
    current_positions: np.ndarray,
    *,
    levels: int = 1,
) -> tuple[GarmentMesh, np.ndarray, MeshRefinementReport]:
    """Refine a simulation shell and its current state without changing topology."""

    if levels < 0:
        raise ValueError("levels must be non-negative")
    if mesh.seam_interfaces:
        raise ValueError("subdivide the unified sewn shell after resolving seam interfaces")

    current = np.asarray(current_positions, dtype=np.float32)
    if current.shape != mesh.vertices.shape:
        raise ValueError("current_positions must match mesh.vertices")

    before = analyze_triangle_mesh(mesh.material_vertices, mesh.faces)
    refined = mesh
    for _ in range(levels):
        refined, current = _subdivide_once(refined, current)
    after = analyze_triangle_mesh(refined.material_vertices, refined.faces)

    invariant_before = (
        before.connected_components,
        before.boundary_loops,
        before.genus,
        before.nonmanifold_edges,
    )
    invariant_after = (
        after.connected_components,
        after.boundary_loops,
        after.genus,
        after.nonmanifold_edges,
    )
    if invariant_after != invariant_before:
        raise RuntimeError(
            "midpoint subdivision changed garment topology: "
            f"{invariant_before} -> {invariant_after}"
        )
    if after.boundary_loops is None or after.genus is None:
        raise RuntimeError("refined garment is not a valid orientable manifold shell")

    report = MeshRefinementReport(
        levels=levels,
        vertices_before=before.vertices,
        vertices_after=after.vertices,
        faces_before=before.faces,
        faces_after=after.faces,
        edges_before=before.edges,
        edges_after=after.edges,
        boundary_loops=after.boundary_loops,
        genus=after.genus,
    )
    return refined, current, report


def _subdivide_once(
    mesh: GarmentMesh, current: np.ndarray
) -> tuple[GarmentMesh, np.ndarray]:
    faces = np.asarray(mesh.faces, dtype=np.int32)
    edge_to_midpoint: dict[tuple[int, int], int] = {}
    edges: list[tuple[int, int]] = []
    for triangle in faces:
        a, b, c = (int(value) for value in triangle)
        for first, second in ((a, b), (b, c), (c, a)):
            edge = (first, second) if first < second else (second, first)
            if edge not in edge_to_midpoint:
                edge_to_midpoint[edge] = len(mesh.vertices) + len(edges)
                edges.append(edge)

    edge_array = np.asarray(edges, dtype=np.int32)

    def add_midpoints(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values)
        midpoints = 0.5 * (values[edge_array[:, 0]] + values[edge_array[:, 1]])
        return np.concatenate([values, midpoints], axis=0).astype(values.dtype, copy=False)

    vertices = add_midpoints(np.asarray(mesh.vertices, dtype=np.float32))
    rest_vertices = add_midpoints(np.asarray(mesh.material_vertices, dtype=np.float32))
    current_refined = add_midpoints(current)
    uvs = None if mesh.uvs is None else add_midpoints(np.asarray(mesh.uvs))

    labels = list(mesh.panel_labels)
    for first, second in edges:
        labels.append(_merge_semantic_labels(labels[first], labels[second]))

    refined_faces: list[tuple[int, int, int]] = []
    for a_raw, b_raw, c_raw in faces:
        a, b, c = int(a_raw), int(b_raw), int(c_raw)
        ab = edge_to_midpoint[tuple(sorted((a, b)))]
        bc = edge_to_midpoint[tuple(sorted((b, c)))]
        ca = edge_to_midpoint[tuple(sorted((c, a)))]
        refined_faces.extend(
            ((a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca))
        )

    pinned = set(int(index) for index in mesh.pinned_vertices)
    for edge, midpoint in edge_to_midpoint.items():
        if edge[0] in pinned and edge[1] in pinned:
            pinned.add(midpoint)

    refined_seams: list[tuple[int, int]] = []
    seam_set = {tuple(sorted((int(a), int(b)))) for a, b in mesh.seam_edges}
    for edge in seam_set:
        midpoint = edge_to_midpoint.get(edge)
        if midpoint is None:
            raise ValueError(f"seam edge {edge} is not present in the triangle mesh")
        refined_seams.extend(((edge[0], midpoint), (midpoint, edge[1])))

    return (
        GarmentMesh(
            vertices=vertices,
            faces=np.asarray(refined_faces, dtype=np.int32),
            pinned_vertices=np.asarray(sorted(pinned), dtype=np.int32),
            panel_labels=labels,
            seam_edges=refined_seams,
            rest_vertices=rest_vertices,
            uvs=uvs,
        ),
        current_refined,
    )


def _merge_semantic_labels(first: str, second: str) -> str:
    if first == second:
        return first
    for region in ("left_sleeve", "right_sleeve", "front_body"):
        matches = [label for label in (first, second) if region in label]
        if matches:
            return matches[0]
    return first
