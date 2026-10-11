from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .data import GarmentMesh
from .mesh_refinement import _merge_semantic_labels
from .topology_correction import analyze_triangle_mesh


@dataclass(frozen=True)
class AdaptiveAMRConfig:
    """Controls one or more conservative intrinsic shell refinement passes."""

    passes: int = 1
    refine_fraction: float = 0.18
    indicator_threshold: float = 0.55
    maximum_faces: int = 20000
    curvature_weight: float = 1.0
    strain_weight: float = 1.0
    material_gradient_weight: float = 1.0
    contact_weight: float = 1.0
    energy_residual_weight: float = 1.0
    contact_clearance_m: float = 0.003
    minimum_face_area_m2: float = 1.0e-10

    def __post_init__(self) -> None:
        if self.passes < 0 or self.maximum_faces <= 0:
            raise ValueError("AMR passes and maximum_faces must be non-negative/positive")
        if not 0.0 < self.refine_fraction <= 1.0:
            raise ValueError("refine_fraction must lie in (0, 1]")
        if not 0.0 <= self.indicator_threshold <= 1.0:
            raise ValueError("indicator_threshold must lie in [0, 1]")
        if self.contact_clearance_m <= 0.0 or self.minimum_face_area_m2 <= 0.0:
            raise ValueError("contact clearance and minimum face area must be positive")
        weights = (
            self.curvature_weight,
            self.strain_weight,
            self.material_gradient_weight,
            self.contact_weight,
            self.energy_residual_weight,
        )
        if any(value < 0.0 for value in weights) or not any(value > 0.0 for value in weights):
            raise ValueError("AMR indicator weights must be non-negative with one positive weight")


@dataclass(frozen=True)
class AdaptiveAMRReport:
    passes_requested: int
    passes_applied: int
    vertices_before: int
    vertices_after: int
    faces_before: int
    faces_after: int
    marked_faces: tuple[int, ...]
    refined_faces: int
    indicator_min: float
    indicator_max: float
    indicator_mean: float
    indicator_components: dict[str, float]
    mass_before_kg: float
    mass_after_kg: float
    linear_momentum_error_kg_m_s: float
    angular_momentum_error_kg_m2_s: float
    topology_preserved: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AdaptiveAMRStateTransfer:
    positions: np.ndarray
    velocities: np.ndarray
    vertex_mass: np.ndarray
    material_field: np.ndarray | None
    linear_momentum_error: float
    angular_momentum_error: float


def intrinsic_face_error_indicator(
    mesh: GarmentMesh,
    current_positions: np.ndarray,
    *,
    material_field: np.ndarray | None = None,
    contact_gap: np.ndarray | None = None,
    energy_residual: np.ndarray | None = None,
    config: AdaptiveAMRConfig = AdaptiveAMRConfig(),
) -> tuple[np.ndarray, dict[str, float]]:
    """Compute a dimensionless, remeshing-invariant face error indicator.

    The indicator is deliberately assembled from intrinsic quantities: edge
    stretch, normal variation, scalar material gradients, contact gap and
    energy residual.  Absolute world position is never used, so a rigid E(3)
    transform cannot change which faces are refined.
    """

    positions = np.asarray(current_positions, dtype=np.float64)
    faces = _validate_mesh_inputs(mesh, positions)
    rest = np.asarray(mesh.material_vertices, dtype=np.float64)
    components: dict[str, np.ndarray] = {}
    components["curvature"] = _face_normal_variation(positions, faces)
    components["strain"] = _face_edge_strain(positions, rest, faces)
    material_active = material_field is not None
    contact_active = contact_gap is not None
    energy_active = energy_residual is not None
    if material_active:
        components["material_gradient"] = _face_scalar_gradient(material_field, faces, len(positions))
    else:
        components["material_gradient"] = np.zeros(len(faces), dtype=np.float64)
    if contact_active:
        gap = _face_scalar_mean(contact_gap, faces, len(positions))
        components["contact"] = np.clip(
            (config.contact_clearance_m - gap) / config.contact_clearance_m,
            0.0,
            1.0,
        )
    else:
        components["contact"] = np.zeros(len(faces), dtype=np.float64)
    if energy_active:
        components["energy_residual"] = _face_scalar_mean(energy_residual, faces, len(positions))
        components["energy_residual"] = _normalize_positive(components["energy_residual"])
    else:
        components["energy_residual"] = np.zeros(len(faces), dtype=np.float64)

    weights = {
        "curvature": config.curvature_weight,
        "strain": config.strain_weight,
        "material_gradient": config.material_gradient_weight,
        "contact": config.contact_weight,
        "energy_residual": config.energy_residual_weight,
    }
    # Do not dilute a real curvature peak with identically-zero channels.
    # The old unconditional strain channel capped a rest-shell curvature-only
    # indicator at 0.5, below the default 0.55 refinement threshold.
    active = {
        name
        for name, component in components.items()
        if weights[name] > 0.0
        and float(component.max(initial=0.0)) > 1.0e-12
    }
    indicator = np.zeros(len(faces), dtype=np.float64)
    weight_sum = 0.0
    summary: dict[str, float] = {}
    for name, weight in weights.items():
        component = np.clip(components[name], 0.0, 1.0)
        if name in active:
            indicator += weight * component
            weight_sum += weight
        summary[name] = float(component.mean()) if len(component) else 0.0
    indicator /= max(weight_sum, 1.0e-12)
    return np.clip(indicator, 0.0, 1.0), summary


def adaptive_refine_garment(
    mesh: GarmentMesh,
    current_positions: np.ndarray,
    *,
    velocities: np.ndarray | None = None,
    vertex_mass: np.ndarray | None = None,
    material_field: np.ndarray | None = None,
    contact_gap: np.ndarray | None = None,
    energy_residual: np.ndarray | None = None,
    config: AdaptiveAMRConfig = AdaptiveAMRConfig(),
) -> tuple[GarmentMesh, AdaptiveAMRStateTransfer, AdaptiveAMRReport]:
    """Locally refine a shell while preserving manifold topology and momentum.

    This is a red/green conforming refinement pass. Marked triangles receive
    red four-way refinement; neighboring triangles receive green one-, two-,
    or three-edge templates, so no hanging midpoint is left on an unrefined
    edge. Coarsening is intentionally not included yet: it needs an error
    hysteresis window and a reversible parent forest to avoid oscillation.
    """

    if mesh.seam_interfaces:
        raise ValueError("AMR must run after seam interfaces are resolved into one shell")
    original_positions = np.asarray(current_positions, dtype=np.float64)
    original_velocities = (
        np.zeros_like(original_positions)
        if velocities is None
        else np.asarray(velocities, dtype=np.float64)
    )
    original_faces = np.asarray(mesh.faces, dtype=np.int32).copy()
    old_positions = original_positions.copy()
    _validate_mesh_inputs(mesh, old_positions)
    old_velocities = original_velocities.copy()
    if old_velocities.shape != old_positions.shape:
        raise ValueError("velocities must match current_positions")
    old_material = _validate_vertex_field(material_field, len(old_positions), "material_field")
    old_mass = (
        _vertex_lumped_mass(old_positions, original_faces)
        if vertex_mass is None
        else np.asarray(vertex_mass, dtype=np.float64)
    )
    if old_mass.shape != (len(old_positions),) or np.any(old_mass <= 0.0):
        raise ValueError("vertex_mass must be positive with shape (V,)")

    initial_topology = analyze_triangle_mesh(mesh.material_vertices, mesh.faces)
    refined = mesh
    positions = old_positions
    velocities_now = old_velocities
    mass_now = old_mass
    material_now = old_material
    all_marked: list[int] = []
    passes_applied = 0
    last_components: dict[str, float] = {}
    for pass_index in range(config.passes):
        if len(refined.faces) * 4 > config.maximum_faces:
            break
        indicator, last_components = intrinsic_face_error_indicator(
            refined,
            positions,
            material_field=material_now,
            contact_gap=contact_gap if contact_gap is not None and len(contact_gap) == len(positions) else None,
            energy_residual=energy_residual if energy_residual is not None and len(energy_residual) == len(positions) else None,
            config=config,
        )
        marked = _select_faces(indicator, config)
        if not len(marked):
            break
        remaining = config.maximum_faces - len(refined.faces)
        if len(marked) * 3 > remaining:
            marked = marked[: max(1, remaining // 3)]
        marked_tuple = tuple(int(value) for value in marked)
        all_marked.extend(marked_tuple)
        old_faces = refined.faces.copy()
        refined, positions, parent_edges = _refine_selected_once(refined, positions, marked)
        transfer = conservative_state_transfer(
            old_positions,
            old_velocities,
            old_mass,
            old_faces,
            positions,
            refined.faces,
            parent_edges,
            material_field=old_material,
        )
        velocities_now = transfer.velocities
        mass_now = transfer.vertex_mass
        material_now = transfer.material_field
        old_positions = positions
        old_velocities = velocities_now
        old_mass = mass_now
        old_material = material_now
        passes_applied += 1
        # Indicators and contact samples are material-point fields. A caller
        # that needs them after AMR can resample them from the event/query
        # layer; never silently copy stale vertex-indexed arrays.
        contact_gap = None
        energy_residual = None

    final_topology = analyze_triangle_mesh(refined.material_vertices, refined.faces)
    topology_preserved = (
        initial_topology.connected_components == final_topology.connected_components
        and initial_topology.boundary_loops == final_topology.boundary_loops
        and initial_topology.genus == final_topology.genus
        and initial_topology.nonmanifold_edges == final_topology.nonmanifold_edges
    )
    if not topology_preserved:
        raise RuntimeError("AMR changed the shell topology invariants")
    final_indicator, _ = intrinsic_face_error_indicator(
        refined,
        positions,
        material_field=material_now,
        config=config,
    )
    linear_error, angular_error = _conservation_errors(
        original_positions,
        original_velocities,
        _vertex_lumped_mass(original_positions, original_faces) if vertex_mass is None else np.asarray(vertex_mass, dtype=np.float64),
        positions,
        velocities_now,
        mass_now,
    )
    report = AdaptiveAMRReport(
        passes_requested=config.passes,
        passes_applied=passes_applied,
        vertices_before=len(mesh.vertices),
        vertices_after=len(refined.vertices),
        faces_before=len(mesh.faces),
        faces_after=len(refined.faces),
        marked_faces=tuple(all_marked),
        refined_faces=len(refined.faces) - len(mesh.faces),
        indicator_min=float(final_indicator.min()) if len(final_indicator) else 0.0,
        indicator_max=float(final_indicator.max()) if len(final_indicator) else 0.0,
        indicator_mean=float(final_indicator.mean()) if len(final_indicator) else 0.0,
        indicator_components=last_components,
        mass_before_kg=float(np.sum(_vertex_lumped_mass(original_positions, original_faces) if vertex_mass is None else vertex_mass)),
        mass_after_kg=float(np.sum(mass_now)),
        linear_momentum_error_kg_m_s=float(linear_error),
        angular_momentum_error_kg_m2_s=float(angular_error),
        topology_preserved=topology_preserved,
    )
    state = AdaptiveAMRStateTransfer(
        positions=positions.astype(np.float32),
        velocities=velocities_now.astype(np.float32),
        vertex_mass=mass_now.astype(np.float32),
        material_field=None if material_now is None else material_now.astype(np.float32),
        linear_momentum_error=float(linear_error),
        angular_momentum_error=float(angular_error),
    )
    return refined, state, report


def conservative_state_transfer(
    old_positions: np.ndarray,
    old_velocities: np.ndarray,
    old_mass: np.ndarray,
    old_faces: np.ndarray,
    new_positions: np.ndarray,
    new_faces: np.ndarray,
    parent_edges: dict[int, tuple[int, int]],
    *,
    material_field: np.ndarray | None = None,
) -> AdaptiveAMRStateTransfer:
    """Transfer vertex fields and correct velocity to preserve P and L."""

    old_positions = np.asarray(old_positions, dtype=np.float64)
    old_velocities = np.asarray(old_velocities, dtype=np.float64)
    old_mass = np.asarray(old_mass, dtype=np.float64)
    positions = np.asarray(new_positions, dtype=np.float64)
    if old_positions.shape != old_velocities.shape or old_mass.shape != (len(old_positions),):
        raise ValueError("old state shapes are incompatible")
    new_mass = _vertex_lumped_mass(positions, np.asarray(new_faces, dtype=np.int32))
    new_mass *= float(old_mass.sum()) / max(float(new_mass.sum()), 1.0e-12)
    velocities = np.empty_like(positions)
    velocities[: len(old_positions)] = old_velocities
    for index, (first, second) in parent_edges.items():
        velocities[index] = 0.5 * (old_velocities[first] + old_velocities[second])
    transferred_material = None
    if material_field is not None:
        old_field = _validate_vertex_field(material_field, len(old_positions), "material_field")
        transferred_material = np.empty((len(positions),) + old_field.shape[1:], dtype=np.float64)
        transferred_material[: len(old_positions)] = old_field
        for index, (first, second) in parent_edges.items():
            transferred_material[index] = 0.5 * (old_field[first] + old_field[second])

    corrected = _match_linear_and_angular_momentum(
        old_positions,
        old_velocities,
        old_mass,
        positions,
        velocities,
        new_mass,
    )
    linear_error, angular_error = _conservation_errors(
        old_positions,
        old_velocities,
        old_mass,
        positions,
        corrected,
        new_mass,
    )
    return AdaptiveAMRStateTransfer(
        # Keep the internal transfer in float64 so a later AMR pass and the
        # conservation report do not measure float32 serialization error.
        # The public state is converted once, after all passes finish.
        positions=positions,
        velocities=corrected,
        vertex_mass=new_mass,
        material_field=transferred_material,
        linear_momentum_error=linear_error,
        angular_momentum_error=angular_error,
    )


def _refine_selected_once(
    mesh: GarmentMesh,
    positions: np.ndarray,
    marked_faces: np.ndarray,
) -> tuple[GarmentMesh, np.ndarray, dict[int, tuple[int, int]]]:
    faces = np.asarray(mesh.faces, dtype=np.int32)
    marked = set(int(value) for value in marked_faces)
    edge_to_midpoint: dict[tuple[int, int], int] = {}
    parent_edges: dict[int, tuple[int, int]] = {}
    for face_index in marked:
        for first, second in _face_edges(faces[face_index]):
            edge = tuple(sorted((first, second)))
            if edge not in edge_to_midpoint:
                index = len(positions) + len(edge_to_midpoint)
                edge_to_midpoint[edge] = index
                parent_edges[index] = edge
    edge_array = np.asarray(list(edge_to_midpoint), dtype=np.int32)
    midpoint_positions = _curvature_reconstructed_midpoints(
        positions, faces, edge_array
    )
    new_positions = np.concatenate([positions, midpoint_positions], axis=0)
    base = np.asarray(mesh.vertices, dtype=np.float64)
    new_base = np.concatenate(
        [base, _curvature_reconstructed_midpoints(base, faces, edge_array)], axis=0
    )
    rest = np.asarray(mesh.material_vertices, dtype=np.float64)
    midpoint_rest = _curvature_reconstructed_midpoints(rest, faces, edge_array)
    new_rest = np.concatenate([rest, midpoint_rest], axis=0)
    new_uvs = None
    if mesh.uvs is not None:
        uvs = np.asarray(mesh.uvs, dtype=np.float64)
        new_uvs = np.concatenate([uvs, 0.5 * (uvs[edge_array[:, 0]] + uvs[edge_array[:, 1]])], axis=0)

    new_faces: list[tuple[int, int, int]] = []
    for triangle in faces:
        a, b, c = (int(value) for value in triangle)
        edge_mid = {
            edge: edge_to_midpoint[edge]
            for edge in _face_edges(triangle)
            if edge in edge_to_midpoint
        }
        if not edge_mid:
            new_faces.append((a, b, c))
        elif len(edge_mid) == 1:
            new_faces.extend(_green_one_edge(a, b, c, edge_mid))
        elif len(edge_mid) == 2:
            new_faces.extend(_green_two_edges(a, b, c, edge_mid))
        else:
            ab, bc, ca = (edge_mid[edge] for edge in _face_edges(triangle))
            new_faces.extend(((a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)))

    labels = list(mesh.panel_labels)
    for first, second in edge_array:
        labels.append(_merge_semantic_labels(labels[int(first)], labels[int(second)]))
    pinned = set(int(index) for index in mesh.pinned_vertices)
    for index, (first, second) in parent_edges.items():
        if first in pinned and second in pinned:
            pinned.add(index)
    seam_edges: list[tuple[int, int]] = []
    for first, second in mesh.seam_edges:
        edge = tuple(sorted((int(first), int(second))))
        midpoint = edge_to_midpoint.get(edge)
        if midpoint is None:
            seam_edges.append((int(first), int(second)))
        else:
            seam_edges.extend(((int(first), midpoint), (midpoint, int(second))))
    refined = GarmentMesh(
        vertices=new_base.astype(np.float32),
        faces=np.asarray(new_faces, dtype=np.int32),
        pinned_vertices=np.asarray(sorted(pinned), dtype=np.int32),
        panel_labels=labels,
        seam_edges=seam_edges,
        rest_vertices=new_rest.astype(np.float32),
        uvs=None if new_uvs is None else new_uvs.astype(np.float32),
    )
    return refined, new_positions, parent_edges


def _curvature_reconstructed_midpoints(
    values: np.ndarray,
    faces: np.ndarray,
    edges: np.ndarray,
) -> np.ndarray:
    """Sample a bounded subdivision surface instead of a flat edge chord.

    Interior edges use the Loop edge stencil and free-boundary edges use the
    interpolating four-point curve stencil. Both are affine E(3)-equivariant.
    The displacement cap keeps a pressed manufacturing fold sharp while still
    adding geometric resolution to curved hems, side silhouettes, and axillae.
    """

    points = np.asarray(values, dtype=np.float64)
    face_array = np.asarray(faces, dtype=np.int32)
    edge_array = np.asarray(edges, dtype=np.int32)
    linear = 0.5 * (points[edge_array[:, 0]] + points[edge_array[:, 1]])
    reconstructed = linear.copy()
    edge_faces: dict[tuple[int, int], list[int]] = {}
    boundary_neighbors: dict[int, set[int]] = {}
    for face_index, triangle in enumerate(face_array):
        for edge in _face_edges(triangle):
            edge_faces.setdefault(edge, []).append(face_index)
    for edge, adjacent in edge_faces.items():
        if len(adjacent) == 1:
            first, second = edge
            boundary_neighbors.setdefault(first, set()).add(second)
            boundary_neighbors.setdefault(second, set()).add(first)

    for index, (first_raw, second_raw) in enumerate(edge_array):
        first, second = int(first_raw), int(second_raw)
        edge = tuple(sorted((first, second)))
        adjacent = edge_faces.get(edge, [])
        candidate = linear[index]
        if len(adjacent) == 2:
            opposites: list[int] = []
            for face_index in adjacent:
                opposites.extend(
                    int(vertex)
                    for vertex in face_array[face_index]
                    if int(vertex) not in edge
                )
            if len(opposites) == 2:
                candidate = (
                    0.375 * (points[first] + points[second])
                    + 0.125 * (points[opposites[0]] + points[opposites[1]])
                )
        elif len(adjacent) == 1:
            before = sorted(boundary_neighbors.get(first, set()) - {second})
            after = sorted(boundary_neighbors.get(second, set()) - {first})
            if len(before) == 1 and len(after) == 1:
                candidate = (
                    -points[before[0]]
                    + 9.0 * points[first]
                    + 9.0 * points[second]
                    - points[after[0]]
                ) / 16.0

        deviation = candidate - linear[index]
        edge_length = float(np.linalg.norm(points[second] - points[first]))
        deviation_length = float(np.linalg.norm(deviation))
        maximum_deviation = 0.20 * edge_length
        if deviation_length > maximum_deviation > 0.0:
            deviation *= maximum_deviation / deviation_length
        reconstructed[index] = linear[index] + deviation
    return reconstructed


def _green_one_edge(a: int, b: int, c: int, edge_mid: dict[tuple[int, int], int]) -> list[tuple[int, int, int]]:
    edge, midpoint = next(iter(edge_mid.items()))
    if edge == tuple(sorted((a, b))):
        return [(a, midpoint, c), (midpoint, b, c)]
    if edge == tuple(sorted((b, c))):
        return [(a, b, midpoint), (a, midpoint, c)]
    return [(a, b, midpoint), (midpoint, b, c)]


def _green_two_edges(a: int, b: int, c: int, edge_mid: dict[tuple[int, int], int]) -> list[tuple[int, int, int]]:
    ab = edge_mid.get(tuple(sorted((a, b))))
    bc = edge_mid.get(tuple(sorted((b, c))))
    ca = edge_mid.get(tuple(sorted((c, a))))
    if ab is not None and bc is not None:
        return [(b, bc, ab), (ab, a, c), (ab, c, bc)]
    if ab is not None and ca is not None:
        return [(a, ab, ca), (ab, b, c), (ab, c, ca)]
    assert bc is not None and ca is not None
    return [(c, ca, bc), (ca, a, b), (ca, b, bc)]


def _select_faces(indicator: np.ndarray, config: AdaptiveAMRConfig) -> np.ndarray:
    if not len(indicator):
        return np.empty(0, dtype=np.int32)
    count = max(1, int(np.ceil(len(indicator) * config.refine_fraction)))
    order = np.argsort(-indicator, kind="stable")
    selected = order[:count]
    selected = selected[indicator[selected] >= config.indicator_threshold]
    return np.asarray(selected, dtype=np.int32)


def _validate_mesh_inputs(mesh: GarmentMesh, positions: np.ndarray) -> np.ndarray:
    if positions.ndim != 2 or positions.shape != mesh.vertices.shape or positions.shape[1] != 3:
        raise ValueError("current_positions must have shape (V,3) matching mesh.vertices")
    faces = np.asarray(mesh.faces, dtype=np.int32)
    if faces.ndim != 2 or faces.shape[1] != 3 or np.any(faces < 0) or np.any(faces >= len(positions)):
        raise ValueError("mesh faces must be a valid (F,3) index array")
    return faces


def _validate_vertex_field(field: np.ndarray | None, vertex_count: int, name: str) -> np.ndarray | None:
    if field is None:
        return None
    array = np.asarray(field, dtype=np.float64)
    if array.shape[0] != vertex_count:
        raise ValueError(f"{name} must have one entry per vertex")
    return array


def _face_edges(triangle: np.ndarray) -> tuple[tuple[int, int], ...]:
    a, b, c = (int(value) for value in triangle)
    return tuple(sorted((a, b)),), tuple(sorted((b, c))), tuple(sorted((c, a)))


def _face_scalar_mean(values: np.ndarray, faces: np.ndarray, vertex_count: int) -> np.ndarray:
    array = _validate_vertex_field(values, vertex_count, "face scalar field")
    assert array is not None
    if array.ndim > 1:
        array = np.linalg.norm(array.reshape(len(array), -1), axis=1)
    return np.mean(array[faces], axis=1)


def _face_scalar_gradient(values: np.ndarray, faces: np.ndarray, vertex_count: int) -> np.ndarray:
    array = _validate_vertex_field(values, vertex_count, "material_field")
    assert array is not None
    vectors = array.reshape(len(array), -1)
    return _normalize_positive(np.max(np.linalg.norm(vectors[faces] - vectors[faces][:, :1], axis=-1), axis=1))


def _face_edge_strain(positions: np.ndarray, rest: np.ndarray, faces: np.ndarray) -> np.ndarray:
    current_edges = np.concatenate([positions[faces[:, i]] - positions[faces[:, (i + 1) % 3]] for i in range(3)], axis=0)
    rest_edges = np.concatenate([rest[faces[:, i]] - rest[faces[:, (i + 1) % 3]] for i in range(3)], axis=0)
    current_length = np.linalg.norm(current_edges, axis=1)
    rest_length = np.linalg.norm(rest_edges, axis=1)
    values = np.abs(current_length / np.maximum(rest_length, 1.0e-12) - 1.0).reshape(3, len(faces))
    return _normalize_positive(np.max(values, axis=0))


def _face_normal_variation(positions: np.ndarray, faces: np.ndarray) -> np.ndarray:
    triangles = positions[faces]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1.0e-12)
    edge_faces: dict[tuple[int, int], list[int]] = {}
    for index, face in enumerate(faces):
        for edge in _face_edges(face):
            edge_faces.setdefault(edge, []).append(index)
    variation = np.zeros(len(faces), dtype=np.float64)
    for adjacent in edge_faces.values():
        if len(adjacent) == 2:
            first, second = adjacent
            value = 0.5 * (1.0 - float(np.clip(np.dot(normals[first], normals[second]), -1.0, 1.0)))
            variation[first] = max(variation[first], value)
            variation[second] = max(variation[second], value)
    return _normalize_positive(variation)


def _normalize_positive(values: np.ndarray) -> np.ndarray:
    values = np.maximum(np.asarray(values, dtype=np.float64), 0.0)
    maximum = float(values.max()) if len(values) else 0.0
    return values / maximum if maximum > 1.0e-12 else np.zeros_like(values)


def _vertex_lumped_mass(positions: np.ndarray, faces: np.ndarray) -> np.ndarray:
    triangles = positions[faces]
    areas = 0.5 * np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1)
    mass = np.zeros(len(positions), dtype=np.float64)
    share = areas / 3.0
    for column in range(3):
        np.add.at(mass, faces[:, column], share)
    return np.maximum(mass, 1.0e-12)


def _match_linear_and_angular_momentum(
    old_positions: np.ndarray,
    old_velocities: np.ndarray,
    old_mass: np.ndarray,
    new_positions: np.ndarray,
    new_velocities: np.ndarray,
    new_mass: np.ndarray,
) -> np.ndarray:
    result = np.asarray(new_velocities, dtype=np.float64).copy()
    old_linear = np.sum(old_mass[:, None] * old_velocities, axis=0)
    old_center = np.sum(old_mass[:, None] * old_positions, axis=0) / old_mass.sum()
    new_center = np.sum(new_mass[:, None] * new_positions, axis=0) / new_mass.sum()
    result += (old_linear - np.sum(new_mass[:, None] * result, axis=0))[None, :] / new_mass.sum()
    old_angular = np.sum(np.cross(old_positions - old_center, old_mass[:, None] * old_velocities), axis=0)
    new_angular = np.sum(np.cross(new_positions - new_center, new_mass[:, None] * result), axis=0)
    inertia = np.zeros((3, 3), dtype=np.float64)
    for radius, mass in zip(new_positions - new_center, new_mass):
        inertia += mass * (np.dot(radius, radius) * np.eye(3) - np.outer(radius, radius))
    try:
        omega = np.linalg.solve(inertia + 1.0e-12 * np.eye(3), old_angular - new_angular)
    except np.linalg.LinAlgError:
        omega = np.zeros(3, dtype=np.float64)
    result += np.cross(omega[None, :], new_positions - new_center)
    # Roundoff in the rotational correction can leave a tiny translational
    # residual. A uniform velocity update has zero angular momentum about the
    # center of mass, so this final cleanup preserves both invariants.
    result += (
        old_linear - np.sum(new_mass[:, None] * result, axis=0)
    )[None, :] / new_mass.sum()
    return result


def _conservation_errors(
    old_positions: np.ndarray,
    old_velocities: np.ndarray,
    old_mass: np.ndarray,
    new_positions: np.ndarray,
    new_velocities: np.ndarray,
    new_mass: np.ndarray,
) -> tuple[float, float]:
    old_center = np.sum(old_mass[:, None] * old_positions, axis=0) / old_mass.sum()
    new_center = np.sum(new_mass[:, None] * new_positions, axis=0) / new_mass.sum()
    old_linear = np.sum(old_mass[:, None] * old_velocities, axis=0)
    new_linear = np.sum(new_mass[:, None] * new_velocities, axis=0)
    # The interpolated state has no canonical one-to-one correspondence for
    # angular momentum. The correction above makes this residual numerical.
    old_angular = np.sum(np.cross(old_positions - old_center, old_mass[:, None] * old_velocities), axis=0)
    new_angular = np.sum(np.cross(new_positions - new_center, new_mass[:, None] * new_velocities), axis=0)
    return float(np.linalg.norm(old_linear - new_linear)), float(np.linalg.norm(old_angular - new_angular))
