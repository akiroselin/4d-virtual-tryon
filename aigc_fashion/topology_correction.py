from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType
from typing import Protocol

import numpy as np


class SurfaceDomain(str, Enum):
    CLOSED_LEVEL_SET = "closed_level_set"
    GARMENT_PANEL_ATLAS = "garment_panel_atlas"
    GARMENT_INTRINSIC_SHELL = "garment_intrinsic_shell"


class TopologyDisposition(str, Enum):
    VERIFIED = "verified"
    REQUIRES_FAST_MARCHING = "requires_fast_marching"
    PRESERVE_PANEL_TOPOLOGY = "preserve_panel_topology"
    PRESERVE_SHELL_TOPOLOGY = "preserve_shell_topology"
    REJECTED = "rejected"


@dataclass(frozen=True)
class MeshTopologyReport:
    vertices: int
    edges: int
    faces: int
    connected_components: int
    boundary_edges: int
    boundary_loops: int | None
    nonmanifold_edges: int
    isolated_vertices: int
    euler_characteristic: int
    genus: int | None

    @property
    def is_closed_manifold(self) -> bool:
        return (
            self.faces > 0
            and self.boundary_edges == 0
            and self.nonmanifold_edges == 0
            and self.isolated_vertices == 0
        )

    @property
    def is_connected_genus_zero(self) -> bool:
        return (
            self.is_closed_manifold
            and self.connected_components == 1
            and self.euler_characteristic == 2
            and self.genus == 0
        )


@dataclass(frozen=True)
class TopologyGateResult:
    disposition: TopologyDisposition
    report: MeshTopologyReport
    reason: str


class FastMarchingTopologyBackend(Protocol):
    def correct(self, signed_distance: np.ndarray, *, threshold: float) -> np.ndarray:
        ...


class ExternalCortexODETopologyBackend:
    """Adapter for an explicitly supplied CortexODE checkout.

    The upstream repository currently has no explicit license file. This adapter
    loads it in place and deliberately does not vendor or redistribute its code.
    Commercial builds must replace it with a legally cleared implementation.
    """

    def __init__(self, checkout: str | Path) -> None:
        self.checkout = Path(checkout).expanduser().resolve()
        module_path = self.checkout / "util" / "tca.py"
        lookup_table = self.checkout / "util" / "critical186LUT.raw.gz"
        if not module_path.is_file() or not lookup_table.is_file():
            raise FileNotFoundError("CortexODE tca.py or topology lookup table is missing")
        self._module = _load_module(module_path)
        self._corrector = _construct_corrector(self._module, lookup_table)

    def correct(self, signed_distance: np.ndarray, *, threshold: float) -> np.ndarray:
        volume = _validate_sdf(signed_distance)
        corrected = self._corrector.apply(volume, threshold=float(threshold))
        corrected = np.asarray(corrected)
        if corrected.shape != volume.shape or not np.isfinite(corrected).all():
            raise RuntimeError("topology backend returned an invalid corrected SDF")
        return corrected


class TopologyCorrectionStage:
    """Required initialization gate before a CortexODE closed-surface flow."""

    def __init__(self, backend: FastMarchingTopologyBackend | None = None) -> None:
        self.backend = backend

    def correct_closed_sdf(
        self,
        signed_distance: np.ndarray,
        *,
        threshold: float,
        domain: SurfaceDomain,
    ) -> np.ndarray:
        if domain is not SurfaceDomain.CLOSED_LEVEL_SET:
            raise ValueError(
                "Fast Marching genus correction must not be applied to an open garment panel atlas or intrinsic shell"
            )
        if self.backend is None:
            raise RuntimeError("closed-surface topology correction requires a configured backend")
        return self.backend.correct(_validate_sdf(signed_distance), threshold=threshold)

    def validate_extracted_mesh(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        *,
        domain: SurfaceDomain,
    ) -> TopologyGateResult:
        report = analyze_triangle_mesh(vertices, faces)
        if domain is SurfaceDomain.CLOSED_LEVEL_SET:
            if report.is_connected_genus_zero:
                return TopologyGateResult(
                    TopologyDisposition.VERIFIED,
                    report,
                    "closed initialization is a connected genus-0 manifold",
                )
            return TopologyGateResult(
                TopologyDisposition.REQUIRES_FAST_MARCHING,
                report,
                "closed CortexODE initialization must be corrected and re-extracted with chi=2",
            )
        if domain is SurfaceDomain.GARMENT_INTRINSIC_SHELL:
            if (
                report.connected_components != 1
                or report.nonmanifold_edges
                or report.isolated_vertices
                or report.genus != 0
                or not report.boundary_loops
            ):
                return TopologyGateResult(
                    TopologyDisposition.REJECTED,
                    report,
                    "intrinsic garment shell must be one orientable genus-0 manifold with boundaries",
                )
            return TopologyGateResult(
                TopologyDisposition.PRESERVE_SHELL_TOPOLOGY,
                report,
                "CortexODE preserves the intrinsic shell faces and garment boundary loops",
            )
        if report.nonmanifold_edges or report.isolated_vertices:
            return TopologyGateResult(
                TopologyDisposition.REJECTED,
                report,
                "garment atlas contains non-manifold edges or isolated vertices",
            )
        return TopologyGateResult(
            TopologyDisposition.PRESERVE_PANEL_TOPOLOGY,
            report,
            "garment boundaries and panel connectivity are material construction data",
        )


def analyze_triangle_mesh(vertices: np.ndarray, faces: np.ndarray) -> MeshTopologyReport:
    vertices_array = np.asarray(vertices)
    faces_array = np.asarray(faces)
    if vertices_array.ndim != 2 or vertices_array.shape[1] != 3:
        raise ValueError("vertices must have shape (V, 3)")
    if faces_array.ndim != 2 or faces_array.shape[1] != 3:
        raise ValueError("faces must have shape (F, 3)")
    if not np.issubdtype(faces_array.dtype, np.integer):
        raise ValueError("faces must contain integer indices")
    vertex_count = len(vertices_array)
    if faces_array.size and (faces_array.min() < 0 or faces_array.max() >= vertex_count):
        raise ValueError("face index is outside the vertex array")
    if faces_array.size and np.any(
        (faces_array[:, 0] == faces_array[:, 1])
        | (faces_array[:, 1] == faces_array[:, 2])
        | (faces_array[:, 2] == faces_array[:, 0])
    ):
        raise ValueError("faces must not repeat a vertex")

    edge_counts: dict[tuple[int, int], int] = {}
    used_vertices: set[int] = set()
    union = _UnionFind(vertex_count)
    for a_raw, b_raw, c_raw in faces_array:
        a, b, c = int(a_raw), int(b_raw), int(c_raw)
        used_vertices.update((a, b, c))
        union.join(a, b)
        union.join(b, c)
        for first, second in ((a, b), (b, c), (c, a)):
            edge = (first, second) if first < second else (second, first)
            edge_counts[edge] = edge_counts.get(edge, 0) + 1

    boundary_edges = [edge for edge, count in edge_counts.items() if count == 1]
    nonmanifold_edges = sum(count > 2 for count in edge_counts.values())
    isolated_vertices = vertex_count - len(used_vertices)
    component_roots = {union.find(vertex) for vertex in used_vertices}
    components = len(component_roots)
    boundary_loops = _count_boundary_loops(boundary_edges)
    chi = vertex_count - len(edge_counts) + len(faces_array)

    genus: int | None = None
    if nonmanifold_edges == 0 and isolated_vertices == 0 and boundary_loops is not None:
        numerator = 2 * components - boundary_loops - chi
        if numerator >= 0 and numerator % 2 == 0:
            genus = numerator // 2

    return MeshTopologyReport(
        vertices=vertex_count,
        edges=len(edge_counts),
        faces=len(faces_array),
        connected_components=components,
        boundary_edges=len(boundary_edges),
        boundary_loops=boundary_loops,
        nonmanifold_edges=nonmanifold_edges,
        isolated_vertices=isolated_vertices,
        euler_characteristic=chi,
        genus=genus,
    )


def _validate_sdf(signed_distance: np.ndarray) -> np.ndarray:
    volume = np.asarray(signed_distance)
    if volume.ndim != 3 or min(volume.shape) < 5:
        raise ValueError("signed_distance must be a 3D volume with every axis at least 5 voxels")
    if not np.issubdtype(volume.dtype, np.floating):
        volume = volume.astype(np.float64)
    if not np.isfinite(volume).all():
        raise ValueError("signed_distance must be finite")
    return volume


def _count_boundary_loops(edges: list[tuple[int, int]]) -> int | None:
    if not edges:
        return 0
    adjacency: dict[int, set[int]] = {}
    for first, second in edges:
        adjacency.setdefault(first, set()).add(second)
        adjacency.setdefault(second, set()).add(first)
    if any(len(neighbors) != 2 for neighbors in adjacency.values()):
        return None
    remaining = set(adjacency)
    loops = 0
    while remaining:
        loops += 1
        stack = [remaining.pop()]
        while stack:
            current = stack.pop()
            for neighbor in adjacency[current]:
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
    return loops


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def join(self, first: int, second: int) -> None:
        root_first = self.find(first)
        root_second = self.find(second)
        if root_first == root_second:
            return
        if self.rank[root_first] < self.rank[root_second]:
            root_first, root_second = root_second, root_first
        self.parent[root_second] = root_first
        if self.rank[root_first] == self.rank[root_second]:
            self.rank[root_first] += 1


def _load_module(module_path: Path) -> ModuleType:
    spec = spec_from_file_location("external_cortexode_tca", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load topology module from {module_path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _construct_corrector(module: ModuleType, lookup_table: Path) -> object:
    # Upstream hard-codes a cwd-relative lookup path. Construct the same object
    # from its public functions while keeping our process cwd unchanged.
    corrector_type = getattr(module, "topology")
    corrector = corrector_type.__new__(corrector_type)
    bit, lookup = module.tca_init_fill(str(lookup_table), threshold=1.0)
    corrector.bit = bit
    corrector.lut = lookup
    return corrector
