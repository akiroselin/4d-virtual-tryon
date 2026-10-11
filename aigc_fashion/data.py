from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class MaterialSpec:
    name: str = "medium cotton jersey"
    density: float = 3.0
    stretch_stiffness: float = 8.0
    bending_stiffness: float = 0.9
    damping: float = 0.97
    thickness: float = 0.002


@dataclass(frozen=True)
class AvatarSpec:
    body_model: str = "commercial_body_mesh"
    chest_cm: float = 92.0
    waist_cm: float = 76.0
    shoulder_cm: float = 44.0
    height_cm: float = 175.0
    pose: str = "a-pose"


@dataclass(frozen=True)
class TshirtSizeSpec:
    chest_circumference_cm: float = 104.0
    body_length_cm: float = 70.0
    shoulder_width_cm: float = 46.0
    sleeve_length_cm: float = 20.0
    sleeve_opening_cm: float = 36.0
    neck_width_cm: float = 18.0
    armhole_circumference_cm: float | None = None

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class GarmentIntent:
    prompt: str
    garment_type: str = "t-shirt"
    silhouette: str = "relaxed"
    sleeve: str = "short"
    neckline: str = "crew"
    fabric: str = "medium cotton jersey"
    style_refs: tuple[str, ...] = ("minimal",)
    body_fit: str = "loose"
    manufacturable: bool = True

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["style_refs"] = list(self.style_refs)
        return data


@dataclass(frozen=True)
class PatternSpec:
    garment_type: str
    width: int
    height: int
    body_width: float
    body_length: float
    shoulder_width: float
    sleeve_length: float
    sleeve_width: float
    neck_width: float
    material: MaterialSpec
    size: TshirtSizeSpec | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["material"] = asdict(self.material)
        if self.size is not None:
            data["size"] = self.size.to_dict()
        else:
            data["size"] = None
        return data


@dataclass(frozen=True)
class SeamInterface:
    """An oriented, continuous correspondence between two panel boundaries.

    The vertex paths are ordered in matching arc-length directions.  They may
    contain different numbers of vertices; callers should evaluate both paths
    at a shared normalized arc-length coordinate instead of assuming a
    one-to-one vertex pairing.
    """

    name: str
    panel_a: str
    panel_b: str
    edge_ids_a: tuple[int, ...]
    edge_ids_b: tuple[int, ...]
    vertices_a: tuple[int, ...]
    vertices_b: tuple[int, ...]
    side_b_reversed: bool = False
    ease: float = 1.0

    def __post_init__(self) -> None:
        if len(self.vertices_a) < 2 or len(self.vertices_b) < 2:
            raise ValueError("a seam interface needs at least two vertices on each side")
        if self.ease <= 0.0:
            raise ValueError("seam ease must be positive")


@dataclass
class GarmentMesh:
    vertices: np.ndarray
    faces: np.ndarray
    pinned_vertices: np.ndarray
    panel_labels: list[str]
    seam_edges: list[tuple[int, int]]
    rest_vertices: np.ndarray | None = None
    uvs: np.ndarray | None = None
    seam_interfaces: list[SeamInterface] = field(default_factory=list)

    @property
    def material_vertices(self) -> np.ndarray:
        """Return the undeformed sewing-pattern geometry."""

        return self.vertices if self.rest_vertices is None else self.rest_vertices

    def summary(self) -> dict[str, Any]:
        return {
            "vertices": int(self.vertices.shape[0]),
            "faces": int(self.faces.shape[0]),
            "pinned_vertices": int(self.pinned_vertices.shape[0]),
            "seam_edges": int(len(self.seam_edges)),
            "seam_interfaces": int(len(self.seam_interfaces)),
            "panels": sorted(set(self.panel_labels)),
            "has_material_rest_state": self.rest_vertices is not None,
            "has_uvs": self.uvs is not None,
        }
