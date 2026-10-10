from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class ArrangementBoundingVolume:
    """Avatar-attached capsule/elliptic cylinder used for panel placement."""

    name: str
    region: str
    start: tuple[float, float, float]
    end: tuple[float, float, float]
    radius_x: float
    radius_y: float

    def __post_init__(self) -> None:
        if self.radius_x <= 0.0 or self.radius_y <= 0.0:
            raise ValueError("arrangement bounding-volume radii must be positive")
        if np.linalg.norm(np.asarray(self.end) - np.asarray(self.start)) <= 1.0e-7:
            raise ValueError("arrangement bounding volume needs a nonzero axis")


@dataclass(frozen=True)
class ArrangementPoint:
    """A Marvelous-Designer-style panel placement frame on an avatar volume."""

    name: str
    region: str
    side: str
    bounding_volume: str
    position: tuple[float, float, float]
    normal: tuple[float, float, float]
    wrap_axis: tuple[float, float, float]
    offset: float
    wrap_direction: str
    panel_roles: tuple[str, ...]

    def __post_init__(self) -> None:
        normal = np.asarray(self.normal, dtype=np.float64)
        axis = np.asarray(self.wrap_axis, dtype=np.float64)
        if abs(float(np.linalg.norm(normal)) - 1.0) > 1.0e-4:
            raise ValueError("arrangement-point normal must be unit length")
        if abs(float(np.linalg.norm(axis)) - 1.0) > 1.0e-4:
            raise ValueError("arrangement-point wrap axis must be unit length")
        if abs(float(np.dot(normal, axis))) > 1.0e-4:
            raise ValueError("arrangement-point normal and wrap axis must be orthogonal")
        if self.offset < 0.0:
            raise ValueError("arrangement-point offset must be non-negative")
        if self.wrap_direction not in {"up", "down", "around"}:
            raise ValueError("wrap direction must be up, down, or around")


@dataclass(frozen=True)
class AvatarArrangement:
    volumes: tuple[ArrangementBoundingVolume, ...]
    points: tuple[ArrangementPoint, ...]

    def point(self, name: str) -> ArrangementPoint:
        for point in self.points:
            if point.name == name:
                return point
        raise KeyError(f"unknown arrangement point: {name}")

    def points_for_role(self, role: str) -> tuple[ArrangementPoint, ...]:
        return tuple(point for point in self.points if role in point.panel_roles)

    def to_dict(self) -> dict[str, object]:
        return {
            "standard": "bounding-volume attached oriented arrangement points",
            "volumes": [asdict(volume) for volume in self.volumes],
            "points": [asdict(point) for point in self.points],
        }


def unit_vector(vector: Iterable[float]) -> tuple[float, float, float]:
    value = np.asarray(tuple(vector), dtype=np.float64)
    norm = float(np.linalg.norm(value))
    if norm <= 1.0e-10:
        raise ValueError("cannot normalize a zero arrangement vector")
    normalized = value / norm
    return tuple(float(component) for component in normalized)
