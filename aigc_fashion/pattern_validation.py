from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PatternValidationReport:
    status: str
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    metrics: dict[str, float]

    @property
    def valid(self) -> bool:
        return self.status == "valid"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def validate_tshirt_pattern_spec(
    spec_path: str | Path,
    max_stitch_ratio: float = 1.35,
) -> PatternValidationReport:
    """Check whether a learned pattern is structurally safe to simulate."""

    path = Path(spec_path)
    errors: list[str] = []
    warnings: list[str] = []
    metrics: dict[str, float] = {}
    try:
        spec = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return PatternValidationReport("invalid", (f"cannot read pattern JSON: {exc}",), (), {})

    pattern = spec.get("pattern", spec)
    panels = pattern.get("panels", {})
    stitches = pattern.get("stitches", [])
    units = float(spec.get("properties", pattern.get("properties", {})).get("units_in_meter", 100.0))
    if units <= 0.0:
        return PatternValidationReport("invalid", ("units_in_meter must be positive",), (), {})
    if not isinstance(panels, dict) or not panels:
        return PatternValidationReport("invalid", ("pattern has no panels",), (), {})

    panel_areas: dict[str, float] = {}
    edge_lengths: dict[tuple[str, int], float] = {}
    for name, panel in panels.items():
        vertices = np.asarray(panel.get("vertices", []), dtype=np.float64)
        if vertices.ndim != 2 or vertices.shape[0] < 3 or vertices.shape[1] < 2:
            errors.append(f"panel {name!r} has fewer than three 2D vertices")
            continue
        vertices = vertices[:, :2] / units
        area = 0.5 * abs(
            float(
                np.dot(vertices[:, 0], np.roll(vertices[:, 1], -1))
                - np.dot(vertices[:, 1], np.roll(vertices[:, 0], -1))
            )
        )
        panel_areas[str(name)] = area
        if area < 1e-5:
            errors.append(f"panel {name!r} is degenerate")

        edges = panel.get("edges") or [
            {"endpoints": [index, (index + 1) % vertices.shape[0]]}
            for index in range(vertices.shape[0])
        ]
        for edge_index, edge in enumerate(edges):
            endpoints = edge.get("endpoints", [])
            if not isinstance(endpoints, list) or len(endpoints) != 2:
                errors.append(f"panel {name!r} edge {edge_index} has invalid endpoints")
                continue
            start_index, end_index = int(endpoints[0]), int(endpoints[1])
            if not (0 <= start_index < vertices.shape[0] and 0 <= end_index < vertices.shape[0]):
                errors.append(f"panel {name!r} edge {edge_index} references a missing vertex")
                continue
            edge_lengths[(str(name), edge_index)] = _edge_length(
                vertices[start_index],
                vertices[end_index],
                edge.get("curvature"),
            )

    sleeve_names = {name for name in panels if "sleeve" in name.lower()}
    body_names = set(panels) - sleeve_names
    if len(body_names) < 2:
        errors.append("T-shirt pattern needs front and back body panels")
    if len(sleeve_names) < 2:
        errors.append("T-shirt pattern needs left and right sleeve panels")

    stitch_ratios: list[float] = []
    sleeve_body_lengths: dict[str, list[tuple[float, float]]] = {name: [] for name in sleeve_names}
    sleeve_body_stitches: dict[str, int] = {name: 0 for name in sleeve_names}
    for stitch_index, stitch in enumerate(stitches):
        if not isinstance(stitch, list) or len(stitch) != 2:
            errors.append(f"stitch {stitch_index} is not an edge pair")
            continue
        keys: list[tuple[str, int]] = []
        for side in stitch:
            key = (str(side.get("panel", "")), int(side.get("edge", -1)))
            if key not in edge_lengths:
                errors.append(f"stitch {stitch_index} references missing edge {key}")
            keys.append(key)
        if any(key not in edge_lengths for key in keys):
            continue
        first_length, second_length = edge_lengths[keys[0]], edge_lengths[keys[1]]
        ratio = max(first_length, second_length) / max(min(first_length, second_length), 1e-9)
        stitch_ratios.append(ratio)
        if ratio > max_stitch_ratio:
            errors.append(f"stitch {stitch_index} length ratio {ratio:.2f} exceeds {max_stitch_ratio:.2f}")

        first_sleeve = keys[0][0] in sleeve_names
        second_sleeve = keys[1][0] in sleeve_names
        if first_sleeve != second_sleeve:
            sleeve_key = keys[0] if first_sleeve else keys[1]
            body_key = keys[1] if first_sleeve else keys[0]
            sleeve_body_lengths[sleeve_key[0]].append(
                (edge_lengths[sleeve_key], edge_lengths[body_key])
            )
            sleeve_body_stitches[sleeve_key[0]] += 1

    for sleeve_name in sorted(sleeve_names):
        pairs = sleeve_body_lengths[sleeve_name]
        if not pairs:
            errors.append(f"sleeve panel {sleeve_name!r} is not stitched to a body armhole")
            continue
        sleeve_length = sum(pair[0] for pair in pairs)
        armhole_length = sum(pair[1] for pair in pairs)
        ratio = max(sleeve_length, armhole_length) / max(min(sleeve_length, armhole_length), 1e-9)
        metrics[f"{sleeve_name}_armhole_length_ratio"] = float(ratio)
        if ratio > max_stitch_ratio:
            errors.append(
                f"sleeve {sleeve_name!r} cap/armhole ratio {ratio:.2f} exceeds {max_stitch_ratio:.2f}"
            )
        if sleeve_body_stitches[sleeve_name] < 2:
            warnings.append(f"sleeve panel {sleeve_name!r} has only one armhole stitch segment")

    metrics.update(
        {
            "panel_count": float(len(panels)),
            "stitch_count": float(len(stitches)),
            "max_stitch_length_ratio": float(max(stitch_ratios, default=0.0)),
            "min_panel_area_m2": float(min(panel_areas.values(), default=0.0)),
        }
    )
    return PatternValidationReport(
        "invalid" if errors else "valid",
        tuple(errors),
        tuple(warnings),
        metrics,
    )


def _edge_length(start: np.ndarray, end: np.ndarray, curvature: Any) -> float:
    if not isinstance(curvature, (list, tuple)) or len(curvature) < 2:
        return float(np.linalg.norm(end - start))
    edge = end - start
    control = start + float(curvature[0]) * edge
    control += float(curvature[1]) * np.asarray([-edge[1], edge[0]], dtype=np.float64)
    t = np.linspace(0.0, 1.0, 65, dtype=np.float64)[:, None]
    points = (1.0 - t) ** 2 * start + 2.0 * (1.0 - t) * t * control + t**2 * end
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
