from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .data import TshirtSizeSpec
from .image_fit import ImageTshirtFit


def write_image_conditioned_tshirt_spec(
    path: str | Path,
    size: TshirtSizeSpec,
    image_fit: ImageTshirtFit | None = None,
) -> Path:
    """Write a sewable four-panel T-shirt program in centimeter units."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    front_vertices = _body_panel_vertices(size)
    back_vertices = list(front_vertices)
    front_lengths = _armhole_edge_lengths(front_vertices)
    back_lengths = _armhole_edge_lengths(back_vertices)
    sleeve_vertices = _sleeve_panel_vertices(size, front_lengths, back_lengths)

    panels = {
        "front": _body_panel(front_vertices, front_neck=True),
        "back": _body_panel(back_vertices, front_neck=False),
        "left_sleeve": _sleeve_panel(sleeve_vertices),
        "right_sleeve": _sleeve_panel(sleeve_vertices),
    }
    stitches = [
        [_edge("front", 1), _edge("back", 1)],
        [_edge("front", 9), _edge("back", 9)],
        [_edge("front", 4), _edge("back", 4)],
        [_edge("front", 6), _edge("back", 6)],
        [_edge("left_sleeve", 6), _edge("front", 8)],
        [_edge("left_sleeve", 5), _edge("front", 7)],
        [_edge("left_sleeve", 4), _edge("back", 7)],
        [_edge("left_sleeve", 3), _edge("back", 8)],
        [_edge("right_sleeve", 6), _edge("front", 2)],
        [_edge("right_sleeve", 5), _edge("front", 3)],
        [_edge("right_sleeve", 4), _edge("back", 3)],
        [_edge("right_sleeve", 3), _edge("back", 2)],
        [_edge("left_sleeve", 0), _edge("left_sleeve", 2)],
        [_edge("right_sleeve", 0), _edge("right_sleeve", 2)],
    ]
    source = {
        "method": "image-conditioned parametric sewing-pattern program",
        "metric_anchor": "avatar measurements plus wearing ease",
    }
    if image_fit is not None:
        source.update(
            {
                "image_bbox_xyxy": list(image_fit.bbox_xyxy),
                "image_silhouette_aspect": image_fit.silhouette_aspect,
            }
        )
    spec = {
        "properties": {
            "units_in_meter": 100,
            "generator": "aigc_fashion.pattern_program",
            "source": source,
        },
        "pattern": {"panels": panels, "stitches": stitches},
    }
    path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    return path


def _body_panel_vertices(size: TshirtSizeSpec) -> list[list[float]]:
    half_width = size.chest_circumference_cm * 0.25
    half_length = size.body_length_cm * 0.5
    shoulder_half = min(size.shoulder_width_cm * 0.5, half_width * 0.93)
    neck_half = min(size.neck_width_cm * 0.5, shoulder_half * 0.62)
    shoulder_y = half_length - min(1.2, 0.08 * size.shoulder_width_cm)
    armhole_depth = float(
        np.clip(size.sleeve_opening_cm * 0.56, size.body_length_cm * 0.18, size.body_length_cm * 0.34)
    )
    if size.armhole_circumference_cm is not None:
        lower = size.body_length_cm * 0.18
        upper = size.body_length_cm * 0.34
        target_panel_armhole = size.armhole_circumference_cm * 0.5
        for _ in range(28):
            candidate = 0.5 * (lower + upper)
            candidate_length = _panel_armhole_length(
                half_width,
                half_length,
                shoulder_half,
                shoulder_y,
                candidate,
            )
            if candidate_length < target_panel_armhole:
                lower = candidate
            else:
                upper = candidate
        armhole_depth = 0.5 * (lower + upper)
    armhole_mid_x = half_width - max(0.7, 0.08 * armhole_depth)
    armhole_mid_y = half_length - 0.53 * armhole_depth
    return [
        [-half_width, -half_length],
        [half_width, -half_length],
        [half_width, half_length - armhole_depth],
        [armhole_mid_x, armhole_mid_y],
        [shoulder_half, shoulder_y],
        [neck_half, half_length],
        [-neck_half, half_length],
        [-shoulder_half, shoulder_y],
        [-armhole_mid_x, armhole_mid_y],
        [-half_width, half_length - armhole_depth],
    ]


def _panel_armhole_length(
    half_width: float,
    half_length: float,
    shoulder_half: float,
    shoulder_y: float,
    armhole_depth: float,
) -> float:
    underarm = np.asarray([half_width, half_length - armhole_depth], dtype=np.float64)
    armhole_mid = np.asarray(
        [half_width - max(0.7, 0.08 * armhole_depth), half_length - 0.53 * armhole_depth],
        dtype=np.float64,
    )
    shoulder = np.asarray([shoulder_half, shoulder_y], dtype=np.float64)
    return _curved_edge_length(underarm, armhole_mid, 0.12) + _curved_edge_length(
        armhole_mid,
        shoulder,
        0.12,
    )


def _curved_edge_length(start: np.ndarray, end: np.ndarray, curvature: float) -> float:
    edge = end - start
    control = start + 0.5 * edge + curvature * np.asarray([-edge[1], edge[0]], dtype=np.float64)
    t = np.linspace(0.0, 1.0, 65, dtype=np.float64)[:, None]
    points = (1.0 - t) ** 2 * start + 2.0 * (1.0 - t) * t * control + t**2 * end
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def _body_panel(vertices: list[list[float]], front_neck: bool) -> dict[str, object]:
    edges = [{"endpoints": [index, (index + 1) % len(vertices)]} for index in range(len(vertices))]
    for edge_id in (2, 3, 7, 8):
        edges[edge_id]["curvature"] = [0.5, 0.12]
    edges[5]["curvature"] = [0.5, 0.34 if front_neck else 0.13]
    return {
        "vertices": vertices,
        "edges": edges,
        "translation": [0, 0, 0],
        "rotation": [0, 0, 0],
    }


def _armhole_edge_lengths(vertices: list[list[float]]) -> tuple[float, float]:
    points = np.asarray(vertices, dtype=np.float64)
    return (
        float(np.linalg.norm(points[3] - points[2])),
        float(np.linalg.norm(points[4] - points[3])),
    )


def _sleeve_panel_vertices(
    size: TshirtSizeSpec,
    front_lengths: tuple[float, float],
    back_lengths: tuple[float, float],
) -> list[list[float]]:
    cap_lengths = (back_lengths[0], back_lengths[1], front_lengths[1], front_lengths[0])
    angles = np.deg2rad(np.asarray([210.0, 240.0, 300.0, 330.0], dtype=np.float64))
    deltas = np.column_stack([np.cos(angles), np.sin(angles)]) * np.asarray(cap_lengths)[:, None]
    cap_points = [-0.5 * np.sum(deltas, axis=0)]
    for delta in deltas[:-1]:
        cap_points.append(cap_points[-1] + delta)
    cap_top, cap_q1, cap_apex, cap_q3 = cap_points
    cap_bottom = cap_q3 + deltas[-1]

    sleeve_length = max(size.sleeve_length_cm, 4.0)
    cuff_half = max(size.sleeve_opening_cm * 0.5, 2.0)
    return [
        cap_bottom.tolist(),
        [sleeve_length, -cuff_half],
        [sleeve_length, cuff_half],
        cap_top.tolist(),
        cap_q1.tolist(),
        cap_apex.tolist(),
        cap_q3.tolist(),
    ]


def _sleeve_panel(vertices: list[list[float]]) -> dict[str, object]:
    edges = [{"endpoints": [index, (index + 1) % len(vertices)]} for index in range(len(vertices))]
    for edge_id in (3, 4, 5, 6):
        edges[edge_id]["curvature"] = [0.5, -0.12]
    return {
        "vertices": vertices,
        "edges": edges,
        "translation": [0, 0, 0],
        "rotation": [0, 0, 0],
    }


def _edge(panel: str, index: int) -> dict[str, object]:
    return {"panel": panel, "edge": index}
