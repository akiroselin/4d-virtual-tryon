from __future__ import annotations

from pathlib import Path

import numpy as np

from .data import AvatarSpec, GarmentIntent, GarmentMesh, MaterialSpec, PatternSpec, TshirtSizeSpec


def material_from_intent(intent: GarmentIntent) -> MaterialSpec:
    fabric = intent.fabric.lower()
    if "silk" in fabric:
        return MaterialSpec(name=intent.fabric, density=1.2, stretch_stiffness=5.0, bending_stiffness=0.35, damping=0.985)
    if "denim" in fabric:
        return MaterialSpec(name=intent.fabric, density=7.0, stretch_stiffness=14.0, bending_stiffness=1.8, damping=0.96)
    if "linen" in fabric:
        return MaterialSpec(name=intent.fabric, density=2.0, stretch_stiffness=9.0, bending_stiffness=0.7, damping=0.975)
    if "wool" in fabric:
        return MaterialSpec(name=intent.fabric, density=3.4, stretch_stiffness=6.5, bending_stiffness=0.65, damping=0.98)
    if "nylon" in fabric or "polyamide" in fabric:
        return MaterialSpec(name=intent.fabric, density=2.4, stretch_stiffness=11.0, bending_stiffness=0.55, damping=0.94)
    return MaterialSpec(name=intent.fabric)


def build_tshirt_pattern(
    intent: GarmentIntent,
    avatar: AvatarSpec,
    resolution: int = 23,
    size: TshirtSizeSpec | None = None,
) -> PatternSpec:
    fit = {
        "slim": 1.02,
        "relaxed": 1.14,
        "oversized": 1.28,
    }.get(intent.silhouette, 1.14)
    sleeve_factor = {
        "sleeveless": 0.18,
        "short": 0.42,
        "long": 1.05,
    }.get(intent.sleeve, 0.42)
    neck_factor = {
        "crew": 0.28,
        "v-neck": 0.34,
        "boat": 0.45,
    }.get(intent.neckline, 0.28)

    if size is not None:
        # Pattern x-width is the flat garment width, while the user-facing
        # measurement is full chest circumference.
        body_width = size.chest_circumference_cm / 200.0
        body_length = size.body_length_cm / 100.0
        shoulder_width = size.shoulder_width_cm / 100.0
        sleeve_length = size.sleeve_length_cm / 100.0
        sleeve_width = size.sleeve_opening_cm / 200.0
        neck_width = size.neck_width_cm / 100.0
    else:
        # A front/back pattern panel is half of the full chest circumference.
        body_width = avatar.chest_cm / 200.0 * fit
        body_length = avatar.height_cm / 100.0 * 0.38
        shoulder_width = avatar.shoulder_cm / 100.0 * fit
        sleeve_length = shoulder_width * sleeve_factor
        sleeve_width = body_width * 0.34
        neck_width = shoulder_width * neck_factor

    return PatternSpec(
        garment_type="t-shirt",
        width=max(11, resolution),
        height=max(13, int(resolution * 1.18)),
        body_width=body_width,
        body_length=body_length,
        shoulder_width=shoulder_width,
        sleeve_length=sleeve_length,
        sleeve_width=sleeve_width,
        neck_width=neck_width,
        material=material_from_intent(intent),
        size=size,
    )


def generate_tshirt_mesh(pattern: PatternSpec) -> GarmentMesh:
    panel_vertices, panel_faces, panel_labels = _generate_tshirt_panel(pattern)
    return _build_tshirt_shell(pattern, panel_vertices, panel_faces, panel_labels)


def _generate_tshirt_panel(pattern: PatternSpec) -> tuple[np.ndarray, np.ndarray, list[str]]:
    x_min = -pattern.shoulder_width * 0.5 - pattern.sleeve_length
    x_max = pattern.shoulder_width * 0.5 + pattern.sleeve_length
    y_min = -pattern.body_length * 0.58
    y_max = pattern.body_length * 0.42
    vertices: list[list[float]] = []
    labels: list[str] = []
    index: dict[tuple[int, int], int] = {}

    for iy in range(pattern.height):
        y = y_min + (y_max - y_min) * iy / (pattern.height - 1)
        for ix in range(pattern.width):
            x = x_min + (x_max - x_min) * ix / (pattern.width - 1)
            label = _tshirt_region(x, y, pattern)
            if label is None:
                continue
            z = 1.0 + 0.01 * np.sin(7.0 * x) * np.cos(5.0 * y)
            index[(ix, iy)] = len(vertices)
            vertices.append([x, y, z])
            labels.append(label)

    faces: list[list[int]] = []
    for iy in range(pattern.height - 1):
        for ix in range(pattern.width - 1):
            cell = [(ix, iy), (ix + 1, iy), (ix, iy + 1), (ix + 1, iy + 1)]
            if not all(point in index for point in cell):
                continue
            a, b, c, d = (index[point] for point in cell)
            faces.append([a, b, c])
            faces.append([b, d, c])

    return np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.int32), labels


def _build_tshirt_shell(
    pattern: PatternSpec,
    panel_vertices: np.ndarray,
    panel_faces: np.ndarray,
    panel_labels: list[str],
) -> GarmentMesh:
    material_panel = panel_vertices.copy()
    material_panel[:, 2] = 0.0
    front = panel_vertices.copy()
    back = panel_vertices.copy()
    front[:, 2] += 0.035
    back[:, 2] -= 0.035
    vertices_np = np.concatenate([front, back], axis=0).astype(np.float32)
    offset = panel_vertices.shape[0]

    front_faces = panel_faces.copy()
    back_faces = (panel_faces[:, [2, 1, 0]] + offset).astype(np.int32)
    faces_np = np.concatenate([front_faces, back_faces], axis=0)
    labels = [f"front_{label}" for label in panel_labels] + [f"back_{label}" for label in panel_labels]
    seam_edges = _shell_seam_pairs(pattern, panel_vertices, panel_faces, offset)

    rest_vertices = np.concatenate([material_panel, material_panel], axis=0).astype(np.float32)
    xy = rest_vertices[:, :2]
    uv_min = xy.min(axis=0)
    uv_span = np.maximum(xy.max(axis=0) - uv_min, 1e-6)
    uvs = ((xy - uv_min) / uv_span).astype(np.float32)

    y_max = pattern.body_length * 0.42
    pinned = _select_wearing_support_pins(vertices_np, pattern, y_max)
    return GarmentMesh(
        vertices=vertices_np,
        faces=faces_np,
        pinned_vertices=pinned,
        panel_labels=labels,
        seam_edges=sorted(seam_edges),
        rest_vertices=rest_vertices,
        uvs=uvs,
    )


def export_obj(path: Path, mesh: GarmentMesh) -> None:
    lines = ["# AIGC fashion prototype garment mesh\n"]
    for x, y, z in mesh.vertices:
        lines.append(f"v {x:.8f} {y:.8f} {z:.8f}\n")
    for a, b, c in mesh.faces:
        lines.append(f"f {a + 1} {b + 1} {c + 1}\n")
    path.write_text("".join(lines), encoding="utf-8")


def _tshirt_region(x: float, y: float, p: PatternSpec) -> str | None:
    body_half = p.body_width * 0.5
    shoulder_half = p.shoulder_width * 0.5
    y_min = -p.body_length * 0.58
    y_max = p.body_length * 0.42
    body_top = y_max * 0.58
    shoulder_low = y_max * 0.32
    body = abs(x) <= body_half and y_min <= y <= body_top
    shoulder = _is_shoulder_region(x, y, p, shoulder_half, shoulder_low, y_max)
    sleeve = _is_short_sleeve_region(x, y, p, body_half, shoulder_half, y_max)
    neck = _is_crew_neck_opening(x, y, p)

    if neck:
        return None
    if body:
        return "front_body"
    if shoulder:
        return "shoulder"
    if sleeve == "left":
        return "left_sleeve"
    if sleeve == "right":
        return "right_sleeve"
    return None


def _is_shoulder_region(
    x: float,
    y: float,
    p: PatternSpec,
    shoulder_half: float,
    shoulder_low: float,
    y_max: float,
) -> bool:
    abs_x = abs(x)
    if abs_x > shoulder_half:
        return False
    shoulder_t = min(abs_x / max(shoulder_half, 1e-6), 1.0)
    slope_drop = (0.010 + p.sleeve_width * 0.46) * shoulder_t**1.35
    top_y = y_max - slope_drop
    armhole_dip = np.clip((shoulder_t - 0.72) / 0.28, 0.0, 1.0) * p.sleeve_width * 0.10
    bottom_y = shoulder_low - armhole_dip
    return bottom_y <= y <= max(bottom_y + 0.020, top_y)


def _collect_seams(seam_edges: set[tuple[int, int]], labels: list[str], edges: list[tuple[int, int]]) -> None:
    for a, b in edges:
        if labels[a] != labels[b]:
            seam_edges.add(tuple(sorted((a, b))))


def _collect_panel_transition_seams(seam_edges: set[tuple[int, int]], labels: list[str], faces: np.ndarray) -> None:
    for a, b, c in faces:
        _collect_seams(seam_edges, labels, [(int(a), int(b)), (int(b), int(c)), (int(c), int(a))])


def _shell_seam_pairs(
    pattern: PatternSpec,
    vertices: np.ndarray,
    faces: np.ndarray,
    offset: int,
) -> set[tuple[int, int]]:
    counts: dict[tuple[int, int], int] = {}
    for a, b, c in faces:
        for edge in ((int(a), int(b)), (int(b), int(c)), (int(c), int(a))):
            key = tuple(sorted(edge))
            counts[key] = counts.get(key, 0) + 1

    seam_edges: set[tuple[int, int]] = set()
    for a, b in sorted(edge for edge, count in counts.items() if count == 1):
        midpoint = (vertices[a] + vertices[b]) * 0.5
        if _is_opening_boundary(float(midpoint[0]), float(midpoint[1]), pattern):
            continue
        seam_edges.add((a, a + offset))
        seam_edges.add((b, b + offset))
    return seam_edges


def _is_opening_boundary(x: float, y: float, p: PatternSpec) -> bool:
    x_min = -p.shoulder_width * 0.5 - p.sleeve_length
    x_max = p.shoulder_width * 0.5 + p.sleeve_length
    y_min = -p.body_length * 0.58
    y_max = p.body_length * 0.42
    hem = y < y_min + 0.045
    sleeve_cuff = x < x_min + 0.045 or x > x_max - 0.045
    neckline = _is_crew_neck_opening(x, y, p, margin=0.025)
    return hem or sleeve_cuff or neckline


def _is_crew_neck_opening(x: float, y: float, p: PatternSpec, margin: float = 0.0) -> bool:
    y_max = p.body_length * 0.42
    neck_half = max(p.neck_width * 0.5 + margin, 1e-6)
    if abs(x) > neck_half:
        return False
    neck_depth = float(np.clip(p.neck_width * 0.28, 0.035, 0.060))
    t = min(abs(x) / neck_half, 1.0)
    boundary = y_max - neck_depth * np.sqrt(max(0.0, 1.0 - t * t))
    return y > boundary - margin


def _is_short_sleeve_region(
    x: float,
    y: float,
    p: PatternSpec,
    body_half: float,
    shoulder_half: float,
    y_max: float,
) -> str | None:
    side = -1.0 if x < 0.0 else 1.0
    abs_x = abs(x)
    attach_x = min(body_half * 0.82, shoulder_half * 0.96)
    outer_x = shoulder_half + p.sleeve_length
    if abs_x < attach_x or abs_x > outer_x:
        return None
    span = max(outer_x - attach_x, 1e-6)
    t = (abs_x - attach_x) / span
    sleeve_center_y = y_max * 0.56 - t * max(0.035, p.sleeve_width * 0.36)
    sleeve_half_height = p.sleeve_width * (0.72 - 0.12 * t)
    if abs(y - sleeve_center_y) <= sleeve_half_height:
        return "left" if side < 0.0 else "right"
    return None


def _select_wearing_support_pins(vertices: np.ndarray, pattern: PatternSpec, y_max: float) -> np.ndarray:
    if vertices.size == 0:
        return np.asarray([], dtype=np.int32)
    inner_shoulder = max(pattern.neck_width * 0.68, pattern.shoulder_width * 0.24)
    targets = [
        (-inner_shoulder, y_max * 0.82),
        (-pattern.shoulder_width * 0.30, y_max * 0.68),
        (pattern.shoulder_width * 0.30, y_max * 0.68),
        (inner_shoulder, y_max * 0.82),
    ]
    pins = []
    for x, y in targets:
        distances = np.linalg.norm(vertices[:, :2] - np.asarray([x, y], dtype=np.float32), axis=1)
        pins.append(int(np.argmin(distances)))
    return np.unique(np.asarray(pins, dtype=np.int32))
