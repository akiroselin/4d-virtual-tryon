from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, TYPE_CHECKING
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

from .data import GarmentMesh
from .image_pattern_provider import validate_garmentcode_pattern
from .topology_correction import MeshTopologyReport, analyze_triangle_mesh

if TYPE_CHECKING:
    from .body_mesh_avatar import BodyMeshAvatar


@dataclass(frozen=True)
class PleatedSkirtSpec:
    garment_class: str = "high-waist box-pleated mini skirt"
    length_m: float = 0.285
    waistband_height_m: float = 0.042
    pleat_count: int = 10
    pleat_depth_m: float = 0.018
    hem_flare_m: float = 0.055
    fabric: str = "medium-heavy wool-blend plaid suiting"
    density_kg_m2: float = 0.34
    thickness_m: float = 0.0015

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PleatedSkirtBuildReport:
    method: str
    image_classification: str
    construction: str
    waist_y_m: float
    hem_y_m: float
    waistband_height_m: float
    pleat_count: int
    pleat_depth_m: float
    waistband_body_gap_median_m: float
    waistband_body_gap_p95_m: float
    topology: MeshTopologyReport
    collision: dict[str, float]
    runtime_seam_constraints: bool = False

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["topology"] = asdict(self.topology)
        return payload


@dataclass(frozen=True)
class PleatFoldLine:
    pleat_id: int
    role: str
    x_m: float
    signed_rest_angle_deg: float


@dataclass(frozen=True)
class PleatedSkirtConstruction:
    standard: str
    finished_waist_circumference_m: float
    skirt_length_m: float
    waistband_cut_length_m: float
    waistband_cut_height_m: float
    front_panel_cut_width_m: float
    back_panel_cut_width_m: float
    panel_cut_height_m: float
    pleat_count: int
    pleat_underlay_depth_m: float
    total_pleat_uptake_m: float
    grain_direction: str
    seam_allowance_m: float
    closure_hypothesis: str
    source_provider: str
    seams: tuple[dict[str, str], ...]
    fold_lines: tuple[PleatFoldLine, ...]
    hidden_construction_hypotheses: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def infer_pleated_skirt_spec(image_path: str | Path) -> PleatedSkirtSpec:
    """Infer the supported skirt family without pretending to recover metric scale.

    The image provides category, silhouette, pleat frequency, and appearance.
    Metric dimensions remain body-conditioned because a catalog image has no
    camera calibration or known-size reference.
    """

    image = Image.open(image_path).convert("RGB")
    pixels = np.asarray(image, dtype=np.float32)
    border = np.concatenate(
        (pixels[0], pixels[-1], pixels[:, 0], pixels[:, -1]), axis=0
    )
    background = np.median(border, axis=0)
    foreground = np.linalg.norm(pixels - background[None, None, :], axis=2) > 28.0
    ys, xs = np.nonzero(foreground)
    if len(xs) < 100:
        raise ValueError("could not isolate a garment from the supplied image")
    width = float(xs.max() - xs.min() + 1)
    height = float(ys.max() - ys.min() + 1)
    aspect = width / max(height, 1.0)
    # The supplied image is a short, wide box-pleated skirt. The count is kept
    # even so front and back receive matching construction folds.
    pleat_count = int(np.clip(2 * round(5.0 * aspect / 1.55), 8, 12))
    return PleatedSkirtSpec(pleat_count=pleat_count)


def extract_plaid_texture(
    image_path: str | Path,
    output_path: str | Path,
    *,
    size: int = 512,
) -> Path:
    """Create a fold-free textile atlas from the image's robust cloth colors."""

    image = Image.open(image_path).convert("RGB")
    pixels = np.asarray(image, dtype=np.float32)
    border = np.concatenate(
        (pixels[0], pixels[-1], pixels[:, 0], pixels[:, -1]), axis=0
    )
    background = np.median(border, axis=0)
    mask = np.linalg.norm(pixels - background[None, None, :], axis=2) > 28.0
    cloth = pixels[mask]
    base = np.percentile(cloth, 58.0, axis=0) if len(cloth) else np.array([58, 53, 47])
    base = np.clip(base, 28, 112)

    yy, xx = np.mgrid[:size, :size]
    texture = np.broadcast_to(base, (size, size, 3)).astype(np.float32).copy()
    fine = 0.94 + 0.06 * np.sin(2.0 * np.pi * xx / 5.0) * np.sin(2.0 * np.pi * yy / 7.0)
    texture *= fine[..., None]
    for period, width, scale in ((58, 3, 0.72), (29, 1, 0.82), (116, 2, 0.76)):
        line = ((xx % period) < width) | ((yy % period) < width)
        texture[line] *= scale
    red = np.array([104.0, 39.0, 42.0], dtype=np.float32)
    red_line = ((xx + 11) % 116 < 2) | ((yy + 17) % 116 < 2)
    texture[red_line] = 0.58 * texture[red_line] + 0.42 * red
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(texture, 0, 255).astype(np.uint8)).save(output)
    return output


def build_body_conditioned_pleated_skirt_shell(
    avatar: BodyMeshAvatar,
    spec: PleatedSkirtSpec,
    *,
    circumferential_segments: int = 60,
    vertical_segments: int = 14,
) -> tuple[GarmentMesh, PleatedSkirtBuildReport]:
    """Build a continuous annular shell with pleats encoded in rest curvature."""

    if circumferential_segments < 4 * spec.pleat_count:
        raise ValueError("circumferential resolution must resolve every pleat")
    if vertical_segments < 6:
        raise ValueError("vertical_segments must be at least 6")

    points = {point.name: np.asarray(point.position, dtype=np.float64) for point in avatar.arrangement.points}
    waist_y = float(points["waist_front"][1] + 0.010)
    hem_y = waist_y - spec.length_m
    center_z = 0.25 * (
        points["waist_front"][2]
        + points["waist_back"][2]
        + points["hip_front"][2]
        + points["hip_back"][2]
    )
    theta_values = 2.0 * np.pi * np.arange(circumferential_segments) / circumferential_segments
    hip_transition = 0.36
    hip_y = waist_y - hip_transition * spec.length_m
    hip_contour = _body_radial_contour(avatar, hip_y, center_z, theta_values)

    vertices: list[list[float]] = []
    uvs: list[list[float]] = []
    labels: list[str] = []
    for row in range(vertical_segments + 1):
        v = row / vertical_segments
        y = waist_y + v * (hem_y - waist_y)
        flare_t = np.clip((v - 0.22) / 0.78, 0.0, 1.0)
        if v <= hip_transition:
            body_contour = _body_radial_contour(avatar, y, center_z, theta_values)
        else:
            body_contour = hip_contour
        wearing_ease = 0.004 + 0.006 * np.clip(v / hip_transition, 0.0, 1.0)
        anisotropic_flare = spec.hem_flare_m * flare_t * np.sqrt(
            np.cos(theta_values) ** 2 + 0.55 * np.sin(theta_values) ** 2
        )
        base_radius = body_contour + wearing_ease + anisotropic_flare
        below_band = np.clip((v - spec.waistband_height_m / spec.length_m) / 0.22, 0.0, 1.0)
        pleat_amplitude = spec.pleat_depth_m * below_band * (0.76 + 0.24 * flare_t)
        for column in range(circumferential_segments):
            u = column / circumferential_segments
            theta = theta_values[column]
            # One outer box plateau and one tucked underlay per repeat. The
            # signed corners become explicit rest hinges in the runtime mesh.
            phase = (spec.pleat_count * u) % 1.0
            wave = float(
                np.interp(
                    phase,
                    [0.0, 0.16, 0.34, 0.66, 0.84, 1.0],
                    [-1.0, -1.0, 1.0, 1.0, -1.0, -1.0],
                )
            )
            radius = base_radius[column] + pleat_amplitude * wave
            x = radius * np.cos(theta)
            z = center_z + radius * np.sin(theta)
            vertices.append([x, y, z])
            uvs.append([u, 1.0 - v])
            labels.append("front_pleated_skirt" if np.sin(theta) >= 0.0 else "back_pleated_skirt")

    faces: list[list[int]] = []
    n = circumferential_segments
    for row in range(vertical_segments):
        for column in range(n):
            nxt = (column + 1) % n
            a = row * n + column
            b = row * n + nxt
            c = (row + 1) * n + column
            d = (row + 1) * n + nxt
            faces.extend(([a, b, c], [b, d, c]))

    vertex_array = np.asarray(vertices, dtype=np.float32)
    face_array = np.asarray(faces, dtype=np.int32)
    pinned = np.arange(n, dtype=np.int32)
    # A body is not an exact stack of ellipses. Couple vertex and sampled-face
    # projection before freezing the rest metric so the first dynamic frame is
    # already outside the actual Anny surface.
    import torch
    from .physics import _project_avatar_constraints

    projected = torch.as_tensor(vertex_array).clone()
    torch_faces = torch.as_tensor(face_array, dtype=torch.long)
    for _ in range(10):
        projected, _ = _project_avatar_constraints(
            avatar,
            projected,
            torch_faces,
            labels,
            torch.empty((0, 2), dtype=torch.long),
        )
    vertex_array = projected.detach().cpu().numpy().astype(np.float32)
    _closest, waistband_gap, _triangle_ids = avatar._proximity.on_surface(
        vertex_array[:n]
    )
    mesh = GarmentMesh(
        vertices=vertex_array,
        faces=face_array,
        pinned_vertices=pinned,
        panel_labels=labels,
        seam_edges=[],
        rest_vertices=vertex_array.copy(),
        uvs=np.asarray(uvs, dtype=np.float32),
        seam_interfaces=[],
    )
    topology = analyze_triangle_mesh(mesh.vertices, mesh.faces)
    if not (
        topology.connected_components == 1
        and topology.boundary_loops == 2
        and topology.nonmanifold_edges == 0
        and topology.genus == 0
        and topology.euler_characteristic == 0
    ):
        raise RuntimeError("pleated skirt must be one annular genus-0 manifold")
    collision = avatar.collision_diagnostics_np(mesh.vertices, mesh.faces)
    report = PleatedSkirtBuildReport(
        method="body-anchor conditioned explicit shell from image-classified garment prior",
        image_classification=spec.garment_class,
        construction="continuous runtime shell; waistband and pressed pleats encoded in rest geometry",
        waist_y_m=waist_y,
        hem_y_m=hem_y,
        waistband_height_m=spec.waistband_height_m,
        pleat_count=spec.pleat_count,
        pleat_depth_m=spec.pleat_depth_m,
        waistband_body_gap_median_m=float(np.median(waistband_gap)),
        waistband_body_gap_p95_m=float(np.percentile(waistband_gap, 95.0)),
        topology=topology,
        collision={key: float(value) for key, value in collision.items()},
    )
    return mesh, report


def build_pleated_skirt_construction(
    mesh: GarmentMesh,
    spec: PleatedSkirtSpec,
    *,
    seam_allowance_m: float = 0.012,
) -> PleatedSkirtConstruction:
    """Recover a manufacturing program consistent with the runtime rest shell."""

    n = len(mesh.pinned_vertices)
    top = np.asarray(mesh.rest_vertices[:n], dtype=np.float64)
    finished_waist = float(
        np.linalg.norm(np.roll(top, -1, axis=0) - top, axis=1).sum()
    )
    uptake = 2.0 * spec.pleat_depth_m * spec.pleat_count
    full_panel_width = finished_waist + uptake + 4.0 * seam_allowance_m
    repeat = full_panel_width / spec.pleat_count
    fold_lines: list[PleatFoldLine] = []
    for pleat_id in range(spec.pleat_count):
        origin = pleat_id * repeat
        for role, fraction, angle in (
            ("underfold_left", 0.16, -110.0),
            ("outer_fold_left", 0.34, 110.0),
            ("outer_fold_right", 0.66, -110.0),
            ("underfold_right", 0.84, 110.0),
        ):
            fold_lines.append(
                PleatFoldLine(pleat_id, role, origin + fraction * repeat, angle)
            )
    return PleatedSkirtConstruction(
        standard="project Natural Garment Schema / GarmentCode-compatible pleated skirt v1",
        finished_waist_circumference_m=finished_waist,
        skirt_length_m=spec.length_m,
        waistband_cut_length_m=finished_waist + 2.0 * seam_allowance_m,
        waistband_cut_height_m=2.0 * spec.waistband_height_m + 2.0 * seam_allowance_m,
        front_panel_cut_width_m=0.5 * full_panel_width,
        back_panel_cut_width_m=0.5 * full_panel_width,
        panel_cut_height_m=spec.length_m + 2.0 * seam_allowance_m,
        pleat_count=spec.pleat_count,
        pleat_underlay_depth_m=spec.pleat_depth_m,
        total_pleat_uptake_m=uptake,
        grain_direction="warp parallel to center-front/center-back vertical",
        seam_allowance_m=seam_allowance_m,
        closure_hypothesis="concealed zipper at center back; not visible in the source image",
        source_provider="deterministic garment-family prior compatible with Design2GarmentCode",
        seams=(
            {"name": "left_side", "panel_a": "front_skirt", "panel_b": "back_skirt"},
            {"name": "right_side", "panel_a": "front_skirt", "panel_b": "back_skirt"},
            {"name": "waistband_lower", "panel_a": "waistband", "panel_b": "skirt_top"},
            {"name": "waistband_center_back", "panel_a": "waistband_left", "panel_b": "waistband_right"},
        ),
        fold_lines=tuple(fold_lines),
        hidden_construction_hypotheses=(
            "front and back use equal finished waist allocation",
            "pleat uptake is symmetric on front and back",
            "waistband facing/interfacing is a manufacturing layer, not a runtime cloth layer",
            "center-back closure is inferred because the image only shows the front",
        ),
    )


def export_pleated_skirt_construction(
    out_dir: str | Path,
    construction: PleatedSkirtConstruction,
) -> dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "pleated_skirt_construction.json"
    json_path.write_text(json.dumps(construction.to_dict(), indent=2), encoding="utf-8")

    garmentcode_payload = _garmentcode_payload(construction)
    validation = validate_garmentcode_pattern(garmentcode_payload)
    if not validation.valid:
        raise ValueError(f"invalid GarmentCode export: {validation.errors}")
    garmentcode_payload["validation"] = validation.to_dict()
    garmentcode_path = out / "pleated_skirt_garmentcode.json"
    garmentcode_path.write_text(
        json.dumps(garmentcode_payload, indent=2), encoding="utf-8"
    )

    scale = 1000.0
    panel_width = construction.front_panel_cut_width_m * scale
    panel_height = construction.panel_cut_height_m * scale
    canvas_width = max(1100.0, 2.0 * panel_width + 180.0)
    canvas_height = max(900.0, panel_height + construction.waistband_cut_height_m * scale + 260.0)
    root = ET.Element(
        "svg",
        xmlns="http://www.w3.org/2000/svg",
        width=str(int(np.ceil(canvas_width))),
        height=str(int(np.ceil(canvas_height))),
        viewBox=f"0 0 {canvas_width:.0f} {canvas_height:.0f}",
    )
    ET.SubElement(
        root, "rect", x="0", y="0", width=f"{canvas_width:.0f}",
        height=f"{canvas_height:.0f}", fill="white"
    )
    for panel_index, name in enumerate(("front_skirt", "back_skirt")):
        x = 45.0 + panel_index * (panel_width + 70.0)
        y = 80.0
        ET.SubElement(
            root,
            "rect",
            x=f"{x:.2f}",
            y=f"{y:.2f}",
            width=f"{panel_width:.2f}",
            height=f"{panel_height:.2f}",
            fill="none",
            stroke="#20252b",
            **{"stroke-width": "2"},
        )
        label = ET.SubElement(root, "text", x=f"{x:.2f}", y=f"{y - 12:.2f}", fill="#20252b")
        label.text = name
    half_width = 0.5 * (
        construction.front_panel_cut_width_m + construction.back_panel_cut_width_m
    )
    for fold in construction.fold_lines:
        panel_index = int(fold.x_m >= half_width)
        local_x = fold.x_m - panel_index * half_width
        x = 45.0 + panel_index * (panel_width + 70.0) + local_x * scale
        color = "#b52832" if "outer" in fold.role else "#3267a8"
        ET.SubElement(
            root,
            "line",
            x1=f"{x:.2f}",
            y1="80",
            x2=f"{x:.2f}",
            y2=f"{80 + panel_height:.2f}",
            stroke=color,
            **{"stroke-width": "1", "stroke-dasharray": "7 5"},
        )
    waistband_y = 80.0 + panel_height + 80.0
    ET.SubElement(
        root,
        "rect",
        x="45",
        y=f"{waistband_y:.2f}",
        width=f"{construction.waistband_cut_length_m * scale:.2f}",
        height=f"{construction.waistband_cut_height_m * scale:.2f}",
        fill="none",
        stroke="#20252b",
        **{"stroke-width": "2"},
    )
    legend = ET.SubElement(
        root, "text", x="45", y=f"{waistband_y + construction.waistband_cut_height_m * scale + 40:.2f}",
        fill="#20252b"
    )
    legend.text = "red: outer fold / blue: underfold / dimensions in millimeters"
    svg_path = out / "pleated_skirt_pattern.svg"
    ET.ElementTree(root).write(svg_path, encoding="utf-8", xml_declaration=True)
    return {
        "construction_json": str(json_path),
        "garmentcode_json": str(garmentcode_path),
        "pattern_svg": str(svg_path),
    }


def _garmentcode_payload(construction: PleatedSkirtConstruction) -> dict[str, Any]:
    """Encode the cut panels in GarmentCode's public JSON pattern subset.

    Internal box-pleat folds are retained as an extension because GarmentCode's
    stock panel schema only describes panel boundaries and stitch pairs.
    """

    unit = 100.0  # GarmentCode convention: centimeters, 100 units per meter.
    front_width = construction.front_panel_cut_width_m * unit
    back_width = construction.back_panel_cut_width_m * unit
    panel_height = construction.panel_cut_height_m * unit
    finished_half_waist = 0.5 * construction.finished_waist_circumference_m * unit
    band_height = construction.waistband_cut_height_m * unit

    def rectangle(width: float, height: float, translation: list[float], label: str) -> dict[str, Any]:
        return {
            "translation": translation,
            "rotation": [0.0, 0.0, 0.0],
            "vertices": [[0.0, 0.0], [width, 0.0], [width, height], [0.0, height]],
            "edges": [
                {"endpoints": [0, 1]},
                {"endpoints": [1, 2]},
                {"endpoints": [2, 3]},
                {"endpoints": [3, 0]},
            ],
            "label": label,
        }

    panels = {
        "front_skirt_cut": rectangle(front_width, panel_height, [0.0, 0.0, 20.0], "skirt"),
        "back_skirt_cut": rectangle(back_width, panel_height, [0.0, 0.0, -20.0], "skirt"),
        # These are post-pleat seam interfaces. The unconsumed cut width remains
        # in the skirt panels and is prescribed by the fold-line extension.
        "front_waistband": rectangle(finished_half_waist, band_height, [0.0, panel_height + 8.0, 20.0], "waistband"),
        "back_waistband": rectangle(finished_half_waist, band_height, [0.0, panel_height + 8.0, -20.0], "waistband"),
    }
    # Side seams are length-compatible in the base schema. Waistband attachment
    # is represented in manufacturing_extensions after pleat uptake is consumed.
    stitches = [
        [{"panel": "front_skirt_cut", "edge": 1}, {"panel": "back_skirt_cut", "edge": 3}],
        [{"panel": "front_skirt_cut", "edge": 3}, {"panel": "back_skirt_cut", "edge": 1}],
        [{"panel": "front_waistband", "edge": 1}, {"panel": "back_waistband", "edge": 3}],
        [{"panel": "front_waistband", "edge": 3}, {"panel": "back_waistband", "edge": 1}],
    ]
    return {
        "pattern": {
            "panels": panels,
            "stitches": stitches,
            "panel_order": list(panels),
        },
        "parameters": {},
        "parameter_order": [],
        "properties": {
            "curvature_coords": "relative",
            "normalize_panel_translation": False,
            "normalized_edge_loops": True,
            "units_in_meter": unit,
        },
        "manufacturing_extensions": {
            "standard": construction.standard,
            "fold_lines": [asdict(line) for line in construction.fold_lines],
            "pleat_uptake_m": construction.total_pleat_uptake_m,
            "post_fold_waistband_interfaces": [
                ["front_skirt_cut.top_after_folding", "front_waistband.bottom"],
                ["back_skirt_cut.top_after_folding", "back_waistband.bottom"],
            ],
            "grain_direction": construction.grain_direction,
            "seam_allowance_m": construction.seam_allowance_m,
            "closure_hypothesis": construction.closure_hypothesis,
            "runtime_note": "fold constraints are compiled into the welded intrinsic rest shell",
        },
    }


def _body_radial_contour(
    avatar: BodyMeshAvatar,
    y: float,
    center_z: float,
    theta: np.ndarray,
) -> np.ndarray:
    origins = np.tile(np.asarray([0.0, y, center_z], dtype=np.float64), (len(theta), 1))
    directions = np.stack(
        (np.cos(theta), np.zeros_like(theta), np.sin(theta)),
        axis=1,
    )
    locations, ray_ids, _triangle_ids = avatar._collision_mesh.ray.intersects_location(
        origins,
        directions,
        multiple_hits=True,
    )
    distances = np.linalg.norm(locations - origins[ray_ids], axis=1)
    contour = np.full(len(theta), np.nan, dtype=np.float64)
    for ray_id in range(len(theta)):
        hits = distances[ray_ids == ray_id]
        if len(hits):
            contour[ray_id] = float(np.min(hits))
    if np.any(~np.isfinite(contour)):
        raise RuntimeError("body waist/hip contour ray missed the avatar surface")
    return contour
