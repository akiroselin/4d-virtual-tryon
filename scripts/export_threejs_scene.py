from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aigc_fashion.body_mesh_avatar import BodyMeshAvatar  # noqa: E402
from aigc_fashion.data import AvatarSpec  # noqa: E402
from aigc_fashion.mesh_refinement import midpoint_subdivide_garment  # noqa: E402
from aigc_fashion.render import front_print_uv_window  # noqa: E402
from examples.run_unified_anny_cloth4d import (  # noqa: E402
    _add_skirt_walking_ease,
    _shirt_mesh_from_validated_shell,
    _skirt_mesh_from_validated_shell,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a physical-time cloth field for Three.js WebGPU."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=(
            ROOT
            / "outputs"
            / "anny_unified_neuralclothsim4d_bundle_ph_v14_walk_v3"
        ),
    )
    parser.add_argument(
        "--skirt-run",
        type=Path,
        default=ROOT / "outputs" / "anny_pleated_skirt_bundle_ph_v14",
    )
    parser.add_argument(
        "--shirt-field-run",
        type=Path,
        default=ROOT / "outputs" / "anny_unified_neuralclothsim4d_walk_step_lie",
        help="Validated dense T-shirt field with the same body clock.",
    )
    parser.add_argument(
        "--shirt-mesh-run",
        type=Path,
        default=None,
        help="Optional trained T-shirt shell directory; defaults to shirt-field-run.",
    )
    parser.add_argument(
        "--skirt-image",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--shirt-texture",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--body-mesh",
        type=Path,
        default=ROOT / "assets" / "body_meshes" / "anny_child_neutral_lowpoly_12k.obj",
        help="Body mesh used for both the exported body geometry and scene bounds.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "web" / "scenes" / "anny-skirt-v14",
    )
    parser.add_argument(
        "--shirt-subdivision-levels",
        type=int,
        choices=(0, 1, 2),
        default=1,
        help="Must match the T-shirt topology used to generate the field.",
    )
    return parser.parse_args()


def _dog_atlas(path: Path, shirt) -> Image.Image:
    source = Image.open(path).convert("RGB")
    pixels = np.asarray(source)
    chroma = pixels.max(axis=2) - pixels.min(axis=2)
    dark = pixels.mean(axis=2) < 232
    ys, xs = np.nonzero((chroma > 18) & dark)
    crop = source.crop(
        (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    )
    size = 1024
    atlas = Image.new("RGB", (size, size), (246, 246, 244))
    print_min, print_max = front_print_uv_window(shirt, source)
    window_width = int(round((print_max[0] - print_min[0]) * size))
    window_height = int(round((print_max[1] - print_min[1]) * size))
    # Contain-fit preserves the dog's aspect ratio. The image-space y origin
    # is top-left, while the UV contract is bottom-left; flip the v window.
    scale = min(window_width / crop.width, window_height / crop.height)
    crop = crop.resize(
        (max(1, int(round(crop.width * scale))), max(1, int(round(crop.height * scale)))),
        Image.Resampling.LANCZOS,
    )
    window_x = int(round(print_min[0] * size))
    window_y = int(round((1.0 - print_max[1]) * size))
    x = window_x + (window_width - crop.width) // 2
    y = window_y + (window_height - crop.height) // 2
    atlas.paste(crop, (x, y))
    return atlas


def _canonical_shirt_uvs(
    shirt,
    *,
    chart_positions: np.ndarray | None = None,
    alignment_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Build an upright, continuous material chart for the front torso."""
    uvs = np.asarray(shirt.uvs, dtype=np.float32).copy()
    labels = np.asarray(shirt.panel_labels)
    front = labels == "front_front_body"
    if not np.any(front):
        return uvs
    material = np.asarray(
        shirt.material_vertices if chart_positions is None else chart_positions,
        dtype=np.float32,
    )
    if material.shape != (len(uvs), 3):
        raise ValueError("shirt chart positions must match the UV vertex count")

    alignment = front.copy()
    if alignment_mask is not None:
        alignment &= np.asarray(alignment_mask, dtype=bool)
    if np.count_nonzero(alignment) < 3:
        alignment = front

    # The reconstructed shell can carry a small x/y shear even before cloth
    # dynamics start. Remove that chart shear so an upright chest graphic is
    # not baked diagonally into the garment; subsequent physical deformation
    # still moves the UVs with the cloth vertices.
    align_x = material[alignment, 0]
    align_y = material[alignment, 1]
    y_variance = float(np.var(align_y))
    shear = 0.0
    if y_variance > 1.0e-10:
        shear = float(np.cov(align_x, align_y, bias=True)[0, 1] / y_variance)
    x = material[front, 0] - shear * (
        material[front, 1] - float(np.mean(align_y))
    )
    y = material[front, 1]
    x_span = max(float(np.ptp(x)), 1.0e-6)
    y_span = max(float(np.ptp(y)), 1.0e-6)
    # The front body occupies one continuous chart. Keep a white margin for
    # the sleeves/back while making the dog texture follow material x/y.
    uvs[front, 0] = 0.20 + 0.60 * (x - float(x.min())) / x_span
    uvs[front, 1] = 0.14 + 0.72 * (y - float(y.min())) / y_span
    return uvs


def _front_print_face_mask(
    shirt,
    faces: np.ndarray,
    rest_positions: np.ndarray,
    *,
    minimum_front_normal: float = 0.25,
) -> np.ndarray:
    """Select only outward-facing front-torso triangles for the print."""
    faces = np.asarray(faces, dtype=np.int32)
    positions = np.asarray(rest_positions, dtype=np.float32)
    labels = np.asarray(shirt.panel_labels)
    if positions.shape != (len(labels), 3):
        raise ValueError("shirt rest positions must match panel semantics")

    torso_vertices = labels == "front_front_body"
    torso_faces = np.all(torso_vertices[faces], axis=1)
    triangles = positions[faces]
    normals = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    normal_lengths = np.linalg.norm(normals, axis=1)
    front_component = normals[:, 2] / np.maximum(normal_lengths, 1.0e-12)
    return torso_faces & (front_component >= minimum_front_normal)


def _display_layer_guard(
    shirt_positions: np.ndarray,
    skirt_positions: np.ndarray,
    *,
    minimum_gap_m: float = 0.0012,
) -> tuple[np.ndarray, float, float]:
    """Keep the rendered hem/waist ordering without altering the physical field."""
    guarded = np.asarray(shirt_positions, dtype=np.float32).copy()
    minimum_before = float("inf")
    maximum_shift = 0.0
    for frame in range(len(guarded)):
        distance = np.linalg.norm(
            guarded[frame, :, None, :] - skirt_positions[frame, None, :, :],
            axis=-1,
        )
        gap = float(distance.min())
        minimum_before = min(minimum_before, gap)
        if gap >= minimum_gap_m:
            continue
        shirt_y = guarded[frame, :, 1]
        lower_hem_weight = np.clip(
            (float(shirt_y.max()) - shirt_y) / 0.10, 0.0, 1.0
        )
        shift = min(0.003, minimum_gap_m - gap)
        guarded[frame, :, 1] += shift * lower_hem_weight
        maximum_shift = max(maximum_shift, float(shift))
    return guarded, minimum_before, maximum_shift


def _write_array(output: Path, name: str, values: np.ndarray, dtype: str) -> str:
    path = output / name
    array = np.ascontiguousarray(values, dtype=np.dtype(dtype).newbyteorder("<"))
    array.tofile(path)
    return path.name


def _bounds(*arrays: np.ndarray) -> dict[str, list[float]]:
    points = np.concatenate([array.reshape(-1, 3) for array in arrays], axis=0)
    return {
        "minimum": points.min(axis=0).astype(float).tolist(),
        "maximum": points.max(axis=0).astype(float).tolist(),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    avatar = BodyMeshAvatar.from_obj(
        args.body_mesh,
        AvatarSpec(
            chest_cm=64,
            waist_cm=56,
            shoulder_cm=30,
            height_cm=128,
            pose="a-pose",
        ),
        source_up_axis="z",
        source_front_axis="-y",
    )
    shirt_mesh_run = args.shirt_field_run if args.shirt_mesh_run is None else args.shirt_mesh_run
    shirt = _shirt_mesh_from_validated_shell(avatar, shirt_mesh_run)
    shirt, _, _ = midpoint_subdivide_garment(
        shirt, shirt.vertices, levels=args.shirt_subdivision_levels
    )
    skirt = _skirt_mesh_from_validated_shell(
        avatar, args.skirt_image, args.skirt_run
    )
    skirt, _, _ = _add_skirt_walking_ease(
        skirt, skirt.vertices, body_center_z=avatar.base_z
    )

    field = np.load(args.run_dir / "unified_neuralclothsim4d_field.npz")
    shirt_field = np.load(
        args.shirt_field_run / "unified_neuralclothsim4d_field.npz"
    )
    times = np.asarray(field["physical_times_s"], dtype=np.float32)
    body_positions = np.asarray(field["body_positions"], dtype=np.float32)
    shirt_times = np.asarray(shirt_field["physical_times_s"], dtype=np.float32)
    shirt_positions = np.asarray(shirt_field["tshirt_positions"], dtype=np.float32)
    shirt_velocities = np.asarray(shirt_field["tshirt_velocities"], dtype=np.float32)
    shirt_faces = np.asarray(shirt_field["tshirt_faces"], dtype=np.int32)
    skirt_positions = np.asarray(field["pleated_skirt_positions"], dtype=np.float32)
    skirt_velocities = np.asarray(field["pleated_skirt_velocities"], dtype=np.float32)
    skirt_faces = np.asarray(field["pleated_skirt_faces"], dtype=np.int32)
    skirt_uvs = np.asarray(
        field["pleated_skirt_uvs"] if "pleated_skirt_uvs" in field else skirt.uvs,
        dtype=np.float32,
    )
    shirt_positions, layer_gap_before_guard_m, layer_guard_shift_m = (
        _display_layer_guard(shirt_positions, skirt_positions)
    )

    if shirt_positions.shape[1:] != (len(shirt.vertices), 3):
        raise ValueError("shirt topology does not match the physical-time field")
    if not np.allclose(shirt_times, times, atol=1.0e-7):
        raise ValueError("shirt and skirt fields must share one physical-time clock")
    if skirt_positions.shape[1:] != (len(skirt_uvs), 3):
        raise ValueError("skirt topology does not match the physical-time field and UV field")
    if shirt_positions.shape[1] == len(shirt.vertices) and not np.array_equal(shirt_faces, shirt.faces):
        raise ValueError("shirt faces do not match the exported field")
    if skirt_positions.shape[1] == len(skirt.vertices) and not np.array_equal(skirt_faces, skirt.faces):
        raise ValueError("skirt faces do not match the exported field")
    if body_positions.shape[1] != len(avatar.mesh_vertices):
        raise ValueError("avatar topology does not match the physical-time field")

    # The implicit shell's legacy panel labels cover both torso surfaces.
    # Resolve the actual outer chest from the physical rest-frame normals so
    # the back and side walls can never receive the print material.
    printed_faces = _front_print_face_mask(
        shirt,
        shirt_faces,
        shirt_positions[0],
    )
    printed_vertices = np.zeros(len(shirt.vertices), dtype=bool)
    printed_vertices[shirt_faces[printed_faces].reshape(-1)] = True
    # The physical field carries the legacy chart for compatibility; the
    # browser export uses an upright chart fitted to the true front surface.
    shirt_uvs = _canonical_shirt_uvs(
        shirt,
        chart_positions=shirt_positions[0],
        alignment_mask=printed_vertices,
    )
    shirt.uvs = shirt_uvs
    ordered_shirt_faces = np.concatenate(
        (shirt_faces[printed_faces], shirt_faces[~printed_faces]), axis=0
    )
    printed_index_count = int(np.count_nonzero(printed_faces) * 3)

    files = {
        "bodyPositions": _write_array(
            args.output_dir, "body-positions.f32", body_positions, "f4"
        ),
        "bodyIndices": _write_array(
            args.output_dir, "body-indices.u32", avatar.mesh_faces, "u4"
        ),
        "shirtPositions": _write_array(
            args.output_dir, "shirt-positions.f32", shirt_positions, "f4"
        ),
        "shirtVelocities": _write_array(
            args.output_dir, "shirt-velocities.f32", shirt_velocities, "f4"
        ),
        "shirtIndices": _write_array(
            args.output_dir, "shirt-indices.u32", ordered_shirt_faces, "u4"
        ),
        "shirtUvs": _write_array(
            args.output_dir, "shirt-uvs.f32", shirt_uvs, "f4"
        ),
        "skirtPositions": _write_array(
            args.output_dir, "skirt-positions.f32", skirt_positions, "f4"
        ),
        "skirtVelocities": _write_array(
            args.output_dir, "skirt-velocities.f32", skirt_velocities, "f4"
        ),
        "skirtIndices": _write_array(
            args.output_dir, "skirt-indices.u32", skirt_faces, "u4"
        ),
        "skirtUvs": _write_array(
            args.output_dir, "skirt-uvs.f32", skirt_uvs, "f4"
        ),
    }
    _dog_atlas(args.shirt_texture, shirt).save(args.output_dir / "shirt-print.png")
    Image.open(args.skirt_run / "image_conditioned_plaid.png").convert("RGB").save(
        args.output_dir / "skirt-plaid.png"
    )
    files["shirtTexture"] = "shirt-print.png"
    files["skirtTexture"] = "skirt-plaid.png"

    source_manifest = json.loads(
        (args.run_dir / "manifest.json").read_text(encoding="utf-8")
    )
    scene_manifest = {
        "format": "NeuralClothSim4D Three.js binary scene v1",
        "renderer": "Three.js WebGPURenderer",
        "threeVersion": "0.185.1",
        "units": "meters",
        "upAxis": "+Y",
        "frontAxis": "+Z",
        "frameCount": int(len(times)),
        "timesSeconds": times.astype(float).tolist(),
        "physicalDurationSeconds": float(times[-1] - times[0]),
        "displayLoopSeconds": 3.2,
        "bounds": _bounds(body_positions, shirt_positions, skirt_positions),
        "files": files,
        "meshes": {
            "body": {
                "vertexCount": int(body_positions.shape[1]),
                "indexCount": int(avatar.mesh_faces.size),
            },
            "shirt": {
                "vertexCount": int(shirt_positions.shape[1]),
                "indexCount": int(ordered_shirt_faces.size),
                "groups": [
                    {
                        "start": 0,
                        "count": printed_index_count,
                        "materialIndex": 0,
                    },
                    {
                        "start": printed_index_count,
                        "count": int(ordered_shirt_faces.size - printed_index_count),
                        "materialIndex": 1,
                    },
                ],
            },
            "skirt": {
                "vertexCount": int(skirt_positions.shape[1]),
                "indexCount": int(skirt_faces.size),
            },
        },
        "physicsAudit": {
            "minimumBodyGapMeters": source_manifest["report"][
                "minimum_body_gap_m"
            ]["pleated_skirt"],
            "minimumInterGarmentGapMeters": source_manifest["report"][
                "minimum_inter_garment_gap_m"
            ].get("tshirt|pleated_skirt"),
            "displayLayerGuard": {
                "enabled": layer_guard_shift_m > 0.0,
                "minimumGapBeforeGuardMeters": layer_gap_before_guard_m,
                "maximumTshirtHemShiftMeters": layer_guard_shift_m,
                "role": "render-only hem/waist ordering; physical field unchanged",
            },
            "skirtViolatingEdges": source_manifest["report"][
                "metric_projection"
            ]["pleated_skirt"]["violating_edges"],
            "skirtEdgeCount": source_manifest["report"]["metric_projection"][
                "pleated_skirt"
            ]["edge_count"],
            "metricProjectionEnabled": bool(
                source_manifest["report"].get("metric_projection_enabled", False)
            ),
            "lieMessageSubsteps": source_manifest["report"][
                "lie_group_message_passing"
            ]["substep_calls"],
            "massDriftKg": source_manifest["report"][
                "bundle_measure_port_hamiltonian"
            ]["pleated_skirt"]["mass_drift_kg"],
            "materialModel": source_manifest.get("runtime_material", {}),
            "noetherAudit": source_manifest["report"].get(
                "noether_audit",
                {
                    "status": "field generated before the Noether audit schema was added",
                    "structure_group": (
                        "diagonal E(3) on body-relative garment configurations; "
                        "SE(3) transport acts on the shared reference frame"
                    ),
                    "strict_conservation_scope": (
                        "internal shell/edge interactions; regenerate with the current "
                        "solver to populate numerical balance traces"
                    ),
                },
            ),
        },
        "source": {
            "runDirectory": str(args.run_dir),
            "field": str(args.run_dir / "unified_neuralclothsim4d_field.npz"),
            "shirtField": str(
                args.shirt_field_run / "unified_neuralclothsim4d_field.npz"
            ),
        },
    }
    manifest_path = args.output_dir / "scene.json"
    manifest_path.write_text(
        json.dumps(scene_manifest, indent=2), encoding="utf-8"
    )
    print(manifest_path)


if __name__ == "__main__":
    main()
