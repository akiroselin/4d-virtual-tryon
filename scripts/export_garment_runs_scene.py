"""Export two separately simulated garments as one Three.js WebGPU outfit scene.

Use this when the T-shirt run (examples/run_anny_cloth4d.py) and the
pleated-skirt run (examples/run_anny_pleated_skirt.py) were simulated on the
same Anny body and do not overlap, so no inter-garment contact is needed (for
overlapping layers use run_unified_anny_cloth4d.py + export_threejs_scene.py).
The script checks the vertical gap between T-shirt hem and skirt waistband
and refuses to export when the garments overlap. Both physical rollouts are
resampled linearly onto one clock spanning the shorter rollout; the body is
static. Writes web/scenes/<name>/ in the format anny-cloth-webgpu.js reads.

    python scripts/export_garment_runs_scene.py --shirt-run outputs/demo_tshirt \\
        --skirt-run outputs/demo_skirt --shirt-texture tee.png --skirt-image skirt.png \\
        --output-dir web/scenes/demo-outfit
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aigc_fashion.body_mesh_avatar import BodyMeshAvatar  # noqa: E402
from aigc_fashion.data import AvatarSpec  # noqa: E402
from aigc_fashion.pleated_skirt import (  # noqa: E402
    build_body_conditioned_pleated_skirt_shell,
    infer_pleated_skirt_spec,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--shirt-run", type=Path, required=True)
    parser.add_argument("--skirt-run", type=Path, required=True)
    parser.add_argument("--shirt-texture", type=Path, required=True)
    parser.add_argument("--skirt-image", type=Path, required=True)
    parser.add_argument("--body-mesh", type=Path, default=ROOT / "assets" / "body_meshes" / "anny_child_neutral.obj")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-layer-gap-m", type=float, default=0.01)
    return parser.parse_args()


def _rollout(run: Path) -> tuple[np.ndarray, np.ndarray, float, dict]:
    positions = np.load(run / "neuralclothsim4d_physical_rollout.npy").astype(np.float32)
    velocities = np.load(run / "neuralclothsim4d_velocities.npy").astype(np.float32)
    report = json.loads((run / "neuralclothsim4d_dynamics.json").read_text(encoding="utf-8"))
    return positions, velocities, float(report["physical_duration_s"]), report


def _resample(frames: np.ndarray, duration: float, times: np.ndarray) -> np.ndarray:
    source = np.linspace(0.0, duration, len(frames))
    out = np.empty((len(times), *frames.shape[1:]), dtype=np.float32)
    for index, t in enumerate(times):
        k = int(np.clip(np.searchsorted(source, t, side="right") - 1, 0, len(frames) - 2))
        u = float(np.clip((t - source[k]) / max(source[k + 1] - source[k], 1e-12), 0.0, 1.0))
        out[index] = (1.0 - u) * frames[k] + u * frames[k + 1]
    return out


def _print_atlas(texture: Path, size: int = 1024) -> Image.Image:
    """Crop the chromatic print from the product image into the centre of a white atlas."""

    source = Image.open(texture).convert("RGB")
    pixels = np.asarray(source).astype(np.int16)
    chroma = pixels.max(axis=2) - pixels.min(axis=2)
    ys, xs = np.nonzero((chroma > 18) & (pixels.mean(axis=2) < 232))
    atlas = Image.new("RGB", (size, size), (246, 246, 242))
    if len(xs) == 0:
        return atlas
    crop = source.crop((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))
    scale = min(0.55 * size / crop.width, 0.55 * size / crop.height)
    crop = crop.resize((max(1, int(crop.width * scale)), max(1, int(crop.height * scale))), Image.Resampling.LANCZOS)
    atlas.paste(crop, ((size - crop.width) // 2, (size - crop.height) // 2))
    return atlas


def _shirt_uvs_and_print_faces(rest: np.ndarray, faces: np.ndarray, body_center_z: float) -> tuple[np.ndarray, np.ndarray]:
    """Planar chart over the whole outward front torso; the atlas margin, not face edges, bounds the print."""

    tri = rest[faces]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    centres = tri.mean(axis=1)
    top, hem = float(rest[:, 1].max()), float(rest[:, 1].min())
    torso_half = 0.5 * float(np.ptp(rest[rest[:, 1] < top - 0.12, 0])) * 0.62
    front_faces = (normals[:, 2] > 0.2) & (centres[:, 2] > body_center_z) & (np.abs(centres[:, 0]) < torso_half)
    uvs = np.full((len(rest), 2), 0.02, dtype=np.float32)  # back and sleeves sample the white margin
    front = np.unique(faces[front_faces])
    span = max(2.0 * torso_half, top - hem)
    centre_y = hem + 0.55 * (top - hem)
    uvs[front, 0] = 0.5 + rest[front, 0] / span
    uvs[front, 1] = 0.5 + (rest[front, 1] - centre_y) / span
    return np.clip(uvs, 0.0, 1.0), front_faces


def _write(output: Path, name: str, values: np.ndarray, dtype: str) -> str:
    np.ascontiguousarray(values, dtype=np.dtype(dtype).newbyteorder("<")).tofile(output / name)
    return name


def main() -> None:
    args = parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    avatar = BodyMeshAvatar.from_obj(
        args.body_mesh,
        AvatarSpec(chest_cm=64, waist_cm=56, shoulder_cm=30, height_cm=128, pose="a-pose"),
        source_x_axis="x",
        source_up_axis="z",
        source_front_axis="-y",
    )
    shirt_positions, shirt_velocities, shirt_duration, shirt_report = _rollout(args.shirt_run)
    skirt_positions, skirt_velocities, skirt_duration, skirt_report = _rollout(args.skirt_run)
    shirt_rest = trimesh.load(args.shirt_run / "intrinsic_rest_shell.obj", process=False)
    skirt_rest = trimesh.load(args.skirt_run / "intrinsic_rest_shell.obj", process=False)
    shirt_faces = np.asarray(shirt_rest.faces, dtype=np.int64)
    skirt_faces = np.asarray(skirt_rest.faces, dtype=np.int64)
    if shirt_positions.shape[1] != len(shirt_rest.vertices) or skirt_positions.shape[1] != len(skirt_rest.vertices):
        raise ValueError("rollout vertex counts do not match the runs' intrinsic rest shells")
    skirt_template, _ = build_body_conditioned_pleated_skirt_shell(avatar, infer_pleated_skirt_spec(args.skirt_image))
    if not np.array_equal(np.asarray(skirt_template.faces), skirt_faces) or skirt_template.uvs is None:
        raise ValueError("skirt run topology does not match the pleated-skirt family for this image")

    layer_gap = float(shirt_positions[:, :, 1].min() - skirt_positions[:, :, 1].max())
    if layer_gap < args.minimum_layer_gap_m:
        raise ValueError(
            f"T-shirt hem and skirt waistband overlap or touch (gap {layer_gap:.4f} m); "
            "simulate them together with run_unified_anny_cloth4d.py instead"
        )

    duration = min(shirt_duration, skirt_duration)
    frame_count = max(len(shirt_positions), 2)
    times = np.linspace(0.0, duration, frame_count)
    shirt_frames = _resample(shirt_positions, shirt_duration, times)
    shirt_vel = _resample(shirt_velocities, shirt_duration, times)
    skirt_frames = _resample(skirt_positions, skirt_duration, times)
    skirt_vel = _resample(skirt_velocities, skirt_duration, times)
    body = np.asarray(avatar.mesh_vertices, dtype=np.float32)
    body_frames = np.repeat(body[None], frame_count, axis=0)

    shirt_uvs, print_faces = _shirt_uvs_and_print_faces(np.asarray(shirt_rest.vertices), shirt_faces, float(avatar.base_z))
    ordered = np.concatenate((shirt_faces[print_faces], shirt_faces[~print_faces]), axis=0)
    printed_count = int(np.count_nonzero(print_faces) * 3)
    files = {
        "bodyPositions": _write(output, "body-positions.f32", body_frames, "f4"),
        "bodyIndices": _write(output, "body-indices.u32", avatar.mesh_faces, "u4"),
        "shirtPositions": _write(output, "shirt-positions.f32", shirt_frames, "f4"),
        "shirtVelocities": _write(output, "shirt-velocities.f32", shirt_vel, "f4"),
        "shirtIndices": _write(output, "shirt-indices.u32", ordered, "u4"),
        "shirtUvs": _write(output, "shirt-uvs.f32", shirt_uvs, "f4"),
        "skirtPositions": _write(output, "skirt-positions.f32", skirt_frames, "f4"),
        "skirtVelocities": _write(output, "skirt-velocities.f32", skirt_vel, "f4"),
        "skirtIndices": _write(output, "skirt-indices.u32", skirt_faces, "u4"),
        "skirtUvs": _write(output, "skirt-uvs.f32", np.asarray(skirt_template.uvs, dtype=np.float32), "f4"),
        "shirtTexture": "shirt-print.png",
        "skirtTexture": "skirt-plaid.png",
    }
    _print_atlas(args.shirt_texture).save(output / "shirt-print.png")
    Image.open(args.skirt_run / "image_conditioned_plaid.png").convert("RGB").save(output / "skirt-plaid.png")
    points = np.concatenate([body_frames.reshape(-1, 3), shirt_frames.reshape(-1, 3), skirt_frames.reshape(-1, 3)])
    manifest = {
        "format": "NeuralClothSim4D Three.js binary scene v1",
        "renderer": "Three.js WebGPURenderer",
        "threeVersion": "0.184.0",
        "units": "meters",
        "upAxis": "+Y",
        "frontAxis": "+Z",
        "frameCount": int(frame_count),
        "timesSeconds": times.astype(float).tolist(),
        "physicalDurationSeconds": float(duration),
        "displayLoopSeconds": 2.0,
        "bounds": {"minimum": points.min(axis=0).tolist(), "maximum": points.max(axis=0).tolist()},
        "files": files,
        "meshes": {
            "body": {"vertexCount": int(len(body)), "indexCount": int(np.asarray(avatar.mesh_faces).size)},
            "shirt": {
                "vertexCount": int(shirt_frames.shape[1]),
                "indexCount": int(ordered.size),
                "groups": [
                    {"start": 0, "count": printed_count, "materialIndex": 0},
                    {"start": printed_count, "count": int(ordered.size - printed_count), "materialIndex": 1},
                ],
            },
            "skirt": {"vertexCount": int(skirt_frames.shape[1]), "indexCount": int(skirt_faces.size)},
        },
        "physicsAudit": {
            "minimumBodyGapMeters": float(min(shirt_report["minimum_contact_gap_m"], skirt_report["minimum_contact_gap_m"])),
            "minimumInterGarmentGapMeters": layer_gap,
            "interGarmentContact": "none needed: garments simulated separately on the same static body with a "
            f"{100 * layer_gap:.1f} cm vertical gap between T-shirt hem and skirt waistband",
            "constitutiveCalibration": {
                "tshirt": shirt_report["constitutive_calibration_status"],
                "pleated_skirt": skirt_report["constitutive_calibration_status"],
            },
            "metricProjectionEnabled": False,
            "skirtViolatingEdges": 0,
            "skirtEdgeCount": 0,
            "lieMessageSubsteps": int(shirt_report["substeps"]),
            "sources": {"tshirt": str(args.shirt_run), "pleated_skirt": str(args.skirt_run)},
            "inputs": {"shirt_texture": str(args.shirt_texture), "skirt_image": str(args.skirt_image)},
        },
    }
    (output / "scene.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"scene": output.name, "frames": frame_count, "duration_s": duration, "layer_gap_m": layer_gap}))


if __name__ == "__main__":
    main()
