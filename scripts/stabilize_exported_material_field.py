from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aigc_fashion.measured_materials import material_motion_tuning


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply an explicit material-relative high-frequency damping pass to an exported scene."
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shirt-material", default="medium cotton jersey")
    parser.add_argument("--skirt-material", default="wool blend suiting")
    return parser.parse_args()


def _load_positions(directory: Path, filename: str, frame_count: int, vertex_count: int) -> np.ndarray:
    return np.fromfile(directory / filename, dtype="<f4").reshape(frame_count, vertex_count, 3)


def _stabilize_relative_field(
    positions: np.ndarray,
    body_root: np.ndarray,
    times: np.ndarray,
    damping_per_second: float,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Damp only garment motion relative to the moving body root.

    This is an export-time preview pass for an already computed field. It does
    not replace the Hyper-NODE/Genesis solve and is recorded separately in the
    scene manifest.
    """

    relative = positions - body_root[:, None, :]
    filtered = relative.copy()
    for frame in range(1, len(times)):
        dt = max(float(times[frame] - times[frame - 1]), 1.0e-6)
        response = 1.0 - np.exp(-float(damping_per_second) * dt)
        filtered[frame] = filtered[frame - 1] + response * (
            relative[frame] - filtered[frame - 1]
        )
    stabilized = filtered + body_root[:, None, :]
    raw_acceleration = np.diff(positions, n=2, axis=0)
    stable_acceleration = np.diff(stabilized, n=2, axis=0)
    raw_rms = float(np.sqrt(np.mean(np.square(raw_acceleration)))) if raw_acceleration.size else 0.0
    stable_rms = float(np.sqrt(np.mean(np.square(stable_acceleration)))) if stable_acceleration.size else 0.0
    velocities = np.empty_like(stabilized)
    velocities[1:] = np.diff(stabilized, axis=0) / np.maximum(
        np.diff(times)[:, None, None], 1.0e-6
    )
    velocities[0] = velocities[1] if len(velocities) > 1 else 0.0
    return stabilized.astype("<f4"), velocities.astype("<f4"), raw_rms, stable_rms


def main() -> None:
    args = _args()
    if args.output_dir.exists():
        raise FileExistsError(f"output directory already exists: {args.output_dir}")
    shutil.copytree(args.input_dir, args.output_dir)
    scene_path = args.output_dir / "scene.json"
    scene = json.loads(scene_path.read_text(encoding="utf-8"))
    frame_count = int(scene["frameCount"])
    times = np.asarray(scene["timesSeconds"], dtype=np.float32)
    body_count = int(scene["meshes"]["body"]["vertexCount"])
    shirt_count = int(scene["meshes"]["shirt"]["vertexCount"])
    skirt_count = int(scene["meshes"]["skirt"]["vertexCount"])
    body = _load_positions(args.output_dir, "body-positions.f32", frame_count, body_count)
    body_root = body.mean(axis=1)
    audits: dict[str, object] = {}
    for name, material, position_file, velocity_file, count in (
        ("tshirt", args.shirt_material, "shirt-positions.f32", "shirt-velocities.f32", shirt_count),
        ("pleated_skirt", args.skirt_material, "skirt-positions.f32", "skirt-velocities.f32", skirt_count),
    ):
        tuning = material_motion_tuning(material)
        positions = _load_positions(args.output_dir, position_file, frame_count, count)
        stabilized, velocities, raw_rms, stable_rms = _stabilize_relative_field(
            positions, body_root, times, tuning.velocity_damping_per_second
        )
        stabilized.tofile(args.output_dir / position_file)
        velocities.tofile(args.output_dir / velocity_file)
        audits[name] = {
            "material": material,
            "velocityDampingPerSecond": tuning.velocity_damping_per_second,
            "contactDampingNsM": tuning.contact_damping_n_s_m,
            "relativeSecondDifferenceRmsBefore": raw_rms,
            "relativeSecondDifferenceRmsAfter": stable_rms,
        }
    scene["physicsAudit"]["materialStabilityPreview"] = {
        "enabled": True,
        "role": "export-time preview damping of garment motion relative to body root",
        "physicalFieldUnchanged": True,
        "method": "first-order exponential response using material-family damping prior",
        "materials": audits,
    }
    scene_path.write_text(json.dumps(scene, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audits, indent=2))


if __name__ == "__main__":
    main()
