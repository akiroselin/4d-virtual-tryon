from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import trimesh


_BARYCENTRICS = np.asarray(
    (
        (2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0),
        (1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0),
        (1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0),
    ),
    dtype=np.float32,
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay the body-SDF face contact constraint on an exported cloth field."
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--clearance-m", type=float, default=0.0035)
    parser.add_argument("--passes", type=int, default=4)
    return parser.parse_args()


def _load(directory: Path, name: str, shape: tuple[int, ...], dtype: str) -> np.ndarray:
    return np.fromfile(directory / name, dtype=dtype).reshape(shape)


def _project_faces(
    positions: np.ndarray,
    faces: np.ndarray,
    query: trimesh.proximity.ProximityQuery,
    body: trimesh.Trimesh,
    clearance: float,
) -> tuple[np.ndarray, float, int]:
    result = positions.copy()
    triangles = result[faces]
    samples = np.einsum("qk,fkj->fqj", _BARYCENTRICS, triangles).reshape(-1, 3)
    closest, distances, triangle_ids = query.on_surface(samples)
    signed = query.signed_distance(samples)
    active = (clearance + signed > 0.0) & (distances < 0.035)
    if not np.any(active):
        return result, 0.0, 0

    normals = np.asarray(body.face_normals[triangle_ids[active]], dtype=np.float32)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1.0e-8)
    correction = closest[active] + clearance * normals - samples[active]
    length = np.linalg.norm(correction, axis=1, keepdims=True)
    correction *= np.minimum(1.0, 0.008 / np.maximum(length, 1.0e-9))

    updates = np.zeros_like(result)
    weights = np.zeros(len(result), dtype=np.float32)
    active_ids = np.flatnonzero(active)
    for value, sample_id in zip(correction, active_ids):
        face_id, bary_id = divmod(int(sample_id), len(_BARYCENTRICS))
        for local_id, weight in enumerate(_BARYCENTRICS[bary_id]):
            vertex_id = int(faces[face_id, local_id])
            updates[vertex_id] += value * float(weight)
            weights[vertex_id] += float(weight)
    movable = weights > 0.0
    result[movable] += updates[movable] / weights[movable, None]
    return result, float(np.maximum(clearance + signed[active], 0.0).max()), int(active.sum())


def main() -> None:
    args = _args()
    if args.clearance_m <= 0.0 or args.passes <= 0:
        raise ValueError("clearance and passes must be positive")
    if args.output_dir.exists():
        raise FileExistsError(f"output directory already exists: {args.output_dir}")
    shutil.copytree(args.input_dir, args.output_dir)
    scene_path = args.output_dir / "scene.json"
    scene = json.loads(scene_path.read_text(encoding="utf-8"))
    frame_count = int(scene["frameCount"])
    body_count = int(scene["meshes"]["body"]["vertexCount"])
    body_faces = _load(
        args.output_dir,
        "body-indices.u32",
        (-1, 3),
        "<u4",
    )
    body_positions = _load(
        args.output_dir,
        "body-positions.f32",
        (frame_count, body_count, 3),
        "<f4",
    )
    repairs: dict[str, object] = {}
    for name in ("shirt", "skirt"):
        counts = scene["meshes"][name]
        vertex_count = int(counts["vertexCount"])
        faces = _load(
            args.output_dir,
            f"{name}-indices.u32",
            (-1, 3),
            "<u4",
        )
        positions = _load(
            args.output_dir,
            f"{name}-positions.f32",
            (frame_count, vertex_count, 3),
            "<f4",
        )
        max_depth = 0.0
        active_samples = 0
        for frame in range(frame_count):
            body_mesh = trimesh.Trimesh(
                vertices=body_positions[frame],
                faces=body_faces,
                process=False,
            )
            body_mesh.fix_normals()
            body_query = trimesh.proximity.ProximityQuery(body_mesh)
            for _ in range(args.passes):
                positions[frame], depth, active = _project_faces(
                    positions[frame], faces, body_query, body_mesh, args.clearance_m
                )
                max_depth = max(max_depth, depth)
                active_samples += active
        positions.astype("<f4").tofile(args.output_dir / f"{name}-positions.f32")
        repairs[name] = {
            "maxInputPenetrationMeters": max_depth,
            "activeFaceSamples": active_samples,
        }

    scene["physicsAudit"]["bodyContactReplay"] = {
        "enabled": True,
        "method": "watertight body SDF + barycentric face contact projection",
        "clearanceMeters": args.clearance_m,
        "passesPerFrame": args.passes,
        "artificialMaterialStabilization": False,
        "velocityFieldUnchanged": True,
        "garments": repairs,
    }
    scene["physicsAudit"].pop("materialStabilityPreview", None)
    scene_path.write_text(json.dumps(scene, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(repairs, indent=2))


if __name__ == "__main__":
    main()
