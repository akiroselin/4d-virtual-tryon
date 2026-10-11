from __future__ import annotations

import argparse
from pathlib import Path
import os

import numpy as np
import roma
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export an Apache-licensed Anny body mesh to OBJ.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--height-cm", type=float, default=178.0)
    parser.add_argument(
        "--topology",
        default="default",
        choices=["default", "makehuman", "notoes", "notoes_collapse3pc", "notoes_collapse5pc", "notoes_collapse10pc", "soma"],
        help="Avoid 'smplx' topology for commercial use; Anny marks that topology non-commercial.",
    )
    parser.add_argument("--rig", default="default")
    parser.add_argument("--local-changes", default="default", choices=["none", "default", "all"])
    parser.add_argument("--gender", type=float, default=0.0, help="Anny gender scalar: 0 male, 1 female.")
    parser.add_argument(
        "--age",
        type=float,
        default=1.0 / 3.0,
        help="Anny age scalar: -1/3 newborn, 0 baby, 1/3 child, 2/3 young adult, 1 old.",
    )
    parser.add_argument("--muscle", type=float, default=0.5, help="Anny muscle scalar; 0.5 is neutral.")
    parser.add_argument("--weight", type=float, default=0.5, help="Anny weight scalar; 0.5 is neutral.")
    parser.add_argument(
        "--hand-pose",
        choices=["relaxed", "open"],
        default="relaxed",
        help="Use a gentle finger curl for try-on renders, or preserve the open rest hand.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.environ.setdefault("ANNY_CACHE_DIR", str(Path("outputs/aigc_fashion_prototype/work/anny_cache").resolve()))
    os.environ.setdefault("WARP_CACHE_PATH", str(Path("outputs/aigc_fashion_prototype/work/warp_cache").resolve()))

    import anny

    if args.topology == "smplx":
        raise ValueError("Do not use Anny 'smplx' topology for commercial runs; use default/soma/notoes topology.")

    model = anny.create_fullbody_model(
        rig=args.rig,
        topology=args.topology,
        local_changes=args.local_changes,
        remove_unattached_vertices=True,
        bone_orientation="blender-rootidentity",
    )
    phenotype_kwargs = _supported_phenotypes(
        model,
        {
            "gender": args.gender,
            "age": args.age,
            "muscle": args.muscle,
            "weight": args.weight,
        },
    )
    pose_parameters = _relaxed_hand_pose(model) if args.hand_pose == "relaxed" else None
    with torch.no_grad():
        output = model(pose_parameters=pose_parameters, phenotype_kwargs=phenotype_kwargs)

    vertices = output["vertices"][0].detach().cpu().numpy().astype(np.float32)
    faces = model.faces.detach().cpu().numpy().astype(np.int32)
    vertices = _scale_to_height(vertices, args.height_cm)
    write_obj(args.output, vertices, faces)
    print(f"Exported Anny body mesh: {args.output}")
    print(f"topology={args.topology} hand_pose={args.hand_pose} vertices={vertices.shape[0]} faces={faces.shape[0]}")
    print("license=Anny Apache-2.0, MakeHuman/MPFB2 assets CC0; no SMPL assets used")
    return 0


def _supported_phenotypes(model: object, values: dict[str, float]) -> dict[str, torch.Tensor]:
    names = set()
    for attr in ("phenotype_labels", "macrodetail_labels", "phenotype_names"):
        candidate = getattr(model, attr, None)
        if candidate is not None:
            names.update(str(item) for item in candidate)
    return {
        name: torch.tensor([value], dtype=torch.float32)
        for name, value in values.items()
        if name in names
    }


def _relaxed_hand_pose(model: object) -> dict[str, torch.Tensor]:
    pose: dict[str, torch.Tensor] = {}
    bone_labels = set(str(label) for label in model.bone_labels)
    for side in ("L", "R"):
        for finger in range(2, 6):
            for joint, scale in ((1, 0.55), (2, 1.0), (3, 1.15)):
                label = f"finger{finger}-{joint}.{side}"
                if label not in bone_labels:
                    continue
                rotation = roma.euler_to_rotmat("x", [18.0 * scale], degrees=True, dtype=torch.float32)
                pose[label] = roma.Rigid(rotation, translation=None).to_homogeneous()[None]
    return pose


def _scale_to_height(vertices: np.ndarray, height_cm: float) -> np.ndarray:
    # Anny uses z as vertical; our simulator remaps axes at import time.
    height = float(vertices[:, 2].max() - vertices[:, 2].min())
    if height > 1e-6:
        vertices = vertices * ((height_cm / 100.0) / height)
    return vertices


def write_obj(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Exported from Apache-licensed Anny body model; no SMPL assets\n"]
    for x, y, z in vertices:
        lines.append(f"v {x:.8f} {y:.8f} {z:.8f}\n")
    for face in faces:
        valid = [int(idx) + 1 for idx in face if int(idx) >= 0]
        if len(valid) >= 3:
            lines.append("f " + " ".join(str(idx) for idx in valid) + "\n")
    path.write_text("".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
