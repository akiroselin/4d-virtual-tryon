from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a licensed SMPL/SMPL-X model folder to an OBJ body mesh for the T-shirt simulator."
    )
    parser.add_argument("--model-folder", type=Path, required=True, help="Official SMPL/SMPL-X model directory.")
    parser.add_argument("--output", type=Path, required=True, help="Output OBJ path.")
    parser.add_argument("--model-type", choices=["smpl", "smplh", "smplx"], default="smplx")
    parser.add_argument("--gender", choices=["neutral", "male", "female"], default="neutral")
    parser.add_argument("--height-cm", type=float, default=178.0, help="Scale exported mesh to this height.")
    parser.add_argument("--num-betas", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        import smplx
    except ImportError:
        print(
            "Missing Python package 'smplx'. Install it with: python3 -m pip install smplx",
            file=sys.stderr,
        )
        return 2

    if not args.model_folder.exists():
        print(f"Model folder does not exist: {args.model_folder}", file=sys.stderr)
        return 2

    try:
        model = smplx.create(
            str(args.model_folder),
            model_type=args.model_type,
            gender=args.gender,
            num_betas=args.num_betas,
            use_pca=False,
            batch_size=1,
        )
    except (AssertionError, FileNotFoundError) as exc:
        expected = args.model_folder / args.model_type
        print(f"Could not load official {args.model_type.upper()} assets: {exc}", file=sys.stderr)
        print(f"Expected model files under: {expected}", file=sys.stderr)
        print("Download the official licensed model from https://smpl-x.is.tue.mpg.de/ or https://smpl.is.tue.mpg.de/", file=sys.stderr)
        return 2
    betas = torch.zeros((1, args.num_betas), dtype=torch.float32)
    body_pose = torch.zeros((1, getattr(model, "NUM_BODY_JOINTS", 21) * 3), dtype=torch.float32)
    global_orient = torch.zeros((1, 3), dtype=torch.float32)
    transl = torch.zeros((1, 3), dtype=torch.float32)

    with torch.no_grad():
        output = model(betas=betas, body_pose=body_pose, global_orient=global_orient, transl=transl)

    vertices = output.vertices[0].detach().cpu().numpy().astype(np.float32)
    height = float(vertices[:, 1].max() - vertices[:, 1].min())
    if height > 1e-6:
        vertices *= (args.height_cm / 100.0) / height
    faces = np.asarray(model.faces, dtype=np.int32)
    write_obj(args.output, vertices, faces)
    print(f"Exported {args.model_type.upper()} {args.gender} mesh: {args.output}")
    print(f"vertices={vertices.shape[0]} faces={faces.shape[0]}")
    return 0


def write_obj(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Exported from licensed SMPL/SMPL-X assets\n"]
    for x, y, z in vertices:
        lines.append(f"v {x:.8f} {y:.8f} {z:.8f}\n")
    for a, b, c in faces:
        lines.append(f"f {int(a) + 1} {int(b) + 1} {int(c) + 1}\n")
    path.write_text("".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
