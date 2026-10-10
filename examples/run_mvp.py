from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aigc_fashion import AvatarSpec, TshirtSizeSpec, run_pipeline  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the AIGC fashion MVP pipeline.")
    parser.add_argument(
        "--prompt",
        default="relaxed minimal streetwear short sleeve crew neck medium cotton jersey t-shirt",
    )
    parser.add_argument("--intent-json", default=None)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "mvp_run")
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--resolution", type=int, default=23)
    parser.add_argument("--flat-pattern", action="store_true", help="Simulate the flat pattern instead of dressing it on the avatar first.")
    parser.add_argument("--avatar-chest-cm", type=float, default=96.0)
    parser.add_argument("--avatar-waist-cm", type=float, default=82.0)
    parser.add_argument("--avatar-shoulder-cm", type=float, default=46.0)
    parser.add_argument("--avatar-height-cm", type=float, default=178.0)
    parser.add_argument("--garment-chest-cm", type=float, default=None, help="T-shirt full chest circumference in cm.")
    parser.add_argument("--garment-length-cm", type=float, default=None)
    parser.add_argument("--garment-shoulder-cm", type=float, default=None)
    parser.add_argument("--sleeve-length-cm", type=float, default=None)
    parser.add_argument("--sleeve-opening-cm", type=float, default=36.0)
    parser.add_argument("--neck-width-cm", type=float, default=18.0)
    parser.add_argument("--texture-image", type=Path, default=None)
    parser.add_argument("--fit-from-image", action="store_true", help="Infer T-shirt panel proportions from --texture-image.")
    parser.add_argument("--learned-pattern", action="store_true", help="Use SewFormer learned sewing-pattern reconstruction when checkpoint is available.")
    parser.add_argument("--learned-pattern-spec", type=Path, default=None, help="Use an existing SewFormer/GarmentCode sewing-pattern JSON directly.")
    parser.add_argument("--sewformer-repo", type=Path, default=ROOT.parents[1] / "work" / "sewformer")
    parser.add_argument("--sewformer-checkpoint", type=Path, default=None)
    parser.add_argument("--body-mesh", type=Path, default=None, help="Commercial-safe OBJ body mesh, e.g. exported from Anny or a scan.")
    parser.add_argument("--body-mesh-source-x-axis", default="x", help="Source mesh axis for left/right, e.g. x or -x.")
    parser.add_argument("--body-mesh-source-up-axis", default="auto", help="Source mesh axis for vertical up, e.g. auto, y, or z.")
    parser.add_argument("--body-mesh-source-front-axis", default="auto", help="Source mesh axis for front depth, e.g. auto, z, or -y.")
    parser.add_argument("--body-mesh-scale", type=float, default=None, help="Optional manual scale before alignment; default auto-scales to avatar height.")
    parser.add_argument("--smpl-mesh", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--smpl-source-x-axis", default="x", help=argparse.SUPPRESS)
    parser.add_argument("--smpl-source-up-axis", default="y", help=argparse.SUPPRESS)
    parser.add_argument("--smpl-source-front-axis", default="z", help=argparse.SUPPRESS)
    parser.add_argument("--smpl-scale", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--strict-preflight",
        action="store_true",
        help="Block cloth physics when category, topology, size, initial strain, seams, or collision checks fail.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    avatar = AvatarSpec(
        chest_cm=args.avatar_chest_cm,
        waist_cm=args.avatar_waist_cm,
        shoulder_cm=args.avatar_shoulder_cm,
        height_cm=args.avatar_height_cm,
    )
    tshirt_size = None
    if args.garment_chest_cm or args.garment_length_cm or args.garment_shoulder_cm or args.sleeve_length_cm:
        tshirt_size = TshirtSizeSpec(
            chest_circumference_cm=args.garment_chest_cm or 104.0,
            body_length_cm=args.garment_length_cm or 70.0,
            shoulder_width_cm=args.garment_shoulder_cm or 46.0,
            sleeve_length_cm=args.sleeve_length_cm or 20.0,
            sleeve_opening_cm=args.sleeve_opening_cm,
            neck_width_cm=args.neck_width_cm,
        )
    summary = run_pipeline(
        prompt=args.prompt,
        intent_json=args.intent_json,
        out_dir=args.out_dir,
        avatar=avatar,
        steps=args.steps,
        resolution=args.resolution,
        dress_on_avatar=not args.flat_pattern,
        tshirt_size=tshirt_size,
        texture_image=args.texture_image,
        fit_from_image=args.fit_from_image,
        learned_pattern=args.learned_pattern,
        learned_pattern_spec=args.learned_pattern_spec,
        sewformer_repo=args.sewformer_repo,
        sewformer_checkpoint=args.sewformer_checkpoint,
        body_mesh=args.body_mesh,
        body_mesh_source_x_axis=args.body_mesh_source_x_axis,
        body_mesh_source_up_axis=args.body_mesh_source_up_axis,
        body_mesh_source_front_axis=args.body_mesh_source_front_axis,
        body_mesh_scale=args.body_mesh_scale,
        smpl_mesh=args.smpl_mesh,
        smpl_source_x_axis=args.smpl_source_x_axis,
        smpl_source_up_axis=args.smpl_source_up_axis,
        smpl_source_front_axis=args.smpl_source_front_axis,
        smpl_scale=args.smpl_scale,
        strict_preflight=args.strict_preflight,
    )
    print(json.dumps(summary, indent=2))
