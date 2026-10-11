from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aigc_fashion import AvatarSpec, BodyMeshAvatar  # noqa: E402
from aigc_fashion.implicit_shell import build_body_sdf_tshirt_shell  # noqa: E402
from aigc_fashion.learned_panel_mesh import garment_mesh_from_sewformer_spec  # noqa: E402
from aigc_fashion.nyx_render import (  # noqa: E402
    export_nyx_scene_assets,
    probe_nyx_backend,
    render_nyx_scene,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export and render the Anny continuous T-shirt shell with Genesis Nyx."
    )
    parser.add_argument(
        "--body-mesh",
        type=Path,
        default=ROOT / "assets" / "body_meshes" / "anny_child_neutral.obj",
    )
    parser.add_argument(
        "--pattern-spec",
        type=Path,
        default=ROOT
        / "outputs"
        / "anny_child_vto_bounded_prefit_final"
        / "image_conditioned_tshirt_spec.json",
    )
    parser.add_argument("--body-source-up-axis", default="z")
    parser.add_argument("--body-source-front-axis", default="-y")
    parser.add_argument(
        "--texture-image",
        type=Path,
        required=True,
    )
    parser.add_argument("--rollout", type=Path)
    parser.add_argument("--frame-index", type=int, default=-1)
    parser.add_argument("--target-faces", type=int, default=700)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs" / "anny_implicit_shell_nyx",
    )
    parser.add_argument("--spp", type=int, default=128)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Export PBR GLBs and manifest without loading the CUDA Nyx backend.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    avatar = BodyMeshAvatar.from_obj(
        args.body_mesh,
        AvatarSpec(
            chest_cm=64.0,
            waist_cm=56.0,
            shoulder_cm=30.0,
            height_cm=128.0,
            pose="a-pose",
        ),
        source_x_axis="x",
        source_up_axis=args.body_source_up_axis,
        source_front_axis=args.body_source_front_axis,
    )
    atlas = garment_mesh_from_sewformer_spec(args.pattern_spec, resolution=23)
    garment, report = build_body_sdf_tshirt_shell(
        avatar,
        atlas,
        target_faces=args.target_faces,
    )
    vertices = garment.vertices
    if args.rollout is not None:
        rollout = np.asarray(np.load(args.rollout), dtype=np.float32)
        if rollout.ndim != 3 or rollout.shape[1:] != garment.vertices.shape:
            raise ValueError(
                f"rollout shape {rollout.shape} does not match shell {(len(garment.vertices), 3)}"
            )
        vertices = rollout[args.frame_index]

    assets = export_nyx_scene_assets(
        args.out_dir / "assets",
        avatar,
        garment,
        vertices,
        args.texture_image,
    )
    payload = {
        "shell": report.to_dict(),
        "assets": assets,
        "nyx_backend": probe_nyx_backend().to_dict(),
    }
    summary_path = args.out_dir / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Nyx assets: {assets['manifest']}")

    if args.prepare_only:
        return
    output = render_nyx_scene(
        assets,
        args.out_dir / "anny_tshirt_nyx.png",
        resolution=(args.resolution, args.resolution),
        spp=args.spp,
    )
    print(f"Nyx render: {output}")


if __name__ == "__main__":
    main()
