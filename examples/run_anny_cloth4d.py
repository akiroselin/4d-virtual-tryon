from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aigc_fashion.anny_cloth4d_simulation import (  # noqa: E402
    AnnyCloth4DConfig,
    run_anny_cloth4d_simulation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optimize the geometric NeuralClothSim 4D field on an Anny T-shirt scene."
    )
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "anny_e3_anchor_cloth4d")
    parser.add_argument("--body-mesh", type=Path, default=ROOT / "assets" / "body_meshes" / "anny_child_neutral.obj")
    parser.add_argument(
        "--body-source-up-axis",
        default="z",
        help="Anny exports are Blender Z-up.",
    )
    parser.add_argument(
        "--body-source-front-axis",
        default="-y",
        help="Anny faces Blender -Y; the sign must not be inferred from extents.",
    )
    parser.add_argument(
        "--pattern-spec",
        type=Path,
        default=ROOT / "outputs" / "anny_child_vto_bounded_prefit_final" / "image_conditioned_tshirt_spec.json",
    )
    parser.add_argument(
        "--warm-start-obj",
        type=Path,
        default=None,
        help="Optional compatible garment OBJ; omitted to use avatar arrangement points.",
    )
    parser.add_argument(
        "--simulation-topology",
        choices=("implicit_shell", "seam_atlas"),
        default="implicit_shell",
        help="Continuous seam-free simulation shell, or the retained panel/seam baseline.",
    )
    parser.add_argument(
        "--placement-mode",
        choices=("arrangement_points", "legacy_predrape"),
        default="arrangement_points",
        help="Panel placement path; legacy_predrape is retained for comparison.",
    )
    parser.add_argument(
        "--texture-image",
        type=Path,
        required=True,
    )
    parser.add_argument("--genesis-repo", type=Path, default=ROOT.parents[1] / "work" / "genesis-world")
    parser.add_argument("--teacher-rollout", type=Path)
    parser.add_argument(
        "--warm-start-checkpoint",
        type=Path,
        help="Compatible pretrained 4D field checkpoint for variational PINN fine-tuning.",
    )
    parser.add_argument("--training-steps", type=int, default=180)
    parser.add_argument("--minimum-training-steps", type=int, default=80)
    parser.add_argument("--shell-physics-interval", type=int, default=6)
    parser.add_argument("--shell-physics-warmup-steps", type=int, default=0)
    parser.add_argument("--implicit-shell-target-faces", type=int, default=1400)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = replace(
        AnnyCloth4DConfig(),
        training_steps=args.training_steps,
        minimum_training_steps=min(args.minimum_training_steps, args.training_steps),
        shell_physics_interval=args.shell_physics_interval,
        shell_physics_warmup_steps=min(
            args.shell_physics_warmup_steps,
            max(args.training_steps - 1, 0),
        ),
        implicit_shell_target_faces=args.implicit_shell_target_faces,
    )
    if args.warm_start_checkpoint is not None:
        checkpoint = torch.load(
            args.warm_start_checkpoint,
            map_location="cpu",
            weights_only=True,
        )
        source = checkpoint["config"]
        config = replace(
            config,
            intrinsic_channels=source["intrinsic_channels"],
            scene_feature_dim=source["scene_feature_dim"],
            material_latent_dim=source["material_latent_dim"],
            hidden_dim=source["hidden_dim"],
        )
    summary = run_anny_cloth4d_simulation(
        out_dir=args.out_dir,
        body_mesh=args.body_mesh,
        pattern_spec=args.pattern_spec,
        texture_image=args.texture_image,
        warm_start_obj=args.warm_start_obj,
        genesis_repo=args.genesis_repo,
        simulation_topology=args.simulation_topology,
        placement_mode=args.placement_mode,
        body_source_up_axis=args.body_source_up_axis,
        body_source_front_axis=args.body_source_front_axis,
        teacher_rollout=args.teacher_rollout,
        warm_start_checkpoint=args.warm_start_checkpoint,
        config=config,
    )
    print(f"simulation: {summary['status']}")
    print(f"animation: {summary['artifacts']['gif']}")
    print(f"comparison: {summary['artifacts']['comparison_gif']}")
    print(f"mean field error: {summary['metrics']['constrained_field_mean_error_m']:.6f} m")


if __name__ == "__main__":
    main()
