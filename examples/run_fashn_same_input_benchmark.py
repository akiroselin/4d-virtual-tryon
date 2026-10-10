from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

from PIL import Image, ImageDraw, ImageOps


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run FASHN VTON v1.5 on the exact Anny/garment benchmark inputs."
    )
    parser.add_argument("--fashn-repo", type=Path, default=ROOT / "work/fashn-vton-1.5")
    parser.add_argument("--weights-dir", type=Path, required=True)
    parser.add_argument(
        "--input-dir", type=Path, default=ROOT / "outputs/vto_benchmark_same_model"
    )
    parser.add_argument(
        "--physics-result",
        type=Path,
        default=ROOT
        / "outputs/anny_tshirt_and_pleated_skirt/anny_tshirt_pleated_skirt_final.png",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=ROOT / "outputs/vto_benchmark_same_model"
    )
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=177)
    parser.add_argument("--device", choices=("cuda", "cpu"), default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = args.weights_dir / "model.safetensors"
    pose_paths = (
        args.weights_dir / "dwpose/yolox_l.onnx",
        args.weights_dir / "dwpose/dw-ll_ucoco_384.onnx",
    )
    missing = [path for path in (model_path, *pose_paths) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "FASHN VTON weights are incomplete: " + ", ".join(str(path) for path in missing)
        )
    if model_path.stat().st_size < 1_000_000_000:
        raise RuntimeError("model.safetensors is an LFS pointer, not the released model weight")

    sys.path.insert(0, str(args.fashn_repo / "src"))
    from fashn_vton import TryOnPipeline

    inputs = args.input_dir
    person_with_skirt = Image.open(inputs / "anny_with_skirt_person.png").convert("RGB")
    person_with_tshirt = Image.open(inputs / "anny_with_tshirt_person.png").convert("RGB")
    tshirt = Image.open(inputs / "tshirt.png").convert(
        "RGB"
    )
    skirt = Image.open(inputs / "pleated_skirt_garment.png").convert("RGB")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    pipeline = TryOnPipeline(weights_dir=args.weights_dir, device=args.device)
    top_result = pipeline(
        person_image=person_with_skirt,
        garment_image=tshirt,
        category="tops",
        garment_photo_type="flat-lay",
        num_samples=1,
        num_timesteps=args.steps,
        seed=args.seed,
        segmentation_free=True,
    ).images[0]
    bottom_result = pipeline(
        person_image=person_with_tshirt,
        garment_image=skirt,
        category="bottoms",
        garment_photo_type="flat-lay",
        num_samples=1,
        num_timesteps=args.steps,
        seed=args.seed,
        segmentation_free=True,
    ).images[0]
    top_path = args.out_dir / "fashn_top_on_skirt.png"
    bottom_path = args.out_dir / "fashn_skirt_on_tshirt.png"
    top_result.save(top_path)
    bottom_result.save(bottom_path)

    comparison_path = args.out_dir / "same_input_vto_comparison.png"
    _comparison_board(
        args.physics_result,
        top_path,
        bottom_path,
        comparison_path,
    )
    manifest = {
        "status": "completed",
        "baseline": "FASHN VTON v1.5 official inference code and weights",
        "seed": args.seed,
        "steps": args.steps,
        "runtime_seconds": time.perf_counter() - started,
        "routes": {
            "tops": "same Anny wearing the physical skirt + original T-shirt product image",
            "bottoms": "same Anny wearing the physical T-shirt + original skirt product image",
        },
        "artifacts": {
            "physics": str(args.physics_result),
            "fashn_top_on_skirt": str(top_path),
            "fashn_skirt_on_tshirt": str(bottom_path),
            "comparison": str(comparison_path),
        },
    }
    (args.out_dir / "fashn_benchmark_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


def _comparison_board(
    physics_path: Path,
    top_path: Path,
    bottom_path: Path,
    output_path: Path,
) -> None:
    items = (
        ("NeuralClothSim4D", physics_path),
        ("FASHN: top on skirt", top_path),
        ("FASHN: skirt on top", bottom_path),
    )
    width, height = 480, 720
    board = Image.new("RGB", (3 * width, height + 58), "white")
    draw = ImageDraw.Draw(board)
    for index, (label, path) in enumerate(items):
        image = Image.open(path).convert("RGB")
        image = ImageOps.contain(image, (width, height), Image.Resampling.LANCZOS)
        x = index * width + (width - image.width) // 2
        y = 48 + (height - image.height) // 2
        board.paste(image, (x, y))
        draw.text((index * width + 16, 16), label, fill=(24, 30, 38))
    board.save(output_path)


if __name__ == "__main__":
    main()
