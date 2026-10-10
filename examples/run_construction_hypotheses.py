from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aigc_fashion.construction_hypotheses import (  # noqa: E402
    generate_physical_structure_hypotheses,
)
from aigc_fashion.data import AvatarSpec, GarmentIntent  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate body-graded garment construction hypotheses")
    parser.add_argument("--garment-type", default="t-shirt")
    parser.add_argument("--silhouette", choices=("slim", "relaxed", "oversized"), default="relaxed")
    parser.add_argument("--fabric", default="medium cotton jersey")
    parser.add_argument("--height-cm", type=float, required=True)
    parser.add_argument("--chest-cm", type=float, required=True)
    parser.add_argument("--waist-cm", type=float, required=True)
    parser.add_argument("--shoulder-cm", type=float, required=True)
    parser.add_argument("--hip-cm", type=float)
    parser.add_argument("--upper-arm-cm", type=float)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    avatar = AvatarSpec(
        height_cm=args.height_cm,
        chest_cm=args.chest_cm,
        waist_cm=args.waist_cm,
        shoulder_cm=args.shoulder_cm,
    )
    intent = GarmentIntent(
        prompt=f"{args.silhouette} {args.fabric} {args.garment_type}",
        garment_type=args.garment_type,
        silhouette=args.silhouette,
        fabric=args.fabric,
        body_fit="close" if args.silhouette == "slim" else "loose",
    )
    overrides = {}
    if args.hip_cm is not None:
        overrides["hip_circumference"] = args.hip_cm
    if args.upper_arm_cm is not None:
        overrides["upper_arm_circumference"] = args.upper_arm_cm
    hypotheses = generate_physical_structure_hypotheses(
        intent,
        avatar,
        measurement_overrides_cm=overrides,
        top_k=args.top_k,
    )
    payload = {
        "method": "body measurements + construction experience database + pattern ontology",
        "cortexode_role": "fixed-topology current-embedding refinement only",
        "hypotheses": [item.to_dict() for item in hypotheses],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(args.out.resolve())


if __name__ == "__main__":
    main()
