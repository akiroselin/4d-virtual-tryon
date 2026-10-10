"""Run SAM 3D Body on one image and save the MHR parameters of each person.

This script runs inside the SAM 3D Body environment (its own Python with
detectron2 etc.), not the main project environment. ``aigc_fashion.body_reconstruction``
calls it as a subprocess and consumes the JSON it writes:

    python scripts/sam3d_body_infer.py --sam3d-repo /path/to/sam-3d-body \\
        --checkpoint /path/to/sam-3d-body-dinov3/model.ckpt \\
        --mhr-model /path/to/sam-3d-body-dinov3/assets/mhr_model.pt \\
        --image person.jpg --output people.json

SAM 3D Body code and weights are under the SAM License
(https://github.com/facebookresearch/sam-3d-body/blob/main/LICENSE); the
checkpoints are gated on Hugging Face and must be requested by the user.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

KEYS = (
    "bbox",
    "focal_length",
    "pred_cam_t",
    "global_rot",
    "body_pose_params",
    "hand_pose_params",
    "scale_params",
    "shape_params",
    "expr_params",
    "mhr_model_params",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sam3d-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--mhr-model", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--detector", default="vitdet", help="Human detector name ('' to use the whole image)")
    parser.add_argument("--bbox-threshold", type=float, default=0.5)
    args = parser.parse_args()

    sys.path.insert(0, str(args.sam3d_repo.resolve()))
    import torch
    from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model, model_cfg = load_sam_3d_body(str(args.checkpoint), device=device, mhr_path=str(args.mhr_model))
    detector = None
    if args.detector:
        from tools.build_detector import HumanDetector

        detector = HumanDetector(name=args.detector, device=device, path="")
    estimator = SAM3DBodyEstimator(sam_3d_body_model=model, model_cfg=model_cfg, human_detector=detector)
    people = estimator.process_one_image(str(args.image), bbox_thr=args.bbox_threshold)

    records = []
    for person in people:
        record = {key: np.asarray(person[key]).tolist() for key in KEYS if key in person}
        records.append(record)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "model": "SAM 3D Body",
                "checkpoint": str(args.checkpoint),
                "device": str(device),
                "image": str(args.image),
                "people": records,
            }
        ),
        encoding="utf-8",
    )
    print(f"{len(records)} person(s) -> {args.output}")


if __name__ == "__main__":
    main()
