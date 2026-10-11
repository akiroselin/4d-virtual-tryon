from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aigc_fashion.panel_instance_grower import (  # noqa: E402
    EvidenceGuidedFloodFill,
    PanelInstanceGrower,
    rasterize_garmentcode_panels,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the FFN panel-growth scheduler against GarmentCode teacher masks."
    )
    parser.add_argument("--pattern", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=160)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    payload = json.loads(args.pattern.read_text(encoding="utf-8"))
    evidence, seeds, names = rasterize_garmentcode_panels(
        payload,
        resolution=args.resolution,
    )
    result = PanelInstanceGrower(EvidenceGuidedFloodFill()).grow(
        evidence,
        seeds,
        source="GarmentCode teacher diagnostic; not image inference",
    )
    summary = result.summary()
    summary["teacher_pattern"] = str(args.pattern.resolve())
    summary["panel_roles"] = list(names)
    summary["instance_iou"] = {
        instance.role: float(
            np.count_nonzero(instance.mask & (evidence[index] > 0.5))
            / max(np.count_nonzero(instance.mask | (evidence[index] > 0.5)), 1)
        )
        for index, instance in enumerate(result.instances)
    }
    summary_path = args.out_dir / "panel_growth_teacher_report.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    np.savez_compressed(
        args.out_dir / "panel_growth_teacher_fields.npz",
        evidence=evidence,
        masks=np.stack([instance.mask for instance in result.instances]),
        signed_distances=np.stack(
            [instance.signed_distance for instance in result.instances]
        ),
        seam_endpoint_probabilities=np.stack(
            [instance.seam_endpoint_probability for instance in result.instances]
        ),
    )
    _render_preview(args.out_dir / "panel_growth_teacher_preview.png", result.instances)
    print(json.dumps(summary, indent=2))


def _render_preview(path: Path, instances: tuple, tile_size: int = 240) -> None:
    columns = 2
    rows = int(np.ceil(len(instances) / columns))
    canvas = Image.new("RGB", (columns * tile_size, rows * tile_size), (246, 247, 249))
    for index, instance in enumerate(instances):
        mask = Image.fromarray(instance.mask.astype(np.uint8) * 255)
        mask = mask.resize((tile_size - 24, tile_size - 48), Image.Resampling.NEAREST)
        tile = Image.new("RGB", (tile_size, tile_size), (246, 247, 249))
        color = Image.new("RGB", mask.size, (60, 112, 136))
        tile.paste(color, (12, 34), mask)
        draw = ImageDraw.Draw(tile)
        draw.text((12, 10), f"{instance.role}  confidence {instance.confidence:.3f}", fill=(28, 34, 40))
        draw.text((12, tile_size - 13), f"fields {instance.visited_fields}", fill=(72, 80, 88))
        canvas.paste(tile, ((index % columns) * tile_size, (index // columns) * tile_size))
    canvas.save(path)


if __name__ == "__main__":
    main()
