"""Generate synthetic product images for the demo outfit.

The original runs used private catalog photos that are not in the repository.
These procedurally drawn stand-ins have the properties the pipeline reads:
a garment silhouette on a plain light background (category, aspect, pleat
count), cloth colours, and a chromatic front print. They are synthetic inputs,
not photographs, and outputs built from them should be labelled as such.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

BACKGROUND = (246, 246, 244)


def tshirt_image(path: Path, size: int = 768) -> Path:
    image = Image.new("RGB", (size, size), BACKGROUND)
    draw = ImageDraw.Draw(image)
    s = size / 768.0
    body = [
        (250, 110), (318, 92), (384, 128), (450, 92), (518, 110),
        (668, 228), (604, 318), (548, 276), (548, 690), (220, 690),
        (220, 276), (164, 318), (100, 228),
    ]
    draw.polygon([(x * s, y * s) for x, y in body], fill=(238, 236, 230), outline=(205, 203, 198))
    draw.arc([318 * s, 60 * s, 450 * s, 168 * s], 20, 160, fill=(205, 203, 198), width=int(6 * s))
    # Front print: a small sunset badge so the texture has a chromatic region.
    cx, cy, r = 384 * s, 360 * s, 88 * s
    for i in range(int(r), 0, -1):
        t = i / r
        colour = (int(236 - 40 * t), int(96 + 70 * (1 - t)), int(64 + 30 * (1 - t)))
        draw.ellipse([cx - i, cy - i, cx + i, cy + i], fill=colour)
    for k in range(5):
        y = cy + (12 + 16 * k) * s
        draw.rectangle([cx - r, y, cx + r, y + 6 * s], fill=(32, 54, 92))
    draw.rectangle([cx - r - 2, cy + 12 * s, cx + r + 2, cy + r + 4], outline=None)
    image = image.filter(ImageFilter.SMOOTH)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path


def skirt_image(path: Path, size: int = 768, pleats: int = 10) -> Path:
    image = Image.new("RGB", (size, size), BACKGROUND)
    draw = ImageDraw.Draw(image)
    waist_y, hem_y = 230, 520
    waist_half, hem_half = 190, 330
    cx = size / 2
    for p in range(pleats):
        a0, a1 = p / pleats, (p + 1) / pleats
        top0, top1 = cx - waist_half + 2 * waist_half * a0, cx - waist_half + 2 * waist_half * a1
        bot0, bot1 = cx - hem_half + 2 * hem_half * a0, cx - hem_half + 2 * hem_half * a1
        shade = 0.82 + 0.18 * math.cos(math.pi * (p % 2))
        base = (int(70 * shade), int(64 * shade), int(58 * shade))
        draw.polygon([(top0, waist_y), (top1, waist_y), (bot1, hem_y), (bot0, hem_y)], fill=base)
    # Plaid overlay.
    for x in range(0, size, 29):
        draw.line([(x, waist_y), (x, hem_y)], fill=(48, 44, 40), width=2)
    for y in range(waist_y, hem_y, 29):
        draw.line([(0, y), (size, y)], fill=(48, 44, 40), width=2)
    for y in range(waist_y + 17, hem_y, 116):
        draw.line([(0, y), (size, y)], fill=(122, 42, 44), width=3)
    # Re-mask to the skirt trapezoid and add the waistband.
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).polygon(
        [(cx - waist_half, waist_y), (cx + waist_half, waist_y), (cx + hem_half, hem_y), (cx - hem_half, hem_y)],
        fill=255,
    )
    clean = Image.new("RGB", (size, size), BACKGROUND)
    clean.paste(image, (0, 0), mask)
    ImageDraw.Draw(clean).rectangle([cx - waist_half, waist_y - 26, cx + waist_half, waist_y], fill=(52, 48, 44))
    path.parent.mkdir(parents=True, exist_ok=True)
    clean.save(path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/demo_inputs"))
    args = parser.parse_args()
    print(tshirt_image(args.out_dir / "synthetic_tshirt.png"))
    print(skirt_image(args.out_dir / "synthetic_pleated_skirt.png"))


if __name__ == "__main__":
    main()
