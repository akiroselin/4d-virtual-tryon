from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .data import AvatarSpec, TshirtSizeSpec


@dataclass(frozen=True)
class ImageTshirtFit:
    bbox_xyxy: tuple[int, int, int, int]
    image_size: tuple[int, int]
    silhouette_aspect: float
    estimated_size: TshirtSizeSpec

    def to_dict(self) -> dict:
        data = asdict(self)
        data["estimated_size"] = self.estimated_size.to_dict()
        return data


def infer_tshirt_size_from_image(
    image_path: str | Path,
    chest_circumference_cm: float = 104.0,
    avatar: AvatarSpec | None = None,
) -> ImageTshirtFit:
    """Infer a practical T-shirt size from a product image silhouette.

    This is a deterministic adapter, not a full SewFormer/GarmentCode model.
    It estimates proportions from the visible garment extent and returns a real
    metric TshirtSizeSpec suitable for our current parametric panel generator.
    """

    image = Image.open(image_path).convert("RGB")
    arr = np.asarray(image, dtype=np.int16)
    brightness = arr.mean(axis=2)
    chroma = arr.max(axis=2) - arr.min(axis=2)
    # White shirt on white background: shadows, seams, print, and folds carry
    # most of the signal. Use a permissive non-background mask.
    mask = (brightness < 248) | (chroma > 18)
    mask = _largest_soft_bbox_mask(mask)
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        bbox = (0, 0, image.width - 1, image.height - 1)
    else:
        bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))

    x0, y0, x1, y1 = bbox
    bbox_w = max(1, x1 - x0 + 1)
    bbox_h = max(1, y1 - y0 + 1)
    aspect = bbox_h / bbox_w

    flat_body_width_cm = chest_circumference_cm * 0.5
    if avatar is None:
        body_length_cm = float(np.clip(flat_body_width_cm * aspect * 1.75, 62.0, 78.0))
        shoulder_width_cm = float(np.clip(flat_body_width_cm * 0.88, 42.0, 50.0))
        sleeve_length_cm = float(np.clip(flat_body_width_cm * 0.38, 16.0, 24.0))
        sleeve_opening_cm = float(np.clip(chest_circumference_cm * 0.16, 15.0, 21.0))
        neck_width_cm = float(np.clip(shoulder_width_cm * 0.38, 16.0, 20.0))
    else:
        age_scale = float(np.clip((avatar.height_cm - 120.0) / 60.0, 0.0, 1.0))
        length_factor = 1.28 + 0.27 * age_scale
        raw_length = flat_body_width_cm * aspect * length_factor
        body_length_cm = float(
            np.clip(
                raw_length,
                avatar.height_cm * (0.29 + 0.04 * age_scale),
                avatar.height_cm * (0.35 + 0.05 * age_scale),
            )
        )
        shoulder_width_cm = float(
            np.clip(flat_body_width_cm * 0.88, avatar.shoulder_cm * 1.02, avatar.shoulder_cm * 1.15)
        )
        sleeve_length_cm = float(
            np.clip(flat_body_width_cm * 0.45, avatar.height_cm * 0.115, avatar.height_cm * 0.15)
        )
        sleeve_opening_cm = float(
            np.clip(chest_circumference_cm * 0.25, chest_circumference_cm * 0.23, chest_circumference_cm * 0.31)
        )
        neck_width_cm = float(
            np.clip(shoulder_width_cm * 0.42, avatar.shoulder_cm * 0.40, avatar.shoulder_cm * 0.48)
        )

    return ImageTshirtFit(
        bbox_xyxy=bbox,
        image_size=image.size,
        silhouette_aspect=float(aspect),
        estimated_size=TshirtSizeSpec(
            chest_circumference_cm=chest_circumference_cm,
            body_length_cm=body_length_cm,
            shoulder_width_cm=shoulder_width_cm,
            sleeve_length_cm=sleeve_length_cm,
            sleeve_opening_cm=sleeve_opening_cm,
            neck_width_cm=neck_width_cm,
        ),
    )


def _largest_soft_bbox_mask(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return mask
    x0, x1 = int(np.percentile(xs, 1)), int(np.percentile(xs, 99))
    y0, y1 = int(np.percentile(ys, 1)), int(np.percentile(ys, 99))
    cleaned = np.zeros_like(mask, dtype=bool)
    cleaned[y0 : y1 + 1, x0 : x1 + 1] = mask[y0 : y1 + 1, x0 : x1 + 1]
    return cleaned
