from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

from .data import GarmentIntent


def parse_garment_intent(prompt: str | None = None, intent_json: str | Path | None = None) -> GarmentIntent:
    """Heuristic placeholder for the FashionCLIP/LLM semantic encoder.

    The production version should replace this with FashionCLIP retrieval plus
    an LLM schema normalizer. The interface stays stable: return a structured
    GarmentIntent that downstream pattern and physics modules consume.
    """

    if intent_json is not None:
        data = json.loads(Path(intent_json).read_text(encoding="utf-8"))
        return _intent_from_dict(data)

    text = prompt or "relaxed short sleeve crew neck cotton t-shirt"
    lower = text.lower()

    garment_type = "t-shirt"
    if "dress" in lower:
        garment_type = "dress"
    elif "skirt" in lower:
        garment_type = "skirt"
    elif "shirt" in lower and "t-shirt" not in lower and "tee" not in lower:
        garment_type = "shirt"

    silhouette = "relaxed"
    if re.search(r"\b(oversized|boxy|wide)\b", lower):
        silhouette = "oversized"
    elif re.search(r"\b(slim|fitted|tight)\b", lower):
        silhouette = "slim"

    sleeve = "short"
    if "sleeveless" in lower:
        sleeve = "sleeveless"
    elif "long sleeve" in lower or "long-sleeve" in lower:
        sleeve = "long"

    neckline = "crew"
    if "v-neck" in lower or "v neck" in lower:
        neckline = "v-neck"
    elif "boat neck" in lower:
        neckline = "boat"

    fabric = "medium cotton jersey"
    if "silk" in lower:
        fabric = "light silk"
    elif "denim" in lower:
        fabric = "heavy denim"
    elif "linen" in lower:
        fabric = "light linen"
    elif "nylon" in lower or "polyamide" in lower:
        fabric = "medium nylon"
    elif "wool" in lower:
        fabric = "soft wool knit"

    styles = []
    for style in ("minimal", "streetwear", "evening", "sport", "vintage", "avant-garde"):
        if style in lower:
            styles.append(style)
    if not styles:
        styles.append("minimal")

    body_fit = "loose" if silhouette in {"relaxed", "oversized"} else "close"
    return GarmentIntent(
        prompt=text,
        garment_type=garment_type,
        silhouette=silhouette,
        sleeve=sleeve,
        neckline=neckline,
        fabric=fabric,
        style_refs=tuple(styles),
        body_fit=body_fit,
    )


def _intent_from_dict(data: dict[str, Any]) -> GarmentIntent:
    fields = GarmentIntent.__dataclass_fields__
    filtered = {key: value for key, value in data.items() if key in fields}
    if "style_refs" in filtered:
        filtered["style_refs"] = tuple(filtered["style_refs"])
    return GarmentIntent(**filtered)
