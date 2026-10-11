from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
from PIL import Image


DEFAULT_LIBRARY_PATH = Path(__file__).resolve().parents[1] / "assets" / "pattern_templates" / "pattern_library.json"


@dataclass(frozen=True)
class PanelTemplate:
    role: str
    shape_family: str
    quantity: int
    mirror: bool
    grainline: str


@dataclass(frozen=True)
class PatternProvider:
    provider_id: str
    name: str
    license: str
    commercial_use: bool
    role: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GarmentPatternTemplate:
    template_id: str
    garment_type: str
    aliases: tuple[str, ...]
    generator: str
    status: str
    layer: str
    material_families: tuple[str, ...]
    required_measurements: tuple[str, ...]
    parameters: dict[str, Any]
    panels: tuple[PanelTemplate, ...]
    seams: tuple[dict[str, str], ...]
    topology: str
    providers: tuple[PatternProvider, ...]

    @property
    def implemented(self) -> bool:
        return self.status == "implemented"

    @property
    def production_provider(self) -> PatternProvider | None:
        return next((provider for provider in self.providers if provider.commercial_use), None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "template_id": self.template_id,
            "garment_type": self.garment_type,
            "aliases": list(self.aliases),
            "generator": self.generator,
            "status": self.status,
            "layer": self.layer,
            "material_families": list(self.material_families),
            "required_measurements": list(self.required_measurements),
            "parameters": self.parameters,
            "panels": [asdict(panel) for panel in self.panels],
            "seams": [dict(seam) for seam in self.seams],
            "topology": self.topology,
            "providers": [provider.to_dict() for provider in self.providers],
            "production_provider": (
                self.production_provider.provider_id if self.production_provider is not None else None
            ),
        }


@dataclass(frozen=True)
class GarmentClassification:
    garment_type: str
    template_id: str
    confidence: float
    method: str
    alternatives: tuple[tuple[str, float], ...]
    image_features: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "garment_type": self.garment_type,
            "template_id": self.template_id,
            "confidence": self.confidence,
            "method": self.method,
            "alternatives": [
                {"garment_type": garment_type, "confidence": confidence}
                for garment_type, confidence in self.alternatives
            ],
            "image_features": dict(self.image_features),
        }


@lru_cache(maxsize=4)
def load_pattern_library(path: str | Path = DEFAULT_LIBRARY_PATH) -> tuple[GarmentPatternTemplate, ...]:
    library_path = Path(path)
    data = json.loads(library_path.read_text(encoding="utf-8"))
    if int(data.get("schema_version", 0)) not in {1, 2}:
        raise ValueError(f"unsupported pattern library schema: {library_path}")
    provider_registry = {
        str(provider_id): PatternProvider(
            provider_id=str(provider_id),
            name=str(raw.get("name", provider_id)),
            license=str(raw.get("license", "unknown")),
            commercial_use=bool(raw.get("commercial_use", False)),
            role=str(raw.get("role", "pattern provider")),
        )
        for provider_id, raw in data.get("providers", {}).items()
    }
    default_provider_priority = tuple(str(value) for value in data.get("default_provider_priority", []))
    templates: list[GarmentPatternTemplate] = []
    for raw in data.get("templates", []):
        provider_ids = tuple(str(value) for value in raw.get("provider_priority", default_provider_priority))
        providers = tuple(provider_registry[value] for value in provider_ids if value in provider_registry)
        templates.append(
            GarmentPatternTemplate(
                template_id=str(raw["id"]),
                garment_type=str(raw["garment_type"]),
                aliases=tuple(str(value) for value in raw.get("aliases", [])),
                generator=str(raw["generator"]),
                status=str(raw["status"]),
                layer=str(raw["layer"]),
                material_families=tuple(str(value) for value in raw.get("material_families", [])),
                required_measurements=tuple(str(value) for value in raw.get("required_measurements", [])),
                parameters=dict(raw.get("parameters", {})),
                panels=tuple(PanelTemplate(**panel) for panel in raw.get("panels", [])),
                seams=tuple(dict(seam) for seam in raw.get("seams", [])),
                topology=str(raw.get("topology", "template_defined")),
                providers=providers,
            )
        )
    if not templates:
        raise ValueError(f"pattern library contains no templates: {library_path}")
    return tuple(templates)


def get_pattern_template(
    garment_type_or_id: str,
    library: tuple[GarmentPatternTemplate, ...] | None = None,
) -> GarmentPatternTemplate:
    templates = library or load_pattern_library()
    query = _normalize_name(garment_type_or_id)
    for template in templates:
        names = (template.template_id, template.garment_type, *template.aliases)
        if query in {_normalize_name(name) for name in names}:
            return template
    raise KeyError(f"no pattern template for garment type: {garment_type_or_id}")


def classify_garment_image(
    image_path: str | Path,
    prompt: str | None = None,
    library: tuple[GarmentPatternTemplate, ...] | None = None,
) -> GarmentClassification:
    """Select a pattern program from product-image shape and an optional text prior."""

    templates = library or load_pattern_library()
    features = _silhouette_features(image_path)
    scores = {template.garment_type: 0.05 for template in templates}

    aspect = features["silhouette_aspect"]
    sleeve_spread = features["upper_to_mid_width_ratio"]
    leg_gap = features["bottom_center_gap"]
    if aspect <= 1.15:
        _add(scores, "t-shirt", 2.6)
        _add(scores, "hoodie", 1.2)
        _add(scores, "button-shirt", 0.9)
        _add(scores, "shorts", 0.8 * leg_gap)
    elif aspect <= 1.65:
        _add(scores, "button-shirt", 1.6)
        _add(scores, "long-sleeve t-shirt", 1.3)
        _add(scores, "tank-top", 1.0)
        _add(scores, "shorts", 1.4 * leg_gap)
        _add(scores, "skirt", 0.8 * (1.0 - leg_gap))
    else:
        _add(scores, "dress", 2.0 * (1.0 - leg_gap))
        _add(scores, "trousers", 2.4 * leg_gap)
        _add(scores, "skirt", 1.2 * (1.0 - leg_gap))
        _add(scores, "jacket", 0.8)

    if sleeve_spread > 1.08:
        _add(scores, "t-shirt", 0.9)
        _add(scores, "button-shirt", 0.5)
        _add(scores, "hoodie", 0.4)
    else:
        _add(scores, "tank-top", 0.6)
        _add(scores, "dress", 0.3)

    prompt_tokens = _normalize_name(prompt or "")
    prompt_matched = False
    for template in templates:
        aliases = (template.garment_type, *template.aliases)
        if any(_contains_phrase(prompt_tokens, _normalize_name(alias)) for alias in aliases):
            scores[template.garment_type] += 4.0
            prompt_matched = True

    ordered_types = sorted(scores, key=scores.get, reverse=True)
    raw = np.asarray([scores[name] for name in ordered_types], dtype=np.float64)
    probabilities = np.exp((raw - raw.max()) / 0.85)
    probabilities /= probabilities.sum()
    ranked = tuple((name, float(value)) for name, value in zip(ordered_types, probabilities))
    selected = get_pattern_template(ranked[0][0], templates)
    return GarmentClassification(
        garment_type=selected.garment_type,
        template_id=selected.template_id,
        confidence=ranked[0][1],
        method="pattern_ontology+silhouette_rules+prompt_prior" if prompt_matched else "pattern_ontology+silhouette_rules",
        alternatives=ranked[1:4],
        image_features=features,
    )


def classify_garment_intent(
    garment_type: str,
    prompt: str | None = None,
    library: tuple[GarmentPatternTemplate, ...] | None = None,
) -> GarmentClassification:
    """Route structured intent to a template when no product image is available."""

    templates = library or load_pattern_library()
    try:
        selected = get_pattern_template(garment_type, templates)
        confidence = 1.0
        method = "structured_garment_type"
    except KeyError:
        normalized_prompt = _normalize_name(prompt or garment_type)
        ranked: list[tuple[GarmentPatternTemplate, int]] = []
        for template in templates:
            names = (template.garment_type, *template.aliases)
            score = max(
                (len(_normalize_name(name)) for name in names if _contains_phrase(normalized_prompt, _normalize_name(name))),
                default=0,
            )
            ranked.append((template, score))
        ranked.sort(key=lambda item: item[1], reverse=True)
        selected, score = ranked[0]
        confidence = 0.75 if score else 0.25
        method = "prompt_alias_router" if score else "template_default"
    alternatives = tuple(
        (template.garment_type, 0.0)
        for template in templates
        if template.template_id != selected.template_id
    )[:3]
    return GarmentClassification(
        garment_type=selected.garment_type,
        template_id=selected.template_id,
        confidence=confidence,
        method=method,
        alternatives=alternatives,
        image_features={},
    )


def _silhouette_features(image_path: str | Path) -> dict[str, float]:
    image = Image.open(image_path).convert("RGB")
    arr = np.asarray(image, dtype=np.int16)
    brightness = arr.mean(axis=2)
    chroma = arr.max(axis=2) - arr.min(axis=2)
    mask = (brightness < 248) | (chroma > 18)
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return {
            "silhouette_aspect": 1.0,
            "upper_to_mid_width_ratio": 1.0,
            "bottom_center_gap": 0.0,
            "foreground_fraction": 0.0,
        }
    x0, x1 = np.percentile(xs, [1.0, 99.0]).astype(int)
    y0, y1 = np.percentile(ys, [1.0, 99.0]).astype(int)
    crop = mask[y0 : y1 + 1, x0 : x1 + 1]
    height, width = crop.shape

    def band_width(start: float, end: float) -> float:
        row0 = int(np.clip(start * height, 0, max(height - 1, 0)))
        row1 = max(row0 + 1, int(np.clip(end * height, 1, height)))
        widths: list[float] = []
        for row in crop[row0:row1]:
            occupied = np.flatnonzero(row)
            if occupied.size >= 2:
                widths.append(float(occupied[-1] - occupied[0] + 1))
        return float(np.median(widths)) if widths else 1.0

    upper_width = band_width(0.12, 0.38)
    mid_width = band_width(0.42, 0.68)
    bottom = crop[int(0.72 * height) :]
    center_half = max(1, int(width * 0.07))
    center = width // 2
    center_occupancy = float(np.mean(bottom[:, center - center_half : center + center_half + 1])) if bottom.size else 1.0
    side_occupancy = float(np.mean(bottom)) if bottom.size else 1.0
    center_gap = float(np.clip((side_occupancy - center_occupancy) / max(side_occupancy, 1e-6), 0.0, 1.0))
    return {
        "silhouette_aspect": float(height / max(width, 1)),
        "upper_to_mid_width_ratio": float(upper_width / max(mid_width, 1.0)),
        "bottom_center_gap": center_gap,
        "foreground_fraction": float(np.mean(crop)),
    }


def _add(scores: dict[str, float], garment_type: str, amount: float) -> None:
    if garment_type in scores:
        scores[garment_type] += float(amount)


def _normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _contains_phrase(text: str, phrase: str) -> bool:
    return bool(phrase) and f" {phrase} " in f" {text} "
