from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt, label
from skimage.measure import find_contours
import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(frozen=True)
class PanelGrowthConfig:
    field_of_view: int = 33
    move_step: int = 12
    move_threshold: float = 0.55
    mask_threshold: float = 0.5
    seed_probability: float = 0.95
    maximum_moves: int = 512
    minimum_panel_pixels: int = 24
    maximum_instances: int = 24

    def __post_init__(self) -> None:
        if self.field_of_view < 9 or self.field_of_view % 2 == 0:
            raise ValueError("field_of_view must be an odd integer of at least 9")
        if self.move_step <= 0 or self.move_step >= self.field_of_view:
            raise ValueError("move_step must be positive and smaller than field_of_view")
        if self.maximum_moves <= 0 or self.minimum_panel_pixels <= 0:
            raise ValueError("growth limits must be positive")
        if not 0.0 < self.move_threshold < 1.0 or not 0.0 < self.mask_threshold < 1.0:
            raise ValueError("growth thresholds must lie in (0, 1)")


@dataclass(frozen=True)
class PanelSeed:
    seed_id: str
    role: str
    role_id: int
    y: int
    x: int
    confidence: float
    evidence_channel: int | None = None

    def __post_init__(self) -> None:
        if not self.seed_id or not self.role:
            raise ValueError("panel seed id and role must be non-empty")
        if self.role_id < 0 or self.y < 0 or self.x < 0:
            raise ValueError("panel seed indices must be non-negative")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("panel seed confidence must lie in [0, 1]")


@dataclass(frozen=True)
class PanelInstance:
    instance_id: str
    role: str
    seed_id: str
    confidence: float
    mask: np.ndarray
    signed_distance: np.ndarray
    seam_endpoint_probability: np.ndarray
    boundary_xy: np.ndarray
    visited_fields: int

    @property
    def area_pixels(self) -> int:
        return int(np.count_nonzero(self.mask))

    def summary(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "role": self.role,
            "seed_id": self.seed_id,
            "confidence": self.confidence,
            "area_pixels": self.area_pixels,
            "boundary_points": int(len(self.boundary_xy)),
            "visited_fields": self.visited_fields,
        }


@dataclass(frozen=True)
class PanelGrowthResult:
    instances: tuple[PanelInstance, ...]
    rejected_seeds: tuple[str, ...]
    source: str
    learned_checkpoint: str | None

    def summary(self) -> dict[str, Any]:
        return {
            "instances": [instance.summary() for instance in self.instances],
            "rejected_seeds": list(self.rejected_seeds),
            "source": self.source,
            "learned_checkpoint": self.learned_checkpoint,
            "production_pattern_claimed": False,
        }


@dataclass(frozen=True)
class PanelFloodFillOutput:
    mask_logits: Tensor
    seam_endpoint_logits: Tensor
    stop_logits: Tensor


class PanelSeedNetwork(nn.Module):
    """Predict unknown-count panel seeds and role logits from image features."""

    def __init__(self, feature_channels: int, role_count: int, hidden_channels: int = 48) -> None:
        super().__init__()
        if feature_channels <= 0 or role_count <= 0:
            raise ValueError("seed-network dimensions must be positive")
        self.role_count = role_count
        self.body = nn.Sequential(
            nn.Conv2d(feature_channels, hidden_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.GELU(),
        )
        self.objectness = nn.Conv2d(hidden_channels, 1, 1)
        self.roles = nn.Conv2d(hidden_channels, role_count, 1)

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
        hidden = self.body(features)
        return self.objectness(hidden), self.roles(hidden)


class PanelImageEncoder(nn.Module):
    """Dense image encoder for canonical panel-growth evidence."""

    def __init__(self, feature_channels: int = 32) -> None:
        super().__init__()
        if feature_channels <= 0:
            raise ValueError("feature_channels must be positive")
        self.feature_channels = feature_channels
        self.network = nn.Sequential(
            nn.Conv2d(3, 24, 5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv2d(24, feature_channels, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(feature_channels, feature_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(feature_channels, feature_channels, 3, padding=1),
        )

    def forward(self, image: Tensor) -> Tensor:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("image must have shape (B,3,H,W)")
        return self.network(image)


class PanelFloodFillNetwork(nn.Module):
    """FFN-style recurrent local update for one canonical 2D panel instance."""

    def __init__(
        self,
        feature_channels: int,
        role_count: int,
        *,
        role_embedding_dim: int = 12,
        hidden_channels: int = 48,
    ) -> None:
        super().__init__()
        if feature_channels <= 0 or role_count <= 0:
            raise ValueError("flood-fill network dimensions must be positive")
        self.feature_channels = feature_channels
        self.role_count = role_count
        self.role_embedding = nn.Embedding(role_count, role_embedding_dim)
        input_channels = feature_channels + 1 + role_embedding_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.GELU(),
        )
        self.mask_head = nn.Conv2d(hidden_channels, 1, 1)
        self.seam_head = nn.Conv2d(hidden_channels, 1, 1)
        self.stop_head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(hidden_channels, 1))

    def forward(
        self,
        visual_patch: Tensor,
        current_mask_logits: Tensor,
        role_ids: Tensor,
    ) -> PanelFloodFillOutput:
        if visual_patch.ndim != 4 or current_mask_logits.shape != (
            visual_patch.shape[0],
            1,
            visual_patch.shape[2],
            visual_patch.shape[3],
        ):
            raise ValueError("visual and mask patches must have compatible BCHW shapes")
        role = self.role_embedding(role_ids.to(torch.long))[:, :, None, None]
        role = role.expand(-1, -1, visual_patch.shape[2], visual_patch.shape[3])
        hidden = self.encoder(torch.cat((visual_patch, current_mask_logits, role), dim=1))
        # Residual mask updates reproduce the recurrent state pathway of FFN.
        mask_logits = current_mask_logits + self.mask_head(hidden)
        return PanelFloodFillOutput(
            mask_logits=mask_logits,
            seam_endpoint_logits=self.seam_head(hidden),
            stop_logits=self.stop_head(hidden),
        )


class PanelInstanceGrowthModel(nn.Module):
    """Trainable image encoder, seed proposer, and recurrent FFN cell."""

    def __init__(self, role_count: int, *, feature_channels: int = 32) -> None:
        super().__init__()
        self.role_count = role_count
        self.feature_channels = feature_channels
        self.image_encoder = PanelImageEncoder(feature_channels)
        self.seed_network = PanelSeedNetwork(feature_channels, role_count)
        self.flood_fill_network = PanelFloodFillNetwork(feature_channels, role_count)

    def encode(self, image: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        features = self.image_encoder(image)
        objectness, roles = self.seed_network(features)
        return features, objectness, roles


class EvidenceGuidedFloodFill(nn.Module):
    """Deterministic updater for tests and teacher diagnostics only.

    It proves the flood-fill scheduler independently from model quality. It is
    never reported as image inference and must not be used as production output.
    """

    def forward(
        self,
        visual_patch: Tensor,
        current_mask_logits: Tensor,
        role_ids: Tensor,
    ) -> PanelFloodFillOutput:
        channels = visual_patch.shape[1]
        selected = torch.stack(
            [visual_patch[index, int(role_ids[index]) % channels] for index in range(len(role_ids))]
        )[:, None]
        probability = selected.clamp(1.0e-4, 1.0 - 1.0e-4)
        logits = torch.logit(probability)
        dilated = F.max_pool2d(probability, 3, stride=1, padding=1)
        eroded = -F.max_pool2d(-probability, 3, stride=1, padding=1)
        seam = torch.logit((dilated - eroded).clamp(1.0e-4, 1.0 - 1.0e-4))
        return PanelFloodFillOutput(
            mask_logits=logits,
            seam_endpoint_logits=seam,
            stop_logits=probability.new_full((len(probability), 1), -8.0),
        )


class PanelInstanceGrower:
    """Seeded local-field scheduler for variable-count panel reconstruction."""

    def __init__(
        self,
        updater: nn.Module,
        *,
        config: PanelGrowthConfig = PanelGrowthConfig(),
        checkpoint_path: str | Path | None = None,
    ) -> None:
        self.updater = updater.eval()
        self.config = config
        self.checkpoint_path = str(Path(checkpoint_path).resolve()) if checkpoint_path else None

    def grow(
        self,
        visual_features: Tensor | np.ndarray,
        seeds: list[PanelSeed] | tuple[PanelSeed, ...],
        *,
        source: str = "learned image features",
    ) -> PanelGrowthResult:
        features = torch.as_tensor(visual_features, dtype=torch.float32)
        if features.ndim == 3:
            features = features.unsqueeze(0)
        if features.ndim != 4 or features.shape[0] != 1:
            raise ValueError("visual_features must have shape (C,H,W) or (1,C,H,W)")
        height, width = features.shape[-2:]
        accepted: list[PanelInstance] = []
        rejected: list[str] = []
        occupied_by_role: dict[str, np.ndarray] = {}
        ordered_seeds = sorted(seeds, key=lambda item: item.confidence, reverse=True)
        for seed in ordered_seeds[: self.config.maximum_instances]:
            if seed.y >= height or seed.x >= width:
                rejected.append(seed.seed_id)
                continue
            instance = self._grow_one(features, seed, len(accepted))
            if instance is None:
                rejected.append(seed.seed_id)
                continue
            existing = occupied_by_role.setdefault(seed.role, np.zeros((height, width), dtype=bool))
            overlap = np.count_nonzero(existing & instance.mask) / max(instance.area_pixels, 1)
            if overlap > 0.65:
                rejected.append(seed.seed_id)
                continue
            existing |= instance.mask
            accepted.append(instance)
        return PanelGrowthResult(
            instances=tuple(accepted),
            rejected_seeds=tuple(rejected),
            source=source,
            learned_checkpoint=self.checkpoint_path,
        )

    def _grow_one(
        self,
        features: Tensor,
        seed: PanelSeed,
        instance_index: int,
    ) -> PanelInstance | None:
        height, width = features.shape[-2:]
        prior = float(np.clip(self.config.seed_probability, 1.0e-4, 1.0 - 1.0e-4))
        logits = torch.full((1, 1, height, width), float(torch.logit(torch.tensor(0.02))))
        logits[0, 0, seed.y, seed.x] = float(torch.logit(torch.tensor(prior)))
        seam_logits = torch.full_like(logits, -8.0)
        queue = [(seed.y, seed.x)]
        queued = {(seed.y, seed.x)}
        visited: set[tuple[int, int]] = set()
        while queue and len(visited) < self.config.maximum_moves:
            center = queue.pop(0)
            if center in visited:
                continue
            visited.add(center)
            feature_patch = _extract_patch(features, center, self.config.field_of_view)
            mask_patch = _extract_patch(logits, center, self.config.field_of_view)
            with torch.no_grad():
                output = self.updater(
                    feature_patch,
                    mask_patch,
                    torch.as_tensor([seed.role_id], dtype=torch.long),
                )
            _write_patch_max(logits, output.mask_logits, center)
            _write_patch_max(seam_logits, output.seam_endpoint_logits, center)
            if bool(torch.sigmoid(output.stop_logits[0, 0]) >= 0.5):
                continue
            probability_patch = torch.sigmoid(output.mask_logits[0, 0])
            for direction, active in _active_frontiers(
                probability_patch,
                threshold=self.config.move_threshold,
            ).items():
                if not active:
                    continue
                dy, dx = direction
                next_center = (
                    int(np.clip(center[0] + dy * self.config.move_step, 0, height - 1)),
                    int(np.clip(center[1] + dx * self.config.move_step, 0, width - 1)),
                )
                if next_center not in visited and next_center not in queued:
                    queue.append(next_center)
                    queued.add(next_center)

        probability = torch.sigmoid(logits)[0, 0].numpy()
        raw_mask = probability >= self.config.mask_threshold
        mask = _seed_connected_component(raw_mask, seed.y, seed.x)
        area = int(np.count_nonzero(mask))
        if area < self.config.minimum_panel_pixels:
            return None
        sdf = distance_transform_edt(~mask) - distance_transform_edt(mask)
        boundary = _longest_boundary(mask)
        confidence = float(seed.confidence * probability[mask].mean())
        return PanelInstance(
            instance_id=f"panel_{instance_index:02d}_{seed.role}",
            role=seed.role,
            seed_id=seed.seed_id,
            confidence=confidence,
            mask=mask,
            signed_distance=sdf.astype(np.float32),
            seam_endpoint_probability=torch.sigmoid(seam_logits)[0, 0].numpy().astype(np.float32),
            boundary_xy=boundary,
            visited_fields=len(visited),
        )


def propose_panel_seeds(
    objectness_logits: Tensor,
    role_logits: Tensor,
    *,
    threshold: float = 0.5,
    nms_radius: int = 5,
    maximum_seeds: int = 24,
    role_names: Mapping[int, str] | None = None,
) -> list[PanelSeed]:
    if objectness_logits.ndim != 4 or objectness_logits.shape[:2] != (1, 1):
        raise ValueError("objectness_logits must have shape (1,1,H,W)")
    if role_logits.ndim != 4 or role_logits.shape[0] != 1:
        raise ValueError("role_logits must have shape (1,R,H,W)")
    probability = torch.sigmoid(objectness_logits)
    pooled = F.max_pool2d(probability, 2 * nms_radius + 1, stride=1, padding=nms_radius)
    peaks = (probability >= threshold) & (probability == pooled)
    coordinates = torch.nonzero(peaks[0, 0], as_tuple=False)
    candidates = []
    for y, x in coordinates.tolist():
        confidence = float(probability[0, 0, y, x])
        role_id = int(torch.argmax(role_logits[0, :, y, x]))
        role = role_names.get(role_id, f"role_{role_id}") if role_names else f"role_{role_id}"
        candidates.append(PanelSeed(f"seed_{y}_{x}", role, role_id, y, x, confidence))
    return sorted(candidates, key=lambda item: item.confidence, reverse=True)[:maximum_seeds]


def reconstruct_panels_from_image_ffn(
    image_path: str | Path,
    checkpoint_path: str | Path,
    *,
    config: PanelGrowthConfig = PanelGrowthConfig(),
    image_size: int = 512,
) -> PanelGrowthResult:
    """Run trained image -> seed -> FFN panel growth without a template fallback."""

    image_path = Path(image_path).expanduser().resolve()
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"panel-growth image not found: {image_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"panel-growth checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    role_names_raw = checkpoint.get("role_names")
    if not isinstance(role_names_raw, (list, tuple)) or not role_names_raw:
        raise ValueError("panel-growth checkpoint must contain non-empty role_names")
    role_names = {index: str(name) for index, name in enumerate(role_names_raw)}
    feature_channels = int(checkpoint.get("feature_channels", 32))
    model = PanelInstanceGrowthModel(len(role_names), feature_channels=feature_channels)
    state_dict = checkpoint.get("model")
    if not isinstance(state_dict, Mapping):
        raise ValueError("panel-growth checkpoint must contain a model state dict")
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    image = Image.open(image_path).convert("RGB")
    image.thumbnail((image_size, image_size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (image_size, image_size), (245, 245, 245))
    offset = ((image_size - image.width) // 2, (image_size - image.height) // 2)
    canvas.paste(image, offset)
    image_tensor = torch.as_tensor(np.asarray(canvas).copy(), dtype=torch.float32)
    image_tensor = image_tensor.permute(2, 0, 1)[None] / 255.0
    with torch.no_grad():
        features, objectness, roles = model.encode(image_tensor)
    seeds = propose_panel_seeds(
        objectness,
        roles,
        maximum_seeds=config.maximum_instances,
        role_names=role_names,
    )
    return PanelInstanceGrower(
        model.flood_fill_network,
        config=config,
        checkpoint_path=checkpoint_path,
    ).grow(
        features,
        seeds,
        source="trained FFN-style image panel instance growth",
    )


def rasterize_garmentcode_panels(
    payload: Mapping[str, Any],
    *,
    resolution: int = 128,
    padding: int = 8,
) -> tuple[np.ndarray, tuple[PanelSeed, ...], tuple[str, ...]]:
    """Create panel-instance teacher evidence from a structured pattern.

    This supports training and scheduler validation. It is not an image-to-panel
    prediction and is deliberately labelled as teacher evidence by callers.
    """

    panels = payload.get("pattern", {}).get("panels", {})
    if not isinstance(panels, Mapping) or not panels:
        raise ValueError("GarmentCode payload contains no panels")
    if resolution < 32 or padding < 2 or 2 * padding >= resolution:
        raise ValueError("invalid panel raster dimensions")
    names = tuple(str(name) for name in panels)
    evidence = np.zeros((len(names), resolution, resolution), dtype=np.float32)
    seeds: list[PanelSeed] = []
    for role_id, name in enumerate(names):
        vertices = np.asarray(panels[name].get("vertices", []), dtype=np.float64)
        if vertices.ndim != 2 or vertices.shape[0] < 3 or vertices.shape[1] != 2:
            raise ValueError(f"panel {name!r} has invalid vertices")
        minimum = vertices.min(axis=0)
        span = np.maximum(vertices.max(axis=0) - minimum, 1.0e-8)
        scale = (resolution - 2 * padding) / max(float(span.max()), 1.0e-8)
        points = (vertices - minimum) * scale
        points += np.array(
            [
                0.5 * (resolution - np.ptp(points[:, 0])),
                0.5 * (resolution - np.ptp(points[:, 1])),
            ]
        )
        image = Image.new("L", (resolution, resolution), 0)
        ImageDraw.Draw(image).polygon([tuple(point) for point in points], fill=255)
        mask = np.asarray(image, dtype=np.float32) / 255.0
        evidence[role_id] = mask
        ys, xs = np.nonzero(mask > 0.5)
        center = np.array([ys.mean(), xs.mean()])
        nearest = int(np.argmin((ys - center[0]) ** 2 + (xs - center[1]) ** 2))
        seeds.append(
            PanelSeed(
                seed_id=f"teacher_{name}",
                role=name,
                role_id=role_id,
                y=int(ys[nearest]),
                x=int(xs[nearest]),
                confidence=1.0,
                evidence_channel=role_id,
            )
        )
    return evidence, tuple(seeds), names


def panel_growth_loss(
    output: PanelFloodFillOutput,
    target_mask: Tensor,
    target_seam_endpoints: Tensor | None = None,
    target_stop: Tensor | None = None,
) -> Tensor:
    if target_mask.shape != output.mask_logits.shape:
        raise ValueError("target_mask must match mask logits")
    mask_loss = F.binary_cross_entropy_with_logits(output.mask_logits, target_mask)
    total = mask_loss
    if target_seam_endpoints is not None:
        if target_seam_endpoints.shape != output.seam_endpoint_logits.shape:
            raise ValueError("target_seam_endpoints must match seam logits")
        total = total + 0.5 * F.binary_cross_entropy_with_logits(
            output.seam_endpoint_logits, target_seam_endpoints
        )
    if target_stop is not None:
        if target_stop.shape != output.stop_logits.shape:
            raise ValueError("target_stop must match stop logits")
        total = total + 0.1 * F.binary_cross_entropy_with_logits(
            output.stop_logits, target_stop
        )
    return total


def _extract_patch(tensor: Tensor, center: tuple[int, int], size: int) -> Tensor:
    radius = size // 2
    padded = F.pad(tensor, (radius, radius, radius, radius), value=0.0)
    y, x = center
    return padded[..., y : y + size, x : x + size]


def _write_patch_max(target: Tensor, patch: Tensor, center: tuple[int, int]) -> None:
    size = patch.shape[-1]
    radius = size // 2
    height, width = target.shape[-2:]
    y, x = center
    y0, y1 = max(0, y - radius), min(height, y + radius + 1)
    x0, x1 = max(0, x - radius), min(width, x + radius + 1)
    py0, px0 = y0 - (y - radius), x0 - (x - radius)
    py1, px1 = py0 + (y1 - y0), px0 + (x1 - x0)
    target[..., y0:y1, x0:x1] = torch.maximum(
        target[..., y0:y1, x0:x1], patch[..., py0:py1, px0:px1]
    )


def _active_frontiers(probability: Tensor, *, threshold: float) -> dict[tuple[int, int], bool]:
    width = max(2, probability.shape[-1] // 8)
    return {
        (-1, 0): bool(torch.max(probability[:width]) >= threshold),
        (1, 0): bool(torch.max(probability[-width:]) >= threshold),
        (0, -1): bool(torch.max(probability[:, :width]) >= threshold),
        (0, 1): bool(torch.max(probability[:, -width:]) >= threshold),
    }


def _seed_connected_component(mask: np.ndarray, seed_y: int, seed_x: int) -> np.ndarray:
    components, count = label(mask, structure=np.ones((3, 3), dtype=np.int8))
    if count == 0:
        return np.zeros_like(mask, dtype=bool)
    component_id = int(components[seed_y, seed_x])
    if component_id == 0:
        nearest = np.argwhere(mask)
        if len(nearest) == 0:
            return np.zeros_like(mask, dtype=bool)
        distances = np.sum((nearest - np.array([seed_y, seed_x])) ** 2, axis=1)
        nearest_y, nearest_x = nearest[int(np.argmin(distances))]
        component_id = int(components[nearest_y, nearest_x])
    return components == component_id


def _longest_boundary(mask: np.ndarray) -> np.ndarray:
    contours = find_contours(mask.astype(np.float32), 0.5)
    if not contours:
        return np.empty((0, 2), dtype=np.float32)
    contour = max(contours, key=len)
    return contour[:, [1, 0]].astype(np.float32)
