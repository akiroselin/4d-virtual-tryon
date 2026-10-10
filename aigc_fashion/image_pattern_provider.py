from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import subprocess
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Design2GarmentCodeStatus:
    repository: str
    commit: str | None
    source_available: bool
    image_entrypoint_available: bool
    parameter_projector_available: bool
    api_credentials_available: bool
    executable_here: bool
    reason: str
    fallback: str = "body-conditioned parametric garment-family prior"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PatternValidationReport:
    valid: bool
    panel_count: int
    stitch_count: int
    closed_panels: int
    maximum_stitch_length_mismatch_ratio: float
    errors: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def probe_design2garmentcode(repo: str | Path) -> Design2GarmentCodeStatus:
    """Report whether the official image-to-GarmentCode path can run locally.

    The official repository contains a complete image entrypoint, but inference
    also needs its Qwen parameter-projector weights and an explicitly supplied
    multimodal API credential. Bundled example credentials are never trusted.
    """

    root = Path(repo).expanduser().resolve()
    source = (root / "lmm_utils" / "predict_garmentcode_picture.py").exists()
    entrypoint = (root / "lmm_utils" / "test_picture_batch.py").exists()
    projector_candidates = (
        root / "lmm_utils" / "Qwen" / "qwen2vl_lora_mlp",
        root / "lmm_utils" / "Qwen" / "Qwen2-VL-2B-Instruct",
    )
    projector = all(path.exists() and any(path.iterdir()) for path in projector_candidates)
    credentials = bool(os.environ.get("OPENAI_API_KEY") or os.environ.get("MODELSCOPE_API_KEY"))
    commit: str | None = None
    if (root / ".git").exists():
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        commit = result.stdout.strip() or None if result.returncode == 0 else None
    executable = source and entrypoint and projector and credentials
    missing: list[str] = []
    if not source or not entrypoint:
        missing.append("official image inference source")
    if not projector:
        missing.append("Qwen base/LoRA parameter-projector weights")
    if not credentials:
        missing.append("user-supplied multimodal API credentials")
    reason = "ready" if not missing else "missing " + ", ".join(missing)
    return Design2GarmentCodeStatus(
        repository=str(root),
        commit=commit,
        source_available=source,
        image_entrypoint_available=entrypoint,
        parameter_projector_available=projector,
        api_credentials_available=credentials,
        executable_here=executable,
        reason=reason,
    )


def load_and_validate_garmentcode_pattern(
    path: str | Path,
) -> tuple[dict[str, Any], PatternValidationReport]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return payload, validate_garmentcode_pattern(payload)


def validate_garmentcode_pattern(payload: dict[str, Any]) -> PatternValidationReport:
    """Validate the structural subset shared by GarmentCode and model outputs."""

    errors: list[str] = []
    pattern = payload.get("pattern")
    if not isinstance(pattern, dict):
        return PatternValidationReport(False, 0, 0, 0, 0.0, ("missing pattern object",))
    panels = pattern.get("panels", {})
    stitches = pattern.get("stitches", [])
    if not isinstance(panels, dict) or not panels:
        errors.append("pattern must contain at least one panel")
        panels = {}
    if not isinstance(stitches, list):
        errors.append("stitches must be a list")
        stitches = []

    closed_panels = 0
    edge_lengths: dict[tuple[str, int], float] = {}
    units_in_meter = float(payload.get("properties", {}).get("units_in_meter", 100.0))
    for name, panel in panels.items():
        vertices = np.asarray(panel.get("vertices", []), dtype=np.float64)
        edges = panel.get("edges", [])
        if vertices.ndim != 2 or vertices.shape[0] < 3 or vertices.shape[1] != 2:
            errors.append(f"panel {name} has invalid vertices")
            continue
        degree = np.zeros(len(vertices), dtype=np.int32)
        for edge_id, edge in enumerate(edges):
            endpoints = edge.get("endpoints", [])
            if len(endpoints) != 2 or any(not 0 <= int(i) < len(vertices) for i in endpoints):
                errors.append(f"panel {name} edge {edge_id} has invalid endpoints")
                continue
            a, b = (int(endpoints[0]), int(endpoints[1]))
            degree[[a, b]] += 1
            edge_lengths[(name, edge_id)] = float(
                np.linalg.norm(vertices[b] - vertices[a]) / units_in_meter
            )
        if len(edges) == len(vertices) and np.all(degree == 2):
            closed_panels += 1
        else:
            errors.append(f"panel {name} boundary is not one closed edge loop")

    mismatches: list[float] = []
    for stitch_id, stitch in enumerate(stitches):
        if not isinstance(stitch, list) or len(stitch) != 2:
            errors.append(f"stitch {stitch_id} must pair exactly two edges")
            continue
        keys = tuple((side.get("panel"), side.get("edge")) for side in stitch)
        if any(key not in edge_lengths for key in keys):
            errors.append(f"stitch {stitch_id} references a missing edge")
            continue
        first, second = (edge_lengths[keys[0]], edge_lengths[keys[1]])
        mismatches.append(abs(first - second) / max(first, second, 1.0e-8))

    max_mismatch = max(mismatches, default=0.0)
    if max_mismatch > 0.02:
        errors.append("one or more stitched edge pairs differ in length by over 2%")
    return PatternValidationReport(
        valid=not errors,
        panel_count=len(panels),
        stitch_count=len(stitches),
        closed_panels=closed_panels,
        maximum_stitch_length_mismatch_ratio=float(max_mismatch),
        errors=tuple(errors),
    )
