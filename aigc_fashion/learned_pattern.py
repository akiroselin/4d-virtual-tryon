from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

from .data import TshirtSizeSpec


@dataclass(frozen=True)
class LearnedPatternResult:
    status: str
    method: str
    message: str
    repo_path: str | None = None
    checkpoint_path: str | None = None
    output_dir: str | None = None
    pattern_spec_path: str | None = None
    pattern_preview_path: str | None = None
    estimated_size: TshirtSizeSpec | None = None

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["estimated_size"] = self.estimated_size.to_dict() if self.estimated_size is not None else None
        return data


def reconstruct_tshirt_pattern_from_image(
    image_path: str | Path,
    work_dir: str | Path,
    repo_path: str | Path = "work/sewformer",
    checkpoint_path: str | Path | None = None,
    python_executable: str | Path | None = None,
) -> LearnedPatternResult:
    """Run SewFormer when its official repo and checkpoint are present.

    The function is intentionally conservative: if learned inference cannot run,
    it returns an explicit unavailable result instead of silently pretending that
    the image-fit fallback is learned reconstruction.
    """

    image_path = Path(image_path).expanduser().resolve()
    work_dir = Path(work_dir).expanduser().resolve()
    repo_path = Path(repo_path).expanduser().resolve()
    sewformer_dir = repo_path / "Sewformer"
    default_ckpt = sewformer_dir / "assets" / "ckpts" / "Detr2d-V6-final-dif-ce-focal-schd-agp_checkpoint_37.pth"
    checkpoint = Path(checkpoint_path).expanduser().resolve() if checkpoint_path is not None else default_ckpt.resolve()

    if not image_path.exists():
        return _unavailable(repo_path, checkpoint, f"image not found: {image_path}")
    if not (sewformer_dir / "inference.py").exists():
        return _unavailable(repo_path, checkpoint, f"SewFormer repo not found: {sewformer_dir}")
    if not checkpoint.exists():
        return _unavailable(
            repo_path,
            checkpoint,
            "SewFormer checkpoint is missing; download liulj/sewformer checkpoint into Sewformer/assets/ckpts or pass --sewformer-checkpoint.",
        )

    input_dir = work_dir / "learned_pattern_input"
    output_dir = work_dir / "learned_pattern_output"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    staged_image = input_dir / image_path.name
    if image_path.resolve() != staged_image.resolve():
        shutil.copy2(image_path, staged_image)

    config_path = sewformer_dir / "configs" / "test.yaml"
    run_config = config_path
    if checkpoint.resolve() != default_ckpt.resolve():
        run_config = work_dir / "sewformer_test_override.yaml"
        text = config_path.read_text(encoding="utf-8")
        text = text.replace(
            "pre-trained: ./assets/ckpts/Detr2d-V6-final-dif-ce-focal-schd-agp_checkpoint_37.pth",
            f"pre-trained: {checkpoint}",
        )
        run_config.write_text(text, encoding="utf-8")

    command = [
        str(python_executable or sys.executable),
        "inference.py",
        "-c",
        str(run_config),
        "-d",
        str(input_dir),
        "-t",
        "real",
        "-o",
        str(output_dir),
    ]
    env = os.environ.copy()
    compatibility_paths = [
        Path("/opt/homebrew/anaconda3/pkgs/numpy-base-1.26.4-py312he047099_0/lib/python3.12/site-packages"),
        Path("/opt/homebrew/anaconda3/pkgs/protobuf-4.25.3-py312h8472c4a_0/lib/python3.12/site-packages"),
    ]
    existing_pythonpath = env.get("PYTHONPATH", "")
    python_paths = [str(path) for path in compatibility_paths if path.exists()]
    if existing_pythonpath:
        python_paths.append(existing_pythonpath)
    if python_paths:
        env["PYTHONPATH"] = os.pathsep.join(python_paths)
    env.setdefault("MPLCONFIGDIR", str(work_dir / ".matplotlib"))
    env.setdefault("XDG_CACHE_HOME", str(work_dir / ".cache"))
    env.setdefault("WANDB_MODE", "disabled")

    completed = subprocess.run(
        command,
        cwd=sewformer_dir,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        return LearnedPatternResult(
            status="failed",
            method="SewFormer single-image sewing-pattern reconstruction",
            message=(completed.stderr or completed.stdout).strip()[-900:],
            repo_path=str(repo_path),
            checkpoint_path=str(checkpoint),
            output_dir=str(output_dir),
        )

    spec_paths = sorted(output_dir.glob("**/*_predicted_single_specification.json"))
    preview_paths = sorted(output_dir.glob("**/*_predicted__pattern.png"))
    if not spec_paths:
        return LearnedPatternResult(
            status="failed",
            method="SewFormer single-image sewing-pattern reconstruction",
            message="SewFormer completed but did not emit a predicted sewing-pattern JSON.",
            repo_path=str(repo_path),
            checkpoint_path=str(checkpoint),
            output_dir=str(output_dir),
        )

    size = estimate_tshirt_size_from_sewformer_spec(spec_paths[0])
    return LearnedPatternResult(
        status="ok",
        method="SewFormer single-image sewing-pattern reconstruction",
        message="learned sewing pattern reconstructed from image",
        repo_path=str(repo_path),
        checkpoint_path=str(checkpoint),
        output_dir=str(output_dir),
        pattern_spec_path=str(spec_paths[0]),
        pattern_preview_path=str(preview_paths[0]) if preview_paths else None,
        estimated_size=size,
    )


def estimate_tshirt_size_from_sewformer_spec(spec_path: str | Path) -> TshirtSizeSpec | None:
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    panels = spec.get("pattern", {}).get("panels", {})
    units_in_meter = float(spec.get("properties", {}).get("units_in_meter", 100.0))
    panel_spans: list[tuple[float, float, float, str]] = []
    for name, panel in panels.items():
        points = np.asarray(panel.get("vertices", []), dtype=np.float32)
        if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] < 2:
            continue
        span = (points[:, :2].max(axis=0) - points[:, :2].min(axis=0)) / units_in_meter
        first = points[1, :2] - points[0, :2]
        second = points[2, :2] - points[0, :2]
        area = float(abs(first[0] * second[1] - first[1] * second[0]))
        panel_spans.append((float(span[0]), float(span[1]), area, str(name)))
    if len(panel_spans) < 2:
        return None

    body_panels = sorted(panel_spans, key=lambda item: item[2], reverse=True)[:2]
    flat_width_m = float(np.median([item[0] for item in body_panels]))
    flat_length_m = float(np.median([item[1] for item in body_panels]))
    if flat_width_m <= 0.0 or flat_length_m <= 0.0:
        return None

    sleeve_panels = [item for item in panel_spans if item not in body_panels]
    sleeve_length_m = max((max(item[0], item[1]) for item in sleeve_panels), default=flat_width_m * 0.32)
    sleeve_opening_m = max((min(item[0], item[1]) for item in sleeve_panels), default=0.18)
    shoulder_cm = min(56.0, max(20.0, flat_width_m * 100.0 * 0.78))
    chest_cm = min(132.0, max(50.0, flat_width_m * 200.0))
    body_length_cm = min(88.0, max(30.0, flat_length_m * 100.0))
    sleeve_length_cm = min(40.0, max(8.0, sleeve_length_m * 100.0))
    return TshirtSizeSpec(
        chest_circumference_cm=chest_cm,
        body_length_cm=body_length_cm,
        shoulder_width_cm=shoulder_cm,
        sleeve_length_cm=sleeve_length_cm,
        sleeve_opening_cm=min(54.0, max(14.0, sleeve_opening_m * 100.0)),
        neck_width_cm=min(24.0, max(12.0, flat_width_m * 100.0 * 0.38)),
    )


def _unavailable(repo_path: Path, checkpoint: Path, message: str) -> LearnedPatternResult:
    return LearnedPatternResult(
        status="unavailable",
        method="SewFormer single-image sewing-pattern reconstruction",
        message=message,
        repo_path=str(repo_path),
        checkpoint_path=str(checkpoint),
    )
