"""Design tools exposed to the fashion-design agent.

Each tool is a plain function that takes JSON-compatible arguments and returns
a JSON-compatible dict, so it can be unit-tested directly and served over MCP
by ``fashion_agent.mcp_server``. The tools wrap the existing ``aigc_fashion``
pipeline; they never claim a learned model ran when it did not.
"""

from __future__ import annotations

from dataclasses import asdict, fields
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SESSIONS_DIR = ROOT / "outputs" / "design_sessions"
OUTFITS_DIR = ROOT / "outputs" / "outfits"
SCENES_DIR = ROOT / "web" / "scenes"
BODY_MESH = ROOT / "assets" / "body_meshes" / "anny_child_neutral.obj"

ProgressCallback = Callable[[int, int, str], None]

# Only this family has an end-to-end pattern -> drape -> fit-gate route in
# ``aigc_fashion.pipeline.run_pipeline``; other families stop at hypotheses.
SIMULATED_GARMENT_TYPES = ("t-shirt",)

INTENT_FIELDS = ("garment_type", "silhouette", "sleeve", "neckline", "fabric", "style_refs", "body_fit")
BODY_FIELDS = ("height_cm", "chest_cm", "waist_cm", "shoulder_cm")
SIZE_FIELDS = (
    "chest_circumference_cm",
    "body_length_cm",
    "shoulder_width_cm",
    "sleeve_length_cm",
    "sleeve_opening_cm",
    "neck_width_cm",
)


class ToolError(ValueError):
    """Raised for invalid tool input; reported to the agent as a tool error."""


def _intent(brief: str | None, intent: dict[str, Any] | None):
    from aigc_fashion.data import GarmentIntent
    from aigc_fashion.semantics import parse_garment_intent

    base = parse_garment_intent(prompt=brief) if brief else GarmentIntent(prompt="")
    values = asdict(base)
    for key, value in (intent or {}).items():
        if key not in INTENT_FIELDS:
            raise ToolError(f"unknown intent field {key!r}; allowed: {', '.join(INTENT_FIELDS)}")
        values[key] = tuple(value) if key == "style_refs" else value
    if not values["prompt"]:
        values["prompt"] = " ".join(str(values[k]) for k in ("silhouette", "fabric", "garment_type"))
    values["style_refs"] = tuple(values["style_refs"])
    if "body_fit" not in (intent or {}):
        values["body_fit"] = "close" if values["silhouette"] == "slim" else "loose"
    return GarmentIntent(**values)


def _avatar(body: dict[str, Any] | None):
    from aigc_fashion.data import AvatarSpec

    body = dict(body or {})
    unknown = set(body) - set(BODY_FIELDS) - {"hip_cm", "upper_arm_cm"}
    if unknown:
        raise ToolError(f"unknown body field(s) {sorted(unknown)}; allowed: {BODY_FIELDS + ('hip_cm', 'upper_arm_cm')}")
    return AvatarSpec(**{k: float(body[k]) for k in BODY_FIELDS if k in body})


def _size(size_cm: dict[str, Any] | None):
    if not size_cm:
        return None
    from aigc_fashion.data import TshirtSizeSpec

    unknown = set(size_cm) - set(SIZE_FIELDS)
    if unknown:
        raise ToolError(f"unknown size field(s) {sorted(unknown)}; allowed: {SIZE_FIELDS}")
    return TshirtSizeSpec(**{k: float(v) for k, v in size_cm.items()})


def _design_dir(design_id: str, sessions_dir: Path) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", design_id):
        raise ToolError("design_id must be 1-64 characters of letters, digits, '-' or '_'")
    return sessions_dir / design_id


def list_garment_templates() -> dict[str, Any]:
    """Pattern templates the router knows, and which ones the agent can simulate."""

    from aigc_fashion.pattern_library import load_pattern_library

    templates = []
    for template in load_pattern_library():
        templates.append(
            {
                "template_id": template.template_id,
                "garment_type": template.garment_type,
                "aliases": list(template.aliases),
                "generator_status": template.status,
                "panels": [panel.role for panel in template.panels],
                "required_measurements": list(template.required_measurements),
                "material_families": list(template.material_families),
                "end_to_end_simulation": template.garment_type in SIMULATED_GARMENT_TYPES,
            }
        )
    return {"templates": templates, "simulated_garment_types": list(SIMULATED_GARMENT_TYPES)}


def list_materials() -> dict[str, Any]:
    """Fabric keywords the solver maps to material cards, plus measured reference profiles."""

    from aigc_fashion.data import GarmentIntent
    from aigc_fashion.measured_materials import MATERIAL_PROFILES
    from aigc_fashion.patterns import material_from_intent

    cards = {}
    for fabric in ("medium cotton jersey", "light silk", "heavy denim", "light linen", "soft wool knit", "medium nylon"):
        cards[fabric] = asdict(material_from_intent(GarmentIntent(prompt=fabric, fabric=fabric)))
    profiles = {
        profile_id: {
            "construction": profile.construction,
            "composition": profile.composition,
            "evidence_status": profile.evidence_status,
            "areal_density": asdict(profile.areal_density),
        }
        for profile_id, profile in MATERIAL_PROFILES.items()
    }
    return {
        "solver_material_cards": cards,
        "measured_reference_profiles": profiles,
        "note": "Fabric keywords (silk, denim, linen, wool, nylon; otherwise cotton jersey) select the card. "
        "Cards are uncalibrated engineering priors, not swatch measurements.",
    }


def parse_design_brief(brief: str, intent: dict[str, Any] | None = None) -> dict[str, Any]:
    """Turn a brief into a structured garment intent and route it to a pattern template."""

    from aigc_fashion.pattern_library import classify_garment_intent, get_pattern_template
    from aigc_fashion.patterns import material_from_intent

    garment_intent = _intent(brief, intent)
    classification = classify_garment_intent(garment_intent.garment_type, prompt=garment_intent.prompt)
    template = get_pattern_template(classification.template_id)
    return {
        "intent": asdict(garment_intent),
        "classification": classification.to_dict(),
        "template_status": template.status,
        "material_card": asdict(material_from_intent(garment_intent)),
        "end_to_end_simulation": garment_intent.garment_type in SIMULATED_GARMENT_TYPES,
        "parser": "keyword heuristic; pass explicit `intent` fields to override anything it misread",
    }


def generate_construction_hypotheses(
    intent: dict[str, Any],
    body: dict[str, Any] | None = None,
    top_k: int = 3,
) -> dict[str, Any]:
    """Body-graded Top-K construction/fit candidates for any template in the library."""

    from aigc_fashion.construction_hypotheses import generate_physical_structure_hypotheses

    garment_intent = _intent(None, intent)
    body = dict(body or {})
    overrides = {}
    if "hip_cm" in body:
        overrides["hip_circumference"] = float(body["hip_cm"])
    if "upper_arm_cm" in body:
        overrides["upper_arm_circumference"] = float(body["upper_arm_cm"])
    hypotheses = generate_physical_structure_hypotheses(
        garment_intent,
        _avatar(body),
        measurement_overrides_cm=overrides,
        top_k=max(1, min(int(top_k), 5)),
    )
    return {"hypotheses": [item.to_dict() for item in hypotheses]}


def run_tryon(
    design_id: str,
    intent: dict[str, Any],
    body: dict[str, Any] | None = None,
    size_cm: dict[str, Any] | None = None,
    texture_image: str | None = None,
    body_mesh: str | None = None,
    body_mesh_axes: dict[str, str] | None = None,
    steps: int = 32,
    resolution: int = 19,
    notes: str = "",
    sessions_dir: Path = DEFAULT_SESSIONS_DIR,
) -> dict[str, Any]:
    """Build the pattern, dress the avatar, simulate, and run the fit gates for one design iteration."""

    from aigc_fashion.pipeline import run_pipeline

    garment_intent = _intent(None, intent)
    if garment_intent.garment_type not in SIMULATED_GARMENT_TYPES:
        raise ToolError(
            f"run_tryon only simulates {SIMULATED_GARMENT_TYPES}; {garment_intent.garment_type!r} "
            "stops at generate_construction_hypotheses in this build"
        )
    design_dir = _design_dir(design_id, sessions_dir)
    iteration = len(_iteration_dirs(design_dir)) + 1
    out_dir = design_dir / f"iter_{iteration:02d}"
    intent_path = out_dir / "intent.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    intent_path.write_text(json.dumps(asdict(garment_intent), indent=2), encoding="utf-8")

    summary = run_pipeline(
        prompt=None,
        intent_json=intent_path,
        out_dir=out_dir,
        avatar=_avatar(body),
        tshirt_size=_size(size_cm),
        texture_image=texture_image,
        fit_from_image=texture_image is not None and size_cm is None,
        body_mesh=body_mesh,
        body_mesh_source_up_axis=(body_mesh_axes or {}).get("source_up_axis", "auto"),
        body_mesh_source_front_axis=(body_mesh_axes or {}).get("source_front_axis", "auto"),
        steps=max(4, min(int(steps), 128)),
        resolution=max(11, min(int(resolution), 31)),
    )
    report = _iteration_report(iteration, summary, notes)
    (out_dir / "agent_iteration.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _iteration_report(iteration: int, summary: dict[str, Any], notes: str) -> dict[str, Any]:
    preflight = summary.get("vto_preflight") or {}
    metrics = summary.get("metrics") or {}
    key_metrics = (
        "p95_edge_stretch",
        "max_edge_stretch",
        "surface_area_ratio",
        "max_seam_gap_m",
        "avatar_collision_max_penetration_last",
    )
    return {
        "iteration": iteration,
        "notes": notes,
        "intent": summary.get("intent"),
        "pattern_cm": {
            key: round(100.0 * float(value), 1)
            for key, value in (summary.get("pattern") or {}).items()
            if key in ("body_width", "body_length", "shoulder_width", "sleeve_length", "sleeve_width", "neck_width")
        },
        "fit_validation": summary.get("fit_validation"),
        "preflight": {
            "status": preflight.get("status"),
            "failed_checks": preflight.get("failed_checks", []),
            "recommendations": preflight.get("recommendations", []),
        },
        "key_metrics": {key: metrics[key] for key in key_metrics if key in metrics},
        "artifacts": {
            key: value
            for key, value in (summary.get("artifacts") or {}).items()
            if key in ("obj", "final_png", "gif", "vto_preflight") and value
        },
        "limits": "Uncalibrated research solver; a pass means the design cleared this build's geometric gates, "
        "not that it is production-validated.",
    }


def _iteration_dirs(design_dir: Path) -> list[Path]:
    if not design_dir.exists():
        return []
    return sorted(path for path in design_dir.glob("iter_*") if path.is_dir())


def compare_iterations(design_id: str, sessions_dir: Path = DEFAULT_SESSIONS_DIR) -> dict[str, Any]:
    """Side-by-side gate status and metrics of every iteration recorded for a design."""

    design_dir = _design_dir(design_id, sessions_dir)
    rows = []
    for path in _iteration_dirs(design_dir):
        report_path = path / "agent_iteration.json"
        if not report_path.exists():
            rows.append({"iteration_dir": str(path), "status": "incomplete"})
            continue
        report = json.loads(report_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "iteration": report["iteration"],
                "notes": report.get("notes", ""),
                "intent": {k: (report.get("intent") or {}).get(k) for k in ("silhouette", "fabric", "sleeve", "neckline")},
                "pattern_cm": report.get("pattern_cm"),
                "fit_status": (report.get("fit_validation") or {}).get("status"),
                "fit_failed": (report.get("fit_validation") or {}).get("failed_checks", []),
                "preflight_status": report["preflight"]["status"],
                "preflight_failed": report["preflight"]["failed_checks"],
                "key_metrics": report.get("key_metrics"),
            }
        )
    if not rows:
        raise ToolError(f"no iterations recorded for design {design_id!r}")
    return {"design_id": design_id, "iterations": rows}


def _run_step(command: list[str], log_path: Path) -> None:
    """Run one physics stage as a subprocess; raise with the log tail on failure."""

    env = dict(os.environ)
    env.setdefault("ANNY_CACHE_DIR", str(ROOT / "work" / "anny_cache"))
    env.setdefault("WARP_CACHE_PATH", str(ROOT / "work" / "warp_cache"))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    if completed.returncode != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace").strip().splitlines()[-12:]
        raise ToolError(f"stage failed ({' '.join(command[1:3])}); log {log_path}:\n" + "\n".join(tail))


def list_outfit_scenes() -> dict[str, Any]:
    """Outfit scenes that the Three.js WebGPU viewer can play."""

    scenes = []
    for manifest_path in sorted(SCENES_DIR.glob("*/scene.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        scenes.append(
            {
                "scene": manifest_path.parent.name,
                "frames": manifest.get("frameCount"),
                "physical_duration_s": manifest.get("physicalDurationSeconds"),
                "physics_audit": manifest.get("physicsAudit"),
                "viewer_url": f"anny-cloth-webgpu.html?scene={manifest_path.parent.name}",
            }
        )
    return {"scenes": scenes}


def simulate_outfit(
    outfit_id: str,
    shirt_image: str,
    skirt_image: str,
    quality: str = "preview",
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Simulate a T-shirt + pleated-skirt outfit on the Anny body and export a WebGPU scene.

    Stages (each one an existing repository runner): image-fitted T-shirt pattern ->
    implicit-shell T-shirt dynamics -> pleated-skirt dynamics -> unified two-garment
    4D scene with layered contact -> Three.js binary scene in web/scenes/<outfit_id>.
    """

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", outfit_id):
        raise ToolError("outfit_id must be 1-64 characters of letters, digits, '-' or '_'")
    if quality not in ("preview", "full"):
        raise ToolError("quality must be 'preview' or 'full'")
    images = {"shirt_image": Path(shirt_image), "skirt_image": Path(skirt_image)}
    for name, path in images.items():
        if not (path if path.is_absolute() else ROOT / path).is_file():
            raise ToolError(f"{name} not found: {path}")
    python = sys.executable
    out = OUTFITS_DIR / outfit_id
    logs = out / "logs"
    preview = quality == "preview"
    shirt_steps = ["--training-steps", "60", "--minimum-training-steps", "30"] if preview else []
    skirt_steps = (
        ["--teacher-steps", "8", "--training-steps", "40", "--minimum-training-steps", "20"] if preview else []
    )
    body = ["--body-mesh", str(BODY_MESH)]
    stages: list[tuple[str, list[str]]] = []
    if not BODY_MESH.is_file():
        stages.append(
            (
                "export Anny child body mesh",
                [python, "scripts/export_anny_body_obj.py", "--output", str(BODY_MESH), "--height-cm", "128", "--gender", "0.5"],
            )
        )
    stages += [
        (
            "fit T-shirt pattern to image and body",
            [python, "examples/run_mvp.py", "--texture-image", shirt_image, "--fit-from-image", *body,
             "--body-mesh-source-up-axis", "z", "--body-mesh-source-front-axis=-y", "--steps", "16",
             # The Anny child body the later stages simulate on; without these the
             # image fit sizes the shirt for the default adult avatar.
             "--avatar-chest-cm", "64", "--avatar-waist-cm", "56", "--avatar-shoulder-cm", "30",
             "--avatar-height-cm", "128",
             "--out-dir", str(out / "pattern")],
        ),
        (
            "simulate implicit-shell T-shirt",
            [python, "examples/run_anny_cloth4d.py", "--texture-image", shirt_image, *body,
             "--pattern-spec", str(out / "pattern" / "image_conditioned_tshirt_spec.json"),
             "--implicit-shell-target-faces", "700", *shirt_steps, "--out-dir", str(out / "tshirt")],
        ),
        (
            "simulate pleated skirt",
            [python, "examples/run_anny_pleated_skirt.py", "--image", skirt_image, *body,
             "--shell-reconstruction", "direct", *skirt_steps, "--out-dir", str(out / "skirt")],
        ),
    ]
    layered = [
        (
            "simulate both garments in one layered 4D scene",
            [python, "examples/run_unified_anny_cloth4d.py", *body, "--shirt-texture", shirt_image,
             "--skirt-image", skirt_image, "--shirt-run", str(out / "tshirt"), "--skirt-run", str(out / "skirt"),
             "--frames", "9" if preview else "25", "--out-dir", str(out / "unified")],
        ),
        (
            "export Three.js WebGPU scene",
            [python, "scripts/export_threejs_scene.py", *body, "--run-dir", str(out / "unified"),
             "--shirt-field-run", str(out / "unified"), "--shirt-mesh-run", str(out / "tshirt"),
             "--skirt-run", str(out / "skirt"), "--skirt-image", skirt_image, "--shirt-texture", shirt_image,
             "--output-dir", str(SCENES_DIR / outfit_id)],
        ),
    ]
    separate = [
        (
            "export Three.js WebGPU scene (garments do not overlap)",
            [python, "scripts/export_garment_runs_scene.py", "--shirt-run", str(out / "tshirt"),
             "--skirt-run", str(out / "skirt"), "--shirt-texture", shirt_image, "--skirt-image", skirt_image,
             *body, "--output-dir", str(SCENES_DIR / outfit_id)],
        ),
    ]
    total = len(stages) + 1
    for index, (label, command) in enumerate(stages):
        if progress is not None:
            progress(index, total, label)
        _run_step(command, logs / f"{index:02d}.log")
    # Overlapping layers need the joint contact solver; a cropped top that ends
    # above the waistband can be exported from the two garment runs directly.
    import numpy as np

    shirt_low = float(np.load(out / "tshirt" / "neuralclothsim4d_physical_rollout.npy")[:, :, 1].min())
    skirt_high = float(np.load(out / "skirt" / "neuralclothsim4d_physical_rollout.npy")[:, :, 1].max())
    layer_gap = shirt_low - skirt_high
    final = separate if layer_gap >= 0.01 else layered
    total = len(stages) + len(final)
    for offset, (label, command) in enumerate(final):
        index = len(stages) + offset
        if progress is not None:
            progress(index, total, label)
        _run_step(command, logs / f"{index:02d}.log")
    if progress is not None:
        progress(total, total, "done")
    manifest = json.loads((SCENES_DIR / outfit_id / "scene.json").read_text(encoding="utf-8"))
    return {
        "outfit_id": outfit_id,
        "scene": outfit_id,
        "quality": quality,
        "frames": manifest.get("frameCount"),
        "physical_duration_s": manifest.get("physicalDurationSeconds"),
        "physics_audit": manifest.get("physicsAudit"),
        "layering": "separate garment runs (no overlap)" if final is separate else "joint layered contact solve",
        "layer_gap_m": layer_gap,
        "run_dir": str(out),
        "viewer_url": f"anny-cloth-webgpu.html?scene={outfit_id}",
        "limits": "Uncalibrated research solver on a single Anny child body; product images give garment family, "
        "silhouette and appearance, while metric scale comes from the body.",
    }


def render_outfit(
    scene: str,
    yaw_degrees: list[float] | None = None,
    phase: float = 0.5,
) -> dict[str, Any]:
    """Render an outfit scene with the Three.js WebGPU viewer in headless Chromium."""

    if not (SCENES_DIR / scene / "scene.json").is_file():
        raise ToolError(f"unknown scene {scene!r}; call list_outfit_scenes")
    node = shutil.which("node")
    if node is None:
        raise ToolError("Node.js is required for render_outfit (scripts/capture_webgpu.mjs)")
    if not (ROOT / "node_modules" / "playwright-core").is_dir():
        raise ToolError("run `npm install` at the repository root first (playwright-core)")
    yaws = yaw_degrees or [0, 35]
    command = [
        node, "scripts/capture_webgpu.mjs", "--scene", scene, "--out", "outputs/renders",
        "--phases", str(float(phase)), "--yaws", ",".join(str(float(y)).rstrip("0").rstrip(".") for y in yaws),
    ]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=900)
    if completed.returncode != 0:
        raise ToolError(f"WebGPU capture failed: {completed.stderr.strip()[-1500:]}")
    result = json.loads(completed.stdout[completed.stdout.index("{"):])
    result["artifacts"] = {"frames": [str(ROOT / frame) for frame in result["frames"]]}
    return result


def reconstruct_body(
    body_id: str,
    image: str | None = None,
    prediction_json: str | None = None,
    person_index: int | None = None,
) -> dict[str, Any]:
    """Photo -> SAM 3D Body -> MHR A-pose body mesh + measurements for try-on."""

    from aigc_fashion.body_reconstruction import reconstruct_body as _reconstruct

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", body_id):
        raise ToolError("body_id must be 1-64 characters of letters, digits, '-' or '_'")
    if (image is None) == (prediction_json is None):
        raise ToolError("pass exactly one of image or prediction_json")
    for value in (image, prediction_json):
        if value is not None and not (Path(value) if Path(value).is_absolute() else ROOT / value).is_file():
            raise ToolError(f"file not found: {value}")
    try:
        result = _reconstruct(
            ROOT / "outputs" / "bodies" / body_id,
            image=Path(image) if image else None,
            prediction_json=Path(prediction_json) if prediction_json else None,
            person_index=person_index,
        )
    except RuntimeError as error:
        raise ToolError(str(error)) from error
    report = result.to_dict()
    report["artifacts"] = {"final_png": result.preview_png}
    report["use_in_run_tryon"] = {
        "body": report["avatar"],
        "body_mesh": result.body_mesh,
        "body_mesh_axes": report["body_mesh_axes"],
    }
    report["limits"] = (
        "Single-view estimate: metric scale comes from SAM 3D Body's priors, clothing hides the true surface, "
        "and girths are measured on the parametric MHR surface. Confirm key measurements with the user."
    )
    return report


_INTENT_SCHEMA = {
    "type": "object",
    "description": "Structured garment intent.",
    "properties": {
        "garment_type": {"type": "string", "description": "e.g. t-shirt, skirt, dress, hoodie (see list_garment_templates)"},
        "silhouette": {"type": "string", "enum": ["slim", "relaxed", "oversized"]},
        "sleeve": {"type": "string", "enum": ["sleeveless", "short", "long"]},
        "neckline": {"type": "string", "enum": ["crew", "v-neck", "boat"]},
        "fabric": {"type": "string", "description": "e.g. medium cotton jersey, light silk, heavy denim"},
        "style_refs": {"type": "array", "items": {"type": "string"}},
        "body_fit": {"type": "string", "enum": ["close", "loose"]},
    },
    "additionalProperties": False,
}
_BODY_SCHEMA = {
    "type": "object",
    "description": "Body measurements in cm; omitted values fall back to an adult default avatar.",
    "properties": {name: {"type": "number"} for name in (*BODY_FIELDS, "hip_cm", "upper_arm_cm")},
    "additionalProperties": False,
}
_SIZE_SCHEMA = {
    "type": "object",
    "description": "Finished T-shirt garment measurements in cm. Omit to let the body grade the size.",
    "properties": {name: {"type": "number"} for name in SIZE_FIELDS},
    "additionalProperties": False,
}

TOOLS: dict[str, tuple[Callable[..., dict[str, Any]], str, dict[str, Any]]] = {
    "list_garment_templates": (
        list_garment_templates,
        "List garment pattern templates, their panels and required measurements, and which garment types "
        "can be simulated end to end.",
        {"type": "object", "properties": {}},
    ),
    "list_materials": (
        list_materials,
        "List fabric keywords the solver understands and the measured reference fabric profiles.",
        {"type": "object", "properties": {}},
    ),
    "parse_design_brief": (
        parse_design_brief,
        "Convert a natural-language design brief into a structured garment intent, routed template and "
        "material card. Pass `intent` fields to correct anything the keyword parser got wrong.",
        {
            "type": "object",
            "properties": {"brief": {"type": "string"}, "intent": _INTENT_SCHEMA},
            "required": ["brief"],
        },
    ),
    "generate_construction_hypotheses": (
        generate_construction_hypotheses,
        "Grade the garment's construction template against body measurements into Top-K fit candidates "
        "(panels, seams, openings, graded parameters, warnings). Works for every template.",
        {
            "type": "object",
            "properties": {
                "intent": _INTENT_SCHEMA,
                "body": _BODY_SCHEMA,
                "top_k": {"type": "integer", "minimum": 1, "maximum": 5, "default": 3},
            },
            "required": ["intent"],
        },
    ),
    "run_tryon": (
        run_tryon,
        "Run one design iteration: build the pattern, dress the avatar, simulate the drape and run the "
        "preflight and fit gates. Saves artifacts under outputs/design_sessions/<design_id>/iter_NN and "
        "returns gate results, failed checks, recommendations and the preview image. T-shirts only. "
        "Takes roughly 10-60 s on CPU.",
        {
            "type": "object",
            "properties": {
                "design_id": {"type": "string", "description": "Stable id for this design; iterations accumulate under it."},
                "intent": _INTENT_SCHEMA,
                "body": _BODY_SCHEMA,
                "size_cm": _SIZE_SCHEMA,
                "texture_image": {"type": "string", "description": "Optional path to a garment/print image."},
                "body_mesh": {"type": "string", "description": "Optional path to an upright body OBJ (e.g. from reconstruct_body)."},
                "body_mesh_axes": {
                    "type": "object",
                    "description": "Axes of body_mesh, e.g. reconstruct_body's {source_up_axis: y, source_front_axis: z}.",
                    "properties": {"source_up_axis": {"type": "string"}, "source_front_axis": {"type": "string"}},
                },
                "steps": {"type": "integer", "minimum": 4, "maximum": 128, "default": 32},
                "resolution": {"type": "integer", "minimum": 11, "maximum": 31, "default": 19},
                "notes": {"type": "string", "description": "What changed in this iteration and why."},
            },
            "required": ["design_id", "intent"],
        },
    ),
    "compare_iterations": (
        compare_iterations,
        "Compare gate status, pattern measurements and metrics across all iterations of a design.",
        {
            "type": "object",
            "properties": {"design_id": {"type": "string"}},
            "required": ["design_id"],
        },
    ),
}


TOOLS.update(
    {
        "list_outfit_scenes": (
            list_outfit_scenes,
            "List outfit scenes (web/scenes/*) the Three.js WebGPU viewer can play, with their physics audit.",
            {"type": "object", "properties": {}},
        ),
        "simulate_outfit": (
            simulate_outfit,
            "Simulate a T-shirt + pleated-skirt outfit on the Anny child body from two product images and export "
            "a Three.js WebGPU scene named after outfit_id. Runs the full physics chain (pattern fit, implicit-shell "
            "T-shirt, pleated skirt; overlapping garments then go through the joint layered contact solver). About "
            "30-45 min on a 4-core CPU for 'preview' when the top ends above the waistband; the layered solver can "
            "take hours. Reports progress per stage.",
            {
                "type": "object",
                "properties": {
                    "outfit_id": {"type": "string"},
                    "shirt_image": {"type": "string", "description": "T-shirt product image (front print is extracted)."},
                    "skirt_image": {"type": "string", "description": "Pleated-skirt product image (plain background)."},
                    "quality": {"type": "string", "enum": ["preview", "full"], "default": "preview"},
                },
                "required": ["outfit_id", "shirt_image", "skirt_image"],
            },
        ),
        "reconstruct_body": (
            reconstruct_body,
            "Reconstruct a person's body from one photo with Meta SAM 3D Body + the MHR parametric body model: "
            "returns an A-pose body mesh (OBJ, metres, +Y up, +Z front), estimated height/chest/waist/hip/"
            "shoulder in cm, a preview image, and ready-to-use run_tryon arguments. Pass `image` (needs the "
            "SAM 3D Body environment: SAM3D_PYTHON, SAM3D_REPO, SAM3D_CHECKPOINT) or `prediction_json` "
            "(a saved SAM 3D Body output).",
            {
                "type": "object",
                "properties": {
                    "body_id": {"type": "string"},
                    "image": {"type": "string", "description": "Photo of the person (full body works best)."},
                    "prediction_json": {"type": "string", "description": "Saved output of scripts/sam3d_body_infer.py."},
                    "person_index": {"type": "integer", "description": "Which detected person; default the largest."},
                },
                "required": ["body_id"],
            },
        ),
        "render_outfit": (
            render_outfit,
            "Render an outfit scene with Three.js on the WebGPU backend (headless Chromium; hardware GPU, else "
            "SwiftShader). Fails rather than falling back to WebGL. Returns the frames as images.",
            {
                "type": "object",
                "properties": {
                    "scene": {"type": "string"},
                    "yaw_degrees": {"type": "array", "items": {"type": "number"}, "description": "Camera orbit angles."},
                    "phase": {"type": "number", "minimum": 0, "maximum": 1, "default": 0.5},
                },
                "required": ["scene"],
            },
        ),
    }
)


def call_tool(
    name: str,
    arguments: dict[str, Any] | None,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    if name not in TOOLS:
        raise ToolError(f"unknown tool {name!r}")
    function = TOOLS[name][0]
    arguments = dict(arguments or {})
    if progress is not None and name == "simulate_outfit":
        arguments["progress"] = progress
    return function(**arguments)


def tool_field_names() -> dict[str, list[str]]:
    """Field names of the pipeline dataclasses the schemas mirror (used by tests)."""

    from aigc_fashion.data import AvatarSpec, GarmentIntent, TshirtSizeSpec

    return {
        "intent": [f.name for f in fields(GarmentIntent)],
        "body": [f.name for f in fields(AvatarSpec)],
        "size": [f.name for f in fields(TshirtSizeSpec)],
    }
