from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .avatar import SmplAvatarProxy
from .data import AvatarSpec, GarmentMesh, TshirtSizeSpec
from .image_fit import ImageTshirtFit, infer_tshirt_size_from_image
from .learned_panel_mesh import garment_mesh_from_sewformer_spec
from .learned_pattern import (
    LearnedPatternResult,
    estimate_tshirt_size_from_sewformer_spec,
    reconstruct_tshirt_pattern_from_image,
)
from .patterns import build_tshirt_pattern, export_obj, generate_tshirt_mesh
from .pattern_program import write_image_conditioned_tshirt_spec
from .panel_instance_grower import PanelGrowthResult, reconstruct_panels_from_image_ffn
from .pattern_library import classify_garment_image, classify_garment_intent, get_pattern_template
from .pattern_validation import PatternValidationReport, validate_tshirt_pattern_spec
from .physics import SimulationConfig, simulate_intrinsic_cloth
from .render import render_frame, render_rollout
from .semantics import parse_garment_intent
from .body_mesh_avatar import BodyMeshAvatar
from .vto_preflight import require_preflight_pass, run_vto_preflight


def run_pipeline(
    prompt: str | None,
    out_dir: Path,
    avatar: AvatarSpec | None = None,
    steps: int = 64,
    resolution: int = 23,
    intent_json: str | Path | None = None,
    dress_on_avatar: bool = True,
    tshirt_size: TshirtSizeSpec | None = None,
    texture_image: str | Path | None = None,
    fit_from_image: bool = False,
    learned_pattern: bool = False,
    learned_pattern_spec: str | Path | None = None,
    sewformer_repo: str | Path = "work/sewformer",
    sewformer_checkpoint: str | Path | None = None,
    panel_growth_checkpoint: str | Path | None = None,
    body_mesh: str | Path | None = None,
    body_mesh_source_x_axis: str = "x",
    body_mesh_source_up_axis: str = "auto",
    body_mesh_source_front_axis: str = "auto",
    body_mesh_scale: float | None = None,
    smpl_mesh: str | Path | None = None,
    smpl_source_x_axis: str = "x",
    smpl_source_up_axis: str = "y",
    smpl_source_front_axis: str = "z",
    smpl_scale: float | None = None,
    strict_preflight: bool = False,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    if texture_image is not None and not Path(texture_image).exists():
        raise FileNotFoundError(f"Texture image not found: {texture_image}")
    avatar = avatar or AvatarSpec()
    user_supplied_tshirt_size = tshirt_size is not None
    intent = parse_garment_intent(prompt=prompt, intent_json=intent_json)

    if intent.garment_type != "t-shirt":
        # The MVP currently normalizes all garment requests to a T-shirt-like
        # template so the downstream pattern and physics path is fully runnable.
        intent = parse_garment_intent(prompt=f"{intent.prompt} t-shirt")

    if texture_image is not None:
        garment_classification = classify_garment_image(texture_image, prompt=intent.prompt)
    else:
        garment_classification = classify_garment_intent(intent.garment_type, prompt=intent.prompt)
    pattern_template = get_pattern_template(garment_classification.template_id)
    panel_growth_result: PanelGrowthResult | None = None
    panel_growth_artifacts: dict[str, str] = {}
    if panel_growth_checkpoint is not None:
        if texture_image is None:
            raise ValueError("panel_growth_checkpoint requires texture_image")
        panel_growth_result = reconstruct_panels_from_image_ffn(
            texture_image,
            panel_growth_checkpoint,
        )
        panel_growth_artifacts = _write_panel_growth_artifacts(
            out_dir,
            panel_growth_result,
        )

    selected_body_mesh = body_mesh or smpl_mesh
    selected_x_axis = body_mesh_source_x_axis if body_mesh is not None else smpl_source_x_axis
    selected_up_axis = body_mesh_source_up_axis if body_mesh is not None else smpl_source_up_axis
    selected_front_axis = body_mesh_source_front_axis if body_mesh is not None else smpl_source_front_axis
    selected_scale = body_mesh_scale if body_mesh is not None else smpl_scale
    if selected_body_mesh is not None:
        avatar_proxy = BodyMeshAvatar.from_obj(
            selected_body_mesh,
            avatar,
            source_x_axis=selected_x_axis,
            source_up_axis=selected_up_axis,
            source_front_axis=selected_front_axis,
            scale=selected_scale,
        )
    else:
        avatar_proxy = SmplAvatarProxy(avatar)
    fit_constraints = (
        avatar_proxy.tshirt_fit_constraints()
        if hasattr(avatar_proxy, "tshirt_fit_constraints")
        else None
    )
    size_adjustments: dict[str, dict[str, float]] = {}

    image_fit: ImageTshirtFit | None = None
    learned_pattern_result: LearnedPatternResult | None = None
    pattern_validation: PatternValidationReport | None = None
    selected_pattern_validation: PatternValidationReport | None = None
    selected_pattern_spec: Path | None = None
    learned_mesh_status: dict[str, Any] = {
        "source": "parametric_fallback",
        "status": "not_requested",
        "message": "using parametric/image-fit T-shirt mesh",
    }
    if learned_pattern_spec is not None:
        spec_path = Path(learned_pattern_spec)
        learned_pattern_result = LearnedPatternResult(
            status="ok" if spec_path.exists() else "unavailable",
            method="provided SewFormer/GarmentCode sewing-pattern specification",
            message="using provided learned sewing-pattern JSON" if spec_path.exists() else f"spec not found: {spec_path}",
            pattern_spec_path=str(spec_path) if spec_path.exists() else None,
            estimated_size=estimate_tshirt_size_from_sewformer_spec(spec_path) if spec_path.exists() else None,
        )
        if learned_pattern_result.estimated_size is not None and tshirt_size is None:
            pattern_validation = validate_tshirt_pattern_spec(spec_path)
            if pattern_validation.valid:
                tshirt_size = learned_pattern_result.estimated_size
    elif learned_pattern and texture_image is not None:
        learned_pattern_result = reconstruct_tshirt_pattern_from_image(
            texture_image,
            out_dir,
            repo_path=sewformer_repo,
            checkpoint_path=sewformer_checkpoint,
        )
        if learned_pattern_result.status == "ok" and learned_pattern_result.pattern_spec_path is not None:
            pattern_validation = validate_tshirt_pattern_spec(learned_pattern_result.pattern_spec_path)
            if pattern_validation.valid and learned_pattern_result.estimated_size is not None:
                tshirt_size = learned_pattern_result.estimated_size

    if fit_from_image and texture_image is not None:
        if user_supplied_tshirt_size:
            image_fit = infer_tshirt_size_from_image(
                texture_image,
                chest_circumference_cm=tshirt_size.chest_circumference_cm,
                avatar=avatar,
            )
        else:
            image_fit = infer_tshirt_size_from_image(
                texture_image,
                chest_circumference_cm=max(avatar.chest_cm + 4.0, avatar.chest_cm * 1.09),
                avatar=avatar,
            )
            tshirt_size = image_fit.estimated_size

    if tshirt_size is not None and fit_constraints is not None:
        original_opening = float(tshirt_size.sleeve_opening_cm)
        original_armhole = tshirt_size.armhole_circumference_cm
        fitted_opening = max(original_opening, fit_constraints["minimum_sleeve_opening_cm"])
        fitted_armhole = max(
            float(original_armhole or 0.0),
            fit_constraints["minimum_armhole_circumference_cm"],
        )
        tshirt_size = replace(
            tshirt_size,
            sleeve_opening_cm=fitted_opening,
            armhole_circumference_cm=fitted_armhole,
        )
        if fitted_opening > original_opening + 1e-6:
            size_adjustments["sleeve_opening_cm"] = {
                "image_or_user_value": original_opening,
                "body_fit_value": fitted_opening,
            }
        if original_armhole is None or fitted_armhole > float(original_armhole) + 1e-6:
            size_adjustments["armhole_circumference_cm"] = {
                "image_or_user_value": float(original_armhole or 0.0),
                "body_fit_value": fitted_armhole,
            }
        if image_fit is not None:
            image_fit = replace(image_fit, estimated_size=tshirt_size)

    pattern = build_tshirt_pattern(intent, avatar, resolution=resolution, size=tshirt_size)
    mesh = None
    if (
        learned_pattern_result is not None
        and learned_pattern_result.status == "ok"
        and learned_pattern_result.pattern_spec_path is not None
    ):
        if pattern_validation is None:
            pattern_validation = validate_tshirt_pattern_spec(learned_pattern_result.pattern_spec_path)
        if not pattern_validation.valid:
            learned_mesh_status = {
                "source": "parametric_fallback",
                "status": "rejected",
                "message": "learned pattern rejected as non-sewable: " + "; ".join(pattern_validation.errors),
                "pattern_spec_path": learned_pattern_result.pattern_spec_path,
            }
        else:
            try:
                mesh = garment_mesh_from_sewformer_spec(learned_pattern_result.pattern_spec_path, resolution=resolution)
                learned_mesh_status = {
                    "source": "learned_sewing_pattern_spec",
                    "status": "ok",
                    "message": "mesh generated from validated learned sewing-pattern panels and stitches",
                    "pattern_spec_path": learned_pattern_result.pattern_spec_path,
                }
            except Exception as exc:
                learned_mesh_status = {
                    "source": "parametric_fallback",
                    "status": "failed",
                    "message": f"learned spec mesh conversion failed: {exc}",
                    "pattern_spec_path": learned_pattern_result.pattern_spec_path,
                }
    if mesh is None and image_fit is not None and tshirt_size is not None:
        selected_pattern_spec = write_image_conditioned_tshirt_spec(
            out_dir / "image_conditioned_tshirt_spec.json",
            tshirt_size,
            image_fit=image_fit,
        )
        selected_pattern_validation = validate_tshirt_pattern_spec(selected_pattern_spec)
        if selected_pattern_validation.valid:
            try:
                mesh = garment_mesh_from_sewformer_spec(selected_pattern_spec, resolution=resolution)
                rejection = ""
                if pattern_validation is not None and not pattern_validation.valid:
                    rejection = " after rejecting the non-sewable SewFormer proposal"
                learned_mesh_status = {
                    "source": "image_conditioned_pattern_program",
                    "status": "ok",
                    "message": "generated a validated four-panel T-shirt with matched sleeve-cap/armhole seams" + rejection,
                    "pattern_spec_path": str(selected_pattern_spec),
                }
            except Exception as exc:
                learned_mesh_status = {
                    "source": "parametric_fallback",
                    "status": "failed",
                    "message": f"validated image-conditioned pattern conversion failed: {exc}",
                    "pattern_spec_path": str(selected_pattern_spec),
                }
    if mesh is None:
        mesh = generate_tshirt_mesh(pattern)
    if dress_on_avatar:
        mesh = avatar_proxy.dress_mesh(mesh)
    mesh_path = out_dir / "garment_mesh.obj"

    active_pattern_spec: str | Path | None = None
    active_pattern_validation: PatternValidationReport | None = None
    if learned_mesh_status["source"] == "learned_sewing_pattern_spec":
        active_pattern_spec = learned_mesh_status.get("pattern_spec_path")
        active_pattern_validation = pattern_validation
    elif learned_mesh_status["source"] == "image_conditioned_pattern_program":
        active_pattern_spec = selected_pattern_spec
        active_pattern_validation = selected_pattern_validation
    preflight = run_vto_preflight(
        mesh=mesh,
        avatar_spec=avatar,
        avatar_proxy=avatar_proxy,
        classification=garment_classification,
        template=pattern_template,
        pattern_source=str(learned_mesh_status["source"]),
        tshirt_size=tshirt_size or pattern.size,
        pattern_spec_path=active_pattern_spec,
        pattern_validation=active_pattern_validation,
        fit_constraints=fit_constraints,
    )
    preflight_path = preflight.write_json(out_dir / "vto_preflight.json")
    if strict_preflight:
        require_preflight_pass(preflight)

    config = SimulationConfig(steps=steps)
    frames, final_vertices, metrics = simulate_intrinsic_cloth(mesh, pattern.material, config, avatar=avatar_proxy)
    fit_validation = _validate_simulated_fit(metrics)
    export_obj(
        mesh_path,
        GarmentMesh(
            vertices=final_vertices,
            faces=mesh.faces,
            pinned_vertices=mesh.pinned_vertices,
            panel_labels=mesh.panel_labels,
            seam_edges=mesh.seam_edges,
            rest_vertices=mesh.rest_vertices,
            uvs=mesh.uvs,
            seam_interfaces=mesh.seam_interfaces,
        ),
    )
    gif_path = out_dir / "intrinsic_tryon_rollout.gif"
    final_path = out_dir / "intrinsic_tryon_final.png"
    texture = Image.open(texture_image).convert("RGB") if texture_image is not None else None
    render_rollout(gif_path, frames, mesh, caption=intent.prompt, avatar=avatar_proxy, texture_image=texture)
    final_image: Image.Image = render_frame(
        final_vertices,
        mesh.faces,
        mesh.pinned_vertices,
        len(frames) - 1,
        intent.prompt,
        avatar=avatar_proxy,
        texture_image=texture,
        mesh=mesh,
    )
    final_image.save(final_path)

    summary = {
        "intent": intent.to_dict(),
        "garment_classification": garment_classification.to_dict(),
        "pattern_template": pattern_template.to_dict(),
        "panel_instance_growth": (
            panel_growth_result.summary() if panel_growth_result is not None else None
        ),
        "avatar": avatar.__dict__,
        "avatar_collision_proxy": avatar_proxy.summary(),
        "pattern": pattern.to_dict(),
        "learned_pattern": learned_pattern_result.to_dict() if learned_pattern_result is not None else None,
        "pattern_validation": pattern_validation.to_dict() if pattern_validation is not None else None,
        "selected_pattern_validation": (
            selected_pattern_validation.to_dict() if selected_pattern_validation is not None else None
        ),
        "learned_pattern_mesh": learned_mesh_status,
        "image_fit": image_fit.to_dict() if image_fit is not None else None,
        "body_fit_constraints": fit_constraints,
        "body_fit_size_adjustments": size_adjustments,
        "vto_preflight": preflight.to_dict(),
        "real_world_units": "meters",
        "mesh": mesh.summary(),
        "physics": {
            "engine": "intrinsic_laplacian_mesh_propagator",
            "operator": "material-space cotangent Laplacian",
            "structural_metric": "body-conditioned fitted metric" if dress_on_avatar else "flat material metric",
            "comparison_baseline": "Genesis PBD.Cloth (separate comparison harness)",
            "steps": config.steps,
            "dt": config.dt,
            "learned_scale": config.learned_scale,
            "stretch_projection_iterations": config.edge_projection_iterations,
            "stretch_tolerance": config.stretch_tolerance,
            "bend_smoothing": config.bend_smoothing,
            "rest_shape_projection": config.rest_shape_projection,
            "gravity_y": config.gravity_y,
            "material_density": pattern.material.density,
            "material_stretch_stiffness": pattern.material.stretch_stiffness,
            "material_bending_stiffness": pattern.material.bending_stiffness,
            "material_thickness": pattern.material.thickness,
            "initial_state": "dressed_on_avatar" if dress_on_avatar else "flat_pattern",
        },
        "metrics": metrics,
        "fit_validation": fit_validation,
        "artifacts": {
            "obj": str(mesh_path),
            "selected_pattern_spec": str(selected_pattern_spec) if selected_pattern_spec is not None else None,
            "vto_preflight": str(preflight_path),
            "gif": str(gif_path),
            "final_png": str(final_path),
            "texture_image": str(texture_image) if texture_image is not None else None,
            "body_mesh": str(selected_body_mesh) if selected_body_mesh is not None else None,
            "deprecated_smpl_mesh_alias_used": smpl_mesh is not None and body_mesh is None,
            "panel_instance_growth": panel_growth_artifacts,
        },
        "implementation_status": {
            "fashion_clip": "template router uses product silhouette plus text prior; learned encoder remains pluggable",
            "vto_guidance": "pre-physics garment/body evidence, semantic landmarks, occlusion plan, size grading, and quality gates",
            "vto_preview": "FASHN VTON v1.5 is the Apache-2.0 production candidate; not yet installed or treated as physical geometry",
            "learned_pattern_reconstruction": "official SewFormer checkpoint integrated as a research proposal generator with sewability rejection",
            "panel_instance_grower": (
                "trained FFN-style image seeds and recurrent local panel fields are active; "
                "outputs remain candidates until sewing-graph and physical validation"
                if panel_growth_result is not None
                else "FFN-style architecture integrated; no trained checkpoint supplied"
            ),
            "pattern_generator": "commercial provider router prefers project-owned/GarmentCode programs; SewFormer is research-only",
            "body_reconstruction": "commercial body mesh adapter; Anny-Fit is research/non-commercial unless separately licensed",
            "physics": "cotangent-Laplacian cloth with collision-prefitted metric, seam-component collision, and radial sleeve SDF constraints",
            "aesthetic_scorer": "specified, not implemented in MVP",
        },
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _write_panel_growth_artifacts(
    out_dir: Path,
    result: PanelGrowthResult,
) -> dict[str, str]:
    summary_path = out_dir / "panel_instance_growth.json"
    payload = result.summary()
    payload["instances"] = [
        {
            **instance.summary(),
            "boundary_xy": instance.boundary_xy.tolist(),
        }
        for instance in result.instances
    ]
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    arrays_path = out_dir / "panel_instance_growth.npz"
    if result.instances:
        np.savez_compressed(
            arrays_path,
            masks=np.stack([instance.mask for instance in result.instances]),
            signed_distances=np.stack(
                [instance.signed_distance for instance in result.instances]
            ),
            seam_endpoint_probabilities=np.stack(
                [instance.seam_endpoint_probability for instance in result.instances]
            ),
        )
    else:
        np.savez_compressed(
            arrays_path,
            masks=np.empty((0, 0, 0), dtype=bool),
            signed_distances=np.empty((0, 0, 0), dtype=np.float32),
            seam_endpoint_probabilities=np.empty((0, 0, 0), dtype=np.float32),
        )
    return {"summary": str(summary_path), "fields": str(arrays_path)}


def _validate_simulated_fit(metrics: dict[str, float]) -> dict[str, Any]:
    """Apply a compact strain/collision/seam acceptance gate."""

    limits = {
        "material_p95_edge_ratio": 1.20,
        "surface_intersecting_face_fraction": 0.005,
        "max_seam_gap_m": 0.003,
    }
    observed = {
        "material_p95_edge_ratio": float(metrics.get("material_embedding_p95_edge_ratio", 1.0)),
        "surface_intersecting_face_fraction": float(
            metrics.get("avatar_surface_intersecting_face_fraction_final", 0.0)
        ),
        "max_seam_gap_m": float(metrics.get("max_seam_gap_m", 0.0)),
    }
    failed_checks = [name for name, value in observed.items() if value > limits[name]]
    return {
        "status": "fail" if failed_checks else "pass",
        "failed_checks": failed_checks,
        "observed": observed,
        "limits": limits,
    }
