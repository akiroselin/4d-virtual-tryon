from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from typing import Any

import numpy as np

from .data import AvatarSpec, GarmentMesh, TshirtSizeSpec
from .pattern_library import GarmentClassification, GarmentPatternTemplate, PatternProvider
from .pattern_validation import PatternValidationReport


@dataclass(frozen=True)
class GarmentEvidence:
    source: str
    garment_type: str
    template_id: str
    classification_confidence: float
    silhouette_features: dict[str, float]
    landmark_source: str


@dataclass(frozen=True)
class BodyEvidence:
    source: str
    pose: str
    measurements_cm: dict[str, float]
    available_regions: tuple[str, ...]


@dataclass(frozen=True)
class LandmarkCorrespondence:
    garment_landmark: str
    body_landmark: str
    visibility: str
    confidence: float
    required: bool = True


@dataclass(frozen=True)
class OcclusionPlan:
    front_to_back: tuple[str, ...]
    protected_body_parts: tuple[str, ...]
    method: str


@dataclass(frozen=True)
class PatternCandidate:
    source: str
    provider: PatternProvider
    template_id: str
    topology: str
    spec_path: str | None
    panel_names: tuple[str, ...]
    panel_roles: tuple[str, ...]
    stitch_count: int


@dataclass(frozen=True)
class PreflightCheck:
    check_id: str
    status: str
    message: str
    observed: Any = None
    limit: Any = None


@dataclass(frozen=True)
class VtoPreflightReport:
    status: str
    method: str
    garment_evidence: GarmentEvidence
    body_evidence: BodyEvidence
    correspondences: tuple[LandmarkCorrespondence, ...]
    occlusion_plan: OcclusionPlan
    pattern_candidate: PatternCandidate
    checks: tuple[PreflightCheck, ...]
    metrics: dict[str, float]
    recommendations: tuple[str, ...]

    @property
    def failed_checks(self) -> tuple[str, ...]:
        return tuple(check.check_id for check in self.checks if check.status == "fail")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["failed_checks"] = list(self.failed_checks)
        return data

    def write_json(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return output


class VtoPreflightError(RuntimeError):
    pass


def run_vto_preflight(
    *,
    mesh: GarmentMesh,
    avatar_spec: AvatarSpec,
    avatar_proxy: Any,
    classification: GarmentClassification,
    template: GarmentPatternTemplate,
    pattern_source: str,
    tshirt_size: TshirtSizeSpec | None,
    pattern_spec_path: str | Path | None = None,
    pattern_validation: PatternValidationReport | None = None,
    fit_constraints: dict[str, float] | None = None,
) -> VtoPreflightReport:
    """Fuse mature VTO evidence with manufacturing checks before cloth physics."""

    candidate = _build_pattern_candidate(
        mesh,
        template,
        pattern_source=pattern_source,
        pattern_spec_path=pattern_spec_path,
    )
    checks: list[PreflightCheck] = []
    metrics = _initial_mesh_metrics(mesh, avatar_proxy)

    category_matches = classification.template_id == template.template_id
    if not category_matches:
        category_status = "fail"
        category_message = "image/text category does not match the selected pattern template"
    elif classification.confidence < 0.55:
        category_status = "warn"
        category_message = "garment category is plausible but needs a stronger image or text prior"
    else:
        category_status = "pass"
        category_message = "garment category and pattern template agree"
    checks.append(
        PreflightCheck(
            "category_template_agreement",
            category_status,
            category_message,
            observed={"template_id": classification.template_id, "confidence": classification.confidence},
            limit={"minimum_confidence": 0.55},
        )
    )

    checks.append(
        PreflightCheck(
            "commercial_provider",
            "pass" if candidate.provider.commercial_use else "fail",
            (
                f"{candidate.provider.name} is allowed for commercial runtime use"
                if candidate.provider.commercial_use
                else f"{candidate.provider.name} remains research-only until its license is separately cleared"
            ),
            observed={"provider": candidate.provider.provider_id, "license": candidate.provider.license},
            limit={"commercial_use": True},
        )
    )

    if pattern_validation is not None:
        checks.append(
            PreflightCheck(
                "sewing_pattern_structure",
                "pass" if pattern_validation.valid else "fail",
                "panel geometry and stitch references are valid" if pattern_validation.valid else "pattern validator rejected the sewing specification",
                observed={"errors": list(pattern_validation.errors), "warnings": list(pattern_validation.warnings)},
                limit={"validator_status": "valid"},
            )
        )
    else:
        checks.append(
            PreflightCheck(
                "sewing_pattern_structure",
                "pass" if candidate.stitch_count > 0 else "fail",
                "mesh contains explicit sewing constraints" if candidate.stitch_count > 0 else "mesh has no sewing constraints",
                observed={"stitch_count": candidate.stitch_count},
                limit={"minimum_stitch_count": 1},
            )
        )

    observed_roles = set(candidate.panel_roles)
    if template.garment_type in {"t-shirt", "long-sleeve t-shirt"}:
        topology_ok, topology_observed, topology_limit = _tshirt_panel_topology(observed_roles)
        missing_roles = topology_observed["missing_roles"]
    else:
        required_roles = {panel.role for panel in template.panels}
        missing_roles = sorted(required_roles - observed_roles)
        topology_ok = not missing_roles
        topology_observed = {
            "panel_roles": sorted(observed_roles),
            "missing_roles": missing_roles,
        }
        topology_limit = {"required_panel_roles": sorted(required_roles)}
    checks.append(
        PreflightCheck(
            "template_panel_topology",
            "pass" if topology_ok else "fail",
            (
                "pattern contains a supported fabrication or simulation sleeve representation"
                if topology_ok
                else "pattern is missing required body or sleeve roles: " + ", ".join(missing_roles)
            ),
            observed=topology_observed,
            limit=topology_limit,
        )
    )

    body_summary = avatar_proxy.summary()
    arrangement_summary = body_summary.get("arrangement", {})
    seams_expected_closed = not (
        isinstance(arrangement_summary, dict)
        and arrangement_summary.get("placement_mode") == "arrangement_points"
    )
    _append_size_checks(checks, avatar_spec, tshirt_size, template, fit_constraints)
    _append_mesh_checks(
        checks,
        metrics,
        seams_expected_closed=seams_expected_closed,
    )
    _append_sleeve_coverage_check(checks, candidate.panel_roles, mesh)

    garment_evidence = GarmentEvidence(
        source="product_image+prompt" if classification.image_features else "structured_text_intent",
        garment_type=classification.garment_type,
        template_id=classification.template_id,
        classification_confidence=float(classification.confidence),
        silhouette_features=dict(classification.image_features),
        landmark_source="template landmarks constrained by image silhouette and body measurements",
    )
    body_evidence = BodyEvidence(
        source=str(body_summary.get("type", "measurement_proxy")),
        pose=avatar_spec.pose,
        measurements_cm={
            "chest": float(avatar_spec.chest_cm),
            "waist": float(avatar_spec.waist_cm),
            "shoulder": float(avatar_spec.shoulder_cm),
            "height": float(avatar_spec.height_cm),
        },
        available_regions=("torso", "neck", "left_upper_arm", "right_upper_arm", "left_hand", "right_hand"),
    )
    correspondences = _tshirt_correspondences(classification.confidence, observed_roles)
    occlusion_plan = OcclusionPlan(
        front_to_back=("hands", "front_sleeves", "front_body", "body", "back_sleeves", "back_body"),
        protected_body_parts=("face", "hair", "hands", "lower_body", "unrelated_clothing"),
        method="human-parsing protection + pose/part ordering + semantic sleeve interfaces",
    )

    failed = [check for check in checks if check.status == "fail"]
    warnings = [check for check in checks if check.status == "warn"]
    status = "fail" if failed else "warn" if warnings else "pass"
    recommendations = tuple(_recommendation(check.check_id) for check in (*failed, *warnings))
    return VtoPreflightReport(
        status=status,
        method="mature_vto_evidence+pattern_program+initial_drape_quality_gate",
        garment_evidence=garment_evidence,
        body_evidence=body_evidence,
        correspondences=correspondences,
        occlusion_plan=occlusion_plan,
        pattern_candidate=candidate,
        checks=tuple(checks),
        metrics=metrics,
        recommendations=tuple(dict.fromkeys(recommendations)),
    )


def require_preflight_pass(report: VtoPreflightReport) -> None:
    if report.status == "fail":
        failed = ", ".join(report.failed_checks)
        raise VtoPreflightError(f"VTO preflight blocked cloth physics: {failed}")


def infer_panel_role(name: str) -> str:
    """Normalize project, GarmentCode, and SewFactory panel names."""

    normalized = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    compact = normalized.replace("_", "")
    is_sleeve = "sleeve" in normalized or "slv" in normalized
    if not is_sleeve:
        if "back" in normalized or compact in {"back", "bbody"}:
            return "back_body"
        if "front" in normalized or compact in {"front", "fbody"}:
            return "front_body"
        return normalized

    side = "right" if (
        "right" in normalized or compact.startswith(("rfsleeve", "rbsleeve", "rslv", "rsleeve"))
    ) else "left" if (
        "left" in normalized or compact.startswith(("lfsleeve", "lbsleeve", "lslv", "lsleeve"))
    ) else "unknown"
    shell = "back" if (
        normalized.startswith("back_") or "_back_" in f"_{normalized}_" or compact.startswith(("lbsleeve", "rbsleeve"))
    ) else "front" if (
        normalized.startswith("front_") or "_front_" in f"_{normalized}_" or compact.startswith(("lfsleeve", "rfsleeve"))
    ) else "whole"
    if side == "unknown":
        return f"{shell}_sleeve" if shell != "whole" else "sleeve"
    if shell == "whole":
        return f"{side}_sleeve"
    return f"{shell}_{side}_sleeve"


def _build_pattern_candidate(
    mesh: GarmentMesh,
    template: GarmentPatternTemplate,
    *,
    pattern_source: str,
    pattern_spec_path: str | Path | None,
) -> PatternCandidate:
    panel_names: list[str] = []
    stitch_count = len(mesh.seam_edges)
    generator = ""
    spec_path: str | None = None
    if pattern_spec_path is not None and Path(pattern_spec_path).exists():
        path = Path(pattern_spec_path)
        raw = json.loads(path.read_text(encoding="utf-8"))
        pattern = raw.get("pattern", raw)
        panel_names = [str(value) for value in pattern.get("panels", {}).keys()]
        stitch_count = len(pattern.get("stitches", []))
        generator = str(raw.get("properties", {}).get("generator", ""))
        spec_path = str(path)
    if not panel_names:
        panel_names = sorted(set(mesh.panel_labels))
    roles = tuple(sorted({infer_panel_role(name) for name in panel_names}))

    provider_id = "project_local"
    source_lower = f"{pattern_source} {generator}".lower()
    if "garmentcode" in source_lower:
        provider_id = "garmentcode"
    elif "sewformer" in source_lower or "learned_sewing_pattern" in source_lower:
        provider_id = "sewformer"
    provider = next((value for value in template.providers if value.provider_id == provider_id), None)
    if provider is None:
        provider = PatternProvider(provider_id, provider_id, "unknown", False, "unregistered provider")
    return PatternCandidate(
        source=pattern_source,
        provider=provider,
        template_id=template.template_id,
        topology=template.topology,
        spec_path=spec_path,
        panel_names=tuple(panel_names),
        panel_roles=roles,
        stitch_count=stitch_count,
    )


def _initial_mesh_metrics(mesh: GarmentMesh, avatar_proxy: Any) -> dict[str, float]:
    faces = np.asarray(mesh.faces, dtype=np.int32)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    rest = np.asarray(mesh.material_vertices, dtype=np.float64)
    edge_pairs = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0)
    edge_pairs = np.unique(np.sort(edge_pairs, axis=1), axis=0)
    rest_lengths = np.linalg.norm(rest[edge_pairs[:, 0]] - rest[edge_pairs[:, 1]], axis=1)
    fitted_lengths = np.linalg.norm(vertices[edge_pairs[:, 0]] - vertices[edge_pairs[:, 1]], axis=1)
    valid = rest_lengths > 1e-8
    ratios = fitted_lengths[valid] / rest_lengths[valid]
    metrics = {
        "initial_material_p95_edge_ratio": float(np.percentile(ratios, 95.0)) if ratios.size else 1.0,
        "initial_material_max_edge_ratio": float(ratios.max(initial=1.0)),
        "initial_material_p05_edge_ratio": float(np.percentile(ratios, 5.0)) if ratios.size else 1.0,
    }

    if mesh.seam_edges:
        seams = np.asarray(mesh.seam_edges, dtype=np.int32)
        seam_gaps = np.linalg.norm(vertices[seams[:, 0]] - vertices[seams[:, 1]], axis=1)
    else:
        seam_gaps = np.empty(0, dtype=np.float64)
    metrics.update(
        {
            "initial_seam_p95_gap_m": float(np.percentile(seam_gaps, 95.0)) if seam_gaps.size else 0.0,
            "initial_seam_max_gap_m": float(seam_gaps.max(initial=0.0)),
        }
    )

    if hasattr(avatar_proxy, "collision_diagnostics_np"):
        collision = avatar_proxy.collision_diagnostics_np(vertices, faces)
        metrics.update(
            {
                "initial_inside_vertices": float(collision.get("avatar_inside_vertices_final", 0.0)),
                "initial_surface_intersecting_face_fraction": float(
                    collision.get("avatar_surface_intersecting_face_fraction_final", 0.0)
                ),
                "initial_surface_max_penetration_m": float(
                    collision.get("avatar_surface_max_penetration_final", 0.0)
                ),
            }
        )
    else:
        metrics.update(_proxy_collision_metrics(vertices, mesh.panel_labels, avatar_proxy))
    return metrics


def _proxy_collision_metrics(vertices: np.ndarray, labels: list[str], avatar_proxy: Any) -> dict[str, float]:
    front = avatar_proxy.support_height_np(vertices[:, :2]) + avatar_proxy.clearance
    back = avatar_proxy.back_height_np(vertices[:, :2]) - avatar_proxy.clearance
    is_back = np.asarray([label.startswith("back_") for label in labels], dtype=bool)
    violation = np.zeros(vertices.shape[0], dtype=np.float64)
    front_active = (~is_back) & np.isfinite(front)
    back_active = is_back & np.isfinite(back)
    violation[front_active] = np.maximum(front[front_active] - vertices[front_active, 2], 0.0)
    violation[back_active] = np.maximum(vertices[back_active, 2] - back[back_active], 0.0)
    return {
        "initial_inside_vertices": float(np.sum(violation > 0.0)),
        "initial_surface_intersecting_face_fraction": 0.0,
        "initial_surface_max_penetration_m": float(violation.max(initial=0.0)),
    }


def _append_size_checks(
    checks: list[PreflightCheck],
    avatar: AvatarSpec,
    size: TshirtSizeSpec | None,
    template: GarmentPatternTemplate,
    fit_constraints: dict[str, float] | None,
) -> None:
    if size is None:
        checks.append(PreflightCheck("body_size_grading", "warn", "no explicit garment measurements were retained"))
        return
    ease = float(size.chest_circumference_cm - avatar.chest_cm)
    ease_range = template.parameters.get("chest_ease_cm", [2.0, 20.0])
    checks.append(
        PreflightCheck(
            "chest_ease",
            "pass" if float(ease_range[0]) <= ease <= float(ease_range[1]) else "fail",
            "garment chest ease is inside the template range" if float(ease_range[0]) <= ease <= float(ease_range[1]) else "garment chest ease is outside the template range",
            observed=ease,
            limit={"min_cm": float(ease_range[0]), "max_cm": float(ease_range[1])},
        )
    )
    shoulder_ratio = float(size.shoulder_width_cm / max(avatar.shoulder_cm, 1e-6))
    checks.append(
        PreflightCheck(
            "shoulder_grading",
            "pass" if 0.92 <= shoulder_ratio <= 1.25 else "fail",
            "shoulder width is compatible with the avatar" if 0.92 <= shoulder_ratio <= 1.25 else "shoulder width is incompatible with the avatar",
            observed=shoulder_ratio,
            limit={"min_ratio": 0.92, "max_ratio": 1.25},
        )
    )
    if fit_constraints is None:
        return
    minimum_opening = float(fit_constraints.get("minimum_sleeve_opening_cm", 0.0))
    checks.append(
        PreflightCheck(
            "sleeve_opening_fit",
            "pass" if size.sleeve_opening_cm >= minimum_opening else "fail",
            "sleeve opening clears the upper arm" if size.sleeve_opening_cm >= minimum_opening else "sleeve opening is too small for the arm",
            observed=float(size.sleeve_opening_cm),
            limit={"minimum_cm": minimum_opening},
        )
    )
    minimum_armhole = float(fit_constraints.get("minimum_armhole_circumference_cm", 0.0))
    armhole = float(size.armhole_circumference_cm or 0.0)
    checks.append(
        PreflightCheck(
            "armhole_fit",
            "pass" if armhole >= minimum_armhole else "fail",
            "armhole circumference clears the upper arm" if armhole >= minimum_armhole else "armhole circumference is missing or too small",
            observed=armhole,
            limit={"minimum_cm": minimum_armhole},
        )
    )


def _append_mesh_checks(
    checks: list[PreflightCheck],
    metrics: dict[str, float],
    *,
    seams_expected_closed: bool = True,
) -> None:
    p95_ratio = metrics["initial_material_p95_edge_ratio"]
    checks.append(
        PreflightCheck(
            "initial_material_strain",
            "pass" if p95_ratio <= 1.20 else "fail",
            "initial dressed mesh preserves the 2D material metric" if p95_ratio <= 1.20 else "initial dressing stretches the pattern before physics",
            observed=p95_ratio,
            limit={"maximum_p95_edge_ratio": 1.20},
        )
    )
    seam_gap = metrics["initial_seam_p95_gap_m"]
    seam_closed = seam_gap <= 0.006
    checks.append(
        PreflightCheck(
            "initial_seam_closure",
            "pass" if seam_closed or not seams_expected_closed else "fail",
            (
                "sewn interfaces begin closed"
                if seam_closed
                else "interfaces are intentionally open in the arrangement stage and close during sewing"
                if not seams_expected_closed
                else "sewn interfaces begin too far apart"
            ),
            observed=seam_gap,
            limit={
                "maximum_p95_gap_m": 0.006,
                "stage": "closed garment" if seams_expected_closed else "open panel arrangement",
            },
        )
    )
    collision_fraction = metrics["initial_surface_intersecting_face_fraction"]
    penetration = metrics["initial_surface_max_penetration_m"]
    collision_ok = collision_fraction <= 0.005 and penetration <= 0.003
    checks.append(
        PreflightCheck(
            "initial_body_collision",
            "pass" if collision_ok else "fail",
            "initial garment is outside the body" if collision_ok else "initial garment intersects the body before physics",
            observed={"face_fraction": collision_fraction, "max_penetration_m": penetration},
            limit={"maximum_face_fraction": 0.005, "maximum_penetration_m": 0.003},
        )
    )


def _append_sleeve_coverage_check(
    checks: list[PreflightCheck],
    roles: tuple[str, ...],
    mesh: GarmentMesh,
) -> None:
    role_set = set(roles)
    split_required = {
        "front_left_sleeve",
        "back_left_sleeve",
        "front_right_sleeve",
        "back_right_sleeve",
    }
    whole_required = {"left_sleeve", "right_sleeve"}
    labels = np.asarray(mesh.panel_labels)
    left_vertices = int(np.sum(np.asarray(["left_sleeve" in label for label in labels])))
    right_vertices = int(np.sum(np.asarray(["right_sleeve" in label for label in labels])))
    representation = (
        "whole_sleeve_panels"
        if whole_required.issubset(role_set)
        else "split_front_back_sleeve_patches"
        if split_required.issubset(role_set)
        else "incomplete"
    )
    covered = representation != "incomplete" and left_vertices >= 4 and right_vertices >= 4
    checks.append(
        PreflightCheck(
            "sleeve_arm_coverage",
            "pass" if covered else "fail",
            (
                "both sleeves have a supported whole-panel or split-patch representation"
                if covered
                else "sleeve topology cannot establish coverage around both arms"
            ),
            observed={
                "representation": representation,
                "roles": sorted(role_set & (whole_required | split_required)),
                "left_vertices": left_vertices,
                "right_vertices": right_vertices,
            },
            limit={
                "accepted_role_sets": [sorted(whole_required), sorted(split_required)],
                "minimum_vertices_per_side": 4,
            },
        )
    )


def _tshirt_correspondences(
    confidence: float,
    observed_roles: set[str],
) -> tuple[LandmarkCorrespondence, ...]:
    base = float(np.clip(confidence, 0.35, 0.95))
    whole_sleeves = {"left_sleeve", "right_sleeve"}.issubset(observed_roles)
    split_sleeves = {
        "front_left_sleeve",
        "back_left_sleeve",
        "front_right_sleeve",
        "back_right_sleeve",
    }.issubset(observed_roles)
    arm_confidence = base if whole_sleeves or split_sleeves else min(base, 0.35)
    return (
        LandmarkCorrespondence("left_shoulder_seam", "left_shoulder", "template_inferred", base),
        LandmarkCorrespondence("right_shoulder_seam", "right_shoulder", "template_inferred", base),
        LandmarkCorrespondence("left_front_armhole", "left_upper_arm_front", "template_inferred", arm_confidence),
        LandmarkCorrespondence("left_back_armhole", "left_upper_arm_back", "template_inferred", arm_confidence),
        LandmarkCorrespondence("right_front_armhole", "right_upper_arm_front", "template_inferred", arm_confidence),
        LandmarkCorrespondence("right_back_armhole", "right_upper_arm_back", "template_inferred", arm_confidence),
        LandmarkCorrespondence("left_cuff", "left_upper_arm", "visible_or_template_inferred", arm_confidence),
        LandmarkCorrespondence("right_cuff", "right_upper_arm", "visible_or_template_inferred", arm_confidence),
        LandmarkCorrespondence("neckline", "neck_base", "visible_or_template_inferred", base),
        LandmarkCorrespondence("hem", "torso_length_anchor", "visible_or_template_inferred", base),
    )


def _recommendation(check_id: str) -> str:
    messages = {
        "category_template_agreement": "rerank category candidates using garment silhouette and semantic landmarks",
        "commercial_provider": "regenerate the pattern with the MIT GarmentCode provider",
        "sewing_pattern_structure": "repair or reject invalid panel edges and stitch references",
        "template_panel_topology": "provide front/back bodies and either whole sleeves or explicitly mapped front/back sleeve patches",
        "body_size_grading": "grade the selected template from avatar measurements before dressing",
        "chest_ease": "adjust template chest ease instead of scaling the sewn 3D mesh",
        "shoulder_grading": "adjust shoulder and armhole parameters jointly",
        "sleeve_opening_fit": "increase cuff opening with material-aware wearing ease",
        "armhole_fit": "solve armhole circumference from the upper-arm section",
        "initial_material_strain": "rerun pattern grading and initial placement; do not ask physics to repair metric distortion",
        "initial_seam_closure": "initialize paired seam vertices on a shared 3D interface",
        "initial_body_collision": "solve collision-free canonical dressing before time integration",
        "sleeve_arm_coverage": "bind each sleeve's front/back cap interfaces to the corresponding body armholes",
    }
    return messages.get(check_id, check_id)


def _tshirt_panel_topology(
    observed_roles: set[str],
) -> tuple[bool, dict[str, Any], dict[str, Any]]:
    body_required = {"front_body", "back_body"}
    whole_sleeves = {"left_sleeve", "right_sleeve"}
    split_sleeves = {
        "front_left_sleeve",
        "back_left_sleeve",
        "front_right_sleeve",
        "back_right_sleeve",
    }
    body_missing = sorted(body_required - observed_roles)
    whole_missing = sorted(whole_sleeves - observed_roles)
    split_missing = sorted(split_sleeves - observed_roles)
    representation = (
        "whole_sleeve_panels"
        if not whole_missing
        else "split_front_back_sleeve_patches"
        if not split_missing
        else "incomplete"
    )
    topology_ok = not body_missing and representation != "incomplete"
    missing_roles = body_missing
    if representation == "incomplete":
        missing_roles = [
            *missing_roles,
            "left_sleeve+right_sleeve or four front/back sleeve patches",
        ]
    observed = {
        "panel_roles": sorted(observed_roles),
        "representation": representation,
        "missing_roles": missing_roles,
    }
    limit = {
        "required_body_roles": sorted(body_required),
        "accepted_sleeve_role_sets": [sorted(whole_sleeves), sorted(split_sleeves)],
    }
    return topology_ok, observed, limit
