from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Mapping

import numpy as np

from .data import AvatarSpec, GarmentIntent, GarmentMesh, MaterialSpec
from .pattern_library import GarmentPatternTemplate, get_pattern_template
from .patterns import material_from_intent


@dataclass(frozen=True)
class MeasurementEvidence:
    name: str
    value_cm: float
    uncertainty_cm: float
    source: str

    @property
    def observed(self) -> bool:
        return self.source in {"user", "body_mesh"}


@dataclass(frozen=True)
class ConstructionParameter:
    name: str
    value: float
    low: float
    high: float
    unit: str
    authority: str


@dataclass(frozen=True)
class PhysicalStructureHypothesis:
    hypothesis_id: str
    rank: int
    score: float
    template_id: str
    garment_type: str
    fit_variant: str
    generator_status: str
    panel_roles: tuple[str, ...]
    seams: tuple[dict[str, str], ...]
    openings: tuple[str, ...]
    closure: str
    measurements: tuple[MeasurementEvidence, ...]
    graded_parameters: tuple[ConstructionParameter, ...]
    material: MaterialSpec
    material_uncertainty: dict[str, tuple[float, float]]
    warnings: tuple[str, ...]

    @property
    def simulation_structure_complete(self) -> bool:
        return self.generator_status == "implemented" and bool(self.panel_roles) and bool(self.seams)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["simulation_structure_complete"] = self.simulation_structure_complete
        payload["authority"] = {
            "rest_metric": "construction hypothesis",
            "panel_topology": "construction hypothesis",
            "sewing_graph": "construction hypothesis",
            "material_prior": "experience database; calibrate when measurements exist",
            "current_embedding": "FEM/IPC/Kirchhoff-Love followed by constrained CortexODE",
        }
        return payload


@dataclass(frozen=True)
class CortexODEAuthorityReport:
    passed: bool
    immutable_faces: bool
    immutable_rest_metric: bool
    immutable_panel_roles: bool
    immutable_seams: bool
    immutable_uvs: bool
    reasons: tuple[str, ...]


def generate_physical_structure_hypotheses(
    intent: GarmentIntent,
    avatar: AvatarSpec,
    *,
    measurement_overrides_cm: Mapping[str, float] | None = None,
    top_k: int = 3,
) -> tuple[PhysicalStructureHypothesis, ...]:
    """Grade a construction template into auditable physical hypotheses.

    Missing measurements are inferred conservatively from anthropometric ratios
    and retain large uncertainty. They are priors for candidate generation, not
    claims about the user's body.
    """

    if top_k <= 0:
        raise ValueError("top_k must be positive")
    template = get_pattern_template(intent.garment_type)
    measurements = _measurement_profile(avatar, measurement_overrides_cm or {})
    target_quantile = {"slim": 0.18, "relaxed": 0.50, "oversized": 0.82}.get(
        intent.silhouette, 0.50
    )
    offsets = (0.0, -0.18, 0.18, -0.32, 0.32)
    quantiles = [float(np.clip(target_quantile + offset, 0.05, 0.95)) for offset in offsets]
    hypotheses = []
    for index, quantile in enumerate(quantiles[:top_k]):
        parameters = _grade_template(template, intent, measurements, quantile)
        inferred_required = [
            name
            for name in template.required_measurements
            if not _measurement_for_requirement(name, measurements).observed
        ]
        warnings = []
        if inferred_required:
            warnings.append("inferred required measurements: " + ", ".join(inferred_required))
        if template.status != "implemented":
            warnings.append("pattern generator is not implemented")
        if not template.seams:
            warnings.append("template sewing graph is incomplete")
        material = material_from_intent(intent)
        uncertainty = _material_uncertainty(material)
        score = 1.0 - 0.32 * abs(quantile - target_quantile)
        score -= 0.04 * len(inferred_required)
        score -= 0.25 if template.status != "implemented" else 0.0
        score -= 0.15 if not template.seams else 0.0
        hypotheses.append(
            PhysicalStructureHypothesis(
                hypothesis_id=f"{template.template_id}:{index:02d}",
                rank=index + 1,
                score=float(np.clip(score, 0.0, 1.0)),
                template_id=template.template_id,
                garment_type=template.garment_type,
                fit_variant=_fit_variant(quantile),
                generator_status=template.status,
                panel_roles=_expanded_panel_roles(template),
                seams=tuple(dict(seam) for seam in template.seams),
                openings=_semantic_openings(template.garment_type),
                closure=_closure_prior(intent, template),
                measurements=tuple(measurements.values()),
                graded_parameters=parameters,
                material=material,
                material_uncertainty=uncertainty,
                warnings=tuple(warnings),
            )
        )
    hypotheses.sort(key=lambda item: item.score, reverse=True)
    return tuple(replace(item, rank=index + 1) for index, item in enumerate(hypotheses))


def validate_cortexode_refinement_authority(
    source: GarmentMesh,
    refined: GarmentMesh,
    *,
    atol: float = 1.0e-7,
) -> CortexODEAuthorityReport:
    """Ensure CortexODE changed only the shell's current 3D embedding."""

    faces = np.array_equal(source.faces, refined.faces)
    rest = np.allclose(source.material_vertices, refined.material_vertices, atol=atol, rtol=0.0)
    roles = source.panel_labels == refined.panel_labels
    seams = source.seam_edges == refined.seam_edges and source.seam_interfaces == refined.seam_interfaces
    if source.uvs is None or refined.uvs is None:
        uvs = source.uvs is None and refined.uvs is None
    else:
        uvs = np.allclose(source.uvs, refined.uvs, atol=atol, rtol=0.0)
    reasons = []
    for passed, reason in (
        (faces, "face connectivity changed"),
        (rest, "construction rest metric changed"),
        (roles, "panel roles changed"),
        (seams, "sewing graph changed"),
        (uvs, "material coordinates changed"),
    ):
        if not passed:
            reasons.append(reason)
    return CortexODEAuthorityReport(
        passed=not reasons,
        immutable_faces=faces,
        immutable_rest_metric=rest,
        immutable_panel_roles=roles,
        immutable_seams=seams,
        immutable_uvs=uvs,
        reasons=tuple(reasons),
    )


def _measurement_profile(
    avatar: AvatarSpec,
    overrides: Mapping[str, float],
) -> dict[str, MeasurementEvidence]:
    base = {
        "height": MeasurementEvidence("height", avatar.height_cm, 0.5, "body_mesh"),
        "chest_circumference": MeasurementEvidence(
            "chest_circumference", avatar.chest_cm, 0.8, "body_mesh"
        ),
        "waist_circumference": MeasurementEvidence(
            "waist_circumference", avatar.waist_cm, 0.8, "body_mesh"
        ),
        "shoulder_width": MeasurementEvidence(
            "shoulder_width", avatar.shoulder_cm, 0.6, "body_mesh"
        ),
    }
    inferred = {
        "hip_circumference": (max(avatar.waist_cm * 1.16, avatar.chest_cm * 0.94), 7.0),
        "body_length": (avatar.height_cm * 0.39, 5.0),
        "upper_arm_circumference": (avatar.chest_cm * 0.31, 3.5),
        "neck_circumference": (avatar.chest_cm * 0.39, 3.0),
        "arm_length": (avatar.height_cm * 0.34, 4.5),
        "wrist_circumference": (avatar.chest_cm * 0.17, 2.0),
        "head_circumference": (avatar.height_cm * 0.32, 3.0),
        "dress_length": (avatar.height_cm * 0.56, 8.0),
        "skirt_length": (avatar.height_cm * 0.27, 6.0),
        "inseam": (avatar.height_cm * 0.45, 6.0),
        "thigh_circumference": (max(avatar.waist_cm * 0.68, 30.0), 5.0),
        "rise": (avatar.height_cm * 0.16, 3.0),
        "back_length": (avatar.height_cm * 0.26, 4.0),
    }
    base.update(
        {
            name: MeasurementEvidence(name, float(value), uncertainty, "anthropometric_prior")
            for name, (value, uncertainty) in inferred.items()
        }
    )
    aliases = {
        "chest_cm": "chest_circumference",
        "waist_cm": "waist_circumference",
        "shoulder_cm": "shoulder_width",
        "height_cm": "height",
    }
    for raw_name, raw_value in overrides.items():
        name = aliases.get(str(raw_name), str(raw_name))
        value = float(raw_value)
        if value <= 0.0 or not np.isfinite(value):
            raise ValueError(f"measurement {raw_name!r} must be finite and positive")
        base[name] = MeasurementEvidence(name, value, 0.3, "user")
    return base


def _measurement_for_requirement(
    requirement: str,
    measurements: Mapping[str, MeasurementEvidence],
) -> MeasurementEvidence:
    if requirement in measurements:
        return measurements[requirement]
    return MeasurementEvidence(requirement, 0.0, float("inf"), "missing")


def _grade_template(
    template: GarmentPatternTemplate,
    intent: GarmentIntent,
    measurements: Mapping[str, MeasurementEvidence],
    quantile: float,
) -> tuple[ConstructionParameter, ...]:
    output = []
    for name, raw_range in template.parameters.items():
        if not isinstance(raw_range, (list, tuple)) or len(raw_range) != 2:
            continue
        low, high = float(raw_range[0]), float(raw_range[1])
        database_value = low + quantile * (high - low)
        value, authority = _body_anchored_parameter(
            name, database_value, low, high, intent, measurements, quantile
        )
        output.append(ConstructionParameter(name, value, low, high, "cm", authority))
    return tuple(output)


def _body_anchored_parameter(
    name: str,
    database_value: float,
    low: float,
    high: float,
    intent: GarmentIntent,
    measurements: Mapping[str, MeasurementEvidence],
    quantile: float,
) -> tuple[float, str]:
    direct = {
        "body_length_cm": "body_length",
        "shoulder_width_cm": "shoulder_width",
        "sleeve_length_cm": "arm_length" if intent.sleeve == "long" else None,
    }.get(name)
    if direct is not None:
        value = measurements[direct].value_cm
        if name == "sleeve_length_cm" and intent.sleeve != "long":
            value *= 0.34
        return float(np.clip(value, low, high)), f"body measurement: {direct}"
    if name == "sleeve_length_cm":
        value = measurements["upper_arm_circumference"].value_cm * 0.58
        return float(np.clip(value, low, high)), "upper-arm proportion prior"
    if name == "sleeve_opening_cm":
        value = measurements["upper_arm_circumference"].value_cm + 0.35 * database_value
        return float(np.clip(value, low, high)), "upper-arm measurement plus wearing ease"
    if name == "armhole_circumference_cm":
        value = 1.35 * measurements["upper_arm_circumference"].value_cm + 0.2 * database_value
        return float(np.clip(value, low, high)), "upper-arm/armhole construction rule"
    if name == "neck_width_cm":
        value = 0.42 * measurements["shoulder_width"].value_cm
        return float(np.clip(value, low, high)), "shoulder-width neckline rule"
    if name == "hem_sweep_cm":
        waist = measurements["waist_circumference"].value_cm
        value = max(database_value, waist * (1.15 + quantile))
        return float(np.clip(value, low, high)), "waist measurement plus silhouette sweep"
    return database_value, "construction experience database"


def _expanded_panel_roles(template: GarmentPatternTemplate) -> tuple[str, ...]:
    roles = []
    for panel in template.panels:
        if panel.quantity == 1:
            roles.append(panel.role)
        else:
            roles.extend(f"{panel.role}_{index + 1}" for index in range(panel.quantity))
    return tuple(roles)


def _semantic_openings(garment_type: str) -> tuple[str, ...]:
    if garment_type in {"t-shirt", "long-sleeve t-shirt", "button-shirt", "hoodie", "jacket"}:
        return ("neck", "hem", "left_cuff", "right_cuff")
    if garment_type == "tank-top":
        return ("neck", "hem", "left_armhole", "right_armhole")
    if garment_type == "skirt":
        return ("waist", "hem")
    if garment_type in {"trousers", "shorts"}:
        return ("waist", "left_leg_hem", "right_leg_hem")
    if garment_type == "dress":
        return ("neck", "hem", "left_armhole", "right_armhole")
    return ("primary_entry", "hem")


def _closure_prior(intent: GarmentIntent, template: GarmentPatternTemplate) -> str:
    garment_type = template.garment_type
    if garment_type in {"t-shirt", "long-sleeve t-shirt", "tank-top", "hoodie"}:
        return "pullover; stretch opening required"
    if garment_type == "button-shirt":
        return "center-front button placket"
    if garment_type in {"skirt", "dress"}:
        return "concealed center-back zipper or elastic waist; retain as Top-K construction choice"
    if garment_type in {"trousers", "shorts"}:
        return "front fly plus waistband closure"
    if garment_type == "jacket":
        return "center-front zipper or buttons; retain as Top-K construction choice"
    return "construction database prior"


def _material_uncertainty(material: MaterialSpec) -> dict[str, tuple[float, float]]:
    return {
        "density": (0.75 * material.density, 1.35 * material.density),
        "stretch_stiffness": (0.55 * material.stretch_stiffness, 1.75 * material.stretch_stiffness),
        "bending_stiffness": (0.4 * material.bending_stiffness, 2.2 * material.bending_stiffness),
        "thickness": (0.65 * material.thickness, 1.6 * material.thickness),
    }


def _fit_variant(quantile: float) -> str:
    if quantile < 0.34:
        return "close"
    if quantile > 0.66:
        return "loose"
    return "regular"
