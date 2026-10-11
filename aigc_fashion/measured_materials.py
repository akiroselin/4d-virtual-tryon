from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class NumericRange:
    minimum: float
    maximum: float
    unit: str

    @property
    def midpoint(self) -> float:
        return 0.5 * (self.minimum + self.maximum)


@dataclass(frozen=True)
class MeasuredFabricProfile:
    """Traceable fabric evidence in solver-facing SI units.

    A range is deliberately not collapsed into a pretend exact material card.
    The exact swatch must be measured before a production simulation is called
    calibrated.
    """

    profile_id: str
    construction: str
    composition: str
    evidence_status: str
    areal_density: NumericRange
    thickness: NumericRange | None
    wale_or_warp_service_strain: NumericRange | None
    course_or_weft_service_strain: NumericRange | None
    shear_rigidity: NumericRange | None
    bending_rigidity_warp: NumericRange | None
    bending_rigidity_weft: NumericRange | None
    source_urls: tuple[str, ...]
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class MaterialMotionTuning:
    """Solver-facing material card assembled from measured family ranges.

    These values are priors for the photographed swatch, not a substitute for
    a KES/FAB test.  Unlike an export-time motion filter, they enter the mass,
    constitutive, and Rayleigh-dissipation terms of the physical step.
    """

    velocity_damping_per_second: float
    contact_damping_n_s_m: float
    seam_damping_n_s_m: float
    friction_coefficient: float
    residual_force_scale: float
    evidence_status: str = "engineering prior; calibrate against a moving swatch"
    areal_density_kg_m2: float = 0.18
    bending_damping_per_second: float = 2.0
    constitutive_force_floor: float = 0.25
    se3_generative_potential_weight: float = 0.0


MATERIAL_MOTION_TUNINGS: dict[str, MaterialMotionTuning] = {
    "cotton": MaterialMotionTuning(
        # Preserve low-frequency body momentum while retaining enough loss
        # for cotton jersey not to behave like silk. Contact damping remains
        # higher because it represents a separate physical port.
        velocity_damping_per_second=2.1,
        contact_damping_n_s_m=7.0,
        seam_damping_n_s_m=1.0,
        friction_coefficient=0.22,
        residual_force_scale=0.006,
        areal_density_kg_m2=0.170,
        bending_damping_per_second=1.4,
        constitutive_force_floor=0.25,
    ),
    "nylon": MaterialMotionTuning(
        velocity_damping_per_second=2.3,
        contact_damping_n_s_m=8.0,
        seam_damping_n_s_m=1.1,
        friction_coefficient=0.18,
        residual_force_scale=0.005,
        areal_density_kg_m2=0.145,
        bending_damping_per_second=1.2,
        constitutive_force_floor=0.22,
    ),
    "wool": MaterialMotionTuning(
        velocity_damping_per_second=2.6,
        contact_damping_n_s_m=9.0,
        seam_damping_n_s_m=1.0,
        friction_coefficient=0.42,
        residual_force_scale=0.005,
        areal_density_kg_m2=0.235,
        bending_damping_per_second=2.0,
        constitutive_force_floor=0.32,
    ),
    "default": MaterialMotionTuning(
        velocity_damping_per_second=2.0,
        contact_damping_n_s_m=5.0,
        seam_damping_n_s_m=0.9,
        friction_coefficient=0.20,
        residual_force_scale=0.006,
        areal_density_kg_m2=0.18,
        bending_damping_per_second=1.3,
        constitutive_force_floor=0.20,
    ),
}


def material_motion_tuning(material_name: str) -> MaterialMotionTuning:
    """Select a family material card from the inferred fabric semantics."""

    name = material_name.lower()
    if "nylon" in name or "polyamide" in name:
        return MATERIAL_MOTION_TUNINGS["nylon"]
    if "cotton" in name or "jersey" in name:
        return MATERIAL_MOTION_TUNINGS["cotton"]
    if "wool" in name or "suiting" in name:
        return MATERIAL_MOTION_TUNINGS["wool"]
    return MATERIAL_MOTION_TUNINGS["default"]


GF_PER_CM_PER_DEGREE_TO_N_PER_M_PER_RAD = 56.1880676
GF_CM2_PER_CM_TO_N_M = 9.80665e-5


MATERIAL_PROFILES: dict[str, MeasuredFabricProfile] = {
    "cotton_single_jersey_reference": MeasuredFabricProfile(
        profile_id="cotton_single_jersey_reference",
        construction="single-jersey weft knit",
        composition="cotton reference family; no elastane assumed",
        evidence_status="literature envelope, not the photographed swatch",
        areal_density=NumericRange(0.120, 0.220, "kg/m^2"),
        thickness=None,
        # Plain single jersey is directionally extensible because yarn loops
        # reorient before yarn tension dominates. These are construction-level
        # ranges, not failure strains or force-strain curves.
        wale_or_warp_service_strain=NumericRange(0.10, 0.20, "strain"),
        course_or_weft_service_strain=NumericRange(0.30, 0.50, "strain"),
        shear_rigidity=NumericRange(
            0.291 * GF_PER_CM_PER_DEGREE_TO_N_PER_M_PER_RAD,
            0.762 * GF_PER_CM_PER_DEGREE_TO_N_PER_M_PER_RAD,
            "N/m/rad",
        ),
        bending_rigidity_warp=NumericRange(
            0.0200 * GF_CM2_PER_CM_TO_N_M,
            0.0200 * GF_CM2_PER_CM_TO_N_M,
            "N*m",
        ),
        bending_rigidity_weft=NumericRange(
            0.0355 * GF_CM2_PER_CM_TO_N_M,
            0.0355 * GF_CM2_PER_CM_TO_N_M,
            "N*m",
        ),
        source_urls=(
            "https://www.sciencedirect.com/topics/materials-science/knit-fabrics",
            "https://ir.bdu.edu.et/bitstream/handle/123456789/9293/"
            "Alemayehu%20Assefa%20Belete%20MTM%20Textile%20Manufacturing..pdf",
            "https://www.mdpi.com/2073-4360/14/9/1873",
        ),
        notes=(
            "Bending values are the reported single-jersey wale/course example.",
            "Areal density and extensibility are family envelopes.",
            "A real swatch still needs KES/FAB force-extension and recovery curves.",
        ),
    ),
    "wool_blend_suiting_reference": MeasuredFabricProfile(
        profile_id="wool_blend_suiting_reference",
        construction="woven suiting",
        composition="wool or wool/polyester reference family",
        evidence_status="published suiting envelope, not the photographed swatch",
        areal_density=NumericRange(0.125, 0.258, "kg/m^2"),
        thickness=None,
        wale_or_warp_service_strain=NumericRange(0.043, 0.051, "strain at KES load"),
        course_or_weft_service_strain=NumericRange(0.075, 0.18, "strain at KES load"),
        shear_rigidity=NumericRange(
            0.50 * GF_PER_CM_PER_DEGREE_TO_N_PER_M_PER_RAD,
            0.65 * GF_PER_CM_PER_DEGREE_TO_N_PER_M_PER_RAD,
            "N/m/rad",
        ),
        bending_rigidity_warp=NumericRange(
            0.20 * GF_CM2_PER_CM_TO_N_M,
            0.20 * GF_CM2_PER_CM_TO_N_M,
            "N*m lower-bound reference",
        ),
        bending_rigidity_weft=None,
        source_urls=(
            "https://doi.org/10.1108/RJTA-04-02-2000-B002",
            "https://www.sciencedirect.com/topics/engineering/"
            "fabric-assurance-by-simple-testing",
            "https://standards.ieee.org/wp-content/uploads/import/governance/iccom/"
            "3DBP-Measurement_Fabric_Properties-Virtual_Simulation.pdf",
        ),
        notes=(
            "The 31-fabric outerwear study reports 125-258 g/m^2.",
            "Service strains and shear range follow published quality envelopes for suiting.",
            "The bending value is a lower-bound mohair/wool quality reference, not a mean.",
        ),
    ),
}


def measured_material_profile(profile_id: str) -> MeasuredFabricProfile:
    try:
        return MATERIAL_PROFILES[profile_id]
    except KeyError as error:
        available = ", ".join(sorted(MATERIAL_PROFILES))
        raise KeyError(f"unknown measured material profile {profile_id!r}; choose {available}") from error
