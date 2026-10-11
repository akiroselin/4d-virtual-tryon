import numpy as np

from aigc_fashion.measured_materials import (
    GF_CM2_PER_CM_TO_N_M,
    GF_PER_CM_PER_DEGREE_TO_N_PER_M_PER_RAD,
    measured_material_profile,
    material_motion_tuning,
)
from aigc_fashion.patterns import material_from_intent
from aigc_fashion.semantics import parse_garment_intent


def test_kes_unit_conversions_are_si() -> None:
    assert np.isclose(GF_CM2_PER_CM_TO_N_M, 9.80665e-5)
    assert np.isclose(GF_PER_CM_PER_DEGREE_TO_N_PER_M_PER_RAD, 56.1880676)


def test_profiles_preserve_ranges_instead_of_claiming_exact_swatches() -> None:
    jersey = measured_material_profile("cotton_single_jersey_reference")
    suiting = measured_material_profile("wool_blend_suiting_reference")

    assert jersey.areal_density.minimum < jersey.areal_density.maximum
    assert suiting.areal_density.minimum < suiting.areal_density.maximum
    assert "not the photographed swatch" in jersey.evidence_status
    assert suiting.wale_or_warp_service_strain is not None


def test_cotton_and_nylon_receive_stable_motion_priors() -> None:
    cotton = material_motion_tuning("medium cotton jersey")
    nylon = material_motion_tuning("medium nylon")
    assert nylon.velocity_damping_per_second > cotton.velocity_damping_per_second
    assert nylon.contact_damping_n_s_m > cotton.contact_damping_n_s_m

    intent = parse_garment_intent("medium nylon walking jacket")
    material = material_from_intent(intent)
    assert "nylon" in material.name
    assert material.damping < 0.97
