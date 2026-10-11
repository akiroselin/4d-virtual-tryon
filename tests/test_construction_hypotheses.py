from dataclasses import replace

import numpy as np

from aigc_fashion.construction_hypotheses import (
    generate_physical_structure_hypotheses,
    validate_cortexode_refinement_authority,
)
from aigc_fashion.data import AvatarSpec, GarmentIntent, GarmentMesh


def _intent(garment_type: str = "t-shirt") -> GarmentIntent:
    return GarmentIntent(
        prompt=f"relaxed cotton {garment_type}",
        garment_type=garment_type,
        silhouette="relaxed",
        fabric="medium cotton jersey",
    )


def test_tshirt_top_k_is_body_graded_and_construction_complete() -> None:
    hypotheses = generate_physical_structure_hypotheses(
        _intent(),
        AvatarSpec(height_cm=128, chest_cm=64, waist_cm=56, shoulder_cm=30),
        measurement_overrides_cm={"upper_arm_circumference": 21.0},
        top_k=3,
    )
    assert len(hypotheses) == 3
    assert [item.rank for item in hypotheses] == [1, 2, 3]
    assert all(
        hypotheses[index].score >= hypotheses[index + 1].score
        for index in range(len(hypotheses) - 1)
    )
    assert all(item.simulation_structure_complete for item in hypotheses)
    assert hypotheses[0].panel_roles == (
        "front_body",
        "back_body",
        "left_sleeve",
        "right_sleeve",
    )
    assert hypotheses[0].openings == ("neck", "hem", "left_cuff", "right_cuff")
    upper_arm = next(
        value for value in hypotheses[0].measurements if value.name == "upper_arm_circumference"
    )
    assert upper_arm.source == "user"
    chest_ease = next(
        value for value in hypotheses[0].graded_parameters if value.name == "chest_ease_cm"
    )
    assert chest_ease.low <= chest_ease.value <= chest_ease.high


def test_unimplemented_skirt_is_not_claimed_simulation_ready() -> None:
    hypothesis = generate_physical_structure_hypotheses(
        _intent("skirt"),
        AvatarSpec(height_cm=128, chest_cm=64, waist_cm=56, shoulder_cm=30),
        top_k=1,
    )[0]
    assert not hypothesis.simulation_structure_complete
    assert hypothesis.openings == ("waist", "hem")
    assert any("sewing graph is incomplete" in warning for warning in hypothesis.warnings)
    hip = next(value for value in hypothesis.measurements if value.name == "hip_circumference")
    assert not hip.observed
    assert hip.uncertainty_cm >= 7.0


def test_cortexode_authority_gate_rejects_rest_metric_mutation() -> None:
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32
    )
    source = GarmentMesh(
        vertices=vertices,
        faces=np.array([[0, 1, 2]], dtype=np.int32),
        pinned_vertices=np.array([0], dtype=np.int32),
        panel_labels=["front"] * 3,
        seam_edges=[],
        rest_vertices=vertices.copy(),
        uvs=vertices[:, :2].copy(),
    )
    moved = replace(source, vertices=vertices + np.array([0.0, 0.0, 0.2], dtype=np.float32))
    assert validate_cortexode_refinement_authority(source, moved).passed

    invalid = replace(moved, rest_vertices=moved.vertices.copy())
    report = validate_cortexode_refinement_authority(source, invalid)
    assert not report.passed
    assert report.reasons == ("construction rest metric changed",)
