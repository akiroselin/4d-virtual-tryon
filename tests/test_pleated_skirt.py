from pathlib import Path
import json
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image
import torch

from aigc_fashion.body_mesh_avatar import BodyMeshAvatar
from aigc_fashion.data import AvatarSpec, MaterialSpec
from aigc_fashion.physics import IntrinsicClothSimulator
from aigc_fashion.pleated_skirt import (
    build_body_conditioned_pleated_skirt_shell,
    build_pleated_skirt_construction,
    export_pleated_skirt_construction,
    extract_plaid_texture,
    infer_pleated_skirt_spec,
)
from aigc_fashion.image_pattern_provider import (
    probe_design2garmentcode,
    validate_garmentcode_pattern,
)


ROOT = Path(__file__).resolve().parents[1]


def _avatar() -> BodyMeshAvatar:
    return BodyMeshAvatar.from_obj(
        ROOT / "assets" / "body_meshes" / "anny_child_neutral.obj",
        AvatarSpec(chest_cm=64, waist_cm=56, shoulder_cm=30, height_cm=128, pose="a-pose"),
        source_up_axis="z",
        source_front_axis="-y",
    )


def test_pleated_skirt_is_annular_rest_curved_shell(tmp_path: Path) -> None:
    source = tmp_path / "skirt.png"
    image = np.full((180, 220, 3), 235, dtype=np.uint8)
    image[55:150, 38:185] = [62, 55, 48]
    Image.fromarray(image).save(source)
    spec = infer_pleated_skirt_spec(source)
    mesh, report = build_body_conditioned_pleated_skirt_shell(_avatar(), spec)

    assert report.topology.boundary_loops == 2
    assert report.topology.euler_characteristic == 0
    assert report.topology.nonmanifold_edges == 0
    assert not mesh.seam_edges
    assert len(mesh.pinned_vertices) == 60
    assert np.array_equal(mesh.vertices, mesh.rest_vertices)
    middle = mesh.vertices.reshape(15, 60, 3)[8]
    radial = np.linalg.norm(middle[:, [0, 2]] - np.array([0.0, middle[:, 2].mean()]), axis=1)
    assert float(np.ptp(radial)) > 0.015


def test_plaid_texture_removes_catalog_background(tmp_path: Path) -> None:
    source = tmp_path / "skirt.png"
    image = np.full((100, 100, 3), 240, dtype=np.uint8)
    image[30:80, 20:85] = [58, 51, 45]
    Image.fromarray(image).save(source)
    output = extract_plaid_texture(source, tmp_path / "plaid.png", size=128)
    texture = np.asarray(Image.open(output))
    assert texture.shape == (128, 128, 3)
    assert float(texture.mean()) < 120.0
    assert float(texture.std()) > 4.0


def test_pressed_pleats_are_force_free_rest_curvature(tmp_path: Path) -> None:
    source = tmp_path / "skirt.png"
    image = np.full((160, 200, 3), 235, dtype=np.uint8)
    image[50:140, 30:175] = [62, 55, 48]
    Image.fromarray(image).save(source)
    mesh, _ = build_body_conditioned_pleated_skirt_shell(
        _avatar(), infer_pleated_skirt_spec(source)
    )
    rest = torch.as_tensor(mesh.rest_vertices)
    model = IntrinsicClothSimulator(
        MaterialSpec(bending_stiffness=2.6),
        rest,
        torch.as_tensor(mesh.faces, dtype=torch.long),
    )
    next_positions, next_velocity = model(
        rest.clone(),
        torch.zeros_like(rest),
        torch.empty(0, dtype=torch.long),
        torch.empty((0, 3)),
        dt=0.006,
        gravity_y=0.0,
    )
    assert torch.allclose(next_positions, rest, atol=1.0e-8)
    assert torch.allclose(next_velocity, torch.zeros_like(rest), atol=1.0e-8)


def test_construction_preserves_box_pleat_uptake_and_fold_roles(tmp_path: Path) -> None:
    source = tmp_path / "skirt.png"
    image = np.full((160, 200, 3), 235, dtype=np.uint8)
    image[50:140, 30:175] = [62, 55, 48]
    Image.fromarray(image).save(source)
    spec = infer_pleated_skirt_spec(source)
    mesh, _ = build_body_conditioned_pleated_skirt_shell(_avatar(), spec)
    construction = build_pleated_skirt_construction(mesh, spec)

    assert len(construction.fold_lines) == 4 * spec.pleat_count
    assert np.isclose(
        construction.total_pleat_uptake_m,
        2.0 * spec.pleat_depth_m * spec.pleat_count,
    )
    assert construction.front_panel_cut_width_m * 2.0 > construction.finished_waist_circumference_m
    roles = {fold.role for fold in construction.fold_lines}
    assert roles == {
        "underfold_left",
        "outer_fold_left",
        "outer_fold_right",
        "underfold_right",
    }


def test_construction_exports_valid_garmentcode_and_svg(tmp_path: Path) -> None:
    source = tmp_path / "skirt.png"
    image = np.full((160, 200, 3), 235, dtype=np.uint8)
    image[50:140, 30:175] = [62, 55, 48]
    Image.fromarray(image).save(source)
    spec = infer_pleated_skirt_spec(source)
    mesh, _ = build_body_conditioned_pleated_skirt_shell(_avatar(), spec)
    construction = build_pleated_skirt_construction(mesh, spec)
    artifacts = export_pleated_skirt_construction(tmp_path / "manufacturing", construction)

    payload = json.loads(Path(artifacts["garmentcode_json"]).read_text())
    validation = validate_garmentcode_pattern(payload)
    assert validation.valid
    assert validation.panel_count == 4
    assert validation.stitch_count == 4
    assert payload["manufacturing_extensions"]["pleat_uptake_m"] > 0.0
    ET.parse(artifacts["pattern_svg"])


def test_design2garmentcode_probe_never_trusts_bundled_credentials(tmp_path: Path) -> None:
    repo = tmp_path / "design2garmentcode"
    (repo / "lmm_utils").mkdir(parents=True)
    (repo / "lmm_utils" / "predict_garmentcode_picture.py").write_text("# source")
    (repo / "lmm_utils" / "test_picture_batch.py").write_text("api_key='bundled-example'")
    status = probe_design2garmentcode(repo)
    assert status.source_available
    assert status.image_entrypoint_available
    assert not status.parameter_projector_available
    assert not status.executable_here
