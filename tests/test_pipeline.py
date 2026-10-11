from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile

import numpy as np
from PIL import Image
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aigc_fashion import AvatarSpec, BodyMeshAvatar, SmplAvatarProxy, TshirtSizeSpec, build_body_sdf_tshirt_shell, parse_garment_intent, run_pipeline  # noqa: E402
from aigc_fashion.data import MaterialSpec  # noqa: E402
from aigc_fashion.learned_panel_mesh import (  # noqa: E402
    _classify_panel_labels,
    _point_in_polygon,
    _sample_panel_edges,
    _subdivide_panel_once,
    garment_mesh_from_sewformer_spec,
)
from aigc_fashion.learned_pattern import reconstruct_tshirt_pattern_from_image  # noqa: E402
from aigc_fashion.image_fit import infer_tshirt_size_from_image  # noqa: E402
from aigc_fashion.implicit_shell import _ordered_boundary_loops  # noqa: E402
from aigc_fashion.pattern_program import write_image_conditioned_tshirt_spec  # noqa: E402
from aigc_fashion.pattern_library import classify_garment_intent, get_pattern_template, load_pattern_library  # noqa: E402
from aigc_fashion.pattern_validation import _edge_length, validate_tshirt_pattern_spec  # noqa: E402
from aigc_fashion.patterns import build_tshirt_pattern, generate_tshirt_mesh  # noqa: E402
from aigc_fashion.physics import (  # noqa: E402
    SimulationConfig,
    _active_seam_edges,
    _unique_edges,
    simulate_intrinsic_cloth,
)
from aigc_fashion.body_mesh_avatar import _load_obj_mesh, _morph_body_measurements, _remap_axes  # noqa: E402
from aigc_fashion.vto_preflight import VtoPreflightError, infer_panel_role, run_vto_preflight  # noqa: E402


def test_intent_parser() -> None:
    intent = parse_garment_intent("oversized silk v-neck streetwear tee")
    assert intent.silhouette == "oversized"
    assert intent.fabric == "light silk"
    assert intent.neckline == "v-neck"


def test_pattern_mesh() -> None:
    intent = parse_garment_intent("relaxed cotton t-shirt")
    pattern = build_tshirt_pattern(intent, AvatarSpec(), resolution=13, size=TshirtSizeSpec(chest_circumference_cm=104.0))
    mesh = generate_tshirt_mesh(pattern)
    assert abs(pattern.body_width - 0.52) < 1e-6
    assert mesh.vertices.shape[0] > 20
    assert mesh.faces.shape[0] > 20
    assert mesh.pinned_vertices.shape[0] > 0
    assert mesh.rest_vertices is not None
    assert mesh.uvs is not None
    assert not np.allclose(mesh.vertices, mesh.rest_vertices)


def test_implicit_pattern_width_uses_half_circumference() -> None:
    intent = parse_garment_intent("relaxed cotton t-shirt")
    avatar = AvatarSpec(chest_cm=100.0)
    implicit = build_tshirt_pattern(intent, avatar, resolution=11)
    explicit = build_tshirt_pattern(
        intent,
        avatar,
        resolution=11,
        size=TshirtSizeSpec(chest_circumference_cm=114.0),
    )
    assert abs(implicit.body_width - explicit.body_width) < 1e-6


def test_avatar_collision_support() -> None:
    intent = parse_garment_intent("relaxed cotton t-shirt")
    pattern = build_tshirt_pattern(intent, AvatarSpec(), resolution=11)
    mesh = generate_tshirt_mesh(pattern)
    avatar = SmplAvatarProxy(AvatarSpec())
    dressed = avatar.dress_mesh(mesh)
    assert dressed.vertices[:, 2].mean() != mesh.vertices[:, 2].mean()
    _, final, metrics = simulate_intrinsic_cloth(dressed, pattern.material, SimulationConfig(steps=3), avatar=avatar)
    front_support = avatar.support_height_np(final[:, :2]) + avatar.clearance - 1e-5
    back_support = avatar.back_height_np(final[:, :2]) - avatar.clearance + 1e-5
    is_back = [label.startswith("back_") for label in dressed.panel_labels]
    for idx, back in enumerate(is_back):
        if back and back_support[idx] < 1.0e5:
            assert final[idx, 2] <= back_support[idx]
        if not back and front_support[idx] > -1.0e5:
            assert final[idx, 2] >= front_support[idx]
    assert "avatar_collision_total_corrections" in metrics
    assert metrics["prefit_metric_growth_max"] >= 1.0


def test_pipeline_outputs() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        summary = run_pipeline("relaxed cotton t-shirt", Path(tmp), steps=4, resolution=11)
        assert Path(summary["artifacts"]["obj"]).exists()
        assert Path(summary["artifacts"]["gif"]).exists()
        assert Path(summary["artifacts"]["final_png"]).exists()
        assert summary["physics"]["engine"] == "intrinsic_laplacian_mesh_propagator"
        assert summary["physics"]["initial_state"] == "dressed_on_avatar"
        assert summary["physics"]["stretch_tolerance"] < 0.03
        assert summary["fit_validation"]["status"] in {"pass", "fail"}
        assert summary["vto_preflight"]["status"] in {"pass", "warn", "fail"}
        assert Path(summary["artifacts"]["vto_preflight"]).exists()


def test_body_mesh_avatar_loader() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        mesh_path = Path(tmp) / "body_mesh.obj"
        _write_body_mesh_obj(mesh_path)
        avatar = BodyMeshAvatar.from_obj(mesh_path, AvatarSpec())
        support = avatar.support_height_np([[0.0, 0.0]])
        back = avatar.back_height_np([[0.0, 0.0]])
        assert avatar.summary()["type"] == "body_mesh"
        assert support[0] > back[0]

        pattern = build_tshirt_pattern(parse_garment_intent("relaxed cotton t-shirt"), AvatarSpec(), resolution=11)
        dressed = avatar.dress_mesh(
            generate_tshirt_mesh(pattern),
            placement_mode="legacy_predrape",
        )
        pinned = set(int(index) for index in dressed.pinned_vertices)
        for side in ("left_sleeve", "right_sleeve"):
            sleeve_pins = [
                index
                for index, label in enumerate(dressed.panel_labels)
                if side in label and index in pinned
            ]
            assert len(sleeve_pins) >= 2
            pin_positions = dressed.vertices[sleeve_pins]
            pairwise = np.linalg.norm(
                pin_positions[:, None, :] - pin_positions[None, :, :],
                axis=-1,
            )
            assert np.max(pairwise) > 0.02


def test_body_measurements_change_mesh_shape() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        mesh_path = Path(tmp) / "body_mesh.obj"
        _write_body_mesh_obj(mesh_path)
        narrow = BodyMeshAvatar.from_obj(mesh_path, AvatarSpec(height_cm=170, chest_cm=72, waist_cm=60, shoulder_cm=36))
        wide = BodyMeshAvatar.from_obj(mesh_path, AvatarSpec(height_cm=170, chest_cm=100, waist_cm=88, shoulder_cm=48))
        assert np.max(np.abs(narrow.mesh_vertices - wide.mesh_vertices)) > 0.015


def test_child_measurement_morph_preserves_arm_span() -> None:
    mesh_path = ROOT / "assets" / "body_meshes" / "anny_child_neutral.obj"
    source, _faces = _load_obj_mesh(mesh_path)
    source = _remap_axes(source, "x", "z", "y")
    morphed = _morph_body_measurements(
        source,
        AvatarSpec(height_cm=128, chest_cm=64, waist_cm=56, shoulder_cm=30),
    )
    assert np.ptp(morphed[:, 0]) >= np.ptp(source[:, 0]) * 0.90
    assert np.linalg.norm(morphed - source, axis=1).max() < 0.05


def test_child_body_mesh_axes_are_inferred() -> None:
    mesh_path = ROOT / "assets" / "body_meshes" / "anny_child_neutral.obj"
    avatar = BodyMeshAvatar.from_obj(
        mesh_path,
        AvatarSpec(height_cm=128, chest_cm=64, waist_cm=56, shoulder_cm=30),
    )
    assert avatar.summary()["source_axes"] == {"x": "x", "up": "z", "front": "y"}
    assert avatar.summary()["shirt_landmarks"]["arm_cuff_x"] < 0.4


def test_sleeve_subdivision_preserves_boundary_paths() -> None:
    points = np.asarray([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
    faces = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    refined_points, refined_faces, refined_edges = _subdivide_panel_once(
        points,
        faces,
        {0: [0, 1], 1: [1, 2], 2: [2, 3], 3: [3, 0]},
    )
    assert refined_points.shape == (9, 2)
    assert refined_faces.shape == (8, 3)
    assert all(len(path) == 3 for path in refined_edges.values())


def test_invalid_learned_sleeve_geometry_is_rejected() -> None:
    report = validate_tshirt_pattern_spec(ROOT / "examples" / "synthetic_learned_tshirt_spec.json")
    assert not report.valid
    assert report.metrics["max_stitch_length_ratio"] > 2.0
    assert any("cap/armhole ratio" in error for error in report.errors)


def test_child_image_fit_writes_sewable_four_panel_pattern() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        image_path = tmp_path / "shirt.png"
        image = np.full((128, 128, 3), 255, dtype=np.uint8)
        image[20:116, 22:106] = 238
        Image.fromarray(image).save(image_path)
        avatar = AvatarSpec(height_cm=128, chest_cm=64, waist_cm=56, shoulder_cm=30)
        fit = infer_tshirt_size_from_image(image_path, chest_circumference_cm=69.76, avatar=avatar)
        assert fit.estimated_size.body_length_cm < 50.0
        assert fit.estimated_size.shoulder_width_cm < 35.0

        spec_path = write_image_conditioned_tshirt_spec(tmp_path / "pattern.json", fit.estimated_size, fit)
        report = validate_tshirt_pattern_spec(spec_path)
        assert report.valid
        assert report.metrics["panel_count"] == 4.0
        assert abs(report.metrics["left_sleeve_armhole_length_ratio"] - 1.0) < 1e-6
        assert abs(report.metrics["right_sleeve_armhole_length_ratio"] - 1.0) < 1e-6


def test_pattern_library_routes_to_commercial_provider_and_fabrication_sleeves() -> None:
    library = load_pattern_library()
    template = get_pattern_template("t-shirt", library)
    assert template.production_provider is not None
    assert template.production_provider.commercial_use
    assert template.production_provider.provider_id == "project_local"
    roles = {panel.role for panel in template.panels}
    assert {"front_body", "back_body", "left_sleeve", "right_sleeve"}.issubset(roles)
    assert template.topology == "front_back_bodice_with_whole_set_in_sleeves"


def test_sewfactory_split_sleeve_names_are_semantic() -> None:
    assert infer_panel_role("lfsleeve") == "front_left_sleeve"
    assert infer_panel_role("lbsleeve") == "back_left_sleeve"
    assert infer_panel_role("rfsleeve") == "front_right_sleeve"
    assert infer_panel_role("rbsleeve") == "back_right_sleeve"
    body = np.asarray([[-2, -3], [2, -3], [2, 3], [-2, 3]], dtype=np.float32)
    sleeve = np.asarray([[0, -1], [2, -1], [2, 1], [0, 1]], dtype=np.float32)
    labels = _classify_panel_labels(
        {
            "front": body,
            "back": body,
            "lfsleeve": sleeve,
            "lbsleeve": sleeve,
            "rfsleeve": sleeve,
            "rbsleeve": sleeve,
        }
    )
    assert labels == {
        "front": "front_front_body",
        "back": "back_front_body",
        "lfsleeve": "front_left_sleeve",
        "lbsleeve": "back_left_sleeve",
        "rfsleeve": "front_right_sleeve",
        "rbsleeve": "back_right_sleeve",
    }


def test_vto_preflight_accepts_open_isometric_arrangement() -> None:
    avatar_spec = AvatarSpec(height_cm=128, chest_cm=64, waist_cm=56, shoulder_cm=30)
    body_path = ROOT / "assets" / "body_meshes" / "anny_child_neutral.obj"
    avatar = BodyMeshAvatar.from_obj(body_path, avatar_spec)
    constraints = avatar.tshirt_fit_constraints()
    size = TshirtSizeSpec(
        chest_circumference_cm=70.0,
        body_length_cm=42.0,
        shoulder_width_cm=31.0,
        sleeve_length_cm=15.0,
        sleeve_opening_cm=constraints["minimum_sleeve_opening_cm"],
        neck_width_cm=13.0,
        armhole_circumference_cm=constraints["minimum_armhole_circumference_cm"],
    )
    with tempfile.TemporaryDirectory() as tmp:
        spec_path = write_image_conditioned_tshirt_spec(Path(tmp) / "pattern.json", size)
        validation = validate_tshirt_pattern_spec(spec_path)
        mesh = garment_mesh_from_sewformer_spec(spec_path, resolution=10)
        dressed = avatar.dress_mesh(mesh)
        classification = classify_garment_intent("t-shirt", "child cotton t-shirt")
        template = get_pattern_template(classification.template_id)
        report = run_vto_preflight(
            mesh=dressed,
            avatar_spec=avatar_spec,
            avatar_proxy=avatar,
            classification=classification,
            template=template,
            pattern_source="image_conditioned_pattern_program",
            tshirt_size=size,
            pattern_spec_path=spec_path,
            pattern_validation=validation,
            fit_constraints=constraints,
        )
        assert report.status == "pass"
        assert "template_panel_topology" not in report.failed_checks
        assert "sleeve_arm_coverage" not in report.failed_checks
        assert "initial_material_strain" not in report.failed_checks
        assert "initial_seam_closure" not in report.failed_checks
        assert "initial_body_collision" not in report.failed_checks
        assert report.metrics["initial_material_p95_edge_ratio"] >= 1.0


def test_strict_preflight_blocks_research_pattern_before_physics() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        spec_path = write_image_conditioned_tshirt_spec(root / "pattern.json", TshirtSizeSpec())
        out_dir = root / "run"
        with pytest.raises(VtoPreflightError, match="commercial_provider"):
            run_pipeline(
                "relaxed cotton t-shirt",
                out_dir,
                learned_pattern_spec=spec_path,
                resolution=9,
                strict_preflight=True,
            )
        assert (out_dir / "vto_preflight.json").exists()
        assert not (out_dir / "garment_mesh.obj").exists()


def test_avatar_conditioned_armhole_matches_upper_arm_fit() -> None:
    mesh_path = ROOT / "assets" / "body_meshes" / "anny_child_neutral.obj"
    avatar_spec = AvatarSpec(height_cm=128, chest_cm=64, waist_cm=56, shoulder_cm=30)
    avatar = BodyMeshAvatar.from_obj(mesh_path, avatar_spec)
    constraints = avatar.tshirt_fit_constraints()
    size = TshirtSizeSpec(
        chest_circumference_cm=69.76,
        body_length_cm=42.2,
        shoulder_width_cm=30.7,
        sleeve_length_cm=15.7,
        sleeve_opening_cm=constraints["minimum_sleeve_opening_cm"],
        neck_width_cm=12.9,
        armhole_circumference_cm=constraints["minimum_armhole_circumference_cm"],
    )
    with tempfile.TemporaryDirectory() as tmp:
        spec_path = write_image_conditioned_tshirt_spec(Path(tmp) / "pattern.json", size)
        report = validate_tshirt_pattern_spec(spec_path)
        assert report.valid
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        sleeve = spec["pattern"]["panels"]["left_sleeve"]
        points = np.asarray(sleeve["vertices"], dtype=np.float32)
        cap_length = sum(
            _edge_length(
                points[sleeve["edges"][edge_id]["endpoints"][0]],
                points[sleeve["edges"][edge_id]["endpoints"][1]],
                sleeve["edges"][edge_id].get("curvature"),
            )
            for edge_id in (3, 4, 5, 6)
        )
        assert cap_length >= constraints["minimum_armhole_circumference_cm"] - 0.05


def test_pipeline_uses_body_mesh_avatar() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        mesh_path = tmp_path / "body_mesh.obj"
        _write_body_mesh_obj(mesh_path)
        summary = run_pipeline("relaxed cotton t-shirt", tmp_path / "run", steps=2, resolution=9, body_mesh=mesh_path)
        assert summary["avatar_collision_proxy"]["type"] == "body_mesh"
        assert summary["artifacts"]["body_mesh"] == str(mesh_path)


def test_learned_pattern_missing_assets_is_explicit() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        image = Path(tmp) / "shirt.png"
        image.write_bytes(b"not-an-image-but-path-exists")
        result = reconstruct_tshirt_pattern_from_image(image, tmp, repo_path=Path(tmp) / "missing-sewformer")
        assert result.status == "unavailable"
        assert "repo not found" in result.message


def test_learned_pattern_spec_generates_mesh() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        spec_path = Path(tmp) / "synthetic_predicted_single_specification.json"
        _write_synthetic_sewformer_spec(spec_path)
        mesh = garment_mesh_from_sewformer_spec(spec_path, resolution=12)
        assert mesh.vertices.shape[0] > 20
        assert mesh.faces.shape[0] > 20
        assert len(mesh.seam_edges) > 0
        assert len(mesh.seam_interfaces) > 0
        assert "front_left_sleeve" in set(mesh.panel_labels)
        assert "front_right_sleeve" in set(mesh.panel_labels)


def test_segmented_armhole_stitches_form_one_ordered_interface() -> None:
    spec_path = ROOT / "outputs" / "anny_child_vto_bounded_prefit_final" / "image_conditioned_tshirt_spec.json"
    mesh = garment_mesh_from_sewformer_spec(spec_path, resolution=23)
    for side in ("left_sleeve", "right_sleeve"):
        front = next(
            interface
            for interface in mesh.seam_interfaces
            if {interface.panel_a, interface.panel_b} == {"front", side}
        )
        back = next(
            interface
            for interface in mesh.seam_interfaces
            if {interface.panel_a, interface.panel_b} == {"back", side}
        )

        def shoulder_sleeve_vertex(interface) -> int:
            if interface.panel_a == side:
                sleeve_path, body_path = interface.vertices_a, interface.vertices_b
            else:
                sleeve_path, body_path = interface.vertices_b, interface.vertices_a
            endpoint = int(
                np.argmax(mesh.material_vertices[np.asarray(body_path)[[0, -1]], 1])
            )
            return int(sleeve_path[0 if endpoint == 0 else -1])

        assert shoulder_sleeve_vertex(front) == shoulder_sleeve_vertex(back)


def test_avatar_arrangement_includes_back_waist_and_limb_frames() -> None:
    avatar = BodyMeshAvatar.from_obj(
        ROOT / "assets" / "body_meshes" / "anny_child_neutral.obj",
        AvatarSpec(chest_cm=64, waist_cm=56, shoulder_cm=30, height_cm=128, pose="a-pose"),
        source_x_axis="x",
        source_up_axis="z",
        source_front_axis="-y",
    )
    names = {point.name for point in avatar.arrangement.points}
    required = {
        "chest_front",
        "chest_back",
        "waist_front",
        "waist_back",
        "waist_side_L",
        "waist_side_R",
        "hip_front",
        "hip_back",
        "upper_arm_L",
        "sleeve_cuff_R",
        "wrist_L",
        "thigh_R",
        "lower_leg_L",
    }
    assert required <= names
    for point in avatar.arrangement.points:
        assert abs(np.dot(point.normal, point.wrap_axis)) < 1.0e-5
        assert point.offset >= avatar.clearance


def test_isometric_arrangement_starts_outside_body_and_preserves_pattern_metric() -> None:
    avatar = BodyMeshAvatar.from_obj(
        ROOT / "assets" / "body_meshes" / "anny_child_neutral.obj",
        AvatarSpec(chest_cm=64, waist_cm=56, shoulder_cm=30, height_cm=128, pose="a-pose"),
        source_x_axis="x",
        source_up_axis="auto",
        source_front_axis="auto",
    )
    mesh = garment_mesh_from_sewformer_spec(
        ROOT / "outputs" / "anny_child_vto_bounded_prefit_final" / "image_conditioned_tshirt_spec.json",
        resolution=23,
    )
    dressed = avatar.dress_mesh(mesh, placement_mode="arrangement_points")
    diagnostics = avatar.collision_diagnostics_np(dressed.vertices, dressed.faces)
    edges = _unique_edges(dressed.faces)
    rest = np.linalg.norm(
        dressed.material_vertices[edges[:, 0]] - dressed.material_vertices[edges[:, 1]],
        axis=1,
    )
    placed = np.linalg.norm(
        dressed.vertices[edges[:, 0]] - dressed.vertices[edges[:, 1]],
        axis=1,
    )
    ratio = placed / np.maximum(rest, 1.0e-8)
    assert diagnostics["avatar_inside_vertices_final"] == 0.0
    assert diagnostics["avatar_surface_intersecting_faces_final"] == 0.0
    assert np.percentile(ratio, 99.0) <= 1.005
    assert np.percentile(ratio, 1.0) >= 0.98
    assert dressed.pinned_vertices.size == 0
    assert avatar.summary()["arrangement"]["placement_mode"] == "arrangement_points"


def test_implicit_tshirt_shell_is_continuous_seam_free_and_collision_free() -> None:
    avatar = BodyMeshAvatar.from_obj(
        ROOT / "assets" / "body_meshes" / "anny_child_neutral.obj",
        AvatarSpec(chest_cm=64, waist_cm=56, shoulder_cm=30, height_cm=128, pose="a-pose"),
        source_x_axis="x",
        source_up_axis="z",
        source_front_axis="-y",
    )
    atlas = garment_mesh_from_sewformer_spec(
        ROOT / "outputs" / "anny_child_vto_bounded_prefit_final" / "image_conditioned_tshirt_spec.json",
        resolution=23,
    )
    shell, report = build_body_sdf_tshirt_shell(
        avatar,
        atlas,
        target_faces=700,
    )
    assert shell.seam_edges == []
    assert shell.seam_interfaces == []
    assert shell.pinned_vertices.size == 0
    assert report.runtime_seam_constraints is False
    assert avatar.summary()["source_axes"]["front"] == "-y"
    assert report.level_set_offset_m == pytest.approx(0.016)
    assert report.cuff_level_set_offset_m == pytest.approx(0.008)
    assert report.learned_cuff_opening_m == pytest.approx(0.1938089, abs=1.0e-5)
    assert report.learned_neck_width_m == pytest.approx(0.1289165, abs=1.0e-5)
    assert report.front_neck_drop_m == pytest.approx(0.026)
    assert report.topology.connected_components == 1
    assert report.topology.boundary_loops == 4
    assert report.topology.genus == 0
    assert report.topology.euler_characteristic == -2
    assert report.topology.nonmanifold_edges == 0
    assert report.collision["avatar_inside_vertices_final"] == 0.0
    assert report.collision["avatar_surface_intersecting_faces_final"] == 0.0
    cuff_circumferences: list[float] = []
    for loop in _ordered_boundary_loops(shell.faces):
        points = shell.vertices[loop]
        if abs(float(np.mean(points[:, 0]))) <= 0.15:
            continue
        closed = np.concatenate((points, points[:1]), axis=0)
        cuff_circumferences.append(
            float(np.linalg.norm(np.diff(closed, axis=0), axis=1).sum())
        )
        signed_distance = avatar._proximity.signed_distance(points)
        assert float(np.min(-signed_distance)) >= avatar.clearance
    assert len(cuff_circumferences) == 2
    assert max(cuff_circumferences) < 0.24


def test_open_stitches_do_not_share_collision_displacement() -> None:
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.10, 0.0, 0.0], [0.0, 0.0, 0.0], [0.01, 0.0, 0.0]],
        dtype=torch.float32,
    )
    seams = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
    active = _active_seam_edges(positions, seams, max_gap=0.025)
    assert active.tolist() == [[2, 3]]


def test_curved_edges_and_polygon_triangulation() -> None:
    source = np.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float32)
    straight, straight_edges = _sample_panel_edges(
        source,
        source,
        [{"endpoints": [0, 1]}, {"endpoints": [1, 2]}, {"endpoints": [2, 3]}, {"endpoints": [3, 0]}],
        0.1,
    )
    curved, curved_edges = _sample_panel_edges(
        source,
        source,
        [
            {"endpoints": [0, 1], "curvature": [0.5, 0.25]},
            {"endpoints": [1, 2]},
            {"endpoints": [2, 3]},
            {"endpoints": [3, 0]},
        ],
        0.1,
    )
    assert curved[curved_edges[0], 1].max() > straight[straight_edges[0], 1].max() + 0.05
    polygon = np.asarray([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]], dtype=np.float32)
    assert _point_in_polygon(np.asarray([1.0, 1.0]), polygon)
    assert not _point_in_polygon(np.asarray([3.0, 1.0]), polygon)


def test_material_and_seam_constraints_affect_rollout() -> None:
    intent = parse_garment_intent("relaxed cotton t-shirt")
    pattern = build_tshirt_pattern(intent, AvatarSpec(), resolution=9)
    mesh = generate_tshirt_mesh(pattern)
    config = SimulationConfig(steps=4, edge_projection_iterations=12)
    _, light, light_metrics = simulate_intrinsic_cloth(
        mesh,
        MaterialSpec(density=0.8, stretch_stiffness=4.0, bending_stiffness=0.2),
        config,
    )
    _, heavy, _ = simulate_intrinsic_cloth(
        mesh,
        MaterialSpec(density=8.0, stretch_stiffness=18.0, bending_stiffness=2.0),
        config,
    )
    assert np.max(np.abs(light - heavy)) > 1e-6
    assert light_metrics["mean_seam_gap_m"] < 0.02


def test_pipeline_rejects_unsewable_learned_pattern_spec() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        spec_path = tmp_path / "synthetic_predicted_single_specification.json"
        _write_synthetic_sewformer_spec(spec_path)
        summary = run_pipeline(
            "child cotton short sleeve t-shirt",
            tmp_path / "run",
            avatar=AvatarSpec(height_cm=128, chest_cm=64, waist_cm=56, shoulder_cm=30),
            steps=2,
            resolution=10,
            learned_pattern_spec=spec_path,
        )
        assert summary["learned_pattern_mesh"]["status"] == "rejected"
        assert summary["learned_pattern_mesh"]["source"] == "parametric_fallback"
        assert summary["pattern_validation"]["status"] == "invalid"
        assert Path(summary["artifacts"]["obj"]).exists()


def _write_body_mesh_obj(path: Path) -> None:
    verts = []
    faces = []
    rings = 18
    cols = 18
    for iy in range(rings):
        y = -0.55 + 1.1 * iy / (rings - 1)
        width = 0.12 + 0.10 * np.exp(-((y - 0.18) ** 2) / 0.08)
        depth = 0.07 + 0.09 * np.exp(-((y - 0.10) ** 2) / 0.10)
        for ix in range(cols):
            theta = 2.0 * np.pi * ix / cols
            verts.append((width * np.cos(theta), y, 0.88 + depth * np.sin(theta)))
    for iy in range(rings - 1):
        for ix in range(cols):
            a = iy * cols + ix
            b = iy * cols + (ix + 1) % cols
            c = (iy + 1) * cols + ix
            d = (iy + 1) * cols + (ix + 1) % cols
            faces.append((a + 1, b + 1, c + 1))
            faces.append((b + 1, d + 1, c + 1))
    text = ["# synthetic body mesh for loader tests\n"]
    text += [f"v {x:.6f} {y:.6f} {z:.6f}\n" for x, y, z in verts]
    text += [f"f {a} {b} {c}\n" for a, b, c in faces]
    path.write_text("".join(text), encoding="utf-8")


def _write_synthetic_sewformer_spec(path: Path) -> None:
    def panel(vertices: list[list[float]]) -> dict[str, object]:
        return {
            "vertices": vertices,
            "edges": [
                {"endpoints": [idx, (idx + 1) % len(vertices)]}
                for idx in range(len(vertices))
            ],
            "translation": [0, 0, 0],
            "rotation": [0, 0, 0],
        }

    spec = {
        "properties": {"units_in_meter": 100},
        "pattern": {
            "panels": {
                "front": panel([[-17.5, -20.0], [17.5, -20.0], [17.5, 20.0], [-17.5, 20.0]]),
                "back": panel([[-17.5, -20.0], [17.5, -20.0], [17.5, 20.0], [-17.5, 20.0]]),
                "left_sleeve": panel([[0.0, -5.5], [15.0, -5.5], [15.0, 5.5], [0.0, 5.5]]),
                "right_sleeve": panel([[0.0, -5.5], [15.0, -5.5], [15.0, 5.5], [0.0, 5.5]]),
            },
            "stitches": [
                [{"panel": "front", "edge": 1}, {"panel": "back", "edge": 1}],
                [{"panel": "front", "edge": 3}, {"panel": "back", "edge": 3}],
                [{"panel": "front", "edge": 2}, {"panel": "back", "edge": 2}],
                [{"panel": "left_sleeve", "edge": 3}, {"panel": "front", "edge": 3}],
                [{"panel": "right_sleeve", "edge": 3}, {"panel": "front", "edge": 1}],
                [{"panel": "left_sleeve", "edge": 0}, {"panel": "left_sleeve", "edge": 2}],
                [{"panel": "right_sleeve", "edge": 0}, {"panel": "right_sleeve", "edge": 2}],
            ],
        },
    }
    path.write_text(json.dumps(spec, indent=2), encoding="utf-8")


if __name__ == "__main__":
    test_intent_parser()
    test_pattern_mesh()
    test_implicit_pattern_width_uses_half_circumference()
    test_avatar_collision_support()
    test_pipeline_outputs()
    test_body_mesh_avatar_loader()
    test_body_measurements_change_mesh_shape()
    test_child_measurement_morph_preserves_arm_span()
    test_invalid_learned_sleeve_geometry_is_rejected()
    test_child_image_fit_writes_sewable_four_panel_pattern()
    test_pipeline_uses_body_mesh_avatar()
    test_learned_pattern_missing_assets_is_explicit()
    test_learned_pattern_spec_generates_mesh()
    test_curved_edges_and_polygon_triangulation()
    test_material_and_seam_constraints_affect_rollout()
    test_pipeline_rejects_unsewable_learned_pattern_spec()
    print("aigc fashion prototype tests passed")
