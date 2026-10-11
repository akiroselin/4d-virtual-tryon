from __future__ import annotations

import numpy as np

from aigc_fashion.adaptive_intrinsic_mesh import (
    AdaptiveAMRConfig,
    _refine_selected_once,
    adaptive_refine_garment,
    intrinsic_face_error_indicator,
)
from aigc_fashion.data import GarmentMesh


def _grid_mesh() -> GarmentMesh:
    vertices = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.5, 0.0],
            [0.5, 0.5, 0.0],
            [1.0, 0.5, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    faces = np.asarray(
        [
            [0, 1, 4], [0, 4, 3],
            [1, 2, 5], [1, 5, 4],
            [3, 4, 7], [3, 7, 6],
            [4, 5, 8], [4, 8, 7],
        ],
        dtype=np.int32,
    )
    return GarmentMesh(
        vertices=vertices,
        faces=faces,
        pinned_vertices=np.asarray([0, 2, 6, 8], dtype=np.int32),
        panel_labels=["front_body"] * len(vertices),
        seam_edges=[],
        rest_vertices=vertices.copy(),
        uvs=vertices[:, :2].copy(),
    )


def test_indicator_is_invariant_to_rigid_transform() -> None:
    mesh = _grid_mesh()
    material = np.asarray([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    baseline, _ = intrinsic_face_error_indicator(
        mesh,
        mesh.vertices,
        material_field=material,
        config=AdaptiveAMRConfig(
            curvature_weight=0.0,
            strain_weight=0.0,
            material_gradient_weight=1.0,
            contact_weight=0.0,
            energy_residual_weight=0.0,
        ),
    )
    angle = 0.73
    rotation = np.asarray(
        [[np.cos(angle), 0.0, -np.sin(angle)], [0.0, 1.0, 0.0], [np.sin(angle), 0.0, np.cos(angle)]]
    )
    translated = mesh.vertices @ rotation.T + np.asarray([3.0, -2.0, 1.0])
    rotated_mesh = GarmentMesh(
        vertices=translated,
        faces=mesh.faces,
        pinned_vertices=mesh.pinned_vertices,
        panel_labels=mesh.panel_labels,
        seam_edges=[],
        rest_vertices=translated.copy(),
        uvs=mesh.uvs,
    )
    transformed, _ = intrinsic_face_error_indicator(
        rotated_mesh,
        translated,
        material_field=material,
        config=AdaptiveAMRConfig(
            curvature_weight=0.0,
            strain_weight=0.0,
            material_gradient_weight=1.0,
            contact_weight=0.0,
            energy_residual_weight=0.0,
        ),
    )
    assert np.allclose(baseline, transformed)


def test_local_amr_preserves_shell_and_state_conservation() -> None:
    mesh = _grid_mesh()
    current = mesh.vertices.copy()
    current[4, 2] = 0.18
    velocities = np.asarray(
        [[0.1, -0.2, 0.05]] * len(current), dtype=np.float64
    )
    material = np.zeros(len(current), dtype=np.float64)
    material[4] = 1.0
    refined, state, report = adaptive_refine_garment(
        mesh,
        current,
        velocities=velocities,
        material_field=material,
        config=AdaptiveAMRConfig(
            passes=1,
            refine_fraction=0.25,
            indicator_threshold=0.2,
            curvature_weight=0.0,
            strain_weight=0.0,
            material_gradient_weight=1.0,
            contact_weight=0.0,
            energy_residual_weight=0.0,
        ),
    )
    assert report.passes_applied == 1
    assert len(refined.vertices) > len(mesh.vertices)
    assert len(refined.faces) > len(mesh.faces)
    assert report.topology_preserved
    assert refined.uvs is not None
    assert state.material_field is not None
    assert np.max(np.abs(state.velocities.mean(axis=0) - velocities.mean(axis=0))) < 1.0e-6
    assert report.linear_momentum_error_kg_m_s < 1.0e-8
    assert report.angular_momentum_error_kg_m2_s < 1.0e-8


def test_curvature_only_rest_shell_crosses_default_refinement_threshold() -> None:
    mesh = _grid_mesh()
    current = mesh.vertices.copy()
    current[4, 2] = 0.18
    curved = GarmentMesh(
        vertices=current,
        faces=mesh.faces,
        pinned_vertices=mesh.pinned_vertices,
        panel_labels=mesh.panel_labels,
        seam_edges=[],
        rest_vertices=current.copy(),
        uvs=mesh.uvs,
    )
    indicator, _ = intrinsic_face_error_indicator(curved, current)

    assert float(indicator.max()) == 1.0
    assert np.any(indicator >= AdaptiveAMRConfig().indicator_threshold)


def test_amr_boundary_midpoint_samples_curved_limit_surface() -> None:
    angles = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    ring = np.stack((np.cos(angles), np.zeros(8), np.sin(angles)), axis=1)
    vertices = np.concatenate((ring, np.zeros((1, 3))), axis=0).astype(np.float32)
    faces = np.asarray(
        [[index, (index + 1) % 8, 8] for index in range(8)], dtype=np.int32
    )
    mesh = GarmentMesh(
        vertices=vertices,
        faces=faces,
        pinned_vertices=np.empty(0, dtype=np.int32),
        panel_labels=["skirt"] * len(vertices),
        seam_edges=[],
        rest_vertices=vertices.copy(),
        uvs=None,
    )

    refined, positions, parents = _refine_selected_once(
        mesh, vertices, np.asarray([0], dtype=np.int32)
    )
    midpoint_index = next(
        index for index, edge in parents.items() if edge == (0, 1)
    )
    linear_midpoint = 0.5 * (vertices[0] + vertices[1])

    assert np.linalg.norm(positions[midpoint_index, [0, 2]]) > np.linalg.norm(
        linear_midpoint[[0, 2]]
    )
    assert np.allclose(refined.material_vertices, positions)


def test_amr_respects_face_budget() -> None:
    mesh = _grid_mesh()
    refined, _, report = adaptive_refine_garment(
        mesh,
        mesh.vertices,
        material_field=np.linspace(0.0, 1.0, len(mesh.vertices)),
        config=AdaptiveAMRConfig(
            maximum_faces=len(mesh.faces) + 1,
            indicator_threshold=0.0,
            curvature_weight=0.0,
            strain_weight=0.0,
            material_gradient_weight=1.0,
            contact_weight=0.0,
            energy_residual_weight=0.0,
        ),
    )
    assert len(refined.faces) <= len(mesh.faces) + 1
    assert report.faces_after == len(refined.faces)
