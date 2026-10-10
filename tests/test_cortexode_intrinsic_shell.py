from dataclasses import replace
from pathlib import Path

import numpy as np

from aigc_fashion.cortexode_intrinsic_shell import (
    CortexODEIntrinsicShellConfig,
    build_coarse_intrinsic_initial_shell,
    reconstruct_intrinsic_shell_cortexode,
)
from aigc_fashion.cortexode_volume_pipeline import (
    CortexODEVolumeConfig,
    build_cortexode_volume_evidence,
    cap_garment_boundaries,
    semantic_topology_from_panel_analysis,
    shell_initialization_from_volume,
)
from aigc_fashion.data import GarmentMesh
from aigc_fashion.topology_correction import analyze_triangle_mesh


ROOT = Path(__file__).resolve().parents[1]


def _annular_shell(columns: int = 16, rows: int = 4) -> GarmentMesh:
    vertices = []
    uvs = []
    for row in range(rows):
        v = row / (rows - 1)
        radius = 0.13 + 0.025 * v
        for column in range(columns):
            u = column / columns
            theta = 2.0 * np.pi * u
            vertices.append([radius * np.cos(theta), 0.25 - 0.24 * v, radius * np.sin(theta)])
            uvs.append([u, 1.0 - v])
    faces = []
    for row in range(rows - 1):
        for column in range(columns):
            nxt = (column + 1) % columns
            a = row * columns + column
            b = row * columns + nxt
            c = (row + 1) * columns + column
            d = (row + 1) * columns + nxt
            faces.extend(([a, b, c], [b, d, c]))
    array = np.asarray(vertices, dtype=np.float32)
    return GarmentMesh(
        vertices=array,
        faces=np.asarray(faces, dtype=np.int32),
        pinned_vertices=np.arange(columns, dtype=np.int32),
        panel_labels=["front_skirt"] * len(array),
        seam_edges=[],
        rest_vertices=array.copy(),
        uvs=np.asarray(uvs, dtype=np.float32),
    )


def test_semantic_capping_converts_annulus_to_closed_genus_zero() -> None:
    garment = _annular_shell()
    topology = semantic_topology_from_panel_analysis(garment)
    vertices, faces = cap_garment_boundaries(garment)
    report = analyze_triangle_mesh(vertices, faces)
    assert report.is_connected_genus_zero
    assert [item["name"] for item in topology["openings"]] == ["waist", "hem"]
    assert all(
        item["physics_compilation"].startswith("free boundary")
        for item in topology["openings"]
    )


def test_cortexode_flow_keeps_open_shell_topology_and_has_physics_losses() -> None:
    target = _annular_shell()
    config = replace(
        CortexODEIntrinsicShellConfig(),
        optimization_steps=2,
        rk4_steps=1,
        hidden_dim=16,
        hidden_layers=2,
        physics_interval=1,
    )
    reconstructed, report, trajectory = reconstruct_intrinsic_shell_cortexode(
        target,
        initial=target,
        config=config,
    )
    topology = analyze_triangle_mesh(reconstructed.vertices, reconstructed.faces)
    assert report.topology_preserved
    assert topology.boundary_loops == 2
    assert topology.genus == 0
    assert trajectory.shape[1:] == target.vertices.shape
    assert any("Kirchhoff-Love" in item for item in report.physics_supervision)
    assert np.array_equal(reconstructed.material_vertices, target.material_vertices)
    assert report.rest_metric_source.startswith("GarmentMesh.material_vertices")
    assert report.seam_constraint_pairs == 0
    assert any("self-collision" in item for item in report.physics_limitations)
    assert report.metric_projection["violating_edges"] == 0
    assert report.material_strain_profile == "elastic_knit"


def test_coarse_cortexode_initialization_preserves_construction_rest_metric() -> None:
    target = _annular_shell()
    rest = target.material_vertices.copy()
    target.vertices[16:32, 2] += 0.02
    initial = build_coarse_intrinsic_initial_shell(target, iterations=2)
    assert np.array_equal(initial.material_vertices, rest)


def test_full_volume_chain_uses_fast_marching_and_restores_open_shell() -> None:
    garment = _annular_shell()
    evidence, report = build_cortexode_volume_evidence(
        garment,
        cortexode_checkout=ROOT.parent.parent / "work" / "cortexode",
        config=CortexODEVolumeConfig(resolution=32, unet_steps=8),
    )
    initial = shell_initialization_from_volume(garment, evidence, smoothing_iterations=2)
    closed = analyze_triangle_mesh(evidence.closed_vertices, evidence.closed_faces)
    opened = analyze_triangle_mesh(initial.vertices, initial.faces)
    assert report.segmentation_dice > 0.7
    assert report.semantic_openings == ("waist", "hem")
    assert closed.is_connected_genus_zero
    assert opened.boundary_loops == 2
    assert opened.genus == 0
