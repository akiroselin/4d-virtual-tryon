from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aigc_fashion.data import GarmentMesh
from aigc_fashion.mesh_refinement import midpoint_subdivide_garment


def test_midpoint_refinement_preserves_shell_topology_and_state() -> None:
    rest = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    mesh = GarmentMesh(
        vertices=rest.copy(),
        faces=np.asarray([[0, 1, 2], [1, 3, 2]], dtype=np.int32),
        pinned_vertices=np.empty(0, dtype=np.int32),
        panel_labels=["front_front_body"] * 4,
        seam_edges=[],
        rest_vertices=rest.copy(),
        uvs=rest[:, :2].copy(),
    )
    current = rest + np.asarray([0.0, 0.0, 0.2], dtype=np.float32)

    refined, refined_current, report = midpoint_subdivide_garment(
        mesh, current, levels=1
    )

    assert refined.vertices.shape == (9, 3)
    assert refined.faces.shape == (8, 3)
    assert refined_current.shape == (9, 3)
    assert refined.uvs is not None and refined.uvs.shape == (9, 2)
    assert report.edges_before == 5
    assert report.edges_after == 16
    assert report.boundary_loops == 1
    assert report.genus == 0
    assert np.allclose(refined_current[4:], refined.vertices[4:] + [0.0, 0.0, 0.2])


def test_midpoint_refinement_rejects_mismatched_dynamic_state() -> None:
    mesh = GarmentMesh(
        vertices=np.zeros((3, 3), dtype=np.float32),
        faces=np.asarray([[0, 1, 2]], dtype=np.int32),
        pinned_vertices=np.empty(0, dtype=np.int32),
        panel_labels=["front_front_body"] * 3,
        seam_edges=[],
    )
    with pytest.raises(ValueError, match="current_positions"):
        midpoint_subdivide_garment(mesh, np.zeros((2, 3), dtype=np.float32))
