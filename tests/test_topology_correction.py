from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aigc_fashion.topology_correction import (  # noqa: E402
    SurfaceDomain,
    TopologyCorrectionStage,
    TopologyDisposition,
    analyze_triangle_mesh,
)


def _tetrahedron() -> tuple[np.ndarray, np.ndarray]:
    vertices = np.array(
        [[1, 1, 1], [-1, -1, 1], [-1, 1, -1], [1, -1, -1]],
        dtype=float,
    )
    faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    return vertices, faces


def test_closed_genus_zero_initialization_passes_gate() -> None:
    vertices, faces = _tetrahedron()
    report = analyze_triangle_mesh(vertices, faces)
    assert report.euler_characteristic == 2
    assert report.genus == 0
    assert report.is_connected_genus_zero
    result = TopologyCorrectionStage().validate_extracted_mesh(
        vertices,
        faces,
        domain=SurfaceDomain.CLOSED_LEVEL_SET,
    )
    assert result.disposition is TopologyDisposition.VERIFIED


def test_open_panel_keeps_material_boundary_instead_of_spherical_correction() -> None:
    vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float)
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    report = analyze_triangle_mesh(vertices, faces)
    assert report.boundary_loops == 1
    assert report.genus == 0
    result = TopologyCorrectionStage().validate_extracted_mesh(
        vertices,
        faces,
        domain=SurfaceDomain.GARMENT_PANEL_ATLAS,
    )
    assert result.disposition is TopologyDisposition.PRESERVE_PANEL_TOPOLOGY


def test_fast_marching_is_for_closed_sdf_only() -> None:
    stage = TopologyCorrectionStage()
    sdf = np.zeros((5, 5, 5), dtype=float)
    with pytest.raises(ValueError, match="open garment panel atlas"):
        stage.correct_closed_sdf(
            sdf,
            threshold=1.0,
            domain=SurfaceDomain.GARMENT_PANEL_ATLAS,
        )
    with pytest.raises(RuntimeError, match="configured backend"):
        stage.correct_closed_sdf(
            sdf,
            threshold=1.0,
            domain=SurfaceDomain.CLOSED_LEVEL_SET,
        )


def test_nonmanifold_panel_is_rejected() -> None:
    vertices = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1]],
        dtype=float,
    )
    faces = np.array([[0, 1, 2], [1, 0, 3], [0, 1, 4]])
    result = TopologyCorrectionStage().validate_extracted_mesh(
        vertices,
        faces,
        domain=SurfaceDomain.GARMENT_PANEL_ATLAS,
    )
    assert result.report.nonmanifold_edges == 1
    assert result.disposition is TopologyDisposition.REJECTED


def test_open_intrinsic_shell_preserves_genus_and_boundary_loops() -> None:
    vertices = np.array(
        [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]], dtype=float
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    result = TopologyCorrectionStage().validate_extracted_mesh(
        vertices,
        faces,
        domain=SurfaceDomain.GARMENT_INTRINSIC_SHELL,
    )
    assert result.disposition is TopologyDisposition.PRESERVE_SHELL_TOPOLOGY
    assert result.report.genus == 0
    assert result.report.boundary_loops == 1
