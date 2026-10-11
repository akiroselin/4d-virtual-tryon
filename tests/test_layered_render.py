from pathlib import Path

import numpy as np
from PIL import Image

from aigc_fashion.data import GarmentMesh
from aigc_fashion.render import (
    _rotate_view_y,
    _smooth_vertex_normals,
    GarmentRenderLayer,
    render_layered_frame,
    render_layered_rollout,
)


def test_three_quarter_camera_exposes_depth_as_screen_horizontal_motion() -> None:
    points = np.asarray([[0.0, 1.0, 2.0]], dtype=np.float32)

    viewed = _rotate_view_y(points, np.pi / 2.0)

    assert np.allclose(viewed, [[2.0, 1.0, 0.0]], atol=1.0e-6)


def test_smooth_vertex_normals_remove_internal_triangle_normal_discontinuity() -> None:
    vertices = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.3]],
        dtype=np.float32,
    )
    faces = np.asarray([[0, 1, 2], [1, 3, 2]], dtype=np.int32)

    normals = _smooth_vertex_normals(vertices, faces)

    assert normals.shape == vertices.shape
    assert np.allclose(np.linalg.norm(normals, axis=1), 1.0)
    assert np.allclose(normals[1], normals[2])
    assert not np.allclose(normals[0], normals[1])


def _triangle(offset_y: float, label: str) -> GarmentMesh:
    vertices = np.asarray(
        [[-0.1, offset_y, 0.1], [0.1, offset_y, 0.1], [0.0, offset_y + 0.2, 0.1]],
        dtype=np.float32,
    )
    return GarmentMesh(
        vertices=vertices,
        faces=np.asarray([[0, 1, 2]], dtype=np.int32),
        pinned_vertices=np.empty(0, dtype=np.int32),
        panel_labels=[label] * 3,
        seam_edges=[],
    )


def test_layered_renderer_keeps_independent_meshes_and_timelines(tmp_path: Path) -> None:
    skirt = _triangle(-0.2, "front_pleated_skirt")
    shirt = _triangle(0.0, "front_front_body")
    image = render_layered_frame(
        [
            GarmentRenderLayer(skirt.vertices, skirt),
            GarmentRenderLayer(shirt.vertices, shirt, closed_sleeve_visibility=True),
        ],
        0,
        "two garment test",
        size=320,
    )
    assert image.size == (320, 320)

    path = tmp_path / "layered.gif"
    render_layered_rollout(
        path,
        [
            ([skirt.vertices, skirt.vertices.copy()], skirt, None, False),
            ([shirt.vertices], shirt, None, True),
        ],
        "two garment test",
    )
    with Image.open(path) as animation:
        assert animation.n_frames == 2
