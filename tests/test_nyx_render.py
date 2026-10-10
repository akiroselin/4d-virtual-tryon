from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aigc_fashion.data import GarmentMesh  # noqa: E402
from aigc_fashion.nyx_render import (  # noqa: E402
    _build_print_atlas,
    export_nyx_scene_assets,
    probe_nyx_backend,
)


def test_nyx_backend_probe_reports_native_boundary() -> None:
    status = probe_nyx_backend()
    assert status.platform
    assert status.machine
    assert "CUDA" in status.requirement
    assert bool(status.reason)
    if status.available:
        assert status.gs_nyx_available
        assert status.plugin_available


def test_nyx_print_atlas_uses_one_white_cloth_base() -> None:
    source = Image.new("RGB", (64, 64), (255, 255, 255))
    pixels = np.asarray(source).copy()
    pixels[18:48, 20:44] = (184, 112, 42)
    atlas = np.asarray(_build_print_atlas(Image.fromarray(pixels), size=128))
    expected_base = np.asarray([244, 244, 241])
    for y, x in ((0, 0), (0, 127), (127, 0), (127, 127), (12, 64)):
        assert np.array_equal(atlas[y, x], expected_base)
    assert np.any(np.linalg.norm(atlas.astype(np.int16) - expected_base, axis=2) > 40)


def test_glb_uv_height_is_not_flipped(tmp_path: Path) -> None:
    avatar = SimpleNamespace(
        mesh_vertices=np.asarray(
            [
                [-0.2, 0.0, -0.1],
                [0.2, 0.0, -0.1],
                [0.0, 0.6, -0.1],
                [0.0, 0.2, 0.2],
            ],
            dtype=np.float32,
        ),
        mesh_faces=np.asarray(
            [[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]],
            dtype=np.int32,
        ),
        base_z=0.0,
    )
    vertices = np.asarray(
        [
            [-0.2, 0.0, 0.1],
            [0.2, 0.0, 0.1],
            [0.2, 0.5, 0.1],
            [-0.2, 0.5, 0.1],
        ],
        dtype=np.float32,
    )
    garment = GarmentMesh(
        vertices=vertices,
        faces=np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int32),
        pinned_vertices=np.empty(0, dtype=np.int32),
        panel_labels=["front_front_body"] * 4,
        seam_edges=[],
        rest_vertices=vertices.copy(),
        uvs=np.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float32),
    )
    texture = tmp_path / "source.png"
    Image.new("RGB", (32, 32), (190, 110, 40)).save(texture)
    assets = export_nyx_scene_assets(tmp_path / "assets", avatar, garment, vertices, texture)
    scene = trimesh.load(assets["garment_glb"], force="scene", process=False)
    printed = next(
        geometry
        for geometry in scene.geometry.values()
        if geometry.visual.material.name == "cotton_jersey_print"
    )
    assert np.corrcoef(printed.vertices[:, 0], printed.visual.uv[:, 0])[0, 1] > 0.99
    assert np.corrcoef(printed.vertices[:, 2], printed.visual.uv[:, 1])[0, 1] > 0.99
    preview = np.asarray(Image.open(assets["base_color_texture"]).convert("RGB"))
    embedded = np.asarray(printed.visual.material.baseColorTexture.convert("RGB"))
    assert np.array_equal(embedded, np.flipud(preview))
