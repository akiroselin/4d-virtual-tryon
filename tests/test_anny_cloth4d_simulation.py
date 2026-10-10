from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aigc_fashion.anny_cloth4d_simulation import (  # noqa: E402
    AnnyCloth4DConfig,
    _cotangent_spectrum,
    _load_conditioner_checkpoint_state,
    _material_chart_label,
    _shell_physics_training_terms,
    _weld_seam_vertices,
    probe_genesis_ipc_backend,
)
from aigc_fashion.cloth4d_geometric_bridge import Cloth4DGeometricConditioner  # noqa: E402
from aigc_fashion.data import MaterialSpec  # noqa: E402
from aigc_fashion.hypernetwork_kirchhoff_love import (  # noqa: E402
    HyperNetworkKirchhoffLoveShell,
    build_kirchhoff_love_shell_geometry,
)
from aigc_fashion.physics_conditioned_flow import SceneMaterialLatent  # noqa: E402


def test_cotangent_spectrum_is_mass_orthonormal() -> None:
    vertices = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    faces = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    values, basis, mass, edges, weights = _cotangent_spectrum(vertices, faces, 3)
    assert values.shape == (3,)
    assert np.all(values >= -1.0e-7)
    assert edges.shape == (5, 2)
    assert weights.shape == (5,)
    gram = basis.T @ (mass[:, None] * basis)
    assert np.allclose(gram, np.eye(3), atol=2.0e-5)


def test_seam_welding_returns_original_to_welded_mapping() -> None:
    vertices = np.asarray(
        [
            [0.0, 0.0, 0.1],
            [1.0, 0.0, 0.1],
            [0.0, 1.0, 0.1],
            [0.0, 0.0, -0.1],
            [1.0, 0.0, -0.1],
            [0.0, 1.0, -0.1],
        ],
        dtype=np.float32,
    )
    faces = np.asarray([[0, 1, 2], [5, 4, 3]], dtype=np.int32)
    welded_vertices, welded_faces, mapping = _weld_seam_vertices(
        vertices,
        faces,
        [(0, 3), (1, 4), (2, 5)],
    )
    assert welded_vertices.shape == (3, 3)
    assert welded_faces.shape == (1, 3)
    assert np.array_equal(mapping[:3], mapping[3:])
    assert np.allclose(welded_vertices[:, 2], 0.0)


def test_genesis_probe_never_silently_selects_stock_pbd(tmp_path: Path) -> None:
    status = probe_genesis_ipc_backend(tmp_path / "missing-genesis")
    assert not status.genesis_source_available
    assert not status.executable_here
    assert not status.stock_pbd_used
    assert status.requested_backend == "Genesis FEM.Cloth + IPC"


def test_runtime_channel_contract_is_explicit() -> None:
    with pytest.raises(ValueError, match="requires at least"):
        AnnyCloth4DConfig(intrinsic_channels=5)


def test_dynamic_metric_projection_has_dense_adaptive_budget() -> None:
    config = AnnyCloth4DConfig()
    assert config.dynamic_metric_projection_adaptive_iterations >= 256


def test_legacy_scene_attention_checkpoint_is_migrated_to_hierarchy() -> None:
    source = Cloth4DGeometricConditioner(
        intrinsic_channels=6,
        scene_feature_dim=8,
        tactile_channels=4,
        material_latent_dim=12,
        hidden_dim=24,
        scene_attention_depth=1,
    )
    legacy = {}
    for key, value in source.state_dict().items():
        if key.startswith("scene.layers.0."):
            key = "scene." + key.removeprefix("scene.layers.0.")
        legacy[key] = value.clone()
    target = Cloth4DGeometricConditioner(
        intrinsic_channels=6,
        scene_feature_dim=8,
        tactile_channels=4,
        material_latent_dim=12,
        hidden_dim=24,
        scene_attention_depth=3,
    )
    report = _load_conditioner_checkpoint_state(target, legacy)
    loaded = target.state_dict()
    assert report["applied"] is True
    assert report["target_layers"] == 3
    for layer_id in range(3):
        assert torch.equal(
            loaded[f"scene.layers.{layer_id}.query.weight"],
            legacy["scene.query.weight"],
        )


def test_bundle_measure_loss_is_active_in_joint_field_physics_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rest = torch.tensor(
        [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.1, 0.1, 0.0]]
    )
    faces = torch.tensor([[0, 1, 2], [1, 3, 2]])
    shell = build_kirchhoff_love_shell_geometry(
        rest[:, :2],
        faces,
        torch.zeros(4, dtype=torch.long),
        reference_positions=rest,
    )
    edges = torch.tensor([[0, 1], [0, 2], [1, 2], [1, 3], [2, 3]])
    frames = torch.stack(
        (
            rest,
            rest + torch.tensor([0.0, 0.0, 0.001]),
            rest + torch.tensor([0.0, 0.0, 0.002]),
        )
    ).requires_grad_(True)
    normal = torch.tensor([[0.0, 0.0, 1.0]]).expand_as(rest)
    geometry = SimpleNamespace(
        shell=shell,
        material_coordinates=rest[:, :2],
        boundary_weight=torch.ones((4, 1)),
        contact_closest=rest - torch.tensor([0.0, 0.0, 0.02]),
        contact_normals=normal,
        edge_pairs=edges,
        rest_lengths=torch.linalg.vector_norm(
            rest[edges[:, 1]] - rest[edges[:, 0]], dim=-1
        ),
    )
    bundle = SimpleNamespace(
        shell_model=HyperNetworkKirchhoffLoveShell(
            1,
            latent_dim=4,
            hyper_hidden_dim=8,
            node_hidden_dim=3,
            node_steps=2,
        ),
        material_latent=SceneMaterialLatent(4),
    )
    config = AnnyCloth4DConfig(
        teacher_steps=3,
        training_steps=2,
        minimum_training_steps=1,
        shell_physics_interval=1,
        dynamic_refit_steps=2,
        dynamic_refit_minimum_steps=1,
        material_latent_dim=4,
    )
    monkeypatch.setattr(
        "aigc_fashion.anny_cloth4d_simulation._conditioning_for_frame",
        lambda *args, **kwargs: None,
    )

    def query(_bundle, _conditioning, _geometry, _canonical, ids, physical_time):
        frame = int(round(float(physical_time / config.frame_dt)))
        return frames[frame, ids]

    monkeypatch.setattr(
        "aigc_fashion.anny_cloth4d_simulation._query_field",
        query,
    )
    terms = _shell_physics_training_terms(
        bundle,
        geometry,
        frames,
        1,
        MaterialSpec(name="cotton jersey knit"),
        config,
    )
    bundle_total, continuity, balance, passivity, dissipation = terms[-5:]

    assert len(terms) == 14
    assert torch.isfinite(torch.stack(terms)).all()
    assert bundle_total > 0.0
    assert continuity > 0.0
    assert balance >= 0.0
    assert passivity >= 0.0
    assert dissipation >= 0.0
    bundle_total.backward()
    assert frames.grad is not None
    assert torch.isfinite(frames.grad).all()


def test_sleeve_visibility_labels_share_one_material_chart() -> None:
    assert _material_chart_label("front_left_sleeve") == "left_sleeve"
    assert _material_chart_label("back_left_sleeve") == "left_sleeve"
    assert _material_chart_label("front_right_sleeve") == "right_sleeve"
    assert _material_chart_label("front_front_body") == "front_front_body"
