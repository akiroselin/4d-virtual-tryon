from pathlib import Path

import numpy as np
import pytest
import torch

from aigc_fashion.panel_instance_grower import (
    EvidenceGuidedFloodFill,
    PanelFloodFillNetwork,
    PanelGrowthConfig,
    PanelImageEncoder,
    PanelInstanceGrower,
    PanelSeed,
    PanelSeedNetwork,
    panel_growth_loss,
    propose_panel_seeds,
    rasterize_garmentcode_panels,
    reconstruct_panels_from_image_ffn,
)


def test_seeded_local_growth_recovers_variable_size_panel_instances() -> None:
    evidence = np.zeros((2, 96, 96), dtype=np.float32)
    evidence[0, 10:80, 10:40] = 1.0
    evidence[1, 20:70, 55:90] = 1.0
    seeds = (
        PanelSeed("front_seed", "front", 0, 20, 20, 1.0),
        PanelSeed("back_seed", "back", 1, 30, 65, 1.0),
    )
    result = PanelInstanceGrower(EvidenceGuidedFloodFill()).grow(
        evidence, seeds, source="teacher scheduler test"
    )
    assert len(result.instances) == 2
    assert [instance.area_pixels for instance in result.instances] == [2100, 1750]
    assert all(instance.visited_fields > 1 for instance in result.instances)
    assert not result.rejected_seeds
    assert not result.summary()["production_pattern_claimed"]


def test_growth_keeps_disconnected_objects_separate() -> None:
    evidence = np.zeros((1, 96, 96), dtype=np.float32)
    evidence[0, 10:35, 10:35] = 1.0
    evidence[0, 60:85, 60:85] = 1.0
    result = PanelInstanceGrower(EvidenceGuidedFloodFill()).grow(
        evidence,
        (
            PanelSeed("first", "panel", 0, 20, 20, 1.0),
            PanelSeed("second", "panel", 0, 70, 70, 0.9),
        ),
        source="teacher scheduler test",
    )
    assert len(result.instances) == 2
    assert all(instance.area_pixels == 625 for instance in result.instances)
    assert not np.any(result.instances[0].mask & result.instances[1].mask)


def test_seed_network_nms_produces_unknown_count_roles() -> None:
    objectness = torch.full((1, 1, 32, 32), -10.0)
    objectness[0, 0, 8, 7] = 8.0
    objectness[0, 0, 24, 25] = 7.0
    roles = torch.zeros((1, 3, 32, 32))
    roles[0, 1, 8, 7] = 4.0
    roles[0, 2, 24, 25] = 4.0
    seeds = propose_panel_seeds(
        objectness,
        roles,
        role_names={0: "front", 1: "sleeve", 2: "back"},
    )
    assert len(seeds) == 2
    assert {seed.role for seed in seeds} == {"sleeve", "back"}


def test_trainable_ffn_cell_is_differentiable() -> None:
    cell = PanelFloodFillNetwork(feature_channels=8, role_count=4, hidden_channels=16)
    features = torch.randn(2, 8, 17, 17, requires_grad=True)
    mask = torch.zeros(2, 1, 17, 17)
    output = cell(features, mask, torch.tensor([0, 3]))
    target = torch.zeros_like(output.mask_logits)
    loss = panel_growth_loss(output, target)
    loss.backward()
    assert features.grad is not None
    assert torch.isfinite(features.grad).all()
    assert output.stop_logits.shape == (2, 1)


def test_garmentcode_teacher_raster_supports_growth() -> None:
    payload = {
        "pattern": {
            "panels": {
                "front": {"vertices": [[0, 0], [40, 0], [35, 70], [5, 70]]},
                "back": {"vertices": [[0, 0], [38, 0], [38, 65], [0, 65]]},
            }
        }
    }
    evidence, seeds, names = rasterize_garmentcode_panels(payload, resolution=96)
    result = PanelInstanceGrower(
        EvidenceGuidedFloodFill(),
        config=PanelGrowthConfig(minimum_panel_pixels=12),
    ).grow(evidence, seeds, source="GarmentCode teacher diagnostic")
    assert names == ("front", "back")
    assert len(result.instances) == 2
    assert all(instance.area_pixels > 1000 for instance in result.instances)


def test_image_checkpoint_entrypoint_refuses_missing_weights(tmp_path: Path) -> None:
    image = tmp_path / "garment.png"
    from PIL import Image

    Image.new("RGB", (64, 64), "white").save(image)
    with pytest.raises(FileNotFoundError, match="checkpoint"):
        reconstruct_panels_from_image_ffn(image, tmp_path / "missing.pt")


def test_encoder_and_seed_heads_keep_dense_feature_grid() -> None:
    encoder = PanelImageEncoder(feature_channels=12)
    seed_network = PanelSeedNetwork(feature_channels=12, role_count=5)
    features = encoder(torch.rand(1, 3, 128, 128))
    objectness, roles = seed_network(features)
    assert features.shape == (1, 12, 32, 32)
    assert objectness.shape == (1, 1, 32, 32)
    assert roles.shape == (1, 5, 32, 32)
