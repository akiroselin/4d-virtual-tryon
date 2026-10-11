from __future__ import annotations

from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aigc_fashion.hypernetwork_kirchhoff_love import (  # noqa: E402
    CONSTITUTIVE_BRANCHES,
    HyperNetworkKirchhoffLoveShell,
    build_kirchhoff_love_shell_geometry,
)
from aigc_fashion.hypernode_training import (  # noqa: E402
    ConstitutiveCurveBatch,
    HyperNODETrajectoryBatch,
    HyperNODETrainer,
    HyperNODETrainingConfig,
    load_hypernode_checkpoint,
    save_hypernode_checkpoint,
)


def _curve_batch() -> ConstitutiveCurveBatch:
    sample_count = 48
    generator = torch.Generator().manual_seed(11)
    coordinates = 2.0 * torch.rand((sample_count, 2), generator=generator) - 1.0
    strains = 0.12 * (2.0 * torch.rand((sample_count, 5), generator=generator) - 1.0)
    stiffness = torch.tensor([150.0, 75.0, 16.0, 180.0, 3.0e-5])
    target = strains * stiffness
    return ConstitutiveCurveBatch(
        material_coordinates=coordinates,
        panel_features=torch.ones((sample_count, 1)),
        material_latent=torch.zeros(3),
        strains=strains,
        target_resultants=target,
    )


def _normalized_curve_error(
    model: HyperNetworkKirchhoffLoveShell,
    batch: ConstitutiveCurveBatch,
) -> float:
    parameters = model.hypernetwork(
        batch.material_coordinates,
        batch.panel_features,
        batch.material_latent,
    )
    prediction = model.constitutive_node.response(
        batch.strains,
        parameters,
        tuple(range(CONSTITUTIVE_BRANCHES)),
    )
    scale = batch.target_resultants.abs().amax(dim=0).clamp_min(1.0e-3)
    return float(
        ((prediction - batch.target_resultants) / scale)
        .square()
        .mean()
        .detach()
    )


def test_staged_curve_pretraining_reduces_material_error() -> None:
    torch.manual_seed(3)
    model = HyperNetworkKirchhoffLoveShell(1, latent_dim=3)
    batch = _curve_batch()
    initial = _normalized_curve_error(model, batch)
    config = HyperNODETrainingConfig(
        homogeneous_steps=12,
        spatial_initialization_steps=18,
        dynamics_steps=0,
        constitutive_learning_rate=2.0e-3,
    )
    trainer = HyperNODETrainer(model, config)
    history = trainer.fit_constitutive_curves([batch])
    final = _normalized_curve_error(model, batch)
    assert len(history) == 30
    assert history[0].phase == "homogeneous_node_pretraining"
    assert history[-1].phase == "hypernetwork_spatial_initialization"
    assert final < initial


def test_hypernode_checkpoint_round_trip(tmp_path: Path) -> None:
    source = HyperNetworkKirchhoffLoveShell(1, latent_dim=3)
    config = HyperNODETrainingConfig(
        homogeneous_steps=0,
        spatial_initialization_steps=0,
        dynamics_steps=0,
    )
    destination = save_hypernode_checkpoint(
        tmp_path / "cotton-jersey.pt",
        source,
        config,
        [],
        metadata={"material": "test cotton jersey"},
    )
    restored = HyperNetworkKirchhoffLoveShell(1, latent_dim=3)
    checkpoint = load_hypernode_checkpoint(destination, restored)
    assert checkpoint["metadata"]["material"] == "test cotton jersey"
    for expected, actual in zip(source.parameters(), restored.parameters()):
        assert torch.equal(expected, actual)


def test_dynamic_training_includes_product_structure_group_loss() -> None:
    rest = torch.tensor(
        [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.1, 0.1, 0.0]]
    )
    faces = torch.tensor([[0, 1, 2], [1, 3, 2]])
    geometry = build_kirchhoff_love_shell_geometry(
        rest[:, :2],
        faces,
        torch.zeros(4, dtype=torch.long),
        reference_positions=rest,
    )
    current = rest.clone()
    current[3, 2] = 0.002
    batch = HyperNODETrajectoryBatch(
        geometry=geometry,
        previous_positions=rest,
        current_positions=current,
        next_positions=current,
        material_latent=torch.zeros(3),
        dt=0.01,
    )
    model = HyperNetworkKirchhoffLoveShell(
        1, latent_dim=3, hyper_hidden_dim=8, node_hidden_dim=3, node_steps=2
    )
    trainer = HyperNODETrainer(
        model,
        HyperNODETrainingConfig(
            homogeneous_steps=0,
            spatial_initialization_steps=0,
            dynamics_steps=1,
            structure_group_weight=0.1,
        ),
    )
    record = trainer.fit_dynamics([batch])[-1]
    assert record.phase == "dynamic_inverse_identification"
    assert record.structure_group >= 0.0
    assert torch.isfinite(torch.tensor(record.total))
