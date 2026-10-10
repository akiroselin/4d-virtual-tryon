from __future__ import annotations

from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aigc_fashion.spatiotemporal_scene_graph import (  # noqa: E402
    HierarchyAwareSpatiotemporalAttention,
    SpatiotemporalEquivariantAttention,
)


def test_scene_attention_is_e3_and_time_translation_equivariant() -> None:
    torch.manual_seed(19)
    model = SpatiotemporalEquivariantAttention(6, hidden_dim=12, relation_count=4)
    features = torch.randn(5, 6)
    positions = torch.randn(5, 3)
    times = torch.tensor([0.0, 0.0, 0.1, 0.1, 0.2])
    levels = torch.tensor([0, 1, 1, 2, 0])
    edge_index = torch.tensor(
        [[0, 0, 1, 2, 2, 3, 4], [1, 2, 2, 1, 3, 4, 0]]
    )
    relation_ids = torch.tensor([1, 2, 0, 0, 1, 3, 2])
    baseline = model(features, positions, times, levels, edge_index, relation_ids)

    rotation = torch.tensor(
        [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]
    )
    moved = model(
        features,
        positions @ rotation.T + torch.tensor([1.0, -2.0, 0.5]),
        times + 7.0,
        levels,
        edge_index,
        relation_ids,
    )
    assert torch.allclose(moved.edge_attention, baseline.edge_attention, atol=2.0e-5, rtol=2.0e-5)
    assert torch.allclose(moved.scalar_features, baseline.scalar_features, atol=2.0e-5, rtol=2.0e-5)
    assert torch.allclose(
        moved.vector_features,
        baseline.vector_features @ rotation.T,
        atol=2.0e-5,
        rtol=2.0e-5,
    )

    for receiver in torch.unique(edge_index[0]):
        mask = edge_index[0] == receiver
        assert torch.allclose(baseline.edge_attention[mask].sum(), torch.tensor(1.0), atol=1.0e-6)


def test_causal_scene_attention_rejects_future_sender() -> None:
    model = SpatiotemporalEquivariantAttention(2, causal=True)
    with torch.no_grad():
        try:
            model(
                torch.zeros(2, 2),
                torch.zeros(2, 3),
                torch.tensor([0.0, 1.0]),
                torch.zeros(2, dtype=torch.long),
                torch.tensor([[0], [1]]),
                torch.zeros(1, dtype=torch.long),
            )
        except ValueError as error:
            assert "future-to-past" in str(error)
        else:
            raise AssertionError("causal attention accepted a future sender")


def test_hierarchy_stack_is_e3_and_time_translation_equivariant() -> None:
    torch.manual_seed(23)
    model = HierarchyAwareSpatiotemporalAttention(
        4,
        depth=3,
        hidden_dim=10,
        relation_count=4,
    )
    features = torch.randn(6, 4)
    positions = torch.randn(6, 3)
    times = torch.tensor([0.0, 0.0, 0.1, 0.1, 0.2, 0.2])
    levels = torch.tensor([0, 1, 2, 3, 2, 1])
    edges = torch.tensor([[0, 1, 2, 3, 4, 5, 2], [1, 2, 3, 4, 5, 0, 4]])
    relations = torch.tensor([1, 1, 2, 3, 0, 0, 2])
    baseline = model(features, positions, times, levels, edges, relations)
    rotation = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    moved = model(
        features,
        positions @ rotation.T + torch.tensor([-0.4, 1.2, 0.7]),
        times + 11.0,
        levels,
        edges,
        relations,
    )
    assert torch.allclose(moved.scalar_features, baseline.scalar_features, atol=3e-5, rtol=3e-5)
    assert torch.allclose(moved.edge_attention, baseline.edge_attention, atol=3e-5, rtol=3e-5)
    assert torch.allclose(
        moved.vector_features,
        baseline.vector_features @ rotation.T,
        atol=3e-5,
        rtol=3e-5,
    )


def test_hierarchy_stack_propagates_information_across_multiple_levels() -> None:
    torch.manual_seed(29)
    model = HierarchyAwareSpatiotemporalAttention(3, depth=3, hidden_dim=8)
    features = torch.zeros(4, 3)
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]
    )
    times = torch.zeros(4)
    levels = torch.arange(4)
    # root(0) <- garment(1) <- panel(2) <- contact(3)
    edges = torch.tensor([[0, 1, 2], [1, 2, 3]])
    relations = torch.tensor([1, 1, 2])
    baseline = model(features, positions, times, levels, edges, relations)
    changed_features = features.clone()
    changed_features[3, 0] = 1.0
    changed = model(changed_features, positions, times, levels, edges, relations)
    assert not torch.allclose(changed.scalar_features[0], baseline.scalar_features[0])
