from __future__ import annotations

from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aigc_fashion.gauge_tactile import (  # noqa: E402
    SO2GaugeTransportLayer,
    gauge_transform,
    transform_connection,
)
from aigc_fashion.genesis_tactile_adapter import (  # noqa: E402
    adapt_genesis_taxel_field,
    discrete_parallel_transport_angles,
    project_tangent_vector,
    reconstruct_tangent_vector,
    rotate_tangent_gauge,
)


def test_genesis_taxel_adapter_is_independent_of_arbitrary_tangent_gauge() -> None:
    torch.manual_seed(23)
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.1], [0.5, 0.8, 0.2]]
    )
    normals = torch.tensor(
        [[0.0, 0.0, 1.0], [0.1, 0.0, 0.995], [0.0, 0.2, 0.98]]
    )
    force = torch.randn(3, 3)
    edges = torch.tensor([[0, 1, 1, 2, 2, 0], [1, 0, 2, 1, 0, 2]])
    adapted = adapt_genesis_taxel_field(positions, normals, force, edges)
    layer = SO2GaugeTransportLayer(1)
    baseline = layer(adapted.tangent_features, edges, adapted.connection_angle).squeeze(-1)
    baseline_vector = reconstruct_tangent_vector(baseline, adapted.tangent_u, adapted.tangent_v)

    frame_angle = torch.tensor([0.3, -0.8, 1.2])
    moved_u, moved_v = rotate_tangent_gauge(adapted.tangent_u, adapted.tangent_v, frame_angle)
    moved_features = project_tangent_vector(force, moved_u, moved_v).unsqueeze(-1)
    moved_connection = discrete_parallel_transport_angles(
        normals / torch.linalg.vector_norm(normals, dim=-1, keepdim=True),
        moved_u,
        moved_v,
        edges,
    )
    assert torch.allclose(
        moved_features,
        gauge_transform(adapted.tangent_features, frame_angle),
        atol=2.0e-5,
        rtol=2.0e-5,
    )
    assert torch.allclose(
        moved_connection,
        transform_connection(adapted.connection_angle, edges, frame_angle),
        atol=2.0e-5,
        rtol=2.0e-5,
    )
    moved = layer(moved_features, edges, moved_connection).squeeze(-1)
    moved_vector = reconstruct_tangent_vector(moved, moved_u, moved_v)
    assert torch.allclose(moved_vector, baseline_vector, atol=3.0e-5, rtol=3.0e-5)
