from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aigc_fashion.gauge_tactile import (  # noqa: E402
    SO2GaugeTransportLayer,
    gauge_transform,
    transform_connection,
)
from aigc_fashion.intrinsic_diffusion import IntrinsicSpectralDiffusion  # noqa: E402
from aigc_fashion.lie_group_diffusion import right_perturb_pose, se3_exp  # noqa: E402


def test_intrinsic_diffusion_is_eigenvector_sign_invariant_and_preserves_constants() -> None:
    layer = IntrinsicSpectralDiffusion(2, initial_time=0.2)
    scale = 2.0**-0.5
    eigenvectors = scale * torch.tensor([[1.0, 1.0], [1.0, -1.0]])
    eigenvalues = torch.tensor([0.0, 2.0])
    mass = torch.ones(2)
    features = torch.tensor([[3.0, 1.0], [3.0, -1.0]], requires_grad=True)
    output = layer(features, eigenvalues, eigenvectors, mass)
    flipped = layer(features, eigenvalues, -eigenvectors, mass)
    assert torch.allclose(output, flipped, atol=1.0e-6)
    assert torch.allclose(output[:, 0], torch.full((2,), 3.0), atol=1.0e-6)
    output.square().sum().backward()
    assert features.grad is not None
    assert layer.raw_time.grad is not None


def test_se3_forward_noising_is_equivariant_to_global_left_action() -> None:
    clean = se3_exp(torch.tensor([0.2, -0.1, 0.3, 0.4, 0.1, -0.2]))
    global_frame = se3_exp(torch.tensor([-0.1, 0.25, 0.05, 2.0, -1.0, 0.5]))
    noise = torch.tensor([0.3, 0.1, -0.2, 0.5, -0.4, 0.2])
    noised = right_perturb_pose(clean, noise, 0.2)
    transformed = right_perturb_pose(global_frame @ clean, noise, 0.2)
    assert torch.allclose(transformed, global_frame @ noised, atol=1.0e-5, rtol=1.0e-5)
    rotation = noised[:3, :3]
    assert torch.allclose(rotation.T @ rotation, torch.eye(3), atol=1.0e-5)
    assert torch.linalg.det(rotation).item() == pytest.approx(1.0, abs=1.0e-5)


def test_tactile_transport_is_equivariant_to_local_so2_gauge_changes() -> None:
    torch.manual_seed(5)
    layer = SO2GaugeTransportLayer(3, charge=1)
    features = torch.complex(torch.randn(4, 3), torch.randn(4, 3))
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3, 3, 0], [1, 0, 2, 1, 3, 2, 0, 3]]
    )
    connection = torch.linspace(-0.4, 0.5, edge_index.shape[1])
    frame_angle = torch.tensor([0.2, -0.7, 0.4, 1.1])
    baseline = layer(features, edge_index, connection)
    transformed_features = gauge_transform(features, frame_angle)
    transformed_connection = transform_connection(connection, edge_index, frame_angle)
    transformed = layer(transformed_features, edge_index, transformed_connection)
    assert torch.allclose(
        transformed,
        gauge_transform(baseline, frame_angle),
        atol=2.0e-5,
        rtol=2.0e-5,
    )
