from __future__ import annotations

from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aigc_fashion.graph_neural_operator import MultiscaleConservativeGraphOperator  # noqa: E402


def _run(
    model: MultiscaleConservativeGraphOperator,
    positions: torch.Tensor,
    scalar_state: torch.Tensor,
    vector_state: torch.Tensor,
):
    fine = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]])
    coarse = torch.tensor([[0, 1], [2, 3]])
    return model(
        positions,
        scalar_state,
        vector_state,
        [fine, coarse],
        [0.1, 0.4],
    )


def test_multiscale_gno_is_conservative_and_e3_equivariant() -> None:
    torch.manual_seed(17)
    model = MultiscaleConservativeGraphOperator(2, hidden_dim=16)
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.1, 0.0], [1.1, 1.0, 0.2], [0.0, 0.9, -0.1]]
    )
    scalar_state = torch.randn(4, 2)
    vector_state = torch.randn(4, 3)
    baseline = _run(model, positions, scalar_state, vector_state)
    assert torch.allclose(baseline.scalar_rate.sum(dim=0), torch.zeros(2), atol=1.0e-6)
    assert torch.allclose(baseline.vector_rate.sum(dim=0), torch.zeros(3), atol=1.0e-6)

    axis = torch.tensor([1.0, 2.0, -0.5])
    axis = axis / torch.linalg.vector_norm(axis)
    angle = torch.tensor(0.63)
    skew = torch.tensor(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]]
    )
    rotation = torch.eye(3) + torch.sin(angle) * skew + (1.0 - torch.cos(angle)) * (skew @ skew)
    translation = torch.tensor([2.0, -1.0, 0.5])
    transformed = _run(
        model,
        positions @ rotation.T + translation,
        scalar_state,
        vector_state @ rotation.T,
    )
    assert torch.allclose(transformed.scalar_rate, baseline.scalar_rate, atol=2.0e-5, rtol=2.0e-5)
    assert torch.allclose(
        transformed.vector_rate,
        baseline.vector_rate @ rotation.T,
        atol=2.0e-5,
        rtol=2.0e-5,
    )


def test_multiscale_gno_backpropagates_on_irregular_particles() -> None:
    model = MultiscaleConservativeGraphOperator(1, hidden_dim=12)
    positions = torch.randn(4, 3, requires_grad=True)
    scalar_state = torch.randn(4, 1, requires_grad=True)
    vector_state = torch.randn(4, 3, requires_grad=True)
    output = _run(model, positions, scalar_state, vector_state)
    loss = output.scalar_rate.square().sum() + output.vector_rate.square().sum()
    loss.backward()
    assert positions.grad is not None and torch.isfinite(positions.grad).all()
    assert scalar_state.grad is not None and torch.isfinite(scalar_state.grad).all()
    assert vector_state.grad is not None and torch.isfinite(vector_state.grad).all()
