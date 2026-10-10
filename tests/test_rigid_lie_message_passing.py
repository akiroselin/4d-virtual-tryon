from __future__ import annotations

from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aigc_fashion.lie_group_diffusion import se3_exp  # noqa: E402
from aigc_fashion.rigid_lie_message_passing import (  # noqa: E402
    SE3JointMessagePassing,
    coadjoint_wrench,
)


def test_joint_messages_are_global_frame_equivariant_and_action_reaction_balanced() -> None:
    torch.manual_seed(13)
    model = SE3JointMessagePassing(5, 3, hidden_dim=16)
    transforms = se3_exp(
        torch.tensor(
            [
                [0.1, 0.0, 0.0, 0.2, 0.0, 0.0],
                [0.0, 0.2, 0.0, 0.5, 0.1, 0.0],
                [0.0, 0.0, -0.1, 0.8, 0.2, 0.1],
            ]
        )
    )
    node_features = torch.randn(3, 5)
    edge_index = torch.tensor([[0, 1], [1, 2]])
    joint_features = torch.randn(2, 3)
    baseline = model(transforms, node_features, edge_index, joint_features)

    global_transform = se3_exp(torch.tensor([0.2, -0.1, 0.3, 1.0, -0.4, 0.5]))
    moved = model(global_transform @ transforms, node_features, edge_index, joint_features)
    assert torch.allclose(moved.relative_twist, baseline.relative_twist, atol=2.0e-5, rtol=2.0e-5)
    assert torch.allclose(moved.node_features, baseline.node_features, atol=2.0e-5, rtol=2.0e-5)
    assert torch.allclose(
        moved.node_world_wrench,
        coadjoint_wrench(global_transform.expand(3, -1, -1), baseline.node_world_wrench),
        atol=3.0e-5,
        rtol=3.0e-5,
    )
    assert torch.allclose(
        baseline.node_world_wrench.sum(dim=0),
        torch.zeros(6),
        atol=1.0e-5,
    )


def test_parameter_count_does_not_depend_on_joint_count() -> None:
    model = SE3JointMessagePassing(4, 2, hidden_dim=12)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    for link_count in (2, 9):
        transforms = se3_exp(torch.zeros(link_count, 6))
        node_features = torch.zeros(link_count, 4)
        edge_index = torch.stack(
            (torch.arange(link_count - 1), torch.arange(1, link_count))
        )
        joint_features = torch.zeros(link_count - 1, 2)
        output = model(transforms, node_features, edge_index, joint_features)
        assert output.node_features.shape == (link_count, 12)
        assert output.node_world_wrench.shape == (link_count, 6)
        assert sum(parameter.numel() for parameter in model.parameters()) == parameter_count
