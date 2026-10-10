from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .lie_group_diffusion import right_invariant_relative, se3_log
from .physics_conditioned_flow import SpectralLinear


@dataclass(frozen=True)
class RigidLieMessageOutput:
    node_features: Tensor
    edge_parent_body_wrench: Tensor
    node_world_wrench: Tensor
    relative_twist: Tensor


class SE3JointMessagePassing(nn.Module):
    """Shared SE(3) joint-edge messages with action-reaction world wrenches.

    Node poses are used only through parent-to-child relative transforms. The
    learned edge function is shared by all joints, so parameter count is
    independent of robot/link count. Genesis still solves the hard joint and
    contact constraints; this module is a prior, residual, or policy encoder.
    """

    def __init__(
        self,
        node_feature_dim: int,
        joint_feature_dim: int,
        *,
        hidden_dim: int = 48,
        spectral_cap: float = 0.9,
        wrench_limit: float = 10.0,
    ) -> None:
        super().__init__()
        if node_feature_dim <= 0 or joint_feature_dim < 0 or hidden_dim <= 0:
            raise ValueError("invalid message-passing dimensions")
        if spectral_cap <= 0.0 or wrench_limit <= 0.0:
            raise ValueError("spectral_cap and wrench_limit must be positive")
        self.node_feature_dim = node_feature_dim
        self.joint_feature_dim = joint_feature_dim
        self.hidden_dim = hidden_dim
        self.wrench_limit = float(wrench_limit)
        edge_input_dim = 2 * node_feature_dim + joint_feature_dim + 6
        self.edge_input = SpectralLinear(edge_input_dim, hidden_dim, max_singular_value=spectral_cap)
        self.edge_hidden = SpectralLinear(hidden_dim, hidden_dim, max_singular_value=spectral_cap)
        self.edge_output = SpectralLinear(hidden_dim, hidden_dim + 6, max_singular_value=spectral_cap)
        self.node_projection = SpectralLinear(
            node_feature_dim,
            hidden_dim,
            max_singular_value=spectral_cap,
        )

    def forward(
        self,
        link_transforms: Tensor,
        node_features: Tensor,
        edge_index: Tensor,
        joint_features: Tensor,
    ) -> RigidLieMessageOutput:
        self._validate(link_transforms, node_features, edge_index, joint_features)
        parent, child = edge_index
        relative_transform = right_invariant_relative(
            link_transforms[parent],
            link_transforms[child],
        )
        relative_twist = se3_log(relative_transform)
        edge_features = torch.cat(
            (
                node_features[parent],
                node_features[child],
                joint_features,
                relative_twist,
            ),
            dim=-1,
        )
        hidden = torch.tanh(self.edge_input(edge_features))
        hidden = torch.tanh(self.edge_hidden(hidden))
        raw = self.edge_output(hidden)
        scalar_message = torch.tanh(raw[:, : self.hidden_dim])
        parent_body_wrench = self.wrench_limit * torch.tanh(raw[:, self.hidden_dim :])

        aggregate = node_features.new_zeros((len(node_features), self.hidden_dim))
        aggregate.index_add_(0, parent, scalar_message)
        aggregate.index_add_(0, child, scalar_message)
        degree = node_features.new_zeros((len(node_features),))
        degree.index_add_(0, parent, torch.ones_like(parent, dtype=node_features.dtype))
        degree.index_add_(0, child, torch.ones_like(child, dtype=node_features.dtype))
        aggregate = aggregate / degree.clamp_min(1.0)[:, None]
        updated_features = torch.tanh(self.node_projection(node_features) + aggregate)

        world_wrench = coadjoint_wrench(link_transforms[parent], parent_body_wrench)
        node_world_wrench = link_transforms.new_zeros((len(link_transforms), 6))
        node_world_wrench.index_add_(0, parent, world_wrench)
        node_world_wrench.index_add_(0, child, -world_wrench)
        return RigidLieMessageOutput(
            node_features=updated_features,
            edge_parent_body_wrench=parent_body_wrench,
            node_world_wrench=node_world_wrench,
            relative_twist=relative_twist,
        )

    def _validate(
        self,
        transforms: Tensor,
        features: Tensor,
        edge_index: Tensor,
        joint_features: Tensor,
    ) -> None:
        if transforms.ndim != 3 or transforms.shape[-2:] != (4, 4):
            raise ValueError("link_transforms must have shape (N, 4, 4)")
        if features.shape != (len(transforms), self.node_feature_dim):
            raise ValueError("node_features shape does not match the model")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape (2, E)")
        if edge_index.dtype not in (torch.int32, torch.int64):
            raise ValueError("edge_index must contain integer indices")
        if joint_features.shape != (edge_index.shape[1], self.joint_feature_dim):
            raise ValueError("joint_features shape does not match the model")
        if edge_index.numel() and (edge_index.min() < 0 or edge_index.max() >= len(transforms)):
            raise ValueError("edge index is outside the link array")


def coadjoint_wrench(transform: Tensor, body_wrench: Tensor) -> Tensor:
    """Map ``(torque, force)`` at a local origin to a world-origin wrench."""

    if transform.shape[-2:] != (4, 4) or body_wrench.shape[-1] != 6:
        raise ValueError("invalid transform or wrench shape")
    rotation = transform[..., :3, :3]
    position = transform[..., :3, 3]
    local_torque = body_wrench[..., :3]
    local_force = body_wrench[..., 3:]
    world_force = (rotation @ local_force[..., None]).squeeze(-1)
    world_torque = (rotation @ local_torque[..., None]).squeeze(-1)
    world_torque = world_torque + torch.linalg.cross(position, world_force, dim=-1)
    return torch.cat((world_torque, world_force), dim=-1)
