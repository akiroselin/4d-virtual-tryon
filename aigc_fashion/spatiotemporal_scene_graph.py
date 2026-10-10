from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import torch
from torch import Tensor, nn

from .physics_conditioned_flow import SpectralLinear


@dataclass(frozen=True)
class SpatiotemporalAttentionOutput:
    scalar_features: Tensor
    vector_features: Tensor
    edge_attention: Tensor


class SpatiotemporalEquivariantAttention(nn.Module):
    """Hierarchy-aware E(3)-equivariant attention on a 4D scene graph.

    Edges are ordered ``receiver <- sender``. Spatial and temporal origins enter
    only through relative quantities. Relation IDs distinguish temporal,
    parent-child, contact, or task edges without assigning parameters per object.
    """

    def __init__(
        self,
        feature_dim: int,
        *,
        hidden_dim: int = 48,
        relation_count: int = 4,
        relation_dim: int = 8,
        spectral_cap: float = 0.9,
        vector_limit: float = 1.0,
        causal: bool = False,
        epsilon: float = 1.0e-8,
    ) -> None:
        super().__init__()
        if feature_dim <= 0 or hidden_dim <= 0 or relation_count <= 0 or relation_dim <= 0:
            raise ValueError("attention dimensions must be positive")
        if vector_limit <= 0.0 or epsilon <= 0.0:
            raise ValueError("vector_limit and epsilon must be positive")
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.relation_count = relation_count
        self.vector_limit = float(vector_limit)
        self.causal = causal
        self.epsilon = float(epsilon)
        self.relation_embedding = nn.Embedding(relation_count, relation_dim)
        self.query = SpectralLinear(feature_dim, hidden_dim, bias=False, max_singular_value=spectral_cap)
        self.key = SpectralLinear(feature_dim, hidden_dim, bias=False, max_singular_value=spectral_cap)
        self.value = SpectralLinear(feature_dim, hidden_dim, max_singular_value=spectral_cap)
        edge_input_dim = 5 + relation_dim
        self.edge_input = SpectralLinear(edge_input_dim, hidden_dim, max_singular_value=spectral_cap)
        self.edge_output = SpectralLinear(hidden_dim, 3, max_singular_value=spectral_cap)
        self.scalar_output = SpectralLinear(hidden_dim, feature_dim, max_singular_value=spectral_cap)

    def forward(
        self,
        scalar_features: Tensor,
        positions: Tensor,
        times: Tensor,
        hierarchy_levels: Tensor,
        edge_index: Tensor,
        relation_ids: Tensor,
    ) -> SpatiotemporalAttentionOutput:
        self._validate(
            scalar_features,
            positions,
            times,
            hierarchy_levels,
            edge_index,
            relation_ids,
        )
        receiver, sender = edge_index
        displacement = positions[sender] - positions[receiver]
        distance = torch.sqrt(displacement.square().sum(dim=-1) + self.epsilon)
        direction = displacement / distance[:, None]
        delta_time = times[sender] - times[receiver]
        if self.causal and torch.any(delta_time > 1.0e-8):
            raise ValueError("causal scene graph contains a future-to-past edge")
        level_delta = hierarchy_levels[sender] - hierarchy_levels[receiver]
        relation = self.relation_embedding(relation_ids)
        edge_invariants = torch.cat(
            (
                distance[:, None],
                torch.log(distance)[:, None],
                delta_time[:, None],
                delta_time.abs()[:, None],
                level_delta.to(dtype=positions.dtype)[:, None],
                relation,
            ),
            dim=-1,
        )
        edge_hidden = torch.tanh(self.edge_input(edge_invariants))
        edge_parameters = self.edge_output(edge_hidden)
        edge_bias = edge_parameters[:, 0]
        scalar_gate = torch.sigmoid(edge_parameters[:, 1])
        vector_coefficient = self.vector_limit * torch.tanh(edge_parameters[:, 2])

        query = self.query(scalar_features)[receiver]
        key = self.key(scalar_features)[sender]
        logits = (query * key).sum(dim=-1) / sqrt(self.hidden_dim) + edge_bias
        attention = _segment_softmax(logits, receiver, len(scalar_features))
        scalar_message = (
            attention[:, None]
            * scalar_gate[:, None]
            * self.value(scalar_features)[sender]
        )
        scalar_aggregate = scalar_features.new_zeros((len(scalar_features), self.hidden_dim))
        scalar_aggregate.index_add_(0, receiver, scalar_message)
        updated_scalar = scalar_features + self.scalar_output(torch.tanh(scalar_aggregate))

        vector_message = attention[:, None] * vector_coefficient[:, None] * direction
        vector_aggregate = positions.new_zeros(positions.shape)
        vector_aggregate.index_add_(0, receiver, vector_message)
        return SpatiotemporalAttentionOutput(
            scalar_features=updated_scalar,
            vector_features=vector_aggregate,
            edge_attention=attention,
        )

    def _validate(
        self,
        features: Tensor,
        positions: Tensor,
        times: Tensor,
        levels: Tensor,
        edge_index: Tensor,
        relation_ids: Tensor,
    ) -> None:
        node_count = len(features)
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise ValueError("scalar_features shape does not match the model")
        if positions.shape != (node_count, 3):
            raise ValueError("positions must have shape (N, 3)")
        if times.shape != (node_count,) or levels.shape != (node_count,):
            raise ValueError("times and hierarchy_levels must have shape (N,)")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape (2, E)")
        edge_count = edge_index.shape[1]
        if relation_ids.shape != (edge_count,):
            raise ValueError("relation_ids must have shape (E,)")
        if relation_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("relation_ids must contain integers")
        if relation_ids.numel() and (relation_ids.min() < 0 or relation_ids.max() >= self.relation_count):
            raise ValueError("relation id is outside the configured relation vocabulary")
        if edge_index.numel() and (edge_index.min() < 0 or edge_index.max() >= node_count):
            raise ValueError("edge index is outside the node array")


class HierarchyAwareSpatiotemporalAttention(nn.Module):
    """Multi-hop E(3) x time-translation attention over the full 4D hierarchy.

    One layer only communicates across one relation edge. Stacking layers lets
    root/body/garment/panel/contact and temporal-copy nodes exchange context
    without flattening their hierarchy or introducing an absolute world frame.
    """

    def __init__(
        self,
        feature_dim: int,
        *,
        depth: int = 3,
        hidden_dim: int = 48,
        relation_count: int = 4,
        relation_dim: int = 8,
        spectral_cap: float = 0.9,
        vector_limit: float = 1.0,
        causal: bool = False,
        epsilon: float = 1.0e-8,
    ) -> None:
        super().__init__()
        if depth <= 0:
            raise ValueError("hierarchy-aware attention depth must be positive")
        self.depth = depth
        self.layers = nn.ModuleList(
            [
                SpatiotemporalEquivariantAttention(
                    feature_dim,
                    hidden_dim=hidden_dim,
                    relation_count=relation_count,
                    relation_dim=relation_dim,
                    spectral_cap=spectral_cap,
                    vector_limit=vector_limit / depth,
                    causal=causal,
                    epsilon=epsilon,
                )
                for _ in range(depth)
            ]
        )

    def forward(
        self,
        scalar_features: Tensor,
        positions: Tensor,
        times: Tensor,
        hierarchy_levels: Tensor,
        edge_index: Tensor,
        relation_ids: Tensor,
    ) -> SpatiotemporalAttentionOutput:
        scalars = scalar_features
        vectors = torch.zeros_like(positions)
        attentions = []
        for layer in self.layers:
            output = layer(
                scalars,
                positions,
                times,
                hierarchy_levels,
                edge_index,
                relation_ids,
            )
            scalars = output.scalar_features
            vectors = vectors + output.vector_features
            attentions.append(output.edge_attention)
        return SpatiotemporalAttentionOutput(
            scalar_features=scalars,
            vector_features=vectors,
            edge_attention=torch.stack(attentions).mean(dim=0),
        )


def _segment_softmax(logits: Tensor, segment: Tensor, segment_count: int) -> Tensor:
    maxima = logits.new_full((segment_count,), -torch.inf)
    maxima.scatter_reduce_(0, segment, logits, reduce="amax", include_self=True)
    exponent = torch.exp(logits - maxima[segment])
    denominator = logits.new_zeros((segment_count,))
    denominator.index_add_(0, segment, exponent)
    return exponent / denominator[segment].clamp_min(1.0e-12)
