from __future__ import annotations

import torch
from torch import Tensor, nn

from .physics_conditioned_flow import SpectralLinear


def gauge_transform(features: Tensor, frame_angle: Tensor, *, charge: int = 1) -> Tensor:
    """Apply the local SO(2) gauge convention ``z -> exp(i m alpha) z``."""

    if not torch.is_complex(features):
        raise ValueError("gauge features must be complex")
    if frame_angle.shape != features.shape[:-1]:
        raise ValueError("frame_angle must match feature batch/vertex dimensions")
    phase = torch.polar(torch.ones_like(frame_angle), charge * frame_angle)
    return phase[..., None] * features


def transform_connection(
    connection_angle: Tensor,
    edge_index: Tensor,
    frame_angle: Tensor,
) -> Tensor:
    """Transform target-to-source transport angles after local frame changes."""

    source, target = edge_index
    transformed = connection_angle + frame_angle[source] - frame_angle[target]
    return torch.atan2(torch.sin(transformed), torch.cos(transformed))


class SO2GaugeTransportLayer(nn.Module):
    """Gauge-equivariant tangent-field message passing with parallel transport."""

    def __init__(self, channels: int, *, charge: int = 1, spectral_cap: float = 0.9) -> None:
        super().__init__()
        if channels <= 0 or charge == 0:
            raise ValueError("channels must be positive and charge non-zero")
        self.channels = channels
        self.charge = charge
        self.self_mix = SpectralLinear(channels, channels, bias=False, max_singular_value=spectral_cap)
        self.neighbor_mix = SpectralLinear(channels, channels, bias=False, max_singular_value=spectral_cap)
        self.gate = SpectralLinear(channels, channels, max_singular_value=spectral_cap)

    def forward(
        self,
        features: Tensor,
        edge_index: Tensor,
        connection_angle: Tensor,
        edge_weight: Tensor | None = None,
    ) -> Tensor:
        if not torch.is_complex(features) or features.ndim != 2 or features.shape[1] != self.channels:
            raise ValueError(f"features must be complex with shape (N, {self.channels})")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape (2, E)")
        edge_count = edge_index.shape[1]
        if connection_angle.shape != (edge_count,):
            raise ValueError("connection_angle must have shape (E,)")
        source, target = edge_index
        phase = torch.polar(
            torch.ones_like(connection_angle),
            self.charge * connection_angle,
        )
        transported = phase[:, None] * features[target]
        if edge_weight is not None:
            if edge_weight.shape != (edge_count,):
                raise ValueError("edge_weight must have shape (E,)")
            transported = transported * edge_weight[:, None]
        aggregate = torch.zeros_like(features)
        aggregate.index_add_(0, source, transported)
        degree = torch.zeros(features.shape[0], device=features.device, dtype=features.real.dtype)
        degree.index_add_(0, source, torch.ones(edge_count, device=features.device, dtype=features.real.dtype))
        aggregate = aggregate / degree.clamp_min(1.0)[:, None]

        mixed = _real_channel_mix(self.self_mix, features) + _real_channel_mix(
            self.neighbor_mix, aggregate
        )
        gate = torch.sigmoid(self.gate(mixed.abs()))
        return gate * mixed


def _real_channel_mix(layer: SpectralLinear, features: Tensor) -> Tensor:
    return torch.complex(layer(features.real), layer(features.imag))
