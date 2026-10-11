from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .gauge_tactile import SO2GaugeTransportLayer
from .genesis_tactile_adapter import (
    discrete_parallel_transport_angles,
    project_tangent_vector,
    tangent_frames_from_normals,
)
from .physics_conditioned_flow import SpectralLinear


@dataclass(frozen=True)
class GaugeContactResponse:
    """Gauge-invariant contact coefficients inferred from a charged tangent field."""

    friction_scale: Tensor
    tangent_speed: Tensor
    active_edge_count: int


class GaugeEquivariantContactCNN(nn.Module):
    """SO(2)-gauge-equivariant message passing on cloth/body contact patches.

    The arbitrary tangent axes of an SDF closest point are a local gauge. The
    network transports charged tangent-velocity features between neighbouring
    contacts, then predicts only a gauge-invariant, bounded friction scale.
    Normal non-penetration and the direction of Coulomb friction remain analytic.
    """

    def __init__(
        self,
        channels: int = 4,
        *,
        spectral_cap: float = 0.9,
        maximum_log_friction_scale: float = 0.35,
    ) -> None:
        super().__init__()
        if channels <= 0 or spectral_cap <= 0.0 or maximum_log_friction_scale < 0.0:
            raise ValueError("contact channels/cap must be positive and scale bound non-negative")
        self.channels = channels
        self.maximum_log_friction_scale = float(maximum_log_friction_scale)
        self.scalar_encoder = SpectralLinear(
            3,
            channels,
            max_singular_value=spectral_cap,
        )
        self.transport = SO2GaugeTransportLayer(
            channels,
            spectral_cap=spectral_cap,
        )
        self.friction_head = SpectralLinear(
            channels + 3,
            1,
            max_singular_value=spectral_cap,
        )
        # A fresh model must reproduce the calibrated analytic contact solver.
        nn.init.zeros_(self.friction_head.weight)
        if self.friction_head.bias is not None:
            nn.init.zeros_(self.friction_head.bias)

    def forward(
        self,
        relative_velocity: Tensor,
        normals: Tensor,
        penetration: Tensor,
        active: Tensor,
        edge_index: Tensor,
        *,
        clearance_m: float,
        tangent_u: Tensor | None = None,
        tangent_v: Tensor | None = None,
    ) -> GaugeContactResponse:
        if relative_velocity.ndim != 2 or relative_velocity.shape[1] != 3:
            raise ValueError("relative_velocity must have shape (V,3)")
        if normals.shape != relative_velocity.shape:
            raise ValueError("normals must match relative_velocity")
        vertex_count = len(relative_velocity)
        if penetration.shape != (vertex_count,) or active.shape != (vertex_count,):
            raise ValueError("penetration and active must have shape (V,)")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape (2,E)")
        if clearance_m <= 0.0:
            raise ValueError("clearance_m must be positive")

        normals = torch.nn.functional.normalize(normals.to(relative_velocity), dim=-1)
        if tangent_u is None or tangent_v is None:
            if tangent_u is not None or tangent_v is not None:
                raise ValueError("both tangent axes must be supplied together")
            tangent_u, tangent_v = tangent_frames_from_normals(normals)
        else:
            tangent_u = tangent_u.to(relative_velocity)
            tangent_v = tangent_v.to(relative_velocity)

        normal_speed = (relative_velocity * normals).sum(dim=-1)
        tangent_velocity = relative_velocity - normal_speed[:, None] * normals
        tangent_feature = project_tangent_vector(
            tangent_velocity,
            tangent_u,
            tangent_v,
        )
        tangent_speed = tangent_feature.abs()
        scalars = torch.stack(
            (
                (penetration.to(relative_velocity) / clearance_m).clamp(0.0, 2.0),
                torch.tanh(-normal_speed),
                torch.tanh(tangent_speed),
            ),
            dim=-1,
        )
        scalar_channels = torch.tanh(self.scalar_encoder(scalars))
        charged_features = tangent_feature[:, None] * scalar_channels.to(
            dtype=tangent_feature.dtype
        )

        contact_edges = _active_bidirectional_edges(edge_index, active, vertex_count)
        connection = discrete_parallel_transport_angles(
            normals,
            tangent_u,
            tangent_v,
            contact_edges,
        )
        transported = self.transport(charged_features, contact_edges, connection)
        invariant_features = torch.cat((transported.abs(), scalars), dim=-1)
        log_scale = self.maximum_log_friction_scale * torch.tanh(
            self.friction_head(invariant_features).squeeze(-1)
        )
        scale = torch.exp(log_scale)
        scale = torch.where(active.to(device=scale.device, dtype=torch.bool), scale, torch.ones_like(scale))
        return GaugeContactResponse(
            friction_scale=scale,
            tangent_speed=tangent_speed,
            active_edge_count=int(contact_edges.shape[1]),
        )


def _active_bidirectional_edges(
    edge_index: Tensor,
    active: Tensor,
    vertex_count: int,
) -> Tensor:
    edges = edge_index.to(device=active.device, dtype=torch.long)
    if edges.numel() and (int(edges.min()) < 0 or int(edges.max()) >= vertex_count):
        raise ValueError("contact edge index is outside the vertex range")
    edges = torch.cat((edges, edges.flip(0)), dim=1)
    mask = active.to(dtype=torch.bool)[edges[0]] & active.to(dtype=torch.bool)[edges[1]]
    return edges[:, mask]
