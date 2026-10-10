from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .physics_conditioned_flow import SpectralLinear


@dataclass(frozen=True)
class GeometricClothPriorOutput:
    internal_force: Tensor
    edge_force_magnitude: Tensor
    edge_stretch: Tensor


class IntrinsicEquivariantClothPrior(nn.Module):
    """E(3)-equivariant learned residual on immutable material edges.

    The network consumes only invariant edge scalars and emits a scalar force
    magnitude. Pairwise antisymmetric vector messages preserve translation and
    rotation/reflection equivariance as well as zero net internal force. This is
    a residual, constitutive prior, or preconditioner; it is not a replacement
    for an orthotropic shell energy or collision solver.
    """

    def __init__(
        self,
        *,
        material_latent_dim: int = 32,
        hidden_dim: int = 48,
        hidden_layers: int = 2,
        spectral_cap: float = 0.9,
        force_limit: float = 2.0,
        epsilon: float = 1.0e-8,
    ) -> None:
        super().__init__()
        if material_latent_dim <= 0 or hidden_dim <= 0 or hidden_layers <= 0:
            raise ValueError("network dimensions must be positive")
        if spectral_cap <= 0.0 or force_limit <= 0.0 or epsilon <= 0.0:
            raise ValueError("caps, limits, and epsilon must be positive")
        self.material_latent_dim = material_latent_dim
        self.force_limit = float(force_limit)
        self.epsilon = float(epsilon)
        # stretch, log length ratio, axial speed, rest length, cotangent weight
        input_dim = material_latent_dim + 5
        self.hidden = nn.ModuleList(
            [
                SpectralLinear(
                    input_dim if index == 0 else hidden_dim,
                    hidden_dim,
                    max_singular_value=spectral_cap,
                )
                for index in range(hidden_layers)
            ]
        )
        self.output = SpectralLinear(
            hidden_dim,
            1,
            max_singular_value=spectral_cap,
        )

    def forward(
        self,
        positions: Tensor,
        velocities: Tensor,
        edge_index: Tensor,
        rest_lengths: Tensor,
        cotangent_weights: Tensor,
        material_latent: Tensor,
    ) -> GeometricClothPriorOutput:
        self._validate_inputs(
            positions,
            velocities,
            edge_index,
            rest_lengths,
            cotangent_weights,
            material_latent,
        )
        source = edge_index[0]
        target = edge_index[1]
        delta = positions[target] - positions[source]
        length = torch.sqrt(delta.square().sum(dim=-1) + self.epsilon)
        direction = delta / length.unsqueeze(-1)
        rest = rest_lengths.to(device=positions.device, dtype=positions.dtype)
        cotangent = cotangent_weights.to(device=positions.device, dtype=positions.dtype)
        velocity_delta = velocities[target] - velocities[source]
        axial_speed = (velocity_delta * direction).sum(dim=-1)
        ratio = length / rest
        stretch = ratio - 1.0
        latent = _edge_latent(
            material_latent,
            edge_count=edge_index.shape[1],
            latent_dim=self.material_latent_dim,
            reference=positions,
        )
        invariants = torch.stack(
            (stretch, torch.log(ratio), axial_speed, rest, cotangent),
            dim=-1,
        )
        hidden = torch.cat((invariants, latent), dim=-1)
        for layer in self.hidden:
            hidden = torch.tanh(layer(hidden))
        magnitude = self.force_limit * torch.tanh(self.output(hidden).squeeze(-1))
        edge_force = magnitude.unsqueeze(-1) * direction

        internal_force = torch.zeros_like(positions)
        internal_force.index_add_(0, source, edge_force)
        internal_force.index_add_(0, target, -edge_force)
        return GeometricClothPriorOutput(
            internal_force=internal_force,
            edge_force_magnitude=magnitude,
            edge_stretch=stretch,
        )

    def _validate_inputs(
        self,
        positions: Tensor,
        velocities: Tensor,
        edge_index: Tensor,
        rest_lengths: Tensor,
        cotangent_weights: Tensor,
        material_latent: Tensor,
    ) -> None:
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError("positions must have shape (N, 3)")
        if velocities.shape != positions.shape:
            raise ValueError("velocities must match positions")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape (2, E)")
        if edge_index.dtype not in (torch.int32, torch.int64):
            raise ValueError("edge_index must contain integer indices")
        edge_count = edge_index.shape[1]
        if rest_lengths.shape != (edge_count,) or cotangent_weights.shape != (edge_count,):
            raise ValueError("edge scalars must have shape (E,)")
        if edge_index.numel() and (edge_index.min() < 0 or edge_index.max() >= len(positions)):
            raise ValueError("edge index is outside the vertex array")
        if torch.any(rest_lengths <= 0.0):
            raise ValueError("rest lengths must be positive")
        valid_latent_shape = material_latent.shape in {
            (self.material_latent_dim,),
            (1, self.material_latent_dim),
            (edge_count, self.material_latent_dim),
        }
        if not valid_latent_shape:
            raise ValueError("material_latent must be global or supplied per edge")


def _edge_latent(
    latent: Tensor,
    *,
    edge_count: int,
    latent_dim: int,
    reference: Tensor,
) -> Tensor:
    latent = latent.to(device=reference.device, dtype=reference.dtype)
    if latent.shape == (latent_dim,):
        return latent.unsqueeze(0).expand(edge_count, -1)
    if latent.shape == (1, latent_dim):
        return latent.expand(edge_count, -1)
    return latent
