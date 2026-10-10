from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .physics_conditioned_flow import SpectralLinear


@dataclass(frozen=True)
class GraphOperatorOutput:
    scalar_rate: Tensor
    vector_rate: Tensor
    scalar_edge_fluxes: tuple[Tensor, ...]
    vector_edge_fluxes: tuple[Tensor, ...]


class MultiscaleConservativeGraphOperator(nn.Module):
    """Conservative E(3)-equivariant neural operator on irregular particles.

    The caller supplies one undirected edge list per spatial scale. A shared
    integral kernel produces antisymmetric scalar and vector fluxes, making the
    global rates conservative by construction while retaining ``O(E)`` cost.
    """

    def __init__(
        self,
        scalar_channels: int,
        *,
        context_dim: int = 0,
        hidden_dim: int = 64,
        spectral_cap: float = 0.9,
        flux_limit: float = 5.0,
        epsilon: float = 1.0e-8,
    ) -> None:
        super().__init__()
        if scalar_channels <= 0 or context_dim < 0 or hidden_dim <= 0:
            raise ValueError("invalid graph-operator dimensions")
        if spectral_cap <= 0.0 or flux_limit <= 0.0 or epsilon <= 0.0:
            raise ValueError("caps, limits, and epsilon must be positive")
        self.scalar_channels = scalar_channels
        self.context_dim = context_dim
        self.flux_limit = float(flux_limit)
        self.epsilon = float(epsilon)
        # distance, log distance, radial speed, scale + mean/abs-difference fields + context
        input_dim = 4 + 2 * scalar_channels + context_dim
        output_dim = scalar_channels + 2
        self.kernel_input = SpectralLinear(input_dim, hidden_dim, max_singular_value=spectral_cap)
        self.kernel_hidden = SpectralLinear(hidden_dim, hidden_dim, max_singular_value=spectral_cap)
        self.kernel_output = SpectralLinear(hidden_dim, output_dim, max_singular_value=spectral_cap)

    def forward(
        self,
        positions: Tensor,
        scalar_state: Tensor,
        vector_state: Tensor,
        edge_indices: Sequence[Tensor],
        scale_values: Sequence[float | Tensor],
        *,
        quadrature_weights: Sequence[Tensor] | None = None,
        context: Tensor | None = None,
    ) -> GraphOperatorOutput:
        self._validate_base(positions, scalar_state, vector_state, edge_indices, scale_values)
        if quadrature_weights is None:
            quadrature_weights = [
                positions.new_ones(edge_index.shape[1]) for edge_index in edge_indices
            ]
        if len(quadrature_weights) != len(edge_indices):
            raise ValueError("one quadrature-weight vector is required per scale")

        context_features = _particle_context(
            context,
            particle_count=len(positions),
            context_dim=self.context_dim,
            reference=positions,
        )
        scalar_rate = torch.zeros_like(scalar_state)
        vector_rate = torch.zeros_like(vector_state)
        all_scalar_fluxes = []
        all_vector_fluxes = []
        for edge_index, scale_value, edge_weight in zip(
            edge_indices,
            scale_values,
            quadrature_weights,
        ):
            source, target = edge_index
            edge_count = edge_index.shape[1]
            if edge_weight.shape != (edge_count,):
                raise ValueError("quadrature weights must have shape (E,)")
            edge_weight = edge_weight.to(device=positions.device, dtype=positions.dtype)
            displacement = positions[target] - positions[source]
            distance = torch.sqrt(displacement.square().sum(dim=-1) + self.epsilon)
            direction = displacement / distance[:, None]
            vector_difference = vector_state[target] - vector_state[source]
            radial_speed = (vector_difference * direction).sum(dim=-1)
            scalar_difference = scalar_state[target] - scalar_state[source]
            scalar_mean = 0.5 * (scalar_state[target] + scalar_state[source])
            scalar_absolute_difference = scalar_difference.abs()
            scale = torch.as_tensor(scale_value, device=positions.device, dtype=positions.dtype)
            scale = scale.reshape(1, 1).expand(edge_count, 1)
            edge_context = 0.5 * (context_features[target] + context_features[source])
            invariants = torch.cat(
                (
                    distance[:, None],
                    torch.log(distance)[:, None],
                    radial_speed[:, None],
                    scale,
                    scalar_mean,
                    scalar_absolute_difference,
                    edge_context,
                ),
                dim=-1,
            )
            hidden = torch.tanh(self.kernel_input(invariants))
            hidden = torch.tanh(self.kernel_hidden(hidden))
            raw = self.kernel_output(hidden)
            diffusivity = F.softplus(raw[:, : self.scalar_channels])
            viscosity = F.softplus(raw[:, self.scalar_channels : self.scalar_channels + 1])
            pressure = self.flux_limit * torch.tanh(raw[:, self.scalar_channels + 1 :])

            scalar_flux = edge_weight[:, None] * diffusivity * scalar_difference
            scalar_flux = self.flux_limit * torch.tanh(scalar_flux / self.flux_limit)
            vector_flux = edge_weight[:, None] * (
                viscosity * vector_difference + pressure * direction
            )
            vector_norm = torch.linalg.vector_norm(vector_flux, dim=-1, keepdim=True)
            bounded_norm = self.flux_limit * torch.tanh(vector_norm / self.flux_limit)
            vector_flux = vector_flux * bounded_norm / vector_norm.clamp_min(self.epsilon)
            scalar_rate.index_add_(0, source, scalar_flux)
            scalar_rate.index_add_(0, target, -scalar_flux)
            vector_rate.index_add_(0, source, vector_flux)
            vector_rate.index_add_(0, target, -vector_flux)
            all_scalar_fluxes.append(scalar_flux)
            all_vector_fluxes.append(vector_flux)

        return GraphOperatorOutput(
            scalar_rate=scalar_rate,
            vector_rate=vector_rate,
            scalar_edge_fluxes=tuple(all_scalar_fluxes),
            vector_edge_fluxes=tuple(all_vector_fluxes),
        )

    def _validate_base(
        self,
        positions: Tensor,
        scalar_state: Tensor,
        vector_state: Tensor,
        edge_indices: Sequence[Tensor],
        scale_values: Sequence[float | Tensor],
    ) -> None:
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError("positions must have shape (N, 3)")
        if scalar_state.shape != (len(positions), self.scalar_channels):
            raise ValueError("scalar_state shape does not match the operator")
        if vector_state.shape != positions.shape:
            raise ValueError("vector_state must match positions")
        if not edge_indices or len(edge_indices) != len(scale_values):
            raise ValueError("edge_indices and scale_values must be non-empty and aligned")
        for edge_index in edge_indices:
            if edge_index.ndim != 2 or edge_index.shape[0] != 2:
                raise ValueError("each edge index must have shape (2, E)")
            if edge_index.dtype not in (torch.int32, torch.int64):
                raise ValueError("edge indices must contain integers")
            if edge_index.numel() and (edge_index.min() < 0 or edge_index.max() >= len(positions)):
                raise ValueError("edge index is outside the particle array")


def _particle_context(
    context: Tensor | None,
    *,
    particle_count: int,
    context_dim: int,
    reference: Tensor,
) -> Tensor:
    if context is None:
        return reference.new_zeros((particle_count, context_dim))
    context = context.to(device=reference.device, dtype=reference.dtype)
    if context.shape == (context_dim,):
        return context.unsqueeze(0).expand(particle_count, -1)
    if context.shape == (particle_count, context_dim):
        return context
    raise ValueError("context must be global or supplied per particle")
