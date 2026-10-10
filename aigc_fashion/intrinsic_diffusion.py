from __future__ import annotations

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class IntrinsicSpectralDiffusion(nn.Module):
    """DiffusionNet-style learned heat propagation on a surface spectrum.

    Eigenvectors must be mass-orthonormal and eigenvalues must use the positive
    ``-Delta`` convention. The learned time is positive for every feature channel.
    """

    def __init__(self, channels: int, *, initial_time: float = 0.1, minimum_time: float = 1.0e-6) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        if initial_time <= minimum_time or minimum_time <= 0.0:
            raise ValueError("initial_time must be greater than a positive minimum_time")
        self.channels = channels
        self.minimum_time = float(minimum_time)
        initial_raw = math.log(math.expm1(initial_time - minimum_time))
        self.raw_time = nn.Parameter(torch.full((channels,), initial_raw))

    @property
    def diffusion_time(self) -> Tensor:
        return F.softplus(self.raw_time) + self.minimum_time

    def forward(
        self,
        features: Tensor,
        eigenvalues: Tensor,
        eigenvectors: Tensor,
        mass: Tensor,
    ) -> Tensor:
        if features.ndim != 2 or features.shape[1] != self.channels:
            raise ValueError(f"features must have shape (V, {self.channels})")
        vertex_count = features.shape[0]
        if eigenvectors.ndim != 2 or eigenvectors.shape[0] != vertex_count:
            raise ValueError("eigenvectors must have shape (V, K)")
        basis_size = eigenvectors.shape[1]
        if eigenvalues.shape != (basis_size,):
            raise ValueError("eigenvalues must have shape (K,)")
        if mass.shape != (vertex_count,):
            raise ValueError("mass must have shape (V,)")
        if torch.any(eigenvalues < -1.0e-7) or torch.any(mass <= 0.0):
            raise ValueError("eigenvalues must be non-negative and mass positive")

        dtype = features.dtype
        device = features.device
        values = eigenvalues.to(device=device, dtype=dtype)
        basis = eigenvectors.to(device=device, dtype=dtype)
        vertex_mass = mass.to(device=device, dtype=dtype)
        coefficients = basis.transpose(0, 1) @ (vertex_mass[:, None] * features)
        heat = torch.exp(-values[:, None] * self.diffusion_time[None, :])
        return basis @ (heat * coefficients)
