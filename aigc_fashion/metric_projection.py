from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import Tensor, nn

from .data import MaterialSpec


@dataclass(frozen=True)
class MaterialStrainLimits:
    profile: str
    maximum_tensile_strain: float
    maximum_compressive_strain: float
    source: str = "material-family engineering prior; requires calibration for production"

    def __post_init__(self) -> None:
        if not 0.0 <= self.maximum_tensile_strain < 1.0:
            raise ValueError("maximum_tensile_strain must lie in [0, 1)")
        if not 0.0 <= self.maximum_compressive_strain < 1.0:
            raise ValueError("maximum_compressive_strain must lie in [0, 1)")


@dataclass(frozen=True)
class MetricProjectionDiagnostics:
    maximum_tensile_strain: float
    maximum_compressive_strain: float
    maximum_limit_violation: float
    violating_edges: int
    edge_count: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def material_strain_limits(material: MaterialSpec) -> MaterialStrainLimits:
    """Return conservative family priors, not lab-certified fabric values."""

    name = material.name.lower()
    if "denim" in name:
        return MaterialStrainLimits("denim_woven", 0.035, 0.025)
    if "linen" in name:
        return MaterialStrainLimits("linen_woven", 0.035, 0.025)
    if "silk" in name and "knit" not in name:
        return MaterialStrainLimits("silk_woven", 0.045, 0.03)
    if any(token in name for token in ("jersey", "interlock", "rib knit", "knit")):
        return MaterialStrainLimits("elastic_knit", 0.18, 0.10)
    if "wool" in name:
        return MaterialStrainLimits("wool_woven_or_felt", 0.075, 0.05)
    if any(token in name for token in ("woven", "cotton", "coating")):
        return MaterialStrainLimits("general_woven", 0.045, 0.03)
    # Stiffness only selects a cautious fallback envelope; it is not treated as
    # a calibrated conversion from simulator units to measured elongation.
    tensile = float(np.clip(0.13 / np.sqrt(max(material.stretch_stiffness, 1.0) / 8.0), 0.04, 0.14))
    return MaterialStrainLimits("uncalibrated_fallback", tensile, 0.6 * tensile)


class DifferentiableMetricProjector(nn.Module):
    """Recover an embedding that escaped Hyper-NODE material strain bounds.

    The immutable edge lengths are taken from the construction rest surface.
    Conflict-free edge-color sweeps keep the operation differentiable while
    avoiding the slow boundary-only convergence of Jacobi updates. This module
    is a failure guard, not a per-step constitutive law. A strict post-check is
    required because incompatible pinned positions can make the constraint set
    empty.
    """

    def __init__(
        self,
        edges: Tensor,
        rest_positions: Tensor,
        pinned_vertices: Tensor,
        limits: MaterialStrainLimits,
        *,
        iterations: int = 12,
        maximum_adaptive_iterations: int = 0,
        relaxation: float = 1.0,
        tolerance: float = 1.0e-3,
    ) -> None:
        super().__init__()
        if edges.ndim != 2 or edges.shape[1] != 2:
            raise ValueError("edges must have shape (E,2)")
        if rest_positions.ndim != 2 or rest_positions.shape[1] != 3:
            raise ValueError("rest_positions must have shape (V,3)")
        if (
            iterations <= 0
            or maximum_adaptive_iterations < 0
            or not 0.0 < relaxation <= 1.0
            or tolerance < 0.0
        ):
            raise ValueError("invalid metric projection settings")
        edges = edges.detach().to(torch.long)
        rest = rest_positions.detach().to(torch.float32)
        rest_lengths = torch.linalg.vector_norm(
            rest[edges[:, 0]] - rest[edges[:, 1]], dim=-1
        )
        if torch.any(rest_lengths <= 1.0e-8):
            raise ValueError("construction rest mesh contains zero-length edges")
        inverse_mass = torch.ones(len(rest), dtype=torch.float32)
        pinned = pinned_vertices.detach().to(torch.long)
        if len(pinned):
            inverse_mass[pinned] = 0.0
        self.register_buffer("edges", edges)
        self.register_buffer("rest_lengths", rest_lengths)
        self.register_buffer("inverse_mass", inverse_mass)
        self.register_buffer("pinned_vertices", pinned)
        self.register_buffer("pinned_positions", rest[pinned].clone())
        edge_colors = _greedy_edge_colors(edges, len(rest))
        self.register_buffer("edge_colors", edge_colors)
        self.edge_color_count = int(edge_colors.max().item()) + 1 if len(edges) else 0
        self.limits = limits
        self.iterations = iterations
        self.maximum_adaptive_iterations = maximum_adaptive_iterations
        self.relaxation = relaxation
        self.tolerance = tolerance

    def forward(self, positions: Tensor, *, pinned_positions: Tensor | None = None) -> Tensor:
        if positions.ndim != 2 or positions.shape != (len(self.inverse_mass), 3):
            raise ValueError("positions must have shape (V,3)")
        projected = positions
        edges = self.edges.to(positions.device)
        rest_lengths = self.rest_lengths.to(positions)
        inverse_mass = self.inverse_mass.to(positions)
        lower = rest_lengths * (1.0 - self.limits.maximum_compressive_strain)
        upper = rest_lengths * (1.0 + self.limits.maximum_tensile_strain)
        fixed = self.pinned_positions.to(positions) if pinned_positions is None else pinned_positions
        if fixed.shape != (len(self.pinned_vertices), 3):
            raise ValueError("pinned_positions has incompatible shape")
        maximum_iterations = self.iterations + self.maximum_adaptive_iterations
        for iteration in range(maximum_iterations):
            for color in range(self.edge_color_count):
                selected = self.edge_colors.to(positions.device) == color
                color_edges = edges[selected]
                a, b = color_edges[:, 0], color_edges[:, 1]
                delta = projected[a] - projected[b]
                length = torch.linalg.vector_norm(delta, dim=-1).clamp_min(1.0e-9)
                color_lower = lower[selected]
                color_upper = upper[selected]
                target = torch.minimum(torch.maximum(length, color_lower), color_upper)
                violation = length - target
                direction = delta / length[:, None]
                denominator = (inverse_mass[a] + inverse_mass[b]).clamp_min(1.0e-8)
                correction = violation[:, None] * direction
                correction_a = -inverse_mass[a, None] / denominator[:, None] * correction
                correction_b = inverse_mass[b, None] / denominator[:, None] * correction
                projected = projected.index_add(
                    0, a, self.relaxation * correction_a
                ).index_add(0, b, self.relaxation * correction_b)
                if len(self.pinned_vertices):
                    mask = torch.zeros(
                        (len(projected), 1), dtype=torch.bool, device=projected.device
                    )
                    mask[self.pinned_vertices.to(projected.device)] = True
                    fixed_full = torch.zeros_like(projected)
                    fixed_full.index_copy_(
                        0, self.pinned_vertices.to(projected.device), fixed
                    )
                    projected = torch.where(mask, fixed_full, projected)
            if (
                iteration + 1 >= self.iterations
                and self.maximum_adaptive_iterations
                and not self._has_violation(projected)
            ):
                break
        return projected

    def _has_violation(self, positions: Tensor) -> bool:
        with torch.no_grad():
            edges = self.edges.to(positions.device)
            lengths = torch.linalg.vector_norm(
                positions[edges[:, 0]] - positions[edges[:, 1]], dim=-1
            )
            ratio = lengths / self.rest_lengths.to(positions)
            violation = torch.maximum(
                ratio - 1.0 - self.limits.maximum_tensile_strain,
                1.0 - ratio - self.limits.maximum_compressive_strain,
            )
            return bool(torch.any(violation > self.tolerance).item())

    def diagnostics(self, positions: Tensor) -> MetricProjectionDiagnostics:
        with torch.no_grad():
            edges = self.edges.to(positions.device)
            lengths = torch.linalg.vector_norm(
                positions[edges[:, 0]] - positions[edges[:, 1]], dim=-1
            )
            ratio = lengths / self.rest_lengths.to(positions)
            tensile = torch.clamp_min(ratio - 1.0, 0.0)
            compressive = torch.clamp_min(1.0 - ratio, 0.0)
            violation = torch.maximum(
                tensile - self.limits.maximum_tensile_strain,
                compressive - self.limits.maximum_compressive_strain,
            ).clamp_min(0.0)
            return MetricProjectionDiagnostics(
                maximum_tensile_strain=float(tensile.max().item()) if len(tensile) else 0.0,
                maximum_compressive_strain=float(compressive.max().item()) if len(compressive) else 0.0,
                maximum_limit_violation=float(violation.max().item()) if len(violation) else 0.0,
                violating_edges=int(torch.count_nonzero(violation > self.tolerance).item()),
                edge_count=int(len(edges)),
            )

    def assert_within_limits(self, positions: Tensor) -> MetricProjectionDiagnostics:
        diagnostics = self.diagnostics(positions)
        if diagnostics.violating_edges:
            raise RuntimeError(
                "metric projection did not reach the hard material strain bounds: "
                f"{diagnostics.violating_edges}/{diagnostics.edge_count} edges, "
                f"max excess={diagnostics.maximum_limit_violation:.6f}"
            )
        return diagnostics


def _greedy_edge_colors(edges: Tensor, vertex_count: int) -> Tensor:
    """Color edges so every vectorized projection batch is vertex-disjoint."""

    used_by_vertex: list[set[int]] = [set() for _ in range(vertex_count)]
    colors: list[int] = []
    for a_value, b_value in edges.detach().cpu().tolist():
        unavailable = used_by_vertex[a_value] | used_by_vertex[b_value]
        color = 0
        while color in unavailable:
            color += 1
        colors.append(color)
        used_by_vertex[a_value].add(color)
        used_by_vertex[b_value].add(color)
    return torch.as_tensor(colors, dtype=torch.long, device=edges.device)
