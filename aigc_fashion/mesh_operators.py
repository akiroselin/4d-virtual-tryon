"""Discrete mesh operators used by the intrinsic cloth solver.

The original build imported these from a sibling ``genesis_geometric`` checkout
that was never published with this repository. ``physics.py`` still prefers
that package when it is importable; otherwise it uses these self-contained
implementations so the project runs from a clean clone.
"""

from __future__ import annotations

import torch


def _as_3d(points: torch.Tensor) -> torch.Tensor:
    if points.shape[-1] == 3:
        return points
    if points.shape[-1] == 2:
        return torch.nn.functional.pad(points, (0, 1))
    raise ValueError(f"expected 2D or 3D vertex coordinates, got shape {tuple(points.shape)}")


def lumped_vertex_areas(rest_positions: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """Barycentric lumped area per vertex: one third of each incident triangle."""

    rest = _as_3d(rest_positions)
    faces = faces.long()
    a, b, c = rest[faces[:, 0]], rest[faces[:, 1]], rest[faces[:, 2]]
    face_area = 0.5 * torch.linalg.norm(torch.cross(b - a, c - a, dim=-1), dim=-1)
    areas = torch.zeros(rest.shape[0], dtype=rest.dtype, device=rest.device)
    share = (face_area / 3.0).repeat_interleave(3)
    areas.index_add_(0, faces.reshape(-1), share)
    return areas.clamp_min(torch.finfo(rest.dtype).eps)


def cotangent_laplacian(
    positions: torch.Tensor,
    rest_positions: torch.Tensor,
    faces: torch.Tensor,
) -> torch.Tensor:
    """Apply the rest-metric cotangent Laplacian to the current positions.

    Weights come from ``rest_positions`` (the immutable material metric) and
    are clamped to be non-negative, then each row is normalised by its weight
    sum. The result ``sum_j w_ij (x_j - x_i) / sum_j w_ij`` is a bounded
    curvature-normal estimate, so ``x + step * L(x)`` is a stable smoothing
    step for ``0 < step <= 1``.
    """

    rest = _as_3d(rest_positions)
    faces = faces.long()
    eps = torch.finfo(rest.dtype).eps
    rows, cols, values = [], [], []
    for k in range(3):
        i = faces[:, (k + 1) % 3]
        j = faces[:, (k + 2) % 3]
        opposite = faces[:, k]
        u = rest[i] - rest[opposite]
        v = rest[j] - rest[opposite]
        cross = torch.linalg.norm(torch.cross(u, v, dim=-1), dim=-1).clamp_min(eps)
        cot = 0.5 * ((u * v).sum(-1) / cross).clamp_min(0.0)
        rows += [i, j]
        cols += [j, i]
        values += [cot, cot]
    rows_t, cols_t = torch.cat(rows), torch.cat(cols)
    w = torch.cat(values).to(positions.dtype).unsqueeze(-1)
    accum = torch.zeros_like(positions).index_add_(0, rows_t, w * (positions[cols_t] - positions[rows_t]))
    row_sum = torch.zeros(positions.shape[0], 1, dtype=positions.dtype, device=positions.device)
    row_sum.index_add_(0, rows_t, w)
    return accum / row_sum.clamp_min(eps)


def normalized_graph_laplacian(positions: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """Uniform umbrella Laplacian ``mean(neighbours) - x`` for batched ``(B, V, 3)`` positions."""

    faces = faces.long()
    num_vertices = positions.shape[-2]
    edges = torch.cat([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], dim=0)
    edges = torch.unique(torch.cat([edges, edges.flip(1)], dim=0), dim=0)
    src, dst = edges[:, 0], edges[:, 1]
    neighbour_sum = torch.zeros_like(positions).index_add_(-2, src, positions[..., dst, :])
    degree = torch.zeros(num_vertices, dtype=positions.dtype, device=positions.device)
    degree.index_add_(0, src, torch.ones_like(src, dtype=positions.dtype))
    degree = degree.unsqueeze(-1)
    neighbour_mean = neighbour_sum / degree.clamp_min(1.0)
    return (degree > 0).to(positions.dtype) * (neighbour_mean - positions)
