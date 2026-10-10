from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class GaugeTactileInput:
    tangent_features: Tensor
    normal_force: Tensor
    tangent_u: Tensor
    tangent_v: Tensor
    connection_angle: Tensor


def adapt_genesis_taxel_field(
    probe_positions: Tensor,
    probe_normals: Tensor,
    force: Tensor,
    edge_index: Tensor,
    *,
    tangent_u: Tensor | None = None,
    tangent_v: Tensor | None = None,
) -> GaugeTactileInput:
    """Convert Genesis link-local taxel data into an SO(2) gauge field.

    ``probe_positions``, ``probe_normals``, and ``force`` must all use the same
    Genesis link-local frame. No canonical tangent x-axis is assumed.
    """

    _validate_vectors(probe_positions, probe_normals, force)
    normals = _normalize(probe_normals)
    if tangent_u is None or tangent_v is None:
        if tangent_u is not None or tangent_v is not None:
            raise ValueError("both tangent axes must be supplied together")
        tangent_u, tangent_v = tangent_frames_from_normals(normals)
    else:
        _validate_tangent_frames(normals, tangent_u, tangent_v)
    normal_force = (force * normals).sum(dim=-1, keepdim=True)
    tangent_features = project_tangent_vector(force, tangent_u, tangent_v).unsqueeze(-1)
    connection = discrete_parallel_transport_angles(normals, tangent_u, tangent_v, edge_index)
    return GaugeTactileInput(
        tangent_features=tangent_features,
        normal_force=normal_force,
        tangent_u=tangent_u,
        tangent_v=tangent_v,
        connection_angle=connection,
    )


def tangent_frames_from_normals(normals: Tensor) -> tuple[Tensor, Tensor]:
    normals = _normalize(normals)
    axes = torch.eye(3, device=normals.device, dtype=normals.dtype)
    axis_index = normals.abs().argmin(dim=-1)
    reference = axes[axis_index]
    tangent_u = _normalize(torch.linalg.cross(normals, reference, dim=-1))
    tangent_v = torch.linalg.cross(normals, tangent_u, dim=-1)
    return tangent_u, tangent_v


def rotate_tangent_gauge(
    tangent_u: Tensor,
    tangent_v: Tensor,
    frame_angle: Tensor,
) -> tuple[Tensor, Tensor]:
    """Change local bases so coordinates transform as ``z -> exp(i alpha) z``."""

    if tangent_u.shape != tangent_v.shape or frame_angle.shape != tangent_u.shape[:-1]:
        raise ValueError("frame angle must match tangent-frame vertices")
    cosine = torch.cos(frame_angle)[..., None]
    sine = torch.sin(frame_angle)[..., None]
    return cosine * tangent_u - sine * tangent_v, sine * tangent_u + cosine * tangent_v


def project_tangent_vector(vector: Tensor, tangent_u: Tensor, tangent_v: Tensor) -> Tensor:
    if vector.shape != tangent_u.shape or tangent_u.shape != tangent_v.shape:
        raise ValueError("vector and tangent frames must have matching shapes")
    component_u = (vector * tangent_u).sum(dim=-1)
    component_v = (vector * tangent_v).sum(dim=-1)
    return torch.complex(component_u, component_v)


def reconstruct_tangent_vector(features: Tensor, tangent_u: Tensor, tangent_v: Tensor) -> Tensor:
    if not torch.is_complex(features) or features.shape != tangent_u.shape[:-1]:
        raise ValueError("complex features must match tangent-frame vertices")
    return features.real[..., None] * tangent_u + features.imag[..., None] * tangent_v


def discrete_parallel_transport_angles(
    normals: Tensor,
    tangent_u: Tensor,
    tangent_v: Tensor,
    edge_index: Tensor,
) -> Tensor:
    """Compute target-to-source Levi-Civita transport angles on mesh edges."""

    _validate_tangent_frames(normals, tangent_u, tangent_v)
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape (2, E)")
    source, target = edge_index
    source_normal = normals[source]
    target_normal = normals[target]
    cosine = (target_normal * source_normal).sum(dim=-1)
    if torch.any(cosine < -1.0 + 1.0e-5):
        raise ValueError("parallel transport is ambiguous across antipodal normals")
    cross = torch.linalg.cross(target_normal, source_normal, dim=-1)
    target_u = tangent_u[target]
    # Minimal Rodrigues rotation mapping target normal to source normal.
    transported_u = target_u + torch.linalg.cross(cross, target_u, dim=-1)
    transported_u = transported_u + torch.linalg.cross(
        cross,
        torch.linalg.cross(cross, target_u, dim=-1),
        dim=-1,
    ) / (1.0 + cosine).clamp_min(1.0e-6)[:, None]
    transported_u = transported_u - (
        transported_u * source_normal
    ).sum(dim=-1, keepdim=True) * source_normal
    transported_u = _normalize(transported_u)
    component_u = (transported_u * tangent_u[source]).sum(dim=-1)
    component_v = (transported_u * tangent_v[source]).sum(dim=-1)
    return torch.atan2(component_v, component_u)


def _validate_vectors(positions: Tensor, normals: Tensor, force: Tensor) -> None:
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("probe_positions must have shape (N, 3)")
    if normals.shape != positions.shape or force.shape != positions.shape:
        raise ValueError("probe_normals and force must match probe_positions")


def _validate_tangent_frames(normals: Tensor, tangent_u: Tensor, tangent_v: Tensor) -> None:
    if normals.shape != tangent_u.shape or tangent_u.shape != tangent_v.shape:
        raise ValueError("normal and tangent frames must have matching shapes")
    normal = _normalize(normals)
    u = _normalize(tangent_u)
    v = _normalize(tangent_v)
    tolerance = 2.0e-4
    if torch.any((normal * u).sum(dim=-1).abs() > tolerance):
        raise ValueError("tangent_u is not tangent to the surface")
    if torch.any((normal * v).sum(dim=-1).abs() > tolerance):
        raise ValueError("tangent_v is not tangent to the surface")
    if torch.any((u * v).sum(dim=-1).abs() > tolerance):
        raise ValueError("tangent axes are not orthogonal")


def _normalize(vector: Tensor) -> Tensor:
    norm = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
    if torch.any(norm < 1.0e-8):
        raise ValueError("frame vectors must be non-zero")
    return vector / norm
