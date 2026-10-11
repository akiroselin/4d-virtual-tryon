from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import Tensor
import torch.nn.functional as F

from .hypernetwork_kirchhoff_love import KirchhoffLoveShellGeometry


@dataclass(frozen=True)
class ReferenceMapFiberState:
    deformation_gradient: Tensor
    metric: Tensor
    metric_change: Tensor
    orientation_quaternion: Tensor


@dataclass(frozen=True)
class ReferenceMapFiberReport:
    representation: str
    vertex_count: int
    face_count: int
    sheet_count: int
    material_map_immutable: bool
    maximum_metric_change: float
    maximum_quaternion_norm_error: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ReferenceMapFiberBundle:
    """Intrinsic material memory plus a quaternionic local-frame fibre.

    Material coordinates and inverse material matrices are the discrete
    reference map. Quaternions carry only the rotational frame; the metric
    retains stretch and shear that no unit quaternion can encode.
    """

    material_coordinates: Tensor
    faces: Tensor
    inverse_material_matrices: Tensor
    rest_positions: Tensor
    face_sheet_ids: Tensor
    rest_metric: Tensor

    def __post_init__(self) -> None:
        vertex_count = len(self.material_coordinates)
        face_count = len(self.faces)
        if self.material_coordinates.shape != (vertex_count, 2):
            raise ValueError("reference material coordinates must have shape (V,2)")
        if self.rest_positions.shape != (vertex_count, 3):
            raise ValueError("reference rest positions must have shape (V,3)")
        if self.faces.shape != (face_count, 3):
            raise ValueError("reference faces must have shape (F,3)")
        if self.inverse_material_matrices.shape != (face_count, 2, 2):
            raise ValueError("inverse material matrices must have shape (F,2,2)")
        if self.face_sheet_ids.shape != (face_count,):
            raise ValueError("one sheet identity is required per face")
        if self.rest_metric.shape != (face_count, 2, 2):
            raise ValueError("reference rest metric must have shape (F,2,2)")
        if torch.any(self.face_sheet_ids < 0):
            raise ValueError("reference sheet identities must be non-negative")

    @classmethod
    def from_shell_geometry(
        cls,
        geometry: KirchhoffLoveShellGeometry,
        rest_positions: Tensor,
    ) -> ReferenceMapFiberBundle:
        shell = geometry.to(rest_positions)
        sheet_ids = shell.face_panel_features.argmax(dim=-1).to(torch.long)
        triangles = rest_positions[shell.faces]
        spatial = torch.stack(
            (triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
            dim=-1,
        )
        deformation = spatial @ shell.inverse_material_matrices
        rest_metric = deformation.transpose(-1, -2) @ deformation
        return cls(
            material_coordinates=shell.vertex_material_coordinates.detach().clone(),
            faces=shell.faces.detach().clone(),
            inverse_material_matrices=shell.inverse_material_matrices.detach().clone(),
            rest_positions=rest_positions.detach().clone(),
            face_sheet_ids=sheet_ids.detach().clone(),
            rest_metric=rest_metric.detach().clone(),
        )

    def sample(self, positions: Tensor) -> ReferenceMapFiberState:
        if positions.shape != self.rest_positions.shape:
            raise ValueError("reference-map sample positions must match the rest shell")
        faces = self.faces.to(device=positions.device)
        inverse = self.inverse_material_matrices.to(positions)
        triangles = positions[faces]
        spatial = torch.stack(
            (triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
            dim=-1,
        )
        deformation = spatial @ inverse
        metric = deformation.transpose(-1, -2) @ deformation
        quaternion = _surface_frame_quaternion(deformation)
        return ReferenceMapFiberState(
            deformation_gradient=deformation,
            metric=metric,
            metric_change=metric - self.rest_metric.to(metric),
            orientation_quaternion=quaternion,
        )

    def report(self, positions: Tensor) -> ReferenceMapFiberReport:
        state = self.sample(positions)
        quaternion_norm_error = (
            torch.linalg.vector_norm(state.orientation_quaternion, dim=-1) - 1.0
        ).abs()
        return ReferenceMapFiberReport(
            representation=(
                "intrinsic material reference map xi=(sheet,u,v) plus Spin(3) "
                "quaternion frame and symmetric shell metric"
            ),
            vertex_count=len(self.material_coordinates),
            face_count=len(self.faces),
            sheet_count=(
                int(self.face_sheet_ids.max().item()) + 1 if len(self.face_sheet_ids) else 0
            ),
            material_map_immutable=True,
            maximum_metric_change=float(state.metric_change.detach().abs().max()),
            maximum_quaternion_norm_error=float(quaternion_norm_error.detach().max()),
        )


def _surface_frame_quaternion(deformation_gradient: Tensor) -> Tensor:
    first = F.normalize(deformation_gradient[:, :, 0], dim=-1, eps=1.0e-10)
    normal = F.normalize(
        torch.linalg.cross(
            deformation_gradient[:, :, 0],
            deformation_gradient[:, :, 1],
            dim=-1,
        ),
        dim=-1,
        eps=1.0e-10,
    )
    second = F.normalize(torch.linalg.cross(normal, first, dim=-1), dim=-1, eps=1.0e-10)
    rotation = torch.stack((first, second, normal), dim=-1)
    return _rotation_matrix_to_quaternion(rotation)


def _rotation_matrix_to_quaternion(rotation: Tensor) -> Tensor:
    if rotation.ndim != 3 or rotation.shape[-2:] != (3, 3):
        raise ValueError("rotation matrices must have shape (N,3,3)")
    m00, m01, m02 = rotation[:, 0, 0], rotation[:, 0, 1], rotation[:, 0, 2]
    m10, m11, m12 = rotation[:, 1, 0], rotation[:, 1, 1], rotation[:, 1, 2]
    m20, m21, m22 = rotation[:, 2, 0], rotation[:, 2, 1], rotation[:, 2, 2]
    half = rotation.new_tensor(0.5)
    qw = half * torch.sqrt((1.0 + m00 + m11 + m22).clamp_min(0.0))
    qx = half * torch.copysign(
        torch.sqrt((1.0 + m00 - m11 - m22).clamp_min(0.0)), m21 - m12
    )
    qy = half * torch.copysign(
        torch.sqrt((1.0 - m00 + m11 - m22).clamp_min(0.0)), m02 - m20
    )
    qz = half * torch.copysign(
        torch.sqrt((1.0 - m00 - m11 + m22).clamp_min(0.0)), m10 - m01
    )
    quaternion = F.normalize(torch.stack((qw, qx, qy, qz), dim=-1), dim=-1)
    return torch.where(quaternion[:, :1] < 0.0, -quaternion, quaternion)
