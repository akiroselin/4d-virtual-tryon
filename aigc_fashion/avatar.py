from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch

from .data import AvatarSpec, GarmentMesh


@dataclass(frozen=True)
class AvatarCollisionReport:
    corrected_vertices: int
    max_penetration: float
    mean_penetration: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


class SmplAvatarProxy:
    """Measurement-driven generic avatar support surface.

    This fallback uses only body measurements and a differentiable torso height
    field. For production, prefer BodyMeshAvatar with a legal commercial body
    mesh such as Apache-licensed Anny default topology or a scanned body mesh.
    """

    def __init__(self, spec: AvatarSpec, clearance: float = 0.018):
        self.spec = spec
        self.clearance = clearance
        self.chest_radius = spec.chest_cm / 100.0 / (2.0 * np.pi)
        self.waist_radius = spec.waist_cm / 100.0 / (2.0 * np.pi)
        self.shoulder_half_width = spec.shoulder_cm / 100.0 * 0.52
        self.torso_half_height = spec.height_cm / 100.0 * 0.19
        self.center_y = -0.02
        self.base_z = 0.88
        self.chest_bulge = max(0.16, self.chest_radius * 1.6)

    def summary(self) -> dict[str, float | str]:
        return {
            "type": "measurement_proxy",
            "body_model": self.spec.body_model,
            "pose": self.spec.pose,
            "chest_radius": float(self.chest_radius),
            "waist_radius": float(self.waist_radius),
            "shoulder_half_width": float(self.shoulder_half_width),
            "torso_half_height": float(self.torso_half_height),
            "clearance": float(self.clearance),
        }

    def support_height_np(self, xy: np.ndarray) -> np.ndarray:
        x = xy[:, 0]
        y = xy[:, 1]
        y_norm = (y - self.center_y) / max(self.torso_half_height, 1e-6)
        width = self._half_width_np(y_norm)
        inside = (np.abs(y_norm) <= 1.0) & (np.abs(x) <= width)
        x_norm = x / np.maximum(width, 1e-6)
        r2 = x_norm * x_norm + 0.72 * y_norm * y_norm
        surface = self.base_z + self.chest_bulge * np.sqrt(np.maximum(0.0, 1.0 - r2))
        return np.where(inside, surface, -np.inf)

    def back_height_np(self, xy: np.ndarray) -> np.ndarray:
        front = self.support_height_np(xy)
        return np.where(np.isfinite(front), self.base_z - 0.55 * (front - self.base_z), np.inf)

    def support_height_torch(self, positions: torch.Tensor) -> torch.Tensor:
        x = positions[:, 0]
        y = positions[:, 1]
        y_norm = (y - self.center_y) / max(self.torso_half_height, 1e-6)
        width = self._half_width_torch(y_norm)
        inside = (torch.abs(y_norm) <= 1.0) & (torch.abs(x) <= width)
        x_norm = x / torch.clamp(width, min=1e-6)
        r2 = x_norm * x_norm + 0.72 * y_norm * y_norm
        surface = self.base_z + self.chest_bulge * torch.sqrt(torch.clamp(1.0 - r2, min=0.0))
        outside = torch.full_like(surface, -1.0e6)
        return torch.where(inside, surface, outside)

    def back_height_torch(self, positions: torch.Tensor) -> torch.Tensor:
        front = self.support_height_torch(positions)
        back = self.base_z - 0.55 * (front - self.base_z)
        outside = torch.full_like(back, 1.0e6)
        return torch.where(front > -1.0e5, back, outside)

    def project_torch(
        self,
        positions: torch.Tensor,
        panel_labels: list[str] | None = None,
        collision_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, AvatarCollisionReport]:
        if panel_labels is not None and any(label.startswith("back_") for label in panel_labels):
            return self._project_shell_torch(positions, panel_labels, collision_mask=collision_mask)

        support = self.support_height_torch(positions) + self.clearance
        penetration = torch.clamp(support - positions[:, 2], min=0.0)
        projected = positions.clone()
        finite = support > -1.0e5
        if collision_mask is not None:
            finite = finite & collision_mask
        projected[finite, 2] = torch.maximum(projected[finite, 2], support[finite])
        active = penetration[finite]
        corrected = int((active > 0.0).sum().item()) if active.numel() else 0
        max_penetration = float(active.max().item()) if active.numel() else 0.0
        mean_penetration = float(active.mean().item()) if active.numel() else 0.0
        return projected, AvatarCollisionReport(corrected, max_penetration, mean_penetration)

    def _project_shell_torch(
        self,
        positions: torch.Tensor,
        panel_labels: list[str],
        collision_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, AvatarCollisionReport]:
        support_front = self.support_height_torch(positions) + self.clearance
        support_back = self.back_height_torch(positions) - self.clearance
        is_back = torch.tensor(
            [label.startswith("back_") for label in panel_labels],
            dtype=torch.bool,
            device=positions.device,
        )
        projected = positions.clone()

        active_mask = collision_mask if collision_mask is not None else torch.ones_like(is_back)
        front_finite = active_mask & (~is_back) & (support_front > -1.0e5)
        front_penetration = torch.clamp(support_front - positions[:, 2], min=0.0)
        projected[front_finite, 2] = torch.maximum(projected[front_finite, 2], support_front[front_finite])

        back_finite = active_mask & is_back & (support_back < 1.0e5)
        back_penetration = torch.clamp(positions[:, 2] - support_back, min=0.0)
        projected[back_finite, 2] = torch.minimum(projected[back_finite, 2], support_back[back_finite])

        active = torch.cat([front_penetration[front_finite], back_penetration[back_finite]])
        corrected = int((active > 0.0).sum().item()) if active.numel() else 0
        max_penetration = float(active.max().item()) if active.numel() else 0.0
        mean_penetration = float(active.mean().item()) if active.numel() else 0.0
        return projected, AvatarCollisionReport(corrected, max_penetration, mean_penetration)

    def dress_mesh(self, mesh: GarmentMesh, garment_offset: float = 0.045) -> GarmentMesh:
        """Wrap a sewn T-shirt shell onto a measurement-driven avatar proxy.

        The front and back panels start from the same 2D sewing-pattern
        coordinates. Before simulation they must be wrapped around the body
        cross-section; otherwise the collision step pulls the front and back
        layers apart and side seams behave like highly stretchable bridges.
        """

        vertices = mesh.vertices.copy()
        is_back = np.asarray([label.startswith("back_") for label in mesh.panel_labels], dtype=bool)
        body_like = np.asarray(
            [
                ("body" in label or "shoulder" in label) and "sleeve" not in label
                for label in mesh.panel_labels
            ],
            dtype=bool,
        )
        if body_like.any():
            garment_half = float(np.percentile(np.abs(vertices[body_like, 0]), 96))
        else:
            garment_half = float(np.percentile(np.abs(vertices[:, 0]), 82))
        garment_half = max(garment_half, 1e-4)

        y = vertices[:, 1]
        y_norm = (y - self.center_y) / max(self.torso_half_height, 1e-6)
        body_half_width = self._half_width_np(y_norm) + self.clearance + garment_offset
        depth = self.chest_bulge + self.clearance + 0.45 * garment_offset
        u_raw = vertices[:, 0] / garment_half
        u = np.clip(u_raw, -1.0, 1.0)
        sleeve_extra = np.sign(u_raw) * np.maximum(np.abs(vertices[:, 0]) - garment_half, 0.0)
        side_blend = np.sqrt(np.maximum(0.0, 1.0 - u * u))

        vertices[:, 0] = u * body_half_width + sleeve_extra
        shell_half_thickness = 0.006
        front_z = self.base_z + depth * side_blend + shell_half_thickness
        back_z = self.base_z - 0.62 * depth * side_blend - shell_half_thickness
        vertices[~is_back, 2] = front_z[~is_back]
        vertices[is_back, 2] = back_z[is_back]

        sleeve_like = np.asarray(["sleeve" in label for label in mesh.panel_labels], dtype=bool)
        vertices[sleeve_like, 1] -= 0.025 * np.minimum(np.abs(sleeve_extra[sleeve_like]) / 0.12, 1.0)
        vertices[sleeve_like & ~is_back, 2] -= 0.035
        vertices[sleeve_like & is_back, 2] += 0.02

        return GarmentMesh(
            vertices=vertices.astype(np.float32),
            faces=mesh.faces.copy(),
            pinned_vertices=mesh.pinned_vertices.copy(),
            panel_labels=list(mesh.panel_labels),
            seam_edges=list(mesh.seam_edges),
            rest_vertices=None if mesh.rest_vertices is None else mesh.rest_vertices.copy(),
            uvs=None if mesh.uvs is None else mesh.uvs.copy(),
            seam_interfaces=list(mesh.seam_interfaces),
        )

    def _half_width_np(self, y_norm: np.ndarray) -> np.ndarray:
        lower = (y_norm + 1.0) * 0.5
        torso = self.waist_radius * (1.0 - lower) + self.chest_radius * 1.12 * lower
        shoulder = self.shoulder_half_width * np.exp(-((y_norm - 0.78) ** 2) / 0.08)
        return np.maximum(torso, shoulder)

    def _half_width_torch(self, y_norm: torch.Tensor) -> torch.Tensor:
        lower = (y_norm + 1.0) * 0.5
        torso = self.waist_radius * (1.0 - lower) + self.chest_radius * 1.12 * lower
        shoulder = self.shoulder_half_width * torch.exp(-((y_norm - 0.78) ** 2) / 0.08)
        return torch.maximum(torso, shoulder)
