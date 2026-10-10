from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .cortexode_volume_pipeline import GarmentUNet3D
from .data import GarmentMesh
from .panel_instance_grower import PanelFloodFillOutput, PanelInstanceGrowthModel
from .topology_correction import analyze_triangle_mesh


@dataclass(frozen=True)
class PanelSemanticGuidance:
    """Soft SewFormer/FFN panel semantics consumed by the volume network.

    ``role_descriptors`` is the category-open path: it can contain text/image
    embeddings for previously unseen roles. ``role_logits`` is the finite
    SewFormer taxonomy fallback. The panel dimension is intentionally dynamic.
    """

    panel_masks: Tensor
    seam_adjacency: Tensor
    valid_panels: Tensor
    role_logits: Tensor | None = None
    role_descriptors: Tensor | None = None
    seam_volume: Tensor | None = None
    opening_volume: Tensor | None = None

    def validate(self) -> None:
        if self.panel_masks.ndim != 4:
            raise ValueError("panel_masks must have shape (B,P,H,W)")
        batch, panels = self.panel_masks.shape[:2]
        if self.seam_adjacency.shape != (batch, panels, panels):
            raise ValueError("seam_adjacency must have shape (B,P,P)")
        if self.valid_panels.shape != (batch, panels):
            raise ValueError("valid_panels must have shape (B,P)")
        if self.role_logits is None and self.role_descriptors is None:
            raise ValueError("role_logits or role_descriptors must be supplied")
        if self.role_logits is not None and self.role_logits.shape[:2] != (batch, panels):
            raise ValueError("role_logits must have shape (B,P,R)")
        if self.role_descriptors is not None and self.role_descriptors.shape[:2] != (batch, panels):
            raise ValueError("role_descriptors must have shape (B,P,E)")
        for name, volume in (("seam_volume", self.seam_volume), ("opening_volume", self.opening_volume)):
            if volume is not None and (volume.ndim != 5 or volume.shape[:2] != (batch, 1)):
                raise ValueError(f"{name} must have shape (B,1,D,H,W)")


@dataclass(frozen=True)
class PhysicsVolumeSupervision:
    """Physics evidence sampled on the common world-space volume grid."""

    residual: Tensor | None = None
    teacher_sdf: Tensor | None = None
    material_target: Tensor | None = None


@dataclass(frozen=True)
class JointShellPrediction:
    closed_proxy_sdf: Tensor
    occupancy_logits: Tensor
    seam_logits: Tensor
    material_field: Tensor
    panel_tokens: Tensor
    seed_objectness_logits: Tensor
    seed_role_logits: Tensor


@dataclass(frozen=True)
class JointShellLossWeights:
    sdf: float = 1.0
    occupancy: float = 0.5
    eikonal: float = 0.08
    body_contact: float = 0.35
    seam_alignment: float = 0.2
    opening: float = 0.2
    physics_residual: float = 0.4
    genesis_teacher: float = 0.35
    material: float = 0.15


@dataclass(frozen=True)
class JointShellLoss:
    total: Tensor
    terms: dict[str, Tensor]

    def detached_summary(self) -> dict[str, float]:
        return {name: float(value.detach()) for name, value in self.terms.items()}


class SewFormerSemanticGraphEncoder(nn.Module):
    """Encode variable-count panel instances and their sewing graph."""

    def __init__(
        self,
        visual_channels: int,
        role_count: int,
        *,
        descriptor_dim: int = 32,
        semantic_dim: int = 32,
        message_steps: int = 2,
    ) -> None:
        super().__init__()
        if min(visual_channels, role_count, descriptor_dim, semantic_dim, message_steps) <= 0:
            raise ValueError("semantic graph dimensions must be positive")
        self.role_count = role_count
        self.descriptor_dim = descriptor_dim
        self.semantic_dim = semantic_dim
        self.message_steps = message_steps
        self.role_table = nn.Parameter(torch.randn(role_count, descriptor_dim) * 0.02)
        self.visual_projection = nn.Linear(visual_channels, semantic_dim)
        self.role_projection = nn.Linear(descriptor_dim, semantic_dim)
        self.message = nn.Sequential(
            nn.Linear(2 * semantic_dim, 2 * semantic_dim),
            nn.GELU(),
            nn.Linear(2 * semantic_dim, semantic_dim),
        )
        self.norm = nn.LayerNorm(semantic_dim)

    def forward(self, visual_features: Tensor, guidance: PanelSemanticGuidance) -> tuple[Tensor, Tensor]:
        guidance.validate()
        if visual_features.ndim != 4 or visual_features.shape[0] != guidance.panel_masks.shape[0]:
            raise ValueError("visual_features must have shape (B,C,H,W) and match guidance batch")
        masks = F.interpolate(
            guidance.panel_masks.float(),
            size=visual_features.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).clamp(0.0, 1.0)
        valid = guidance.valid_panels.to(visual_features.dtype)
        masks = masks * valid[:, :, None, None]
        normalizer = masks.sum(dim=(-2, -1), keepdim=True).clamp_min(1.0e-6)
        visual_tokens = torch.einsum("bphw,bchw->bpc", masks / normalizer, visual_features)

        if guidance.role_descriptors is not None:
            descriptors = guidance.role_descriptors.to(visual_features)
            if descriptors.shape[-1] != self.descriptor_dim:
                raise ValueError(
                    f"role_descriptors last dimension must be {self.descriptor_dim}"
                )
        else:
            if guidance.role_logits is None or guidance.role_logits.shape[-1] != self.role_count:
                raise ValueError(f"role_logits last dimension must be {self.role_count}")
            descriptors = torch.softmax(guidance.role_logits, dim=-1) @ self.role_table

        tokens = self.visual_projection(visual_tokens) + self.role_projection(descriptors)
        adjacency = guidance.seam_adjacency.to(tokens.dtype)
        adjacency = adjacency * valid[:, :, None] * valid[:, None, :]
        degree = adjacency.sum(dim=-1, keepdim=True).clamp_min(1.0)
        for _ in range(self.message_steps):
            neighbour = adjacency @ tokens / degree
            tokens = self.norm(tokens + self.message(torch.cat((tokens, neighbour), dim=-1)))
            tokens = tokens * valid[:, :, None]

        dense = torch.einsum("bps,bphw->bshw", tokens, masks)
        dense = dense / masks.sum(dim=1, keepdim=True).clamp_min(1.0)
        return tokens, dense


class SimulationReadyIntrinsicShellNet(nn.Module):
    """FFN + semantic sewing graph + 3D U-Net shell reconstruction.

    The network predicts a closed topology-correction proxy. Semantic openings
    and sewing links are restored after extraction, then the existing
    physics-conditioned CortexODE deforms the fixed intrinsic shell.
    """

    def __init__(
        self,
        role_count: int,
        *,
        coarse_volume_channels: int = 1,
        feature_channels: int = 32,
        descriptor_dim: int = 32,
        semantic_dim: int = 32,
        lift_channels: int = 8,
        material_channels: int = 4,
        sdf_temperature: float = 0.025,
    ) -> None:
        super().__init__()
        if coarse_volume_channels <= 0 or material_channels <= 0 or sdf_temperature <= 0.0:
            raise ValueError("volume dimensions and sdf_temperature must be positive")
        self.coarse_volume_channels = coarse_volume_channels
        self.material_channels = material_channels
        self.sdf_temperature = sdf_temperature
        self.panel_model = PanelInstanceGrowthModel(role_count, feature_channels=feature_channels)
        self.semantic_encoder = SewFormerSemanticGraphEncoder(
            feature_channels,
            role_count,
            descriptor_dim=descriptor_dim,
            semantic_dim=semantic_dim,
        )
        self.image_lift = nn.Conv2d(feature_channels, lift_channels, 1)
        self.semantic_lift = nn.Conv2d(semantic_dim, lift_channels, 1)
        input_channels = coarse_volume_channels + 1 + 2 * lift_channels
        self.volume_network = GarmentUNet3D(
            input_channels=input_channels,
            output_channels=2 + material_channels,
        )

    def refine_panel_patch(
        self,
        visual_patch: Tensor,
        current_mask_logits: Tensor,
        role_ids: Tensor,
    ) -> PanelFloodFillOutput:
        """Expose the trainable recurrent FFN cell for scheduled rollouts."""

        return self.panel_model.flood_fill_network(visual_patch, current_mask_logits, role_ids)

    def forward(
        self,
        image: Tensor,
        coarse_volume: Tensor,
        body_sdf: Tensor,
        guidance: PanelSemanticGuidance,
    ) -> JointShellPrediction:
        guidance.validate()
        if coarse_volume.ndim != 5 or coarse_volume.shape[1] != self.coarse_volume_channels:
            raise ValueError("coarse_volume has incompatible shape")
        if body_sdf.shape != (coarse_volume.shape[0], 1, *coarse_volume.shape[-3:]):
            raise ValueError("body_sdf must have shape (B,1,D,H,W) and match coarse_volume")
        features, objectness, seed_roles = self.panel_model.encode(image)
        tokens, semantic_map = self.semantic_encoder(features, guidance)
        target_hw = coarse_volume.shape[-2:]
        image_map = F.interpolate(self.image_lift(features), target_hw, mode="bilinear", align_corners=False)
        semantic_map = F.interpolate(
            self.semantic_lift(semantic_map), target_hw, mode="bilinear", align_corners=False
        )
        depth = coarse_volume.shape[-3]
        image_volume = image_map[:, :, None].expand(-1, -1, depth, -1, -1)
        semantic_volume = semantic_map[:, :, None].expand(-1, -1, depth, -1, -1)
        raw = self.volume_network(
            torch.cat((coarse_volume, body_sdf, image_volume, semantic_volume), dim=1)
        )
        sdf = raw[:, :1]
        material = F.softplus(raw[:, 1 : 1 + self.material_channels])
        seam_logits = raw[:, 1 + self.material_channels : 2 + self.material_channels]
        return JointShellPrediction(
            closed_proxy_sdf=sdf,
            occupancy_logits=-sdf / self.sdf_temperature,
            seam_logits=seam_logits,
            material_field=material,
            panel_tokens=tokens,
            seed_objectness_logits=objectness,
            seed_role_logits=seed_roles,
        )


def simulation_ready_shell_loss(
    prediction: JointShellPrediction,
    *,
    target_sdf: Tensor,
    body_sdf: Tensor,
    guidance: PanelSemanticGuidance,
    physics: PhysicsVolumeSupervision = PhysicsVolumeSupervision(),
    weights: JointShellLossWeights = JointShellLossWeights(),
    voxel_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    contact_clearance: float = 0.003,
    surface_band: float = 0.015,
) -> JointShellLoss:
    """Differentiable supervision before Marching Cubes/topology restoration."""

    sdf = prediction.closed_proxy_sdf
    if target_sdf.shape != sdf.shape or body_sdf.shape != sdf.shape:
        raise ValueError("target_sdf and body_sdf must match predicted SDF")
    if min(*voxel_spacing, contact_clearance, surface_band) <= 0.0:
        raise ValueError("spacing, clearance and surface_band must be positive")
    guidance.validate()

    occupancy_target = (target_sdf < 0.0).to(sdf.dtype)
    gradients = torch.gradient(sdf, spacing=voxel_spacing, dim=(-3, -2, -1))
    gradient_norm = torch.sqrt(sum(component.square() for component in gradients) + 1.0e-8)
    surface_density = torch.exp(-sdf.abs() / surface_band)
    surface_density = surface_density / surface_density.mean().detach().clamp_min(1.0e-6)

    terms: dict[str, Tensor] = {
        "sdf": F.smooth_l1_loss(sdf, target_sdf),
        "occupancy": F.binary_cross_entropy_with_logits(
            prediction.occupancy_logits, occupancy_target
        ),
        "eikonal": ((gradient_norm - 1.0).square() * surface_density).mean(),
        "body_contact": (
            F.relu(contact_clearance - body_sdf).square() * surface_density
        ).mean()
        / (contact_clearance**2),
    }
    zero = sdf.new_zeros(())
    if guidance.seam_volume is not None:
        seam_target = F.interpolate(
            guidance.seam_volume.to(sdf), size=sdf.shape[-3:], mode="trilinear", align_corners=False
        ).clamp(0.0, 1.0)
        terms["seam_alignment"] = (
            F.binary_cross_entropy_with_logits(prediction.seam_logits, seam_target)
            + (sdf.abs() * seam_target).sum() / seam_target.sum().clamp_min(1.0)
        )
    else:
        terms["seam_alignment"] = zero
    if guidance.opening_volume is not None:
        opening = F.interpolate(
            guidance.opening_volume.to(sdf), size=sdf.shape[-3:], mode="trilinear", align_corners=False
        ).clamp(0.0, 1.0)
        terms["opening"] = (torch.sigmoid(prediction.occupancy_logits) * opening).sum() / opening.sum().clamp_min(1.0)
    else:
        terms["opening"] = zero
    if physics.residual is not None:
        if physics.residual.shape != sdf.shape:
            raise ValueError("physics residual must match predicted SDF")
        terms["physics_residual"] = (physics.residual.to(sdf).square() * surface_density).mean()
    else:
        terms["physics_residual"] = zero
    if physics.teacher_sdf is not None:
        if physics.teacher_sdf.shape != sdf.shape:
            raise ValueError("teacher_sdf must match predicted SDF")
        terms["genesis_teacher"] = F.smooth_l1_loss(sdf, physics.teacher_sdf.to(sdf))
    else:
        terms["genesis_teacher"] = zero
    if physics.material_target is not None:
        if physics.material_target.shape != prediction.material_field.shape:
            raise ValueError("material_target must match predicted material field")
        terms["material"] = F.smooth_l1_loss(
            prediction.material_field, physics.material_target.to(sdf)
        )
    else:
        terms["material"] = zero

    coefficients = asdict(weights)
    total = sum(coefficients[name] * value for name, value in terms.items())
    terms["total"] = total
    return JointShellLoss(total=total, terms=terms)


@dataclass(frozen=True)
class SimulationReadyShellGate:
    passed: bool
    finite: bool
    connected_components: int
    nonmanifold_edges: int
    boundary_loops: int
    expected_boundary_loops: int
    degenerate_faces: int
    body_intersecting_faces: int
    semantic_link_count: int
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def assess_simulation_ready_shell(
    garment: GarmentMesh,
    *,
    expected_boundary_loops: int,
    body_intersecting_faces: int = 0,
    semantic_link_count: int = 0,
) -> SimulationReadyShellGate:
    """Hard gate applied after semantic opening restoration and CortexODE."""

    if expected_boundary_loops < 0 or body_intersecting_faces < 0 or semantic_link_count < 0:
        raise ValueError("gate counts must be non-negative")
    finite = bool(np.isfinite(garment.vertices).all())
    triangles = garment.vertices[garment.faces]
    double_area = np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=1,
    )
    degenerate_faces = int(np.count_nonzero(double_area <= 1.0e-12))
    topology = analyze_triangle_mesh(garment.vertices, garment.faces)
    components = int(topology.connected_components or 0)
    nonmanifold = int(topology.nonmanifold_edges)
    loops = int(topology.boundary_loops or 0)
    reasons = []
    if not finite:
        reasons.append("non-finite vertices")
    if components != 1:
        reasons.append("shell is not one connected component")
    if nonmanifold:
        reasons.append("non-manifold edges")
    if loops != expected_boundary_loops:
        reasons.append("semantic opening count mismatch")
    if degenerate_faces:
        reasons.append("degenerate faces")
    if body_intersecting_faces:
        reasons.append("body intersections")
    return SimulationReadyShellGate(
        passed=not reasons,
        finite=finite,
        connected_components=components,
        nonmanifold_edges=nonmanifold,
        boundary_loops=loops,
        expected_boundary_loops=expected_boundary_loops,
        degenerate_faces=degenerate_faces,
        body_intersecting_faces=body_intersecting_faces,
        semantic_link_count=semantic_link_count,
        reasons=tuple(reasons),
    )
