from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .e3_anchor_field import E3AnchorFieldOutput, E3AnchorNeuralClothField
from .gauge_tactile import SO2GaugeTransportLayer
from .genesis_tactile_adapter import reconstruct_tangent_vector
from .geometric_cloth_prior import GeometricClothPriorOutput, IntrinsicEquivariantClothPrior
from .intrinsic_diffusion import IntrinsicSpectralDiffusion
from .physics_conditioned_flow import SpectralLinear
from .spatiotemporal_scene_graph import (
    HierarchyAwareSpatiotemporalAttention,
    SpatiotemporalAttentionOutput,
)


@dataclass(frozen=True)
class SE3BodyAnchorState:
    positions: Tensor
    normals: Tensor
    velocities: Tensor


@dataclass(frozen=True)
class Cloth4DGeometricConditioning:
    cloth_anchor_features: Tensor
    query_scalar_features: Tensor
    query_vectors: Tensor
    query_vector_features: Tensor
    body_anchor_features: Tensor
    body_tactile_vectors: Tensor
    particle: GeometricClothPriorOutput
    scene: SpatiotemporalAttentionOutput


class GaugeTactileVectorEncoder(nn.Module):
    """Transport a charged SO(2) field and reconstruct one gauge-free 3D vector."""

    def __init__(
        self,
        channels: int,
        *,
        charge: int = 1,
        spectral_cap: float = 0.9,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.transport = SO2GaugeTransportLayer(
            channels,
            charge=charge,
            spectral_cap=spectral_cap,
        )
        self.channel_mix = SpectralLinear(
            channels,
            1,
            bias=False,
            max_singular_value=spectral_cap,
        )

    def forward(
        self,
        tangent_features: Tensor,
        edge_index: Tensor,
        connection_angle: Tensor,
        tangent_u: Tensor,
        tangent_v: Tensor,
        edge_weight: Tensor | None = None,
    ) -> Tensor:
        transported = self.transport(
            tangent_features,
            edge_index,
            connection_angle,
            edge_weight,
        )
        mixed = torch.complex(
            self.channel_mix(transported.real),
            self.channel_mix(transported.imag),
        ).squeeze(-1)
        return reconstruct_tangent_vector(mixed, tangent_u, tangent_v)


class Cloth4DGeometricConditioner(nn.Module):
    """Typed bridge from the geometric stack into the E(3) anchor field.

    The returned scalar/vector tensors map directly to
    ``E3AnchorNeuralClothField`` with:

    - ``cloth_anchor_feature_dim = intrinsic_channels``;
    - ``query_feature_dim = scene_feature_dim``;
    - ``query_vector_feature_dim = 2`` (particle versus scene vector role);
    - ``body_anchor_feature_dim = 1`` (normal tactile force).
    """

    def __init__(
        self,
        intrinsic_channels: int,
        scene_feature_dim: int,
        tactile_channels: int,
        *,
        material_latent_dim: int = 32,
        hidden_dim: int = 48,
        relation_count: int = 4,
        scene_attention_depth: int = 3,
        spectral_cap: float = 0.9,
        causal_scene_graph: bool = False,
    ) -> None:
        super().__init__()
        self.intrinsic_channels = intrinsic_channels
        self.scene_feature_dim = scene_feature_dim
        self.intrinsic = IntrinsicSpectralDiffusion(intrinsic_channels)
        self.particle = IntrinsicEquivariantClothPrior(
            material_latent_dim=material_latent_dim,
            hidden_dim=hidden_dim,
            spectral_cap=spectral_cap,
        )
        self.scene = HierarchyAwareSpatiotemporalAttention(
            scene_feature_dim,
            depth=scene_attention_depth,
            hidden_dim=hidden_dim,
            relation_count=relation_count,
            spectral_cap=spectral_cap,
            causal=causal_scene_graph,
        )
        self.tactile = GaugeTactileVectorEncoder(
            tactile_channels,
            spectral_cap=spectral_cap,
        )

    def forward(
        self,
        *,
        surface_features: Tensor,
        laplace_eigenvalues: Tensor,
        laplace_eigenvectors: Tensor,
        vertex_mass: Tensor,
        particle_positions: Tensor,
        particle_velocities: Tensor,
        particle_edge_index: Tensor,
        rest_lengths: Tensor,
        cotangent_weights: Tensor,
        material_latent: Tensor,
        scene_scalar_features: Tensor,
        scene_positions: Tensor,
        scene_times: Tensor,
        scene_hierarchy_levels: Tensor,
        scene_edge_index: Tensor,
        scene_relation_ids: Tensor,
        query_scene_node_ids: Tensor,
        tactile_features: Tensor,
        tactile_edge_index: Tensor,
        tactile_connection_angle: Tensor,
        tactile_tangent_u: Tensor,
        tactile_tangent_v: Tensor,
        tactile_normal_force: Tensor | None = None,
        tactile_edge_weight: Tensor | None = None,
        tactile_to_body_anchor: Tensor | None = None,
        body_anchor_count: int | None = None,
    ) -> Cloth4DGeometricConditioning:
        vertex_count = len(particle_positions)
        if surface_features.shape != (vertex_count, self.intrinsic_channels):
            raise ValueError("surface_features must align with particle vertices")
        intrinsic = self.intrinsic(
            surface_features,
            laplace_eigenvalues,
            laplace_eigenvectors,
            vertex_mass,
        )
        particle = self.particle(
            particle_positions,
            particle_velocities,
            particle_edge_index,
            rest_lengths,
            cotangent_weights,
            material_latent,
        )
        scene = self.scene(
            scene_scalar_features,
            scene_positions,
            scene_times,
            scene_hierarchy_levels,
            scene_edge_index,
            scene_relation_ids,
        )
        if query_scene_node_ids.shape != (vertex_count,) or query_scene_node_ids.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError("query_scene_node_ids must be integer with shape (V,)")
        if query_scene_node_ids.numel() and (
            query_scene_node_ids.min() < 0
            or query_scene_node_ids.max() >= len(scene_scalar_features)
        ):
            raise ValueError("query scene-node id is outside the scene graph")
        query_scene_node_ids = query_scene_node_ids.to(device=particle_positions.device)
        query_scalar = scene.scalar_features[query_scene_node_ids]
        query_vectors = torch.stack(
            (
                particle.internal_force,
                scene.vector_features[query_scene_node_ids],
            ),
            dim=1,
        )
        vector_roles = particle_positions.new_tensor([[1.0, 0.0], [0.0, 1.0]])
        query_vector_features = vector_roles[None, :, :].expand(vertex_count, -1, -1)

        tactile_vectors = self.tactile(
            tactile_features,
            tactile_edge_index,
            tactile_connection_angle,
            tactile_tangent_u,
            tactile_tangent_v,
            tactile_edge_weight,
        )
        normal_force = _normal_force_column(
            tactile_normal_force,
            len(tactile_vectors),
            tactile_vectors,
        )
        body_features, body_tactile = _scatter_tactile_to_body_anchors(
            normal_force,
            tactile_vectors,
            tactile_to_body_anchor,
            body_anchor_count,
        )
        return Cloth4DGeometricConditioning(
            cloth_anchor_features=intrinsic,
            query_scalar_features=query_scalar,
            query_vectors=query_vectors,
            query_vector_features=query_vector_features,
            body_anchor_features=body_features,
            body_tactile_vectors=body_tactile,
            particle=particle,
            scene=scene,
        )


def query_conditioned_cloth4d(
    field: E3AnchorNeuralClothField,
    conditioning: Cloth4DGeometricConditioning,
    *,
    material_coordinates: Tensor,
    physical_time: float | Tensor,
    panel_ids: Tensor,
    material_latent: Tensor,
    cloth_anchor_positions: Tensor,
    cloth_anchor_material_coordinates: Tensor,
    cloth_anchor_panel_ids: Tensor,
    body_anchor_positions: Tensor,
    query_reference_positions: Tensor | None = None,
    body_anchor_normals: Tensor,
    body_anchor_velocities: Tensor,
    body_anchor_mask: Tensor | None = None,
    initial_velocity: Tensor | None = None,
    boundary_weight: Tensor | None = None,
) -> E3AnchorFieldOutput:
    """Query the anchor field with every typed geometric-conditioning branch."""

    return field(
        material_coordinates=material_coordinates,
        physical_time=physical_time,
        panel_ids=panel_ids,
        material_latent=material_latent,
        cloth_anchor_positions=cloth_anchor_positions,
        cloth_anchor_material_coordinates=cloth_anchor_material_coordinates,
        cloth_anchor_panel_ids=cloth_anchor_panel_ids,
        body_anchor_positions=body_anchor_positions,
        query_reference_positions=query_reference_positions,
        cloth_anchor_features=conditioning.cloth_anchor_features,
        body_anchor_features=conditioning.body_anchor_features,
        body_anchor_normals=body_anchor_normals,
        body_anchor_velocities=body_anchor_velocities,
        body_tactile_vectors=conditioning.body_tactile_vectors,
        body_anchor_mask=body_anchor_mask,
        query_scalar_features=conditioning.query_scalar_features,
        query_vectors=conditioning.query_vectors,
        query_vector_features=conditioning.query_vector_features,
        initial_velocity=initial_velocity,
        boundary_weight=boundary_weight,
    )


def body_anchors_from_se3(
    local_positions: Tensor,
    local_normals: Tensor,
    link_transforms: Tensor,
    anchor_link_ids: Tensor,
    *,
    local_velocities: Tensor | None = None,
) -> SE3BodyAnchorState:
    """Attach a body-anchor bank to an SE(3) link-pose trajectory.

    ``link_transforms`` may be ``(L,4,4)`` for one state or ``(N,L,4,4)`` for a
    physical-time trajectory. Applying a global left SE(3) action to every link
    applies the same action to every returned anchor position.
    """

    if local_positions.ndim != 2 or local_positions.shape[1] != 3:
        raise ValueError("local_positions must have shape (A, 3)")
    anchor_count = len(local_positions)
    if local_normals.shape != local_positions.shape:
        raise ValueError("local_normals must match local_positions")
    if anchor_link_ids.shape != (anchor_count,) or anchor_link_ids.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise ValueError("anchor_link_ids must be integer with shape (A,)")
    squeeze_batch = link_transforms.ndim == 3
    if squeeze_batch:
        link_transforms = link_transforms[None, ...]
    if link_transforms.ndim != 4 or link_transforms.shape[-2:] != (4, 4):
        raise ValueError("link_transforms must have shape (L,4,4) or (N,L,4,4)")
    link_count = link_transforms.shape[1]
    if anchor_link_ids.numel() and (
        anchor_link_ids.min() < 0 or anchor_link_ids.max() >= link_count
    ):
        raise ValueError("anchor link id is outside the transform bank")
    device = link_transforms.device
    dtype = link_transforms.dtype
    ids = anchor_link_ids.to(device=device)
    transforms = link_transforms[:, ids]
    rotation = transforms[..., :3, :3]
    translation = transforms[..., :3, 3]
    points = local_positions.to(device=device, dtype=dtype)
    normals = local_normals.to(device=device, dtype=dtype)
    positions = (rotation @ points[None, :, :, None]).squeeze(-1) + translation
    world_normals = (rotation @ normals[None, :, :, None]).squeeze(-1)
    world_normals = world_normals / torch.linalg.vector_norm(
        world_normals,
        dim=-1,
        keepdim=True,
    ).clamp_min(1.0e-8)
    if local_velocities is None:
        velocities = torch.zeros_like(positions)
    else:
        if local_velocities.shape != local_positions.shape:
            raise ValueError("local_velocities must match local_positions")
        local_velocity = local_velocities.to(device=device, dtype=dtype)
        velocities = (rotation @ local_velocity[None, :, :, None]).squeeze(-1)
    state = SE3BodyAnchorState(positions, world_normals, velocities)
    if squeeze_batch:
        return SE3BodyAnchorState(
            state.positions[0],
            state.normals[0],
            state.velocities[0],
        )
    return state


def _normal_force_column(value: Tensor | None, count: int, reference: Tensor) -> Tensor:
    if value is None:
        return reference.new_zeros((count, 1))
    value = value.to(device=reference.device, dtype=reference.dtype)
    if value.shape == (count,):
        value = value[:, None]
    if value.shape != (count, 1):
        raise ValueError("tactile_normal_force must have shape (T,) or (T,1)")
    return value


def _scatter_tactile_to_body_anchors(
    normal_force: Tensor,
    tactile_vectors: Tensor,
    mapping: Tensor | None,
    body_anchor_count: int | None,
) -> tuple[Tensor, Tensor]:
    tactile_count = len(tactile_vectors)
    if mapping is None:
        if body_anchor_count not in (None, tactile_count):
            raise ValueError("body_anchor_count must match taxels when no mapping is supplied")
        return normal_force, tactile_vectors
    if mapping.shape != (tactile_count,) or mapping.dtype not in (torch.int32, torch.int64):
        raise ValueError("tactile_to_body_anchor must be integer with shape (T,)")
    if body_anchor_count is None or body_anchor_count <= 0:
        raise ValueError("a positive body_anchor_count is required with a tactile mapping")
    mapping = mapping.to(device=tactile_vectors.device)
    if mapping.numel() and (mapping.min() < 0 or mapping.max() >= body_anchor_count):
        raise ValueError("tactile mapping is outside the body-anchor bank")
    features = normal_force.new_zeros((body_anchor_count, 1))
    vectors = tactile_vectors.new_zeros((body_anchor_count, 3))
    count = normal_force.new_zeros((body_anchor_count, 1))
    features.index_add_(0, mapping, normal_force)
    vectors.index_add_(0, mapping, tactile_vectors)
    count.index_add_(0, mapping, torch.ones_like(normal_force))
    denominator = count.clamp_min(1.0)
    return features / denominator, vectors / denominator
