from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .physics_conditioned_flow import SpectralLinear


@dataclass(frozen=True)
class E3AnchorFieldOutput:
    positions: Tensor
    displacement: Tensor
    residual_vector: Tensor
    cloth_reference: Tensor
    cloth_reference_weights: Tensor
    cloth_attention: Tensor
    body_attention: Tensor
    query_vector_attention: Tensor


class _SpectralScalarNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        hidden_dim: int,
        hidden_layers: int,
        spectral_cap: float,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
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
            output_dim,
            max_singular_value=spectral_cap,
        )

    def forward(self, features: Tensor) -> Tensor:
        hidden = features
        for layer in self.layers:
            hidden = torch.tanh(layer(hidden))
        return self.output(hidden)


class E3AnchorNeuralClothField(nn.Module):
    """Multi-chart 4D cloth field assembled from typed E(3) vector bases.

    The learned networks emit invariant scalar attention and amplitudes only.
    Spatial output is a sum of relative cloth/body directions, body SDF normals,
    body velocity/tactile vectors, and caller-supplied polar query vectors. Under
    a joint ``x -> Qx + a`` transformation with every polar vector transformed by
    ``Q``, the returned positions transform by ``Qx + a`` for every ``Q in O(3)``.

    Cloth anchors are canonical panel samples. ``query_reference_positions`` may
    provide the exact canonical point for each query when the anchor bank is
    sparse; it remains an E(3)-typed point and therefore preserves equivariance.
    Body anchors may be supplied per query so they can follow an SE(3) pose
    trajectory and retain derivatives with respect to physical time. Body normals
    must be polar SDF-gradient normals.
    """

    def __init__(
        self,
        panel_count: int,
        *,
        material_latent_dim: int = 32,
        query_feature_dim: int = 0,
        cloth_anchor_feature_dim: int = 0,
        body_anchor_feature_dim: int = 0,
        query_vector_feature_dim: int = 0,
        panel_embedding_dim: int = 8,
        hidden_dim: int = 64,
        hidden_layers: int = 3,
        spectral_cap: float = 0.9,
        material_bandwidth: float = 0.08,
        material_scale: float = 0.5,
        spatial_scale: float = 1.0,
        velocity_scale: float = 1.0,
        time_scale: float = 1.0,
        displacement_limit: float = 0.5,
        epsilon: float = 1.0e-4,
    ) -> None:
        super().__init__()
        integer_dimensions = (
            panel_count,
            material_latent_dim,
            panel_embedding_dim,
            hidden_dim,
            hidden_layers,
        )
        optional_dimensions = (
            query_feature_dim,
            cloth_anchor_feature_dim,
            body_anchor_feature_dim,
            query_vector_feature_dim,
        )
        if any(value <= 0 for value in integer_dimensions):
            raise ValueError("panel/network dimensions must be positive")
        if any(value < 0 for value in optional_dimensions):
            raise ValueError("optional feature dimensions must be non-negative")
        positive_scales = (
            spectral_cap,
            material_bandwidth,
            material_scale,
            spatial_scale,
            velocity_scale,
            time_scale,
            displacement_limit,
            epsilon,
        )
        if any(value <= 0.0 for value in positive_scales):
            raise ValueError("caps, scales, limits, and epsilon must be positive")

        self.panel_count = panel_count
        self.material_latent_dim = material_latent_dim
        self.query_feature_dim = query_feature_dim
        self.cloth_anchor_feature_dim = cloth_anchor_feature_dim
        self.body_anchor_feature_dim = body_anchor_feature_dim
        self.query_vector_feature_dim = query_vector_feature_dim
        self.material_bandwidth = float(material_bandwidth)
        self.material_scale = float(material_scale)
        self.spatial_scale = float(spatial_scale)
        self.velocity_scale = float(velocity_scale)
        self.time_scale = float(time_scale)
        self.displacement_limit = float(displacement_limit)
        self.epsilon = float(epsilon)

        self.panel_embedding = nn.Embedding(panel_count, panel_embedding_dim)
        nn.init.normal_(self.panel_embedding.weight, std=0.02)
        # latent, material coordinate, panel, two time channels, boundary,
        # query features, and interpolated intrinsic cloth features
        context_dim = (
            material_latent_dim
            + 2
            + panel_embedding_dim
            + 2
            + 1
            + query_feature_dim
            + cloth_anchor_feature_dim
        )
        cloth_relation_dim = context_dim + 4 + cloth_anchor_feature_dim
        # distance, normal alignment, radial/normal speed, speed norm,
        # tactile norm, and tactile/radial alignment
        body_relation_dim = context_dim + 7 + body_anchor_feature_dim
        query_vector_relation_dim = context_dim + 1 + query_vector_feature_dim
        self.cloth_coefficients = _SpectralScalarNetwork(
            cloth_relation_dim,
            2,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            spectral_cap=spectral_cap,
        )
        self.body_coefficients = _SpectralScalarNetwork(
            body_relation_dim,
            5,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            spectral_cap=spectral_cap,
        )
        self.query_vector_coefficients = _SpectralScalarNetwork(
            query_vector_relation_dim,
            2,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            spectral_cap=spectral_cap,
        )

    def forward(
        self,
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
        cloth_anchor_features: Tensor | None = None,
        body_anchor_features: Tensor | None = None,
        body_anchor_normals: Tensor | None = None,
        body_anchor_velocities: Tensor | None = None,
        body_tactile_vectors: Tensor | None = None,
        body_anchor_mask: Tensor | None = None,
        query_scalar_features: Tensor | None = None,
        query_vectors: Tensor | None = None,
        query_vector_features: Tensor | None = None,
        initial_velocity: Tensor | None = None,
        boundary_weight: Tensor | None = None,
    ) -> E3AnchorFieldOutput:
        batch_size = _validate_query(material_coordinates, panel_ids, self.panel_count)
        reference = material_coordinates
        time = _time_column(physical_time, batch_size, reference)
        boundary = _boundary_column(boundary_weight, batch_size, reference)
        latent = _batch_features(
            material_latent,
            batch_size,
            self.material_latent_dim,
            reference,
            "material_latent",
        )
        query_features = _optional_query_features(
            query_scalar_features,
            batch_size,
            self.query_feature_dim,
            reference,
            "query_scalar_features",
        )
        cloth_features = _anchor_scalar_features(
            cloth_anchor_features,
            len(cloth_anchor_positions),
            self.cloth_anchor_feature_dim,
            reference,
            "cloth_anchor_features",
        )
        cloth_data = self._cloth_reference(
            material_coordinates,
            panel_ids,
            cloth_anchor_positions,
            cloth_anchor_material_coordinates,
            cloth_anchor_panel_ids,
            cloth_features,
        )
        interpolated_reference, reference_weights, cloth_delta_material, cloth_mask = cloth_data
        cloth_reference = _query_reference_positions(
            query_reference_positions,
            interpolated_reference,
        )
        intrinsic_query = reference_weights @ cloth_features
        panel = torch.tanh(self.panel_embedding(panel_ids))
        scaled_time = time / self.time_scale
        bounded_time = torch.tanh(scaled_time)
        context = torch.cat(
            (
                torch.tanh(latent),
                torch.tanh(material_coordinates / self.material_scale),
                panel,
                bounded_time,
                bounded_time.square(),
                boundary,
                torch.tanh(query_features),
                torch.tanh(intrinsic_query),
            ),
            dim=-1,
        )

        cloth_relative = (
            cloth_anchor_positions.to(device=reference.device, dtype=reference.dtype)[None, :, :]
            - cloth_reference[:, None, :]
        )
        cloth_distance, cloth_unit = _norm_and_unit(cloth_relative, self.epsilon)
        cloth_relations = torch.cat(
            (
                _expand_context(context, cloth_relative.shape[1]),
                torch.tanh(cloth_distance / self.spatial_scale),
                torch.tanh(cloth_delta_material / self.material_scale),
                torch.tanh(
                    torch.linalg.vector_norm(cloth_delta_material, dim=-1, keepdim=True)
                    / self.material_scale
                ),
                torch.tanh(cloth_features)[None, :, :].expand(batch_size, -1, -1),
            ),
            dim=-1,
        )
        cloth_raw = self.cloth_coefficients(cloth_relations)
        cloth_attention = _masked_softmax(cloth_raw[..., 0], cloth_mask)
        cloth_amplitude = self.displacement_limit * torch.tanh(cloth_raw[..., 1])
        cloth_vector = (
            cloth_attention[..., None] * cloth_amplitude[..., None] * cloth_unit
        ).sum(dim=1)

        body = _expand_anchor_vectors(
            body_anchor_positions,
            batch_size,
            reference,
            "body_anchor_positions",
        )
        body_count = body.shape[1]
        body_mask = _anchor_mask(body_anchor_mask, batch_size, body_count, reference)
        body_features = _batched_anchor_scalar_features(
            body_anchor_features,
            batch_size,
            body_count,
            self.body_anchor_feature_dim,
            reference,
            "body_anchor_features",
        )
        body_normals = _optional_anchor_vectors(
            body_anchor_normals,
            batch_size,
            body_count,
            reference,
            "body_anchor_normals",
        )
        body_velocities = _optional_anchor_vectors(
            body_anchor_velocities,
            batch_size,
            body_count,
            reference,
            "body_anchor_velocities",
        )
        tactile_vectors = _optional_anchor_vectors(
            body_tactile_vectors,
            batch_size,
            body_count,
            reference,
            "body_tactile_vectors",
        )
        body_relative = body - cloth_reference[:, None, :]
        body_distance, body_unit = _norm_and_unit(body_relative, self.epsilon)
        _, normal_unit = _norm_and_unit(body_normals, self.epsilon)
        body_speed, velocity_unit = _norm_and_unit(body_velocities, self.epsilon)
        tactile_norm, tactile_unit = _norm_and_unit(tactile_vectors, self.epsilon)
        normal_alignment = (normal_unit * body_unit).sum(dim=-1, keepdim=True)
        radial_speed = (body_velocities * body_unit).sum(dim=-1, keepdim=True)
        normal_speed = (body_velocities * normal_unit).sum(dim=-1, keepdim=True)
        tactile_alignment = (tactile_unit * body_unit).sum(dim=-1, keepdim=True)
        body_relations = torch.cat(
            (
                _expand_context(context, body_count),
                torch.tanh(body_distance / self.spatial_scale),
                normal_alignment,
                torch.tanh(radial_speed / self.velocity_scale),
                torch.tanh(normal_speed / self.velocity_scale),
                torch.tanh(body_speed / self.velocity_scale),
                torch.tanh(tactile_norm / self.velocity_scale),
                tactile_alignment,
                torch.tanh(body_features),
            ),
            dim=-1,
        )
        body_raw = self.body_coefficients(body_relations)
        body_attention = _masked_softmax(body_raw[..., 0], body_mask)
        body_amplitude = self.displacement_limit * torch.tanh(body_raw[..., 1:])
        body_basis = torch.stack(
            (body_unit, normal_unit, velocity_unit, tactile_unit),
            dim=-2,
        )
        body_vector = (
            body_attention[..., None, None]
            * body_amplitude[..., None]
            * body_basis
        ).sum(dim=(1, 2))

        query_vector, query_attention = self._query_vector_contribution(
            context,
            query_vectors,
            query_vector_features,
            reference,
        )
        residual = _radial_limit(
            cloth_vector + body_vector + query_vector,
            self.displacement_limit,
            self.epsilon,
        )
        velocity0 = _initial_velocity(initial_velocity, batch_size, reference)
        initial_envelope = 1.0 - torch.exp(-scaled_time.square())
        displacement = time * velocity0 + boundary * initial_envelope * residual
        positions = cloth_reference + displacement
        return E3AnchorFieldOutput(
            positions=positions,
            displacement=displacement,
            residual_vector=residual,
            cloth_reference=cloth_reference,
            cloth_reference_weights=reference_weights,
            cloth_attention=cloth_attention,
            body_attention=body_attention,
            query_vector_attention=query_attention,
        )

    def _cloth_reference(
        self,
        material_coordinates: Tensor,
        panel_ids: Tensor,
        anchor_positions: Tensor,
        anchor_material: Tensor,
        anchor_panels: Tensor,
        anchor_features: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        anchor_count = len(anchor_positions)
        if anchor_positions.ndim != 2 or anchor_positions.shape[1] != 3:
            raise ValueError("cloth_anchor_positions must have shape (A, 3)")
        if anchor_material.shape != (anchor_count, 2):
            raise ValueError("cloth_anchor_material_coordinates must have shape (A, 2)")
        if anchor_panels.shape != (anchor_count,) or anchor_panels.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError("cloth_anchor_panel_ids must be integer with shape (A,)")
        if anchor_count == 0:
            raise ValueError("at least one cloth anchor is required")
        device = material_coordinates.device
        dtype = material_coordinates.dtype
        positions = anchor_positions.to(device=device, dtype=dtype)
        anchor_material = anchor_material.to(device=device, dtype=dtype)
        anchor_panels = anchor_panels.to(device=device)
        delta_material = anchor_material[None, :, :] - material_coordinates[:, None, :]
        squared_distance = delta_material.square().sum(dim=-1)
        mask = panel_ids[:, None] == anchor_panels[None, :]
        if torch.any(~mask.any(dim=-1)):
            raise ValueError("each query panel needs at least one cloth anchor")
        logits = -0.5 * squared_distance / self.material_bandwidth**2
        weights = _masked_softmax(logits, mask)
        reference = weights @ positions
        if anchor_features.shape[0] != anchor_count:
            raise ValueError("cloth anchor features do not match the anchor bank")
        return reference, weights, delta_material, mask

    def _query_vector_contribution(
        self,
        context: Tensor,
        vectors: Tensor | None,
        vector_features: Tensor | None,
        reference: Tensor,
    ) -> tuple[Tensor, Tensor]:
        batch_size = len(context)
        if vectors is None:
            if vector_features is not None:
                raise ValueError("query_vector_features require query_vectors")
            return reference.new_zeros((batch_size, 3)), reference.new_zeros((batch_size, 0))
        vector_bank = _expand_anchor_vectors(vectors, batch_size, reference, "query_vectors")
        vector_count = vector_bank.shape[1]
        features = _batched_anchor_scalar_features(
            vector_features,
            batch_size,
            vector_count,
            self.query_vector_feature_dim,
            reference,
            "query_vector_features",
        )
        vector_norm, vector_unit = _norm_and_unit(vector_bank, self.epsilon)
        relations = torch.cat(
            (
                _expand_context(context, vector_count),
                torch.tanh(vector_norm / self.velocity_scale),
                torch.tanh(features),
            ),
            dim=-1,
        )
        raw = self.query_vector_coefficients(relations)
        attention = torch.softmax(raw[..., 0], dim=-1)
        amplitude = self.displacement_limit * torch.tanh(raw[..., 1])
        vector = (attention[..., None] * amplitude[..., None] * vector_unit).sum(dim=1)
        return vector, attention


def _validate_query(material: Tensor, panel_ids: Tensor, panel_count: int) -> int:
    if material.ndim != 2 or material.shape[1] != 2:
        raise ValueError("material_coordinates must have shape (N, 2)")
    batch_size = len(material)
    if panel_ids.shape != (batch_size,) or panel_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("panel_ids must be integer with shape (N,)")
    if panel_ids.numel() and (panel_ids.min() < 0 or panel_ids.max() >= panel_count):
        raise ValueError("panel id is outside the configured panel vocabulary")
    return batch_size


def _time_column(value: float | Tensor, batch_size: int, reference: Tensor) -> Tensor:
    value = torch.as_tensor(value, device=reference.device, dtype=reference.dtype)
    if value.ndim == 0:
        return value.reshape(1, 1).expand(batch_size, 1)
    if value.shape == (batch_size,):
        return value[:, None]
    if value.shape == (batch_size, 1):
        return value
    raise ValueError("physical_time must be scalar or have shape (N,) or (N, 1)")


def _boundary_column(value: Tensor | None, batch_size: int, reference: Tensor) -> Tensor:
    if value is None:
        return reference.new_ones((batch_size, 1))
    value = value.to(device=reference.device, dtype=reference.dtype)
    if value.shape == (batch_size,):
        value = value[:, None]
    if value.shape != (batch_size, 1):
        raise ValueError("boundary_weight must have shape (N,) or (N, 1)")
    if torch.any(value < 0.0) or torch.any(value > 1.0):
        raise ValueError("boundary_weight must lie in [0, 1]")
    return value


def _batch_features(
    value: Tensor,
    batch_size: int,
    feature_dim: int,
    reference: Tensor,
    name: str,
) -> Tensor:
    value = value.to(device=reference.device, dtype=reference.dtype)
    if value.shape == (feature_dim,):
        return value[None, :].expand(batch_size, -1)
    if value.shape == (1, feature_dim):
        return value.expand(batch_size, -1)
    if value.shape == (batch_size, feature_dim):
        return value
    raise ValueError(f"{name} must be global or have shape (N, {feature_dim})")


def _optional_query_features(
    value: Tensor | None,
    batch_size: int,
    feature_dim: int,
    reference: Tensor,
    name: str,
) -> Tensor:
    if value is None:
        return reference.new_zeros((batch_size, feature_dim))
    value = value.to(device=reference.device, dtype=reference.dtype)
    if value.shape != (batch_size, feature_dim):
        raise ValueError(f"{name} must have shape (N, {feature_dim})")
    return value


def _query_reference_positions(value: Tensor | None, interpolated: Tensor) -> Tensor:
    """Use exact per-query canonical positions when a sparse anchor bank is used.

    The explicit reference is a typed point input: callers must transform it with
    the same E(3) action as every cloth/body anchor. Anchor interpolation still
    supplies intrinsic features and relation weights, while this point guarantees
    an exact initial condition at arbitrary mesh resolution.
    """

    if value is None:
        return interpolated
    value = value.to(device=interpolated.device, dtype=interpolated.dtype)
    if value.shape != interpolated.shape:
        raise ValueError("query_reference_positions must have shape (N, 3)")
    return value


def _anchor_scalar_features(
    value: Tensor | None,
    anchor_count: int,
    feature_dim: int,
    reference: Tensor,
    name: str,
) -> Tensor:
    if value is None:
        return reference.new_zeros((anchor_count, feature_dim))
    value = value.to(device=reference.device, dtype=reference.dtype)
    if value.shape != (anchor_count, feature_dim):
        raise ValueError(f"{name} must have shape (A, {feature_dim})")
    return value


def _batched_anchor_scalar_features(
    value: Tensor | None,
    batch_size: int,
    anchor_count: int,
    feature_dim: int,
    reference: Tensor,
    name: str,
) -> Tensor:
    if value is None:
        return reference.new_zeros((batch_size, anchor_count, feature_dim))
    value = value.to(device=reference.device, dtype=reference.dtype)
    if value.shape == (anchor_count, feature_dim):
        return value[None, :, :].expand(batch_size, -1, -1)
    if value.shape == (batch_size, anchor_count, feature_dim):
        return value
    raise ValueError(f"{name} must be shared or have shape (N, A, {feature_dim})")


def _expand_anchor_vectors(
    value: Tensor,
    batch_size: int,
    reference: Tensor,
    name: str,
) -> Tensor:
    value = value.to(device=reference.device, dtype=reference.dtype)
    if value.ndim == 2 and value.shape[1] == 3:
        return value[None, :, :].expand(batch_size, -1, -1)
    if value.ndim == 3 and value.shape[0] == batch_size and value.shape[2] == 3:
        return value
    raise ValueError(f"{name} must have shape (A, 3) or (N, A, 3)")


def _optional_anchor_vectors(
    value: Tensor | None,
    batch_size: int,
    anchor_count: int,
    reference: Tensor,
    name: str,
) -> Tensor:
    if value is None:
        return reference.new_zeros((batch_size, anchor_count, 3))
    expanded = _expand_anchor_vectors(value, batch_size, reference, name)
    if expanded.shape[1] != anchor_count:
        raise ValueError(f"{name} does not match body_anchor_positions")
    return expanded


def _anchor_mask(
    value: Tensor | None,
    batch_size: int,
    anchor_count: int,
    reference: Tensor,
) -> Tensor:
    if anchor_count == 0:
        raise ValueError("at least one body anchor is required")
    if value is None:
        return torch.ones((batch_size, anchor_count), device=reference.device, dtype=torch.bool)
    value = value.to(device=reference.device, dtype=torch.bool)
    if value.shape == (anchor_count,):
        value = value[None, :].expand(batch_size, -1)
    if value.shape != (batch_size, anchor_count):
        raise ValueError("body_anchor_mask must have shape (A,) or (N, A)")
    if torch.any(~value.any(dim=-1)):
        raise ValueError("each query needs at least one active body anchor")
    return value


def _initial_velocity(value: Tensor | None, batch_size: int, reference: Tensor) -> Tensor:
    if value is None:
        return reference.new_zeros((batch_size, 3))
    value = value.to(device=reference.device, dtype=reference.dtype)
    if value.shape == (3,):
        return value[None, :].expand(batch_size, -1)
    if value.shape == (batch_size, 3):
        return value
    raise ValueError("initial_velocity must have shape (3,) or (N, 3)")


def _expand_context(context: Tensor, anchor_count: int) -> Tensor:
    return context[:, None, :].expand(-1, anchor_count, -1)


def _norm_and_unit(vector: Tensor, epsilon: float) -> tuple[Tensor, Tensor]:
    norm = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
    regularized_norm = torch.sqrt(norm.square() + epsilon**2)
    return norm, vector / regularized_norm


def _masked_softmax(logits: Tensor, mask: Tensor) -> Tensor:
    masked = logits.masked_fill(~mask, -torch.inf)
    weights = torch.softmax(masked, dim=-1)
    return torch.where(mask, weights, torch.zeros_like(weights))


def _radial_limit(vector: Tensor, limit: float, epsilon: float) -> Tensor:
    norm = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
    bounded = limit * torch.tanh(norm / limit)
    return vector * bounded / norm.clamp_min(epsilon)
