from __future__ import annotations

from dataclasses import dataclass, fields
from enum import Enum
from math import prod
from typing import Mapping

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class SpectralLinear(nn.Module):
    """Linear map with an exact, differentiable operator-norm cap.

    ``torch.nn.utils.spectral_norm`` estimates the largest singular value with
    power iteration. This small research model instead evaluates it exactly so
    the configured Lipschitz cap is also testable.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = True,
        max_singular_value: float = 1.0,
    ) -> None:
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise ValueError("feature dimensions must be positive")
        if max_singular_value <= 0.0:
            raise ValueError("max_singular_value must be positive")
        self.in_features = in_features
        self.out_features = out_features
        self.max_singular_value = float(max_singular_value)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def effective_weight(self) -> Tensor:
        sigma = torch.linalg.matrix_norm(self.weight, ord=2)
        scale = torch.clamp(sigma / self.max_singular_value, min=1.0)
        return self.weight / scale

    def forward(self, inputs: Tensor) -> Tensor:
        return F.linear(inputs, self.effective_weight(), self.bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"max_singular_value={self.max_singular_value:g}"
        )


class SceneMaterialLatent(nn.Module):
    """One optimisable material code for one NeuralClothSim scene."""

    def __init__(self, latent_dim: int = 32, *, initial_scale: float = 0.01) -> None:
        super().__init__()
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        if initial_scale < 0.0:
            raise ValueError("initial_scale must be non-negative")
        self.latent_dim = latent_dim
        self.value = nn.Parameter(torch.empty(latent_dim))
        nn.init.normal_(self.value, std=initial_scale)

    def forward(self, batch_size: int | None = None) -> Tensor:
        if batch_size is None:
            return self.value
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        return self.value.unsqueeze(0).expand(batch_size, -1)

    def prior_loss(self) -> Tensor:
        return self.value.square().mean()


@dataclass(frozen=True)
class MaterialConditioning:
    gains: Tensor
    biases: Tensor


class BoundedMaterialHyperNetwork(nn.Module):
    """Maps material coordinates and a scene latent to bounded FiLM adapters.

    The hypernetwork never emits complete flow-network weights. Gains are in
    ``(0, 1)`` and biases are bounded, so conditioning cannot invalidate the
    state-Lipschitz cap of the deformation field.
    """

    def __init__(
        self,
        *,
        latent_dim: int = 32,
        material_coordinate_dim: int = 2,
        panel_context_dim: int = 4,
        hidden_dim: int = 32,
        target_layers: int = 3,
        target_hidden_dim: int = 64,
        spectral_cap: float = 1.0,
        bias_limit: float = 0.25,
    ) -> None:
        super().__init__()
        dimensions = (
            latent_dim,
            material_coordinate_dim,
            panel_context_dim,
            hidden_dim,
            target_layers,
            target_hidden_dim,
        )
        if any(dimension <= 0 for dimension in dimensions):
            raise ValueError("all hypernetwork dimensions must be positive")
        if bias_limit < 0.0:
            raise ValueError("bias_limit must be non-negative")
        self.latent_dim = latent_dim
        self.material_coordinate_dim = material_coordinate_dim
        self.panel_context_dim = panel_context_dim
        self.target_layers = target_layers
        self.target_hidden_dim = target_hidden_dim
        self.bias_limit = float(bias_limit)
        input_dim = latent_dim + material_coordinate_dim + panel_context_dim
        output_dim = target_layers * 2 * target_hidden_dim
        self.input_layer = SpectralLinear(
            input_dim, hidden_dim, max_singular_value=spectral_cap
        )
        self.output_layer = SpectralLinear(
            hidden_dim, output_dim, max_singular_value=spectral_cap
        )

    def forward(
        self,
        material_latent: Tensor,
        material_coordinates: Tensor,
        panel_context: Tensor,
    ) -> MaterialConditioning:
        batch_size = material_coordinates.shape[0]
        latent = _batch_features(
            material_latent,
            batch_size,
            self.latent_dim,
            material_coordinates,
            "material_latent",
        )
        panel = _batch_features(
            panel_context,
            batch_size,
            self.panel_context_dim,
            material_coordinates,
            "panel_context",
        )
        if material_coordinates.ndim != 2 or material_coordinates.shape[1] != self.material_coordinate_dim:
            raise ValueError(
                "material_coordinates must have shape "
                f"(N, {self.material_coordinate_dim})"
            )
        features = torch.cat((latent, material_coordinates, panel), dim=-1)
        hidden = torch.tanh(self.input_layer(features))
        raw = self.output_layer(hidden).reshape(
            batch_size,
            self.target_layers,
            2,
            self.target_hidden_dim,
        )
        gains = torch.sigmoid(raw[:, :, 0, :])
        biases = self.bias_limit * torch.tanh(raw[:, :, 1, :])
        return MaterialConditioning(gains=gains, biases=biases)


class PhysicsConditionedCortexODE(nn.Module):
    """Latent-conditioned E(3)-equivariant CortexODE velocity field.

    ``tau`` is canonical-dressing pseudo-time, not NeuralClothSim physical time.
    The field is a warm start for a fixed panel/seam graph. Hybrid contact and
    fold events must stop and restart the external ODE integrator. Learned
    scalar networks consume no absolute world XYZ: they see intrinsic material
    coordinates, invariant norms/alignments, and scalar SDF samples. Their
    coefficients multiply centered radial and SDF-gradient polar-vector bases.

    ``body_context`` is a sequence of geometric ``(scalar, vector_xyz)`` fields.
    The default eight channels represent fine/coarse ``(SDF, grad SDF)`` pairs.
    """

    def __init__(
        self,
        *,
        state_dim: int = 3,
        latent_dim: int = 32,
        material_coordinate_dim: int = 2,
        body_context_dim: int = 8,
        panel_context_dim: int = 4,
        hidden_dim: int = 64,
        hidden_layers: int = 3,
        spectral_cap: float = 0.9,
        velocity_limit: float = 0.25,
    ) -> None:
        super().__init__()
        dimensions = (
            state_dim,
            latent_dim,
            material_coordinate_dim,
            body_context_dim,
            panel_context_dim,
            hidden_dim,
            hidden_layers,
        )
        if any(dimension <= 0 for dimension in dimensions):
            raise ValueError("all flow dimensions must be positive")
        if state_dim != 3:
            raise ValueError("the E(3) CortexODE state must be a 3D point")
        if body_context_dim % 4 != 0:
            raise ValueError("body_context_dim must contain (scalar, vector_xyz) groups")
        if spectral_cap <= 0.0 or velocity_limit <= 0.0:
            raise ValueError("spectral_cap and velocity_limit must be positive")
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        self.material_coordinate_dim = material_coordinate_dim
        self.body_context_dim = body_context_dim
        self.panel_context_dim = panel_context_dim
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.spectral_cap = float(spectral_cap)
        self.velocity_limit = float(velocity_limit)
        self.body_field_count = body_context_dim // 4
        self.geometry_epsilon = 1.0e-8

        flow_input_dim = (
            1
            + 1
            + material_coordinate_dim
            + 3 * self.body_field_count
            + panel_context_dim
        )
        self.flow_layers = nn.ModuleList(
            [
                SpectralLinear(
                    flow_input_dim if index == 0 else hidden_dim,
                    hidden_dim,
                    max_singular_value=spectral_cap,
                )
                for index in range(hidden_layers)
            ]
        )
        self.output_layer = SpectralLinear(
            hidden_dim,
            1 + self.body_field_count,
            max_singular_value=spectral_cap,
        )
        self.material_hypernetwork = BoundedMaterialHyperNetwork(
            latent_dim=latent_dim,
            material_coordinate_dim=material_coordinate_dim,
            panel_context_dim=panel_context_dim,
            target_layers=hidden_layers,
            target_hidden_dim=hidden_dim,
            spectral_cap=spectral_cap,
        )

    def forward(
        self,
        state: Tensor,
        tau: float | Tensor,
        *,
        material_coordinates: Tensor,
        material_latent: Tensor,
        body_context: Tensor | None = None,
        panel_context: Tensor | None = None,
    ) -> Tensor:
        if state.shape[-1] != self.state_dim:
            raise ValueError(f"state must end in dimension {self.state_dim}")
        leading_shape = state.shape[:-1]
        flat_state = state.reshape(-1, self.state_dim)
        batch_size = flat_state.shape[0]
        coordinates = _flatten_features(
            material_coordinates,
            leading_shape,
            self.material_coordinate_dim,
            state,
            "material_coordinates",
        )
        body = _optional_flat_features(
            body_context,
            leading_shape,
            self.body_context_dim,
            state,
            "body_context",
        )
        panel = _optional_flat_features(
            panel_context,
            leading_shape,
            self.panel_context_dim,
            state,
            "panel_context",
        )
        time = _time_features(tau, leading_shape, state)
        conditioning = self.material_hypernetwork(
            material_latent,
            coordinates,
            panel,
        )

        center = flat_state.mean(dim=0, keepdim=True)
        radial = flat_state - center
        radial_norm, radial_unit = _equivariant_norm_and_unit(
            radial,
            self.geometry_epsilon,
        )
        body_fields = body.reshape(batch_size, self.body_field_count, 4)
        body_scalars = body_fields[..., 0]
        body_vectors = body_fields[..., 1:]
        body_norms, body_units = _equivariant_norm_and_unit(
            body_vectors,
            self.geometry_epsilon,
        )
        radial_alignment = (body_units * radial_unit[:, None, :]).sum(dim=-1)
        geometric_invariants = torch.cat(
            (radial_norm, body_scalars, body_norms, radial_alignment),
            dim=-1,
        )

        hidden = torch.cat((geometric_invariants, time, coordinates, panel), dim=-1)
        for index, layer in enumerate(self.flow_layers):
            hidden = torch.tanh(layer(hidden))
            hidden = torch.tanh(
                conditioning.gains[:, index, :] * hidden
                + conditioning.biases[:, index, :]
            )
        coefficients = self.velocity_limit * torch.tanh(self.output_layer(hidden))
        vector_bases = torch.cat((radial_unit[:, None, :], body_units), dim=1)
        velocity = (coefficients[..., None] * vector_bases).sum(dim=1)
        velocity = _bounded_vector_norm(
            velocity,
            self.velocity_limit,
            self.geometry_epsilon,
        )
        return velocity.reshape(*leading_shape, self.state_dim)

    def state_lipschitz_upper_bound(self) -> float:
        """Bound for the learned scalar-coefficient path.

        The complete geometric field additionally contains fixed analytic norm,
        centering, and vector-basis maps; no uncapped learned affine XYZ path is
        present.
        """

        return self.velocity_limit * prod(
            layer.max_singular_value
            for layer in (*self.flow_layers, self.output_layer)
        )


def _equivariant_norm_and_unit(vectors: Tensor, epsilon: float) -> tuple[Tensor, Tensor]:
    norm = torch.sqrt(vectors.square().sum(dim=-1) + epsilon)
    return norm.reshape(*norm.shape, 1) if vectors.ndim == 2 else norm, vectors / norm[..., None]


def _bounded_vector_norm(vectors: Tensor, limit: float, epsilon: float) -> Tensor:
    norm = torch.sqrt(vectors.square().sum(dim=-1, keepdim=True) + epsilon)
    bounded = limit * torch.tanh(norm / limit)
    return vectors * bounded / norm


class GenesisMPMPriorMode(str, Enum):
    DISABLED = "disabled"
    DISTILLATION = "distillation"
    DIFFERENTIABLE = "differentiable"


@dataclass(frozen=True)
class PhysicsLossWeights:
    shell: float = 1.0
    equilibrium: float = 0.1
    traction: float = 0.1
    seam: float = 1.0
    contact: float = 10.0
    material_prior: float = 1.0e-3
    genesis_mpm: float = 0.0

    def __post_init__(self) -> None:
        for item in fields(self):
            value = float(getattr(self, item.name))
            if value < 0.0:
                raise ValueError(f"{item.name} loss weight must be non-negative")


@dataclass(frozen=True)
class PhysicsLossResult:
    total: Tensor
    raw_terms: Mapping[str, Tensor]
    weighted_terms: Mapping[str, Tensor]


def physics_constrained_loss(
    *,
    material_latent: Tensor,
    shell_residual: Tensor | None = None,
    equilibrium_residual: Tensor | None = None,
    traction_residual: Tensor | None = None,
    seam_residual: Tensor | None = None,
    contact_gap: Tensor | None = None,
    genesis_mpm_residual: Tensor | None = None,
    genesis_mpm_mode: GenesisMPMPriorMode = GenesisMPMPriorMode.DISABLED,
    mpm_shell_equivalence_passed: bool = False,
    weights: PhysicsLossWeights = PhysicsLossWeights(),
) -> PhysicsLossResult:
    """Compose shell, boundary, contact, and optional Genesis MPM priors.

    ``genesis_mpm_residual`` may retain an online Genesis gradient or contain a
    detached teacher residual. It stays disabled until a thick-shell equivalence
    benchmark passes because stock MPM is not a sewn anisotropic cloth shell.
    """

    if weights.genesis_mpm > 0.0:
        if genesis_mpm_mode is GenesisMPMPriorMode.DISABLED:
            raise ValueError("Genesis MPM weight requires an enabled prior mode")
        if not mpm_shell_equivalence_passed:
            raise ValueError("Genesis MPM prior requires a passed shell-equivalence gate")
        if genesis_mpm_residual is None:
            raise ValueError("Genesis MPM prior requires genesis_mpm_residual")

    zero = material_latent.new_zeros(())
    raw_terms = {
        "shell": _mean_square(shell_residual, zero),
        "equilibrium": _mean_square(equilibrium_residual, zero),
        "traction": _mean_square(traction_residual, zero),
        "seam": _mean_square(seam_residual, zero),
        "contact": (
            F.relu(-contact_gap).square().mean() if contact_gap is not None else zero
        ),
        "material_prior": material_latent.square().mean(),
        "genesis_mpm": _mean_square(genesis_mpm_residual, zero),
    }
    weighted_terms = {
        "shell": weights.shell * raw_terms["shell"],
        "equilibrium": weights.equilibrium * raw_terms["equilibrium"],
        "traction": weights.traction * raw_terms["traction"],
        "seam": weights.seam * raw_terms["seam"],
        "contact": weights.contact * raw_terms["contact"],
        "material_prior": weights.material_prior * raw_terms["material_prior"],
        "genesis_mpm": weights.genesis_mpm * raw_terms["genesis_mpm"],
    }
    total = torch.stack(tuple(weighted_terms.values())).sum()
    return PhysicsLossResult(total=total, raw_terms=raw_terms, weighted_terms=weighted_terms)


def _mean_square(value: Tensor | None, zero: Tensor) -> Tensor:
    return value.square().mean() if value is not None else zero


def _batch_features(
    value: Tensor,
    batch_size: int,
    feature_dim: int,
    reference: Tensor,
    name: str,
) -> Tensor:
    value = value.to(device=reference.device, dtype=reference.dtype)
    if value.ndim == 1 and value.shape[0] == feature_dim:
        return value.unsqueeze(0).expand(batch_size, -1)
    if value.ndim == 2 and value.shape == (1, feature_dim):
        return value.expand(batch_size, -1)
    if value.ndim == 2 and value.shape == (batch_size, feature_dim):
        return value
    raise ValueError(f"{name} must have shape ({feature_dim},) or (N, {feature_dim})")


def _flatten_features(
    value: Tensor,
    leading_shape: torch.Size,
    feature_dim: int,
    reference: Tensor,
    name: str,
) -> Tensor:
    value = value.to(device=reference.device, dtype=reference.dtype)
    expected = (*leading_shape, feature_dim)
    if tuple(value.shape) != expected:
        raise ValueError(f"{name} must have shape {expected}")
    return value.reshape(-1, feature_dim)


def _optional_flat_features(
    value: Tensor | None,
    leading_shape: torch.Size,
    feature_dim: int,
    reference: Tensor,
    name: str,
) -> Tensor:
    if value is None:
        return reference.new_zeros((prod(leading_shape), feature_dim))
    return _flatten_features(value, leading_shape, feature_dim, reference, name)


def _time_features(
    tau: float | Tensor,
    leading_shape: torch.Size,
    reference: Tensor,
) -> Tensor:
    batch_size = prod(leading_shape)
    value = torch.as_tensor(tau, device=reference.device, dtype=reference.dtype)
    if value.ndim == 0:
        return value.reshape(1, 1).expand(batch_size, 1)
    if tuple(value.shape) == tuple(leading_shape):
        return value.reshape(batch_size, 1)
    if tuple(value.shape) == (*leading_shape, 1):
        return value.reshape(batch_size, 1)
    raise ValueError(f"tau must be scalar or have shape {tuple(leading_shape)}")
