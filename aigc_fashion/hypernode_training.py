from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .hypernetwork_kirchhoff_love import (
    CONSTITUTIVE_BRANCHES,
    HyperNetworkKirchhoffLoveShell,
    KirchhoffLoveShellGeometry,
    LocalNODEParameters,
    kirchhoff_love_dynamics_loss,
)
from .structure_group_4d import (
    deterministic_training_group,
    hypernode_structure_group_loss,
)


@dataclass(frozen=True)
class ConstitutiveCurveBatch:
    """Full-field material samples for one known material latent."""

    material_coordinates: Tensor
    panel_features: Tensor
    material_latent: Tensor
    strains: Tensor
    target_resultants: Tensor
    sample_weights: Tensor | None = None

    def validate(self, model: HyperNetworkKirchhoffLoveShell) -> None:
        sample_count = len(self.material_coordinates)
        panel_count = model.hypernetwork.panel_count
        latent_dim = model.hypernetwork.latent_dim
        if self.material_coordinates.shape != (sample_count, 2):
            raise ValueError("material_coordinates must have shape (N, 2)")
        if self.panel_features.shape != (sample_count, panel_count):
            raise ValueError(f"panel_features must have shape (N, {panel_count})")
        if self.strains.shape != (sample_count, CONSTITUTIVE_BRANCHES):
            raise ValueError(
                f"strains must have shape (N, {CONSTITUTIVE_BRANCHES})"
            )
        if self.target_resultants.shape != self.strains.shape:
            raise ValueError("target_resultants must match strains")
        if self.material_latent.shape not in {
            (latent_dim,),
            (1, latent_dim),
            (sample_count, latent_dim),
        }:
            raise ValueError(
                f"material_latent must have shape ({latent_dim},), "
                f"(1, {latent_dim}), or (N, {latent_dim})"
            )
        if self.sample_weights is not None and self.sample_weights.shape not in {
            (sample_count,),
            self.strains.shape,
        }:
            raise ValueError("sample_weights must have shape (N,) or (N, 5)")


@dataclass(frozen=True)
class HyperNODETrajectoryBatch:
    geometry: KirchhoffLoveShellGeometry
    previous_positions: Tensor
    current_positions: Tensor
    next_positions: Tensor
    material_latent: Tensor
    dt: float
    contact_closest: Tensor | None = None
    contact_normals: Tensor | None = None
    prescribed_free_boundary_traction: Tensor | None = None
    pinned_vertices: Tensor | None = None
    areal_density_kg_m2: float = 0.18
    maximum_tensile_strain: float | None = None
    maximum_compressive_strain: float | None = None


@dataclass(frozen=True)
class HyperNODETrainingConfig:
    homogeneous_steps: int = 200
    spatial_initialization_steps: int = 200
    dynamics_steps: int = 400
    constitutive_learning_rate: float = 5.0e-4
    dynamics_learning_rate: float = 2.0e-4
    weight_decay: float = 1.0e-6
    gradient_clip: float = 1.0
    spatial_parameter_weight: float = 0.02
    zero_response_weight: float = 0.1
    structure_group_weight: float = 0.1
    seed: int = 73

    def __post_init__(self) -> None:
        steps = (
            self.homogeneous_steps,
            self.spatial_initialization_steps,
            self.dynamics_steps,
        )
        if any(value < 0 for value in steps):
            raise ValueError("training step counts must be non-negative")
        positive = (
            self.constitutive_learning_rate,
            self.dynamics_learning_rate,
            self.gradient_clip,
        )
        if any(value <= 0.0 for value in positive):
            raise ValueError("learning rates and gradient_clip must be positive")
        if min(
            self.weight_decay,
            self.spatial_parameter_weight,
            self.zero_response_weight,
            self.structure_group_weight,
        ) < 0.0:
            raise ValueError("regularization weights must be non-negative")


@dataclass(frozen=True)
class HyperNODETrainingRecord:
    phase: str
    step: int
    total: float
    constitutive: float = 0.0
    spatial: float = 0.0
    zero_response: float = 0.0
    equilibrium: float = 0.0
    traction: float = 0.0
    contact: float = 0.0
    structure_group: float = 0.0


class HyperNODETrainer:
    """Staged material trainer; physical-time fields are intentionally absent."""

    def __init__(
        self,
        model: HyperNetworkKirchhoffLoveShell,
        config: HyperNODETrainingConfig | None = None,
    ) -> None:
        self.model = model
        self.config = config or HyperNODETrainingConfig()
        self.history: list[HyperNODETrainingRecord] = []

    def fit_constitutive_curves(
        self,
        batches: Sequence[ConstitutiveCurveBatch],
    ) -> list[HyperNODETrainingRecord]:
        if not batches:
            raise ValueError("at least one constitutive curve batch is required")
        for batch in batches:
            batch.validate(self.model)
        torch.manual_seed(self.config.seed)
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.constitutive_learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self._fit_curve_phase(
            "homogeneous_node_pretraining",
            batches,
            self.config.homogeneous_steps,
            optimizer,
            homogenize_coordinates=True,
        )
        self._fit_curve_phase(
            "hypernetwork_spatial_initialization",
            batches,
            self.config.spatial_initialization_steps,
            optimizer,
            homogenize_coordinates=False,
        )
        return list(self.history)

    def fit_dynamics(
        self,
        batches: Sequence[HyperNODETrajectoryBatch],
        *,
        extra_parameters: Iterable[nn.Parameter] = (),
    ) -> list[HyperNODETrainingRecord]:
        if not batches:
            raise ValueError("at least one trajectory batch is required")
        parameters = list(self.model.parameters()) + list(extra_parameters)
        optimizer = torch.optim.AdamW(
            parameters,
            lr=self.config.dynamics_learning_rate,
            weight_decay=self.config.weight_decay,
        )
        for step in range(self.config.dynamics_steps):
            batch = batches[step % len(batches)]
            optimizer.zero_grad(set_to_none=True)
            current = batch.current_positions
            if not current.requires_grad:
                current = current.detach().requires_grad_(True)
            losses = kirchhoff_love_dynamics_loss(
                self.model,
                batch.geometry,
                batch.previous_positions,
                current,
                batch.next_positions,
                batch.material_latent,
                dt=batch.dt,
                contact_closest=batch.contact_closest,
                contact_normals=batch.contact_normals,
                prescribed_free_boundary_traction=(
                    batch.prescribed_free_boundary_traction
                ),
                pinned_vertices=batch.pinned_vertices,
                areal_density_kg_m2=batch.areal_density_kg_m2,
                maximum_tensile_strain=batch.maximum_tensile_strain,
                maximum_compressive_strain=batch.maximum_compressive_strain,
            )
            group = deterministic_training_group(step, current)
            pose = group.world_pose
            transformed_current = pose.transform_points(current)
            if not transformed_current.requires_grad:
                transformed_current = transformed_current.detach().requires_grad_(True)
            gravity = pose.transform_vectors(
                current.new_tensor((0.0, -9.81, 0.0))
            )
            transformed_losses = kirchhoff_love_dynamics_loss(
                self.model,
                batch.geometry,
                pose.transform_points(batch.previous_positions),
                transformed_current,
                pose.transform_points(batch.next_positions),
                batch.material_latent,
                dt=batch.dt,
                contact_closest=(
                    None
                    if batch.contact_closest is None
                    else pose.transform_points(batch.contact_closest)
                ),
                contact_normals=(
                    None
                    if batch.contact_normals is None
                    else pose.transform_vectors(batch.contact_normals)
                ),
                prescribed_free_boundary_traction=(
                    None
                    if batch.prescribed_free_boundary_traction is None
                    else pose.transform_vectors(
                        batch.prescribed_free_boundary_traction
                    )
                ),
                pinned_vertices=batch.pinned_vertices,
                areal_density_kg_m2=batch.areal_density_kg_m2,
                gravity_m_s2=tuple(float(value) for value in gravity.detach()),
                maximum_tensile_strain=batch.maximum_tensile_strain,
                maximum_compressive_strain=batch.maximum_compressive_strain,
            )
            group_loss = hypernode_structure_group_loss(
                losses.output,
                transformed_losses.output,
                current,
                world_rotation=pose.real,
                original_acceleration=losses.acceleration,
                transformed_acceleration=transformed_losses.acceleration,
            )
            total = (
                0.5 * (losses.total + transformed_losses.total)
                + self.config.structure_group_weight * group_loss.total
            )
            total.backward()
            torch.nn.utils.clip_grad_norm_(parameters, self.config.gradient_clip)
            optimizer.step()
            self.history.append(
                HyperNODETrainingRecord(
                    phase="dynamic_inverse_identification",
                    step=step + 1,
                    total=float(total.detach()),
                    constitutive=float(losses.shell_energy.detach()),
                    equilibrium=float(losses.equilibrium.detach()),
                    traction=float(losses.traction.detach()),
                    contact=float(losses.contact.detach()),
                    structure_group=float(group_loss.total.detach()),
                )
            )
        return list(self.history)

    def _fit_curve_phase(
        self,
        phase: str,
        batches: Sequence[ConstitutiveCurveBatch],
        steps: int,
        optimizer: torch.optim.Optimizer,
        *,
        homogenize_coordinates: bool,
    ) -> None:
        branch_ids = tuple(range(CONSTITUTIVE_BRANCHES))
        for step in range(steps):
            batch = batches[step % len(batches)]
            coordinates = (
                torch.zeros_like(batch.material_coordinates)
                if homogenize_coordinates
                else batch.material_coordinates
            )
            optimizer.zero_grad(set_to_none=True)
            parameters = self.model.hypernetwork(
                coordinates,
                batch.panel_features,
                batch.material_latent,
            )
            prediction = self.model.constitutive_node.response(
                batch.strains,
                parameters,
                branch_ids,
            )
            scale = batch.target_resultants.detach().abs().amax(dim=0).clamp_min(1.0e-3)
            residual = (prediction - batch.target_resultants) / scale
            weights = batch.sample_weights
            if weights is not None:
                weights = weights.to(residual)
                if weights.ndim == 1:
                    weights = weights[:, None]
                residual = residual * torch.sqrt(weights.clamp_min(0.0))
            constitutive = F.smooth_l1_loss(residual, torch.zeros_like(residual))
            zero_response = self.model.constitutive_node.response(
                torch.zeros_like(batch.strains),
                parameters,
                branch_ids,
            ).square().mean()
            spatial = _local_parameter_variance(parameters)
            total = (
                constitutive
                + self.config.zero_response_weight * zero_response
                + self.config.spatial_parameter_weight * spatial
            )
            total.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.gradient_clip,
            )
            optimizer.step()
            self.history.append(
                HyperNODETrainingRecord(
                    phase=phase,
                    step=step + 1,
                    total=float(total.detach()),
                    constitutive=float(constitutive.detach()),
                    spatial=float(spatial.detach()),
                    zero_response=float(zero_response.detach()),
                )
            )


def save_hypernode_checkpoint(
    path: str | Path,
    model: HyperNetworkKirchhoffLoveShell,
    config: HyperNODETrainingConfig,
    history: Sequence[HyperNODETrainingRecord],
    *,
    metadata: dict[str, object] | None = None,
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "aigc_fashion.hypernode.v1",
            "panel_count": model.hypernetwork.panel_count,
            "latent_dim": model.hypernetwork.latent_dim,
            "model_state_dict": model.state_dict(),
            "training_config": asdict(config),
            "history": [asdict(record) for record in history],
            "metadata": dict(metadata or {}),
        },
        destination,
    )
    return destination


def load_hypernode_checkpoint(
    path: str | Path,
    model: HyperNetworkKirchhoffLoveShell,
    *,
    strict: bool = True,
) -> dict[str, object]:
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    if checkpoint.get("format") != "aigc_fashion.hypernode.v1":
        raise ValueError("unsupported HyperNODE checkpoint format")
    if int(checkpoint["panel_count"]) != model.hypernetwork.panel_count:
        raise ValueError("checkpoint panel_count does not match the model")
    if int(checkpoint["latent_dim"]) != model.hypernetwork.latent_dim:
        raise ValueError("checkpoint latent_dim does not match the model")
    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    return checkpoint


def _local_parameter_variance(parameters: LocalNODEParameters) -> Tensor:
    values = (
        parameters.first_weight,
        parameters.first_bias,
        parameters.hidden_weight,
        parameters.hidden_bias,
        parameters.output_weight,
        parameters.output_bias,
        torch.log(parameters.stiffness),
    )
    terms = [value.var(dim=0, unbiased=False).mean() for value in values]
    return torch.stack(terms).mean()
