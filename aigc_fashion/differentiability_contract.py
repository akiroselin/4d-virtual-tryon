from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from typing import Callable, Iterable, Mapping

import torch
from torch import Tensor


class GradientStage(str, Enum):
    MATERIAL_TO_STRESS = "material_to_stress"
    PHYSICS_ROLLOUT = "physics_rollout"
    CONTACT_COUPLER = "contact_coupler"
    SENSOR_READOUT = "sensor_readout"
    RASTER_RENDER = "raster_render"
    PATH_RENDER = "path_render"
    EVENT_LOCALIZATION = "event_localization"


class ProbeStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    UNSUPPORTED = "unsupported"
    NOT_RUN = "not_run"


@dataclass(frozen=True)
class GradientProbeResult:
    stage: GradientStage
    status: ProbeStatus
    analytic_directional_derivative: float | None = None
    finite_difference_derivative: float | None = None
    relative_error: float | None = None
    reason: str = ""


@dataclass(frozen=True)
class DifferentiabilityManifest:
    probes: Mapping[GradientStage, GradientProbeResult] = field(default_factory=dict)

    def require(self, stages: Iterable[GradientStage]) -> None:
        failures = []
        for stage in stages:
            result = self.probes.get(stage)
            if result is None or result.status is not ProbeStatus.PASSED:
                status = result.status.value if result is not None else ProbeStatus.NOT_RUN.value
                failures.append(f"{stage.value}={status}")
        if failures:
            raise RuntimeError("incomplete differentiable path: " + ", ".join(failures))

    @property
    def passed_stages(self) -> frozenset[GradientStage]:
        return frozenset(
            stage
            for stage, result in self.probes.items()
            if result.status is ProbeStatus.PASSED
        )


def probe_directional_gradient(
    stage: GradientStage,
    loss_function: Callable[[Tensor], Tensor],
    point: Tensor,
    *,
    direction: Tensor | None = None,
    epsilon: float = 1.0e-4,
    relative_tolerance: float = 5.0e-2,
) -> GradientProbeResult:
    """Compare autograd and central differences for one replayable stage."""

    if epsilon <= 0.0 or relative_tolerance < 0.0:
        raise ValueError("epsilon must be positive and tolerance non-negative")
    variable = point.detach().clone().requires_grad_(True)
    if direction is None:
        direction = torch.randn_like(variable)
    direction = direction.detach().to(device=variable.device, dtype=variable.dtype)
    norm = torch.linalg.vector_norm(direction)
    if norm == 0.0:
        raise ValueError("probe direction must be non-zero")
    direction = direction / norm

    try:
        loss = loss_function(variable)
        if loss.ndim != 0:
            raise ValueError("loss_function must return a scalar")
        gradient = torch.autograd.grad(loss, variable, allow_unused=True)[0]
        if gradient is None:
            return GradientProbeResult(stage, ProbeStatus.FAILED, reason="autograd path is disconnected")
        analytic = float((gradient * direction).sum().detach())
        with torch.no_grad():
            upper = float(loss_function(variable.detach() + epsilon * direction))
            lower = float(loss_function(variable.detach() - epsilon * direction))
        finite_difference = (upper - lower) / (2.0 * epsilon)
    except (RuntimeError, ValueError) as error:
        return GradientProbeResult(stage, ProbeStatus.FAILED, reason=str(error))

    if not isfinite(analytic) or not isfinite(finite_difference):
        return GradientProbeResult(
            stage,
            ProbeStatus.FAILED,
            analytic,
            finite_difference,
            reason="non-finite derivative",
        )
    denominator = max(abs(analytic), abs(finite_difference), 1.0e-8)
    relative_error = abs(analytic - finite_difference) / denominator
    status = ProbeStatus.PASSED if relative_error <= relative_tolerance else ProbeStatus.FAILED
    reason = "" if status is ProbeStatus.PASSED else "autograd and finite differences disagree"
    return GradientProbeResult(
        stage=stage,
        status=status,
        analytic_directional_derivative=analytic,
        finite_difference_derivative=finite_difference,
        relative_error=relative_error,
        reason=reason,
    )
