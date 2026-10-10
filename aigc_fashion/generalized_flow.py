from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum

import torch
from torch import Tensor


class GeneralizedFlowRole(str, Enum):
    CONSERVATIVE = "conservative"
    DISSIPATIVE = "dissipative"
    CONTROL = "control"
    CONSTRAINT = "constraint"
    LEARNED_RESIDUAL = "learned_residual"


class GeneralizedFlowRepresentation(str, Enum):
    REAL_SCALAR = "real_scalar"
    E3_VECTOR = "e3_vector"
    SO2_COMPLEX = "so2_complex"
    SE3_WRENCH_TWIST = "se3_wrench_twist"


@dataclass(frozen=True)
class GeneralizedFlowPort:
    """Typed effort-flow pair on one directed scene-graph relation.

    Positive power enters ``target`` from ``source``. Internal conservative
    storage, dissipative response, controls, and constraints share this power
    interface without sharing an inappropriate coordinate representation.
    """

    relation_id: str
    source: str
    target: str
    role: GeneralizedFlowRole
    representation: GeneralizedFlowRepresentation
    effort: Tensor
    flow: Tensor
    time: float

    def __post_init__(self) -> None:
        if not self.relation_id or not self.source or not self.target:
            raise ValueError("generalized-flow relation and endpoint names are required")
        if self.effort.shape != self.flow.shape:
            raise ValueError("generalized-flow effort and flow must have matching shapes")
        if self.representation is GeneralizedFlowRepresentation.SO2_COMPLEX:
            if not torch.is_complex(self.effort) or not torch.is_complex(self.flow):
                raise ValueError("SO(2) effort-flow fields must be complex")
        elif torch.is_complex(self.effort) or torch.is_complex(self.flow):
            raise ValueError("only SO(2) effort-flow fields may be complex")
        if self.representation is GeneralizedFlowRepresentation.E3_VECTOR:
            if self.effort.ndim == 0 or self.effort.shape[-1] != 3:
                raise ValueError("E(3) effort-flow fields must end in three components")
        if self.representation is GeneralizedFlowRepresentation.SE3_WRENCH_TWIST:
            if self.effort.ndim == 0 or self.effort.shape[-1] != 6:
                raise ValueError("SE(3) effort-flow fields must end in six components")

    def power(self) -> Tensor:
        """Return invariant real power delivered to the target."""

        if self.representation is GeneralizedFlowRepresentation.SO2_COMPLEX:
            return (self.effort.conj() * self.flow).sum().real
        return (self.effort * self.flow).sum()


@dataclass(frozen=True)
class GeneralizedImpulseTransfer:
    """Measure-valued generalized flow for a momentum reset or projection."""

    relation_id: str
    source: str
    target: str
    role: GeneralizedFlowRole
    representation: GeneralizedFlowRepresentation
    impulse: Tensor
    work: Tensor
    time: float

    def __post_init__(self) -> None:
        if not self.relation_id or not self.source or not self.target:
            raise ValueError("generalized impulse relation and endpoints are required")
        if self.work.ndim != 0 or torch.is_complex(self.work):
            raise ValueError("generalized impulse work must be a real scalar")
        if self.representation is GeneralizedFlowRepresentation.E3_VECTOR:
            if self.impulse.ndim == 0 or self.impulse.shape[-1] != 3:
                raise ValueError("E(3) impulse fields must end in three components")
        if self.representation is GeneralizedFlowRepresentation.SO2_COMPLEX:
            if not torch.is_complex(self.impulse):
                raise ValueError("SO(2) impulses must be complex tangent fields")
        elif torch.is_complex(self.impulse):
            raise ValueError("only SO(2) impulses may be complex")
        if self.representation is GeneralizedFlowRepresentation.SE3_WRENCH_TWIST:
            if self.impulse.ndim == 0 or self.impulse.shape[-1] != 6:
                raise ValueError("SE(3) impulses must end in six components")


@dataclass(frozen=True)
class GeneralizedFlowAudit:
    continuous_work_j: dict[str, float]
    impulse_work_j: dict[str, float]
    sample_count: dict[str, int]
    representation_count: dict[str, int]
    net_work_j: float
    maximum_positive_dissipative_power_w: float
    dissipative_sign_violations: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class GeneralizedFlowLedger:
    """Detached rollout audit for continuous and measure-valued energy flow."""

    def __init__(self, *, dissipative_tolerance_w: float = 1.0e-7) -> None:
        if dissipative_tolerance_w < 0.0:
            raise ValueError("dissipative power tolerance must be non-negative")
        self.dissipative_tolerance_w = float(dissipative_tolerance_w)
        self.reset()

    def reset(self) -> None:
        self._continuous_work = {role.value: 0.0 for role in GeneralizedFlowRole}
        self._impulse_work = {role.value: 0.0 for role in GeneralizedFlowRole}
        self._sample_count = {role.value: 0 for role in GeneralizedFlowRole}
        self._representation_count = {
            representation.value: 0 for representation in GeneralizedFlowRepresentation
        }
        self._maximum_positive_dissipative_power = 0.0
        self._dissipative_sign_violations = 0

    def checkpoint(self) -> dict[str, object]:
        """Capture detached audit state before a speculative event step."""

        return {
            "continuous_work": dict(self._continuous_work),
            "impulse_work": dict(self._impulse_work),
            "sample_count": dict(self._sample_count),
            "representation_count": dict(self._representation_count),
            "maximum_positive_dissipative_power": (
                self._maximum_positive_dissipative_power
            ),
            "dissipative_sign_violations": self._dissipative_sign_violations,
        }

    def restore(self, checkpoint: dict[str, object]) -> None:
        """Discard generalized flows produced by a rejected trial step."""

        required = {
            "continuous_work",
            "impulse_work",
            "sample_count",
            "representation_count",
            "maximum_positive_dissipative_power",
            "dissipative_sign_violations",
        }
        if set(checkpoint) != required:
            raise ValueError("invalid generalized-flow ledger checkpoint")
        self._continuous_work = dict(checkpoint["continuous_work"])
        self._impulse_work = dict(checkpoint["impulse_work"])
        self._sample_count = dict(checkpoint["sample_count"])
        self._representation_count = dict(checkpoint["representation_count"])
        self._maximum_positive_dissipative_power = float(
            checkpoint["maximum_positive_dissipative_power"]
        )
        self._dissipative_sign_violations = int(
            checkpoint["dissipative_sign_violations"]
        )

    def record_port(self, port: GeneralizedFlowPort, *, duration_s: float) -> Tensor:
        if duration_s <= 0.0:
            raise ValueError("continuous generalized-flow duration must be positive")
        power = port.power()
        value = float(power.detach())
        role = port.role.value
        self._continuous_work[role] += value * float(duration_s)
        self._sample_count[role] += 1
        self._representation_count[port.representation.value] += 1
        if port.role is GeneralizedFlowRole.DISSIPATIVE and value > self.dissipative_tolerance_w:
            self._maximum_positive_dissipative_power = max(
                self._maximum_positive_dissipative_power, value
            )
            self._dissipative_sign_violations += 1
        return power

    def record_impulse(self, transfer: GeneralizedImpulseTransfer) -> Tensor:
        value = float(transfer.work.detach())
        role = transfer.role.value
        self._impulse_work[role] += value
        self._sample_count[role] += 1
        self._representation_count[transfer.representation.value] += 1
        if (
            transfer.role is GeneralizedFlowRole.DISSIPATIVE
            and value > self.dissipative_tolerance_w
        ):
            self._dissipative_sign_violations += 1
        return transfer.work

    def audit(self) -> GeneralizedFlowAudit:
        net = sum(self._continuous_work.values()) + sum(self._impulse_work.values())
        return GeneralizedFlowAudit(
            continuous_work_j=dict(self._continuous_work),
            impulse_work_j=dict(self._impulse_work),
            sample_count=dict(self._sample_count),
            representation_count=dict(self._representation_count),
            net_work_j=net,
            maximum_positive_dissipative_power_w=(
                self._maximum_positive_dissipative_power
            ),
            dissipative_sign_violations=self._dissipative_sign_violations,
        )


def kinetic_energy_from_velocity(velocity: Tensor, lumped_mass: Tensor) -> Tensor:
    if velocity.ndim != 2 or velocity.shape[1] != 3:
        raise ValueError("velocity must have shape (V,3)")
    if lumped_mass.shape != (len(velocity),) or torch.any(lumped_mass <= 0.0):
        raise ValueError("lumped mass must be positive with shape (V,)")
    return 0.5 * (lumped_mass[:, None] * velocity.square()).sum()
