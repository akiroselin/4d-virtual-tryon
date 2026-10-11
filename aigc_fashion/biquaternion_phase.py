from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .geometric_phase_graph import IntrinsicClothPhaseState


@dataclass(frozen=True)
class BiquaternionVertexPhase:
    """Complex-quaternion chart of extended cloth phase space.

    ``real=(t, dq_x, dq_y, dq_z)`` and
    ``imag=(p_t, p_x, p_y, p_z)``.  This is a general biquaternion, not a unit
    rotation quaternion.  The eighth real coordinate is the momentum conjugate
    to time; on the physical constraint surface its sum is ``-H``.
    """

    real: Tensor
    imag: Tensor
    lumped_mass: Tensor
    shared_time: float

    def __post_init__(self) -> None:
        if self.real.ndim != 2 or self.real.shape[1] != 4:
            raise ValueError("biquaternion real part must have shape (V,4)")
        if self.imag.shape != self.real.shape:
            raise ValueError("biquaternion imaginary part must match its real part")
        if self.lumped_mass.shape != (len(self.real),) or torch.any(self.lumped_mass <= 0.0):
            raise ValueError("biquaternion phase mass must be positive with shape (V,)")
        expected = self.real.new_full((len(self.real),), float(self.shared_time))
        if not torch.allclose(self.real[:, 0], expected, atol=1.0e-7, rtol=0.0):
            raise ValueError("all vertex biquaternions must share one physical time")

    @classmethod
    def from_canonical_phase(
        cls,
        state: IntrinsicClothPhaseState,
        reference_positions: Tensor,
        *,
        hamiltonian: Tensor,
        energy_partition: Tensor | None = None,
    ) -> BiquaternionVertexPhase:
        if reference_positions.shape != state.positions.shape:
            raise ValueError("reference positions must match canonical phase positions")
        if hamiltonian.ndim != 0:
            raise ValueError("Hamiltonian must be scalar")
        vertex_count = len(state.positions)
        if energy_partition is None:
            weights = state.lumped_mass / state.lumped_mass.sum()
        else:
            weights = energy_partition.to(state.positions)
            if weights.shape != (vertex_count,) or torch.any(weights < 0.0):
                raise ValueError("energy partition must be non-negative with shape (V,)")
            weights = weights / weights.sum().clamp_min(1.0e-12)
        displacement = state.positions - reference_positions.to(state.positions)
        time = state.positions.new_full((vertex_count, 1), state.time)
        conjugate_energy = -(weights * hamiltonian).reshape(-1, 1)
        return cls(
            real=torch.cat((time, displacement), dim=-1),
            imag=torch.cat((conjugate_energy, state.momenta), dim=-1),
            lumped_mass=state.lumped_mass,
            shared_time=state.time,
        )

    @property
    def displacement(self) -> Tensor:
        return self.real[:, 1:]

    @property
    def momenta(self) -> Tensor:
        return self.imag[:, 1:]

    @property
    def time_momenta(self) -> Tensor:
        return self.imag[:, 0]

    def extended_hamiltonian_constraint(self, hamiltonian: Tensor) -> Tensor:
        """Return ``sum(p_t) + H``, zero on the physical extended phase shell."""

        if hamiltonian.ndim != 0:
            raise ValueError("Hamiltonian must be scalar")
        return self.time_momenta.sum() + hamiltonian


@dataclass(frozen=True)
class ComplexKirchhoffLoveStrain:
    """Trace plus charge-2 complex deviator of a symmetric 2D tensor."""

    trace: Tensor
    deviator: Tensor

    def __post_init__(self) -> None:
        if self.trace.ndim != 1 or self.deviator.shape != self.trace.shape:
            raise ValueError("complex KL strain fields must have shape (F,)")
        if not torch.is_complex(self.deviator):
            raise ValueError("KL deviator must be complex")


def membrane_strain_to_complex(membrane_strains: Tensor) -> ComplexKirchhoffLoveStrain:
    """Convert ``(e11,e22,2e12,area)`` to an SO(2) charge-2 field."""

    if membrane_strains.ndim != 2 or membrane_strains.shape[1] < 3:
        raise ValueError("membrane strains must have shape (F,>=3)")
    e11 = membrane_strains[:, 0]
    e22 = membrane_strains[:, 1]
    engineering_shear = membrane_strains[:, 2]
    return ComplexKirchhoffLoveStrain(
        trace=e11 + e22,
        deviator=torch.complex(e11 - e22, engineering_shear),
    )


def complex_kirchhoff_love_quadratic_energy(
    strain: ComplexKirchhoffLoveStrain,
    area: Tensor,
    *,
    bulk_modulus: Tensor | float,
    shear_modulus: Tensor | float,
) -> Tensor:
    """Real scalar energy from complex KL invariants.

    This is an isotropic/Hermitian audit energy.  The production Hyper-NODE
    remains authoritative for nonlinear orthotropic constitutive response.
    """

    if area.shape != strain.trace.shape:
        raise ValueError("KL quadrature area must match strain faces")
    bulk = torch.as_tensor(bulk_modulus, device=area.device, dtype=area.dtype)
    shear = torch.as_tensor(shear_modulus, device=area.device, dtype=area.dtype)
    density = 0.5 * bulk * strain.trace.square() + 0.5 * shear * strain.deviator.abs().square()
    return area * density


def hamilton_product(left: Tensor, right: Tensor) -> Tensor:
    """Hamilton product for arrays of real quaternions ``(...,4)``."""

    if left.shape[-1] != 4 or right.shape[-1] != 4:
        raise ValueError("Hamilton product operands must end in four components")
    aw, av = left[..., :1], left[..., 1:]
    bw, bv = right[..., :1], right[..., 1:]
    scalar = aw * bw - (av * bv).sum(dim=-1, keepdim=True)
    vector = aw * bv + bw * av + torch.linalg.cross(av, bv)
    return torch.cat((scalar, vector), dim=-1)


def biquaternion_product(
    left: BiquaternionVertexPhase,
    right: BiquaternionVertexPhase,
) -> tuple[Tensor, Tensor]:
    """Return real/imaginary parts of ``(a+i b)(c+i d)``."""

    if left.real.shape != right.real.shape:
        raise ValueError("biquaternion products require matching shapes")
    real = hamilton_product(left.real, right.real) - hamilton_product(left.imag, right.imag)
    imag = hamilton_product(left.real, right.imag) + hamilton_product(left.imag, right.real)
    return real, imag
