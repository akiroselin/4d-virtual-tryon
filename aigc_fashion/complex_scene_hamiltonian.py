from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import torch
from torch import Tensor

from .geometric_phase_graph import IntrinsicClothPhaseState


@dataclass(frozen=True)
class KahlerClothPhase:
    """Mass-scaled complex chart for a cloth cotangent phase.

    For angular frequency ``omega``, ``Q=sqrt(M omega) (q-q_ref)`` and
    ``P=p/sqrt(M omega)`` form a canonical pair represented by
    ``z=(Q+iP)/sqrt(2)``. A real spatial rotation acts on both vector parts,
    while global translation cancels against the reference embedding.
    """

    coordinates: Tensor
    lumped_mass: Tensor
    reference_angular_frequency: float
    reference_positions: Tensor
    material_coordinates: Tensor
    time: float

    def __post_init__(self) -> None:
        if not torch.is_complex(self.coordinates):
            raise ValueError("Kahler phase coordinates must be complex")
        if self.coordinates.ndim != 2 or self.coordinates.shape[1] != 3:
            raise ValueError("Kahler cloth coordinates must have shape (V,3)")
        vertex_count = len(self.coordinates)
        if self.lumped_mass.shape != (vertex_count,) or torch.any(self.lumped_mass <= 0.0):
            raise ValueError("Kahler phase mass must be positive with shape (V,)")
        if self.reference_positions.shape != (vertex_count, 3):
            raise ValueError("reference positions must have shape (V,3)")
        if self.material_coordinates.shape != (vertex_count, 2):
            raise ValueError("material coordinates must have shape (V,2)")
        if self.reference_angular_frequency <= 0.0:
            raise ValueError("reference angular frequency must be positive")

    @classmethod
    def from_canonical_phase(
        cls,
        state: IntrinsicClothPhaseState,
        reference_positions: Tensor,
        *,
        reference_angular_frequency: float,
    ) -> KahlerClothPhase:
        if reference_positions.shape != state.positions.shape:
            raise ValueError("reference positions must match cloth positions")
        if reference_angular_frequency <= 0.0:
            raise ValueError("reference angular frequency must be positive")
        mass = state.lumped_mass.to(state.positions)
        scale = torch.sqrt(mass * reference_angular_frequency)[:, None]
        canonical_position = scale * (
            state.positions - reference_positions.to(state.positions)
        )
        canonical_momentum = state.momenta / scale
        coordinates = torch.complex(canonical_position, canonical_momentum) / sqrt(2.0)
        return cls(
            coordinates=coordinates,
            lumped_mass=mass,
            reference_angular_frequency=float(reference_angular_frequency),
            reference_positions=reference_positions.to(state.positions),
            material_coordinates=state.material_coordinates.to(state.positions),
            time=state.time,
        )

    def to_canonical_phase(self) -> IntrinsicClothPhaseState:
        scale = torch.sqrt(
            self.lumped_mass * self.reference_angular_frequency
        )[:, None]
        displacement = sqrt(2.0) * self.coordinates.real / scale
        momenta = sqrt(2.0) * self.coordinates.imag * scale
        return IntrinsicClothPhaseState(
            material_coordinates=self.material_coordinates,
            positions=self.reference_positions + displacement,
            momenta=momenta,
            lumped_mass=self.lumped_mass,
            time=self.time,
        )


def wirtinger_hamiltonian_vector_field(hamiltonian: Tensor, coordinates: Tensor) -> Tensor:
    """Return ``z_dot=-i partial H / partial z_bar`` for real ``H(z,z_bar)``.

    PyTorch returns twice the conjugate Wirtinger derivative for a real scalar
    loss, hence the factor one half below.
    """

    if hamiltonian.ndim != 0 or torch.is_complex(hamiltonian):
        raise ValueError("physical Hamiltonian must be a real scalar")
    if not torch.is_complex(coordinates):
        raise ValueError("Wirtinger coordinates must be complex")
    gradient = torch.autograd.grad(
        hamiltonian,
        coordinates,
        create_graph=True,
        retain_graph=True,
    )[0]
    return -0.5j * gradient


def hermitian_power_pairing(effort: Tensor, flow: Tensor) -> Tensor:
    """Real port power ``Re <effort, flow>`` in a local complex coframe."""

    if effort.shape != flow.shape:
        raise ValueError("port effort and flow must have matching shapes")
    if not torch.is_complex(effort) or not torch.is_complex(flow):
        raise ValueError("port effort and flow must be complex")
    return (effort.conj() * flow).sum().real


def port_hamiltonian_vector_field(
    hamiltonian: Tensor,
    coordinates: Tensor,
    *,
    dissipation: Tensor | None = None,
    external_flow: Tensor | None = None,
) -> Tensor:
    """Complex Hamiltonian flow with positive-semidefinite dissipation ports.

    Contact impulses and other discontinuous events remain explicit reset maps
    and should be supplied separately.
    """

    conservative = wirtinger_hamiltonian_vector_field(hamiltonian, coordinates)
    gradient = 2.0j * conservative
    flow = conservative
    if dissipation is not None:
        coefficient = dissipation.to(coordinates.real)
        if torch.any(coefficient < 0.0):
            raise ValueError("port-Hamiltonian dissipation must be non-negative")
        flow = flow - 0.5 * coefficient * gradient
    if external_flow is not None:
        if external_flow.shape != coordinates.shape or not torch.is_complex(external_flow):
            raise ValueError("external complex flow must match phase coordinates")
        flow = flow + external_flow
    return flow


@dataclass(frozen=True)
class ComplexSceneHamiltonian:
    """Real energy ledger over a hierarchy-aware complex 4D scene graph."""

    total: Tensor
    node_total: Tensor
    relation_total: Tensor
    constraint_total: Tensor
    hierarchy_energy: dict[int, Tensor]
    node_energies: Tensor
    relation_energies: Tensor


def _real_energy_field(value: Tensor, *, name: str) -> Tensor:
    if torch.is_complex(value):
        imaginary_error = (
            value.imag.abs().max() if value.numel() else value.real.new_zeros(())
        )
        if float(imaginary_error.detach()) > 1.0e-7:
            raise ValueError(f"{name} must be real-valued")
        return value.real
    return value


def assemble_complex_scene_hamiltonian(
    node_energies: Tensor,
    relation_energies: Tensor,
    node_hierarchy_levels: Tensor,
    relation_index: Tensor,
    *,
    constraint_energy: Tensor | None = None,
) -> ComplexSceneHamiltonian:
    """Aggregate typed node/relation energies without flattening their charts.

    A relation is charged to the deeper endpoint hierarchy level. Energy
    producers may use cloth Kahler coordinates, ``T*SE(3)`` charts, or SO(2)
    contact fibres; only their real scalar energies cross this API.
    """

    nodes = _real_energy_field(node_energies, name="node energies")
    relations = _real_energy_field(relation_energies, name="relation energies")
    if nodes.ndim != 1 or relations.ndim != 1:
        raise ValueError("scene node and relation energies must be one-dimensional")
    if node_hierarchy_levels.shape != nodes.shape:
        raise ValueError("one hierarchy level is required per scene node")
    if relation_index.shape != (2, len(relations)):
        raise ValueError("relation index must have shape (2,E)")
    index = relation_index.to(device=nodes.device, dtype=torch.long)
    if index.numel() and (torch.any(index < 0) or torch.any(index >= len(nodes))):
        raise ValueError("scene relation index is out of range")
    levels = node_hierarchy_levels.to(device=nodes.device, dtype=torch.long)
    if torch.any(levels < 0):
        raise ValueError("scene hierarchy levels must be non-negative")
    constraint = (
        nodes.new_zeros(())
        if constraint_energy is None
        else _real_energy_field(constraint_energy, name="constraint energy")
    )
    if constraint.ndim != 0:
        raise ValueError("constraint energy must be scalar")

    hierarchy: dict[int, Tensor] = {}
    relation_levels = (
        torch.maximum(levels[index[0]], levels[index[1]])
        if len(relations)
        else levels.new_empty((0,))
    )
    for level in torch.unique(levels).tolist():
        hierarchy[int(level)] = nodes[levels == level].sum()
    for level in torch.unique(relation_levels).tolist():
        key = int(level)
        hierarchy[key] = hierarchy.get(key, nodes.new_zeros(())) + relations[
            relation_levels == level
        ].sum()
    node_total = nodes.sum()
    relation_total = relations.sum()
    total = node_total + relation_total + constraint
    return ComplexSceneHamiltonian(
        total=total,
        node_total=node_total,
        relation_total=relation_total,
        constraint_total=constraint,
        hierarchy_energy=hierarchy,
        node_energies=nodes,
        relation_energies=relations,
    )
