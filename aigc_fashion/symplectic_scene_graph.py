from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class SymplecticSceneNode:
    """One hierarchy node in a canonical cotangent chart.

    ``q`` is a chart coordinate vector for the node and ``p`` is its
    cotangent momentum.  A cloth node currently uses its intrinsic vertex
    embedding as a chart; a future rigid node can use a local se(3) chart.
    """

    node_id: str
    kind: str
    q: Tensor
    p: Tensor
    mass: Tensor
    hierarchy_level: int = 0
    parent_index: int | None = None

    def __post_init__(self) -> None:
        if not self.node_id or not self.kind:
            raise ValueError("symplectic scene nodes need an id and kind")
        if self.q.ndim != 1 or self.p.shape != self.q.shape:
            raise ValueError("node q and p must be matching rank-one vectors")
        if self.mass.shape != self.q.shape or torch.any(self.mass <= 0.0):
            raise ValueError("node mass must match q and remain positive")
        if self.hierarchy_level < 0:
            raise ValueError("hierarchy level must be non-negative")


@dataclass(frozen=True)
class SymplecticSceneGraphState:
    """The 4D scene graph lifted to a finite-dimensional symplectic manifold.

    The graph and physical time are bookkeeping on the base scene.  The
    actual symplectic manifold is the product cotangent bundle ``T*Q`` with
    canonical coordinates ``(q,p)``.  Edges enter the Hamiltonian and learned
    attention, while the canonical two-form is independent of a chart's
    parameterisation.
    """

    nodes: tuple[SymplecticSceneNode, ...]
    edge_index: Tensor
    relation_ids: Tensor
    time: float

    def __post_init__(self) -> None:
        if not self.nodes:
            raise ValueError("symplectic scene graph needs at least one node")
        if self.edge_index.ndim != 2 or self.edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape (2,E)")
        if self.relation_ids.shape != (self.edge_index.shape[1],):
            raise ValueError("relation_ids must match the edge count")
        if self.relation_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("relation_ids must contain integers")
        if self.edge_index.numel() and (
            self.edge_index.min() < 0 or self.edge_index.max() >= len(self.nodes)
        ):
            raise ValueError("scene edge index is outside the node array")

    @property
    def configuration_dimension(self) -> int:
        return sum(int(node.q.numel()) for node in self.nodes)

    @property
    def phase_dimension(self) -> int:
        return 2 * self.configuration_dimension

    def canonical_coordinates(self) -> tuple[Tensor, Tensor, Tensor]:
        q = torch.cat([node.q for node in self.nodes])
        p = torch.cat([node.p for node in self.nodes])
        mass = torch.cat([node.mass for node in self.nodes])
        return q, p, mass

    def canonical_vector(self) -> Tensor:
        q, p, _mass = self.canonical_coordinates()
        return torch.cat((q, p))

    def symplectic_form(self) -> Tensor:
        """Return the canonical matrix for ``omega = sum_i dq_i ^ dp_i``."""

        dimension = self.configuration_dimension
        identity = torch.eye(
            dimension,
            dtype=self.nodes[0].q.dtype,
            device=self.nodes[0].q.device,
        )
        zero = torch.zeros_like(identity)
        return torch.cat(
            (torch.cat((zero, identity), dim=1),
             torch.cat((-identity, zero), dim=1)),
            dim=0,
        )

    def poisson_matrix(self) -> Tensor:
        """Return the inverse canonical Poisson tensor ``J``."""

        return -self.symplectic_form()

    def kinetic_energy(self) -> Tensor:
        _q, p, mass = self.canonical_coordinates()
        return 0.5 * (p.square() / mass).sum()

    def hamiltonian(self, potential_energy: Tensor) -> Tensor:
        if potential_energy.ndim != 0:
            raise ValueError("scene potential energy must be scalar")
        return self.kinetic_energy() + potential_energy

    def with_canonical_coordinates(
        self,
        q: Tensor,
        p: Tensor,
        *,
        time: float | None = None,
    ) -> SymplecticSceneGraphState:
        expected = self.configuration_dimension
        if q.shape != (expected,) or p.shape != (expected,):
            raise ValueError("canonical q and p have the wrong dimension")
        nodes: list[SymplecticSceneNode] = []
        offset = 0
        for node in self.nodes:
            end = offset + len(node.q)
            nodes.append(
                SymplecticSceneNode(
                    node.node_id,
                    node.kind,
                    q[offset:end],
                    p[offset:end],
                    node.mass,
                    node.hierarchy_level,
                    node.parent_index,
                )
            )
            offset = end
        return SymplecticSceneGraphState(
            tuple(nodes), self.edge_index, self.relation_ids,
            self.time if time is None else float(time),
        )

    def symplectic_euler(
        self,
        force: Tensor,
        dt: float,
    ) -> SymplecticSceneGraphState:
        """Perform canonical kick-drift for conservative scene forces."""

        if dt <= 0.0:
            raise ValueError("symplectic scene time step must be positive")
        q, p, mass = self.canonical_coordinates()
        if force.shape != q.shape:
            raise ValueError("scene force must match canonical q")
        p_next = p + float(dt) * force
        q_next = q + float(dt) * p_next / mass
        return self.with_canonical_coordinates(q_next, p_next, time=self.time + dt)


def from_geometric_phase_graph(
    graph: object,
    *,
    node_prefix: str = "cloth",
    hierarchy_levels: tuple[int, ...] | None = None,
    edge_index: Tensor | None = None,
    relation_ids: Tensor | None = None,
) -> SymplecticSceneGraphState:
    """Lift existing ``GeometricPhaseGraphState`` cloth charts into ``T*Q``."""

    cloth = tuple(getattr(graph, "cloth"))
    if not cloth:
        raise ValueError("geometric phase graph contains no cloth charts")
    levels = (0,) * len(cloth) if hierarchy_levels is None else hierarchy_levels
    if len(levels) != len(cloth):
        raise ValueError("one hierarchy level is required per cloth chart")
    nodes = tuple(
        SymplecticSceneNode(
            f"{node_prefix}:{index}",
            "cloth",
            state.positions.reshape(-1),
            state.momenta.reshape(-1),
            state.lumped_mass[:, None].expand(-1, 3).reshape(-1),
            levels[index],
        )
        for index, state in enumerate(cloth)
    )
    if edge_index is None:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    if relation_ids is None:
        relation_ids = torch.empty((edge_index.shape[1],), dtype=torch.long)
    return SymplecticSceneGraphState(
        nodes, edge_index, relation_ids, float(getattr(graph, "time"))
    )
