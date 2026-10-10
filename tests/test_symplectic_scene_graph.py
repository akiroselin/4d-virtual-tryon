import torch

from aigc_fashion.geometric_phase_graph import (
    GeometricPhaseGraphState,
    IntrinsicClothPhaseState,
)
from aigc_fashion.symplectic_scene_graph import from_geometric_phase_graph


def _graph() -> GeometricPhaseGraphState:
    cloth = IntrinsicClothPhaseState.from_velocity(
        torch.zeros((2, 2)),
        torch.zeros((2, 3)),
        torch.zeros((2, 3)),
        torch.ones(2),
        0.0,
    )
    return GeometricPhaseGraphState((cloth,), None, None, 0.0)


def test_phase_graph_lifts_to_cotangent_bundle_and_has_canonical_form():
    state = from_geometric_phase_graph(_graph())
    assert state.configuration_dimension == 6
    assert state.phase_dimension == 12
    omega = state.symplectic_form()
    assert torch.allclose(omega.T, -omega)
    assert torch.allclose(state.poisson_matrix(), -omega)


def test_symplectic_euler_kick_drift_moves_under_gravity():
    state = from_geometric_phase_graph(_graph())
    force = torch.zeros(6)
    force[1::3] = -9.81
    next_state = state.symplectic_euler(force, 0.01)
    q, p, _mass = next_state.canonical_coordinates()
    assert torch.all(p[1::3] < 0.0)
    assert torch.all(q[1::3] < 0.0)


def test_graph_edges_are_metadata_for_the_canonical_form():
    graph = _graph()
    state = from_geometric_phase_graph(
        graph,
        edge_index=torch.tensor([[0], [0]]),
        relation_ids=torch.tensor([2]),
    )
    assert state.edge_index.shape == (2, 1)
    assert state.relation_ids.tolist() == [2]
