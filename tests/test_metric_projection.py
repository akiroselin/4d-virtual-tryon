import pytest
import torch

from aigc_fashion.data import MaterialSpec
from aigc_fashion.metric_projection import (
    DifferentiableMetricProjector,
    MaterialStrainLimits,
    material_strain_limits,
)


def test_material_family_selects_different_hard_strain_envelopes() -> None:
    jersey = material_strain_limits(MaterialSpec(name="cotton jersey knit"))
    denim = material_strain_limits(MaterialSpec(name="heavy denim"))
    assert jersey.profile == "elastic_knit"
    assert denim.profile == "denim_woven"
    assert jersey.maximum_tensile_strain > denim.maximum_tensile_strain
    assert jersey.maximum_compressive_strain > denim.maximum_compressive_strain


def test_metric_projection_enforces_bounds_and_backpropagates() -> None:
    rest = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    projector = DifferentiableMetricProjector(
        torch.tensor([[0, 1]]),
        rest,
        torch.empty(0, dtype=torch.long),
        MaterialStrainLimits("test", 0.05, 0.03),
        iterations=2,
        tolerance=1.0e-6,
    )
    deformed = torch.tensor([[0.0, 0.0, 0.0], [1.8, 0.0, 0.0]], requires_grad=True)
    projected = projector(deformed)
    diagnostics = projector.assert_within_limits(projected)
    assert diagnostics.maximum_tensile_strain == pytest.approx(0.05, abs=1.0e-6)
    assert diagnostics.violating_edges == 0
    projected.square().sum().backward()
    assert deformed.grad is not None
    assert torch.isfinite(deformed.grad).all()


def test_pinned_positions_are_preserved_during_projection() -> None:
    rest = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    projector = DifferentiableMetricProjector(
        torch.tensor([[0, 1], [1, 2]]),
        rest,
        torch.tensor([0]),
        MaterialStrainLimits("test", 0.04, 0.02),
        iterations=64,
    )
    positions = torch.tensor([[0.0, 2.0, 0.0], [1.8, 2.0, 0.0], [3.6, 2.0, 0.0]])
    fixed = positions[[0]].clone()
    projected = projector(positions, pinned_positions=fixed)
    assert torch.equal(projected[[0]], fixed)
    projector.assert_within_limits(projected)


def test_incompatible_pins_fail_the_hard_constraint_audit() -> None:
    rest = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    projector = DifferentiableMetricProjector(
        torch.tensor([[0, 1]]),
        rest,
        torch.tensor([0, 1]),
        MaterialStrainLimits("test", 0.02, 0.02),
        iterations=4,
    )
    incompatible = torch.tensor([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]])
    projected = projector(incompatible, pinned_positions=incompatible)
    with pytest.raises(RuntimeError, match="hard material strain bounds"):
        projector.assert_within_limits(projected)


def test_edge_colored_projection_resolves_uniform_grid_strain() -> None:
    side = 8
    rest = torch.tensor(
        [[float(x), float(y), 0.0] for y in range(side) for x in range(side)]
    )
    edges = []
    for y in range(side):
        for x in range(side):
            vertex = y * side + x
            if x + 1 < side:
                edges.append((vertex, vertex + 1))
            if y + 1 < side:
                edges.append((vertex, vertex + side))
    projector = DifferentiableMetricProjector(
        torch.tensor(edges),
        rest,
        torch.empty(0, dtype=torch.long),
        MaterialStrainLimits("test", 0.18, 0.10),
        iterations=64,
        tolerance=1.5e-3,
    )
    deformed = rest.clone()
    deformed[:, 0] *= 1.35
    deformed[:, 1] *= 0.82
    projected = projector(deformed)
    diagnostics = projector.assert_within_limits(projected)
    assert diagnostics.violating_edges == 0


def test_residual_driven_sweeps_finish_dense_projection_without_relaxing_limits() -> None:
    side = 8
    rest = torch.tensor(
        [[float(x), float(y), 0.0] for y in range(side) for x in range(side)]
    )
    edges = []
    for y in range(side):
        for x in range(side):
            vertex = y * side + x
            if x + 1 < side:
                edges.append((vertex, vertex + 1))
            if y + 1 < side:
                edges.append((vertex, vertex + side))
    limits = MaterialStrainLimits("test", 0.18, 0.10)
    fixed_only = DifferentiableMetricProjector(
        torch.tensor(edges),
        rest,
        torch.empty(0, dtype=torch.long),
        limits,
        iterations=1,
        tolerance=1.0e-3,
    )
    adaptive = DifferentiableMetricProjector(
        torch.tensor(edges),
        rest,
        torch.empty(0, dtype=torch.long),
        limits,
        iterations=1,
        maximum_adaptive_iterations=127,
        tolerance=1.0e-3,
    )
    deformed = rest * torch.tensor([1.35, 0.82, 1.0])

    assert fixed_only.diagnostics(fixed_only(deformed)).violating_edges > 0
    projected = adaptive(deformed)
    diagnostics = adaptive.assert_within_limits(projected)
    assert diagnostics.violating_edges == 0
    assert diagnostics.maximum_limit_violation <= 1.0e-3
