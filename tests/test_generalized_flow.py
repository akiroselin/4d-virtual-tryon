import pytest
import torch

from aigc_fashion.generalized_flow import (
    GeneralizedFlowLedger,
    GeneralizedFlowPort,
    GeneralizedFlowRepresentation,
    GeneralizedFlowRole,
    GeneralizedImpulseTransfer,
)


def test_e3_force_velocity_power_is_rotation_invariant() -> None:
    force = torch.tensor([[1.0, -2.0, 0.5], [-0.4, 0.3, 0.2]])
    velocity = torch.tensor([[0.2, 0.1, -0.3], [0.5, -0.2, 0.4]])
    rotation = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    base = GeneralizedFlowPort(
        "shell", "shell_energy", "cloth_momentum",
        GeneralizedFlowRole.CONSERVATIVE,
        GeneralizedFlowRepresentation.E3_VECTOR,
        force, velocity, 0.0,
    )
    moved = GeneralizedFlowPort(
        "shell", "shell_energy", "cloth_momentum",
        GeneralizedFlowRole.CONSERVATIVE,
        GeneralizedFlowRepresentation.E3_VECTOR,
        force @ rotation.T, velocity @ rotation.T, 0.0,
    )
    assert torch.allclose(base.power(), moved.power(), atol=1.0e-6)


def test_so2_effort_flow_power_is_gauge_invariant() -> None:
    effort = torch.tensor([1.0 + 0.3j, -0.2 + 0.8j])
    flow = torch.tensor([0.4 - 0.1j, 0.6 + 0.5j])
    gauge = torch.exp(torch.tensor(0.71j))
    base = GeneralizedFlowPort(
        "friction", "contact", "cloth",
        GeneralizedFlowRole.DISSIPATIVE,
        GeneralizedFlowRepresentation.SO2_COMPLEX,
        effort, flow, 0.0,
    )
    changed = GeneralizedFlowPort(
        "friction", "contact", "cloth",
        GeneralizedFlowRole.DISSIPATIVE,
        GeneralizedFlowRepresentation.SO2_COMPLEX,
        gauge * effort, gauge * flow, 0.0,
    )
    assert torch.allclose(base.power(), changed.power(), atol=1.0e-6)


def test_ledger_separates_continuous_power_and_impulse_work() -> None:
    ledger = GeneralizedFlowLedger()
    damping = GeneralizedFlowPort(
        "damping", "cloth_momentum", "thermal_sink",
        GeneralizedFlowRole.DISSIPATIVE,
        GeneralizedFlowRepresentation.E3_VECTOR,
        torch.tensor([[-2.0, 0.0, 0.0]]),
        torch.tensor([[0.5, 0.0, 0.0]]),
        0.0,
    )
    ledger.record_port(damping, duration_s=0.2)
    ledger.record_impulse(
        GeneralizedImpulseTransfer(
            "contact_reset", "cloth_momentum", "contact_sink",
            GeneralizedFlowRole.DISSIPATIVE,
            GeneralizedFlowRepresentation.E3_VECTOR,
            torch.tensor([[-0.1, 0.0, 0.0]]),
            torch.tensor(-0.03), 0.2,
        )
    )
    audit = ledger.audit()
    assert audit.continuous_work_j["dissipative"] == pytest.approx(-0.2)
    assert audit.impulse_work_j["dissipative"] == pytest.approx(-0.03)
    assert audit.net_work_j == pytest.approx(-0.23)
    assert audit.dissipative_sign_violations == 0


def test_ledger_detects_positive_dissipative_power() -> None:
    ledger = GeneralizedFlowLedger()
    ledger.record_port(
        GeneralizedFlowPort(
            "bad_damping", "sink", "cloth",
            GeneralizedFlowRole.DISSIPATIVE,
            GeneralizedFlowRepresentation.REAL_SCALAR,
            torch.tensor([1.0]), torch.tensor([0.5]), 0.0,
        ),
        duration_s=0.1,
    )
    assert ledger.audit().dissipative_sign_violations == 1


def test_ledger_can_discard_a_speculative_event_step() -> None:
    ledger = GeneralizedFlowLedger()
    checkpoint = ledger.checkpoint()
    ledger.record_port(
        GeneralizedFlowPort(
            "trial", "field", "cloth",
            GeneralizedFlowRole.CONTROL,
            GeneralizedFlowRepresentation.REAL_SCALAR,
            torch.tensor([2.0]), torch.tensor([3.0]), 0.0,
        ),
        duration_s=0.25,
    )
    ledger.restore(checkpoint)
    audit = ledger.audit()
    assert audit.net_work_j == 0.0
    assert audit.sample_count["control"] == 0
