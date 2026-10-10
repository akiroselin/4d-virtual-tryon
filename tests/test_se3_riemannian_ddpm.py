from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aigc_fashion.lie_group_diffusion import (  # noqa: E402
    SE3GenerativeDynamicsHead,
    SE3RiemannianDDPM,
    se3_exp,
)


def test_se3_ddpm_forward_and_reverse_are_global_left_equivariant() -> None:
    diffusion = SE3RiemannianDDPM(
        8,
        rotation_scale=0.3,
        translation_scale=0.7,
    )
    clean = se3_exp(torch.tensor([0.1, -0.2, 0.05, 0.4, -0.1, 0.2]))
    global_pose = se3_exp(torch.tensor([-0.2, 0.1, 0.3, 1.0, -0.5, 0.2]))
    noise = torch.tensor([0.3, -0.4, 0.2, 0.1, 0.5, -0.2])
    baseline = diffusion.forward_step(clean, 4, noise=noise)
    moved = diffusion.forward_step(global_pose @ clean, 4, noise=noise)
    assert torch.allclose(moved.pose, global_pose @ baseline.pose, atol=2.0e-5, rtol=2.0e-5)
    assert torch.allclose(moved.transition_score, baseline.transition_score)

    recovered = diffusion.reverse_step(
        baseline.pose,
        baseline.transition_score,
        4,
        stochastic=False,
    )
    recovered_moved = diffusion.reverse_step(
        moved.pose,
        moved.transition_score,
        4,
        stochastic=False,
    )
    assert torch.allclose(recovered, clean, atol=3.0e-5, rtol=3.0e-5)
    assert torch.allclose(recovered_moved, global_pose @ clean, atol=3.0e-5, rtol=3.0e-5)


def test_se3_ddpm_uses_distinct_rotation_and_translation_metric_scales() -> None:
    diffusion = SE3RiemannianDDPM(
        2,
        beta_start=0.01,
        beta_end=0.02,
        rotation_scale=0.2,
        translation_scale=2.0,
    )
    identity = torch.eye(4)
    step = diffusion.forward_step(identity, 0, noise=torch.ones(6))
    assert torch.allclose(step.tangent_increment[:3], torch.full((3,), 0.02))
    assert torch.allclose(step.tangent_increment[3:], torch.full((3,), 0.2))
    assert torch.allclose(step.transition_score[:3], torch.full((3,), -50.0))
    assert torch.allclose(step.transition_score[3:], torch.full((3,), -5.0))


def test_se3_ddpm_rejects_invalid_schedule_and_step() -> None:
    with pytest.raises(ValueError, match="betas"):
        SE3RiemannianDDPM(beta_start=0.1, beta_end=1.0)
    diffusion = SE3RiemannianDDPM(3)
    with pytest.raises(ValueError, match="step"):
        diffusion.forward_step(torch.eye(4), 3)


def test_fresh_se3_generative_head_is_identity_and_group_valued() -> None:
    head = SE3GenerativeDynamicsHead(context_dim=4, hidden_dim=8)
    poses = torch.eye(4).repeat(2, 1, 1)
    context = torch.zeros((2, 4))
    diffusion_time = torch.zeros((2, 1))
    proposal = head(poses, diffusion_time, context)

    assert torch.allclose(proposal.pose, poses)
    assert torch.allclose(proposal.score, torch.zeros_like(proposal.score))
    assert torch.allclose(
        proposal.proposal_twist, torch.zeros_like(proposal.proposal_twist)
    )
