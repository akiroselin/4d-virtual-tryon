from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .physics_conditioned_flow import SpectralLinear


def so3_hat(vector: Tensor) -> Tensor:
    if vector.shape[-1] != 3:
        raise ValueError("SO(3) vector must end in dimension 3")
    x, y, z = vector.unbind(dim=-1)
    zero = torch.zeros_like(x)
    return torch.stack(
        (
            zero,
            -z,
            y,
            z,
            zero,
            -x,
            -y,
            x,
            zero,
        ),
        dim=-1,
    ).reshape(*vector.shape[:-1], 3, 3)


def so3_exp(rotation_vector: Tensor) -> Tensor:
    skew = so3_hat(rotation_vector)
    theta_squared = rotation_vector.square().sum(dim=-1, keepdim=True)
    theta = torch.sqrt(theta_squared)
    safe_theta = theta.clamp_min(1.0e-8)
    safe_squared = theta_squared.clamp_min(1.0e-16)
    a_regular = torch.sin(safe_theta) / safe_theta
    b_regular = (1.0 - torch.cos(safe_theta)) / safe_squared
    a_taylor = 1.0 - theta_squared / 6.0 + theta_squared.square() / 120.0
    b_taylor = 0.5 - theta_squared / 24.0 + theta_squared.square() / 720.0
    small = theta_squared < 1.0e-8
    a = torch.where(small, a_taylor, a_regular)[..., None]
    b = torch.where(small, b_taylor, b_regular)[..., None]
    identity = torch.eye(3, device=rotation_vector.device, dtype=rotation_vector.dtype)
    identity = identity.expand(*rotation_vector.shape[:-1], 3, 3)
    return identity + a * skew + b * (skew @ skew)


def so3_log(rotation: Tensor) -> Tensor:
    if rotation.shape[-2:] != (3, 3):
        raise ValueError("rotation must end in shape (3, 3)")
    trace = rotation.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    cosine = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
    theta = torch.acos(cosine)
    vee = torch.stack(
        (
            rotation[..., 2, 1] - rotation[..., 1, 2],
            rotation[..., 0, 2] - rotation[..., 2, 0],
            rotation[..., 1, 0] - rotation[..., 0, 1],
        ),
        dim=-1,
    )
    sine = torch.sin(theta)
    regular = theta / (2.0 * sine.clamp_min(1.0e-8))
    factor = torch.where(theta.abs() < 1.0e-5, torch.full_like(theta, 0.5), regular)
    return factor[..., None] * vee


def se3_exp(twist: Tensor) -> Tensor:
    """SE(3) exponential for twists ordered as ``(omega, linear)``."""

    if twist.shape[-1] != 6:
        raise ValueError("SE(3) twist must end in dimension 6")
    omega = twist[..., :3]
    linear = twist[..., 3:]
    skew = so3_hat(omega)
    theta_squared = omega.square().sum(dim=-1, keepdim=True)
    theta = torch.sqrt(theta_squared)
    safe_theta = theta.clamp_min(1.0e-8)
    safe_squared = theta_squared.clamp_min(1.0e-16)
    b_regular = (1.0 - torch.cos(safe_theta)) / safe_squared
    c_regular = (safe_theta - torch.sin(safe_theta)) / (safe_theta * safe_squared)
    b_taylor = 0.5 - theta_squared / 24.0 + theta_squared.square() / 720.0
    c_taylor = 1.0 / 6.0 - theta_squared / 120.0 + theta_squared.square() / 5040.0
    small = theta_squared < 1.0e-8
    b = torch.where(small, b_taylor, b_regular)[..., None]
    c = torch.where(small, c_taylor, c_regular)[..., None]
    identity = torch.eye(3, device=twist.device, dtype=twist.dtype).expand(*twist.shape[:-1], 3, 3)
    rotation = so3_exp(omega)
    left_jacobian = identity + b * skew + c * (skew @ skew)
    translation = (left_jacobian @ linear[..., None]).squeeze(-1)
    transform = torch.zeros((*twist.shape[:-1], 4, 4), device=twist.device, dtype=twist.dtype)
    transform[..., :3, :3] = rotation
    transform[..., :3, 3] = translation
    transform[..., 3, 3] = 1.0
    return transform


def se3_log(transform: Tensor) -> Tensor:
    """SE(3) logarithm ordered as ``(omega, linear)``.

    The implementation is stable around the identity. Rotations arbitrarily
    close to pi should use a dedicated robust logarithm in production.
    """

    _validate_transform(transform)
    rotation = transform[..., :3, :3]
    translation = transform[..., :3, 3]
    omega = so3_log(rotation)
    skew = so3_hat(omega)
    theta_squared = omega.square().sum(dim=-1, keepdim=True)
    theta = torch.sqrt(theta_squared)
    safe_theta = theta.clamp_min(1.0e-8)
    safe_squared = theta_squared.clamp_min(1.0e-16)
    denominator = 2.0 * safe_theta * torch.sin(safe_theta).clamp_min(1.0e-8)
    d_regular = 1.0 / safe_squared - (1.0 + torch.cos(safe_theta)) / denominator
    d_taylor = 1.0 / 12.0 + theta_squared / 720.0 + theta_squared.square() / 30240.0
    d = torch.where(theta_squared < 1.0e-8, d_taylor, d_regular)[..., None]
    identity = torch.eye(3, device=transform.device, dtype=transform.dtype)
    identity = identity.expand(*transform.shape[:-2], 3, 3)
    inverse_left_jacobian = identity - 0.5 * skew + d * (skew @ skew)
    linear = (inverse_left_jacobian @ translation[..., None]).squeeze(-1)
    return torch.cat((omega, linear), dim=-1)


def se3_inverse(transform: Tensor) -> Tensor:
    _validate_transform(transform)
    rotation = transform[..., :3, :3]
    translation = transform[..., :3, 3]
    inverse = torch.zeros_like(transform)
    inverse_rotation = rotation.transpose(-1, -2)
    inverse[..., :3, :3] = inverse_rotation
    inverse[..., :3, 3] = -(inverse_rotation @ translation[..., None]).squeeze(-1)
    inverse[..., 3, 3] = 1.0
    return inverse


def right_invariant_relative(reference: Tensor, pose: Tensor) -> Tensor:
    _validate_transform(reference)
    _validate_transform(pose)
    return se3_inverse(reference) @ pose


def right_perturb_pose(clean_pose: Tensor, tangent_noise: Tensor, sigma: float | Tensor) -> Tensor:
    _validate_transform(clean_pose)
    scaled_noise = torch.as_tensor(sigma, device=tangent_noise.device, dtype=tangent_noise.dtype) * tangent_noise
    return clean_pose @ se3_exp(scaled_noise)


@dataclass(frozen=True)
class SE3DiffusionStep:
    pose: Tensor
    tangent_increment: Tensor
    transition_score: Tensor
    beta: Tensor


class SE3RiemannianDDPM(nn.Module):
    """Discrete right-invariant random-walk diffusion on SE(3).

    Each forward step applies ``T_k = T_{k-1} Exp(sqrt(beta_k) S eps)``.
    ``S`` gives separate rotation and translation metric scales, avoiding the
    dimensionally invalid treatment of radians and metres as identical Euclidean
    coordinates. The transition score is right-trivialized in ``se(3)``.

    This is a local wrapped-Gaussian approximation to the SE(3) heat kernel. It
    preserves the group exactly but does not claim a closed-form multi-step
    marginal or the exponential-map Jacobian correction needed far from the
    injectivity region.
    """

    def __init__(
        self,
        steps: int = 100,
        *,
        beta_start: float = 1.0e-4,
        beta_end: float = 2.0e-2,
        rotation_scale: float = 1.0,
        translation_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if steps <= 0:
            raise ValueError("steps must be positive")
        if not 0.0 < beta_start <= beta_end < 1.0:
            raise ValueError("betas must satisfy 0 < beta_start <= beta_end < 1")
        if rotation_scale <= 0.0 or translation_scale <= 0.0:
            raise ValueError("SE(3) metric scales must be positive")
        self.steps = steps
        self.register_buffer("betas", torch.linspace(beta_start, beta_end, steps))
        self.register_buffer(
            "tangent_scale",
            torch.tensor(
                [rotation_scale] * 3 + [translation_scale] * 3,
                dtype=torch.float32,
            ),
        )

    def forward_step(
        self,
        pose: Tensor,
        step: int,
        *,
        noise: Tensor | None = None,
    ) -> SE3DiffusionStep:
        _validate_transform(pose)
        beta, scale = self._step_parameters(step, pose)
        expected = (*pose.shape[:-2], 6)
        if noise is None:
            noise = torch.randn(expected, device=pose.device, dtype=pose.dtype)
        else:
            noise = noise.to(device=pose.device, dtype=pose.dtype)
            if noise.shape != expected:
                raise ValueError(f"noise must have shape {expected}")
        increment = torch.sqrt(beta) * scale * noise
        transition_score = -increment / (beta * scale.square())
        noised_pose = pose @ se3_exp(increment)
        return SE3DiffusionStep(
            pose=noised_pose,
            tangent_increment=increment,
            transition_score=transition_score,
            beta=beta,
        )

    def reverse_step(
        self,
        pose: Tensor,
        right_trivialized_score: Tensor,
        step: int,
        *,
        noise: Tensor | None = None,
        stochastic: bool = True,
    ) -> Tensor:
        _validate_transform(pose)
        expected = (*pose.shape[:-2], 6)
        if right_trivialized_score.shape != expected:
            raise ValueError(f"right_trivialized_score must have shape {expected}")
        beta, scale = self._step_parameters(step, pose)
        drift = beta * scale.square() * right_trivialized_score
        if stochastic and step > 0:
            if noise is None:
                noise = torch.randn(expected, device=pose.device, dtype=pose.dtype)
            else:
                noise = noise.to(device=pose.device, dtype=pose.dtype)
                if noise.shape != expected:
                    raise ValueError(f"noise must have shape {expected}")
            increment = drift + torch.sqrt(beta) * scale * noise
        else:
            increment = drift
        return pose @ se3_exp(increment)

    def _step_parameters(self, step: int, reference: Tensor) -> tuple[Tensor, Tensor]:
        if not isinstance(step, int) or isinstance(step, bool) or not 0 <= step < self.steps:
            raise ValueError(f"step must be an integer in [0, {self.steps})")
        beta = self.betas[step].to(device=reference.device, dtype=reference.dtype)
        scale = self.tangent_scale.to(device=reference.device, dtype=reference.dtype)
        return beta, scale


class SE3BodyScoreHead(nn.Module):
    """Right-trivialized score head for a Lie-group diffusion model.

    Inputs are body-frame relative twists. Global left multiplication of every
    pose leaves these coordinates unchanged, avoiding flat quaternion/translation
    denoising. A complete model still needs a manifold noise schedule and reverse
    SDE/ODE integrator.
    """

    def __init__(self, context_dim: int, *, hidden_dim: int = 64, spectral_cap: float = 0.9) -> None:
        super().__init__()
        if context_dim < 0 or hidden_dim <= 0:
            raise ValueError("invalid score-head dimensions")
        self.context_dim = context_dim
        self.input = SpectralLinear(7 + context_dim, hidden_dim, max_singular_value=spectral_cap)
        self.hidden = SpectralLinear(hidden_dim, hidden_dim, max_singular_value=spectral_cap)
        self.output = SpectralLinear(hidden_dim, 6, max_singular_value=spectral_cap)

    def forward(self, relative_twist: Tensor, diffusion_time: Tensor, context: Tensor | None = None) -> Tensor:
        if relative_twist.shape[-1] != 6:
            raise ValueError("relative_twist must end in dimension 6")
        if diffusion_time.shape != (*relative_twist.shape[:-1], 1):
            raise ValueError("diffusion_time must match relative_twist batch shape")
        if context is None:
            context = relative_twist.new_zeros((*relative_twist.shape[:-1], self.context_dim))
        if context.shape != (*relative_twist.shape[:-1], self.context_dim):
            raise ValueError("context shape does not match score head")
        features = torch.cat((relative_twist, diffusion_time, context), dim=-1)
        hidden = torch.tanh(self.input(features))
        hidden = torch.tanh(self.hidden(hidden))
        return self.output(hidden)


@dataclass(frozen=True)
class SE3ScoreProposal:
    """One right-trivialized pose proposal in the Lie algebra."""

    pose: Tensor
    relative_twist: Tensor
    score: Tensor
    proposal_twist: Tensor


class SE3GenerativeDynamicsHead(nn.Module):
    """Physics-facing wrapper around the SE(3) score head.

    The output layer is zero-initialized deliberately. A fresh checkpoint is
    therefore an identity proposal and cannot inject random motion into the
    cloth. Once trained with denoising and physics losses, the proposal is
    generated in ``se(3)`` and can be passed to a Hamiltonian/contact port.
    """

    def __init__(
        self,
        context_dim: int,
        *,
        diffusion_steps: int = 8,
        score_scale: float = 1.0,
        rotation_scale: float = 0.3,
        translation_scale: float = 0.03,
        hidden_dim: int = 64,
        spectral_cap: float = 0.9,
    ) -> None:
        super().__init__()
        if score_scale < 0.0:
            raise ValueError("score_scale must be non-negative")
        self.score_scale = float(score_scale)
        self.diffusion = SE3RiemannianDDPM(
            diffusion_steps,
            rotation_scale=rotation_scale,
            translation_scale=translation_scale,
        )
        self.score_head = SE3BodyScoreHead(
            context_dim,
            hidden_dim=hidden_dim,
            spectral_cap=spectral_cap,
        )
        nn.init.zeros_(self.score_head.output.weight)
        if self.score_head.output.bias is not None:
            nn.init.zeros_(self.score_head.output.bias)

    def forward(
        self,
        relative_pose: Tensor,
        diffusion_time: Tensor,
        context: Tensor,
        *,
        step: int = 0,
    ) -> SE3ScoreProposal:
        relative_twist = se3_log(relative_pose)
        score = self.score_head(relative_twist, diffusion_time, context)
        scaled_score = self.score_scale * score
        proposed_pose = self.diffusion.reverse_step(
            relative_pose,
            scaled_score,
            step,
            stochastic=False,
        )
        proposal_twist = se3_log(se3_inverse(relative_pose) @ proposed_pose)
        return SE3ScoreProposal(
            pose=proposed_pose,
            relative_twist=relative_twist,
            score=score,
            proposal_twist=proposal_twist,
        )


def _validate_transform(transform: Tensor) -> None:
    if transform.shape[-2:] != (4, 4):
        raise ValueError("SE(3) transform must end in shape (4, 4)")
