from __future__ import annotations

import torch

from .model import SteeringBatch


def _group_normalized(values: torch.Tensor, group_ids: list[str], eps: float = 1e-6) -> torch.Tensor:
    out = torch.zeros_like(values)
    for group_id in sorted(set(group_ids)):
        idx = [i for i, item in enumerate(group_ids) if item == group_id]
        group = values[idx]
        out[idx] = (group - group.mean()) / (group.std(unbiased=False) + eps)
    return out


def tb_vargrad_loss(batch: SteeringBatch, *, beta: float) -> tuple[torch.Tensor, dict[str, float]]:
    targets = batch.log_ref - batch.log_pi.detach() + batch.rewards / beta
    z_hat = torch.zeros_like(targets)
    for group_id in sorted(set(batch.group_ids)):
        idx = [i for i, item in enumerate(batch.group_ids) if item == group_id]
        z_hat[idx] = targets[idx].mean()
    residual = z_hat.detach() + batch.log_pi - batch.log_ref - batch.rewards / beta
    loss = residual.square().mean()
    return loss, _diagnostics(batch, loss, residual=residual)


def grpo_loss(batch: SteeringBatch, *, beta: float, clip_epsilon: float) -> tuple[torch.Tensor, dict[str, float]]:
    returns = batch.rewards - beta * (batch.log_pi.detach() - batch.log_ref)
    advantages = _group_normalized(returns, batch.group_ids)
    ratio = torch.exp(batch.log_pi - batch.log_pi.detach())
    clipped = ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon)
    loss = -torch.minimum(ratio * advantages, clipped * advantages).mean()
    return loss, _diagnostics(batch, loss, advantages=advantages)


def actor_critic_loss(
    batch: SteeringBatch,
    *,
    beta: float,
    value_loss_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    returns = batch.rewards - beta * (batch.log_pi.detach() - batch.log_ref)
    advantages = returns - batch.values
    policy_loss = -(advantages.detach() * batch.log_pi).mean()
    value_loss = (batch.values - returns.detach()).square().mean()
    loss = policy_loss + value_loss_weight * value_loss
    diagnostics = _diagnostics(batch, loss, advantages=advantages.detach())
    diagnostics["policy_loss"] = float(policy_loss.detach().cpu())
    diagnostics["value_loss"] = float(value_loss.detach().cpu())
    return loss, diagnostics


def _diagnostics(
    batch: SteeringBatch,
    loss: torch.Tensor,
    *,
    residual: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
) -> dict[str, float]:
    kl = batch.log_pi.detach() - batch.log_ref
    diagnostics = {
        "loss": float(loss.detach().cpu()),
        "reward_mean": float(batch.rewards.detach().mean().cpu()),
        "log_pi_mean": float(batch.log_pi.detach().mean().cpu()),
        "log_ref_mean": float(batch.log_ref.detach().mean().cpu()),
        "kl_mean": float(kl.mean().cpu()),
        "valid_tokens_mean": float(batch.num_valid_tokens.detach().mean().cpu()),
    }
    if residual is not None:
        diagnostics["tb_residual_abs_mean"] = float(residual.detach().abs().mean().cpu())
    if advantages is not None:
        diagnostics["advantage_mean"] = float(advantages.detach().mean().cpu())
        diagnostics["advantage_std"] = float(advantages.detach().std(unbiased=False).cpu())
    return diagnostics

