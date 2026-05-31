import torch

from itds.model import SteeringBatch
from itds.objectives import _token_monte_carlo_returns, actor_critic_loss


def _batch() -> SteeringBatch:
    token_log_pi = torch.tensor([-1.0, -2.0, -3.0], requires_grad=True)
    token_log_ref = torch.tensor([-1.5, -1.5, -1.5])
    token_values = torch.tensor([0.1, 0.2, 0.3], requires_grad=True)
    return SteeringBatch(
        log_pi=token_log_pi.sum().reshape(1),
        log_ref=token_log_ref.sum().reshape(1),
        values=token_values.mean().reshape(1),
        rewards=torch.tensor([1.0]),
        group_ids=["0"],
        num_valid_tokens=torch.tensor([3.0]),
        token_log_pi=token_log_pi,
        token_log_ref=token_log_ref,
        token_values=token_values,
        token_to_sequence=torch.tensor([0, 0, 0]),
        token_is_terminal=torch.tensor([0.0, 0.0, 1.0]),
    )


def test_token_returns_put_correctness_reward_only_on_terminal_token():
    batch = _batch()
    token_rewards = -0.5 * (batch.token_log_pi.detach() - batch.token_log_ref)
    token_rewards = token_rewards + batch.token_is_terminal * batch.rewards[batch.token_to_sequence]

    assert torch.allclose(token_rewards, torch.tensor([-0.25, 0.25, 1.75]))
    assert torch.allclose(_token_monte_carlo_returns(token_rewards, batch.token_to_sequence), torch.tensor([1.75, 2.0, 1.75]))


def test_actor_critic_loss_is_token_level_and_differentiable():
    loss, diagnostics = actor_critic_loss(_batch(), beta=0.5, value_loss_weight=0.1)

    assert loss.requires_grad
    assert diagnostics["num_tokens"] == 3.0
    assert "token_reward_mean" in diagnostics
    assert "return_mean" in diagnostics


def test_token_basis_small_initialization_without_loading_base(monkeypatch):
    from types import SimpleNamespace

    import itds.model as model_module

    class TinyBase(torch.nn.Module):
        config = SimpleNamespace(hidden_size=4, vocab_size=1000)

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

    monkeypatch.setattr(model_module, "AutoModelForCausalLM", TinyBase)
    model = model_module.TopKLowRankSteering("tiny", rank=8, token_basis_init_std=1e-4)

    assert model.token_basis.weight.std().item() < 5e-4
