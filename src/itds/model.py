from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import torch
from torch import nn
from transformers import AutoModelForCausalLM


@dataclass
class SteeringBatch:
    log_pi: torch.Tensor
    log_ref: torch.Tensor
    values: torch.Tensor
    rewards: torch.Tensor
    group_ids: list[str]
    num_valid_tokens: torch.Tensor
    token_log_pi: torch.Tensor
    token_log_ref: torch.Tensor
    token_values: torch.Tensor
    token_to_sequence: torch.Tensor
    token_is_terminal: torch.Tensor


class ResidualSiluBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.linear = nn.Linear(width, width)
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.activation(self.linear(x))


def _build_actor(hidden_size: int, rank: int, depth: int) -> nn.Sequential:
    if depth < 1:
        raise ValueError("actor_depth must be >= 1")
    layers: list[nn.Module] = [nn.Linear(hidden_size, rank)]
    layers.extend(ResidualSiluBlock(rank) for _ in range(depth - 1))
    return nn.Sequential(*layers)


def _build_critic(hidden_size: int, rank: int, depth: int) -> nn.Sequential:
    if depth < 2:
        raise ValueError("critic_depth must be >= 2")
    layers: list[nn.Module] = [nn.Linear(hidden_size, rank)]
    layers.extend(ResidualSiluBlock(rank) for _ in range(depth - 2))
    layers.extend([nn.SiLU(), nn.Linear(rank, 1)])
    return nn.Sequential(*layers)


class TopKLowRankSteering(nn.Module):
    def __init__(
        self,
        model_name_or_path: str,
        *,
        top_k: int = 64,
        rank: int = 32,
        actor_depth: int = 10,
        critic_depth: int = 10,
        alpha: float = 1.0,
        token_basis_init_std: float = 1e-3,
        torch_dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.top_k = top_k
        self.rank = rank
        self.actor_depth = actor_depth
        self.critic_depth = critic_depth
        self.alpha = alpha
        self.token_basis_init_std = token_basis_init_std
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )
        for param in self.base_model.parameters():
            param.requires_grad_(False)
        hidden_size = self.base_model.config.hidden_size
        vocab_size = self.base_model.config.vocab_size
        self.state_projector = _build_actor(hidden_size, rank, actor_depth)
        self.token_basis = nn.Embedding(vocab_size, rank)
        nn.init.normal_(self.token_basis.weight, mean=0.0, std=token_basis_init_std)
        self.value_head = _build_critic(hidden_size, rank, critic_depth)

    def trainable_parameter_counts(self) -> tuple[int, int]:
        trainable = sum(param.numel() for param in self.parameters() if param.requires_grad)
        total = sum(param.numel() for param in self.parameters())
        return trainable, total

    def steering_state_dict(self) -> dict[str, torch.Tensor]:
        return {
            key: value
            for key, value in self.state_dict().items()
            if key.startswith(("state_projector.", "token_basis.", "value_head."))
        }

    def load_steering_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        self.load_state_dict(state_dict, strict=False)

    def _last_hidden_state(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if hasattr(self.base_model, "model") and hasattr(self.base_model, "lm_head"):
            output = self.base_model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
            return output.last_hidden_state[:, :-1, :].detach()

        output = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        return output.hidden_states[-1][:, :-1, :].detach()

    def _logits_from_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        if hasattr(self.base_model, "lm_head"):
            return self.base_model.lm_head(hidden)
        return self.base_model.get_output_embeddings()(hidden)

    def encode_batch(self, tokenizer, examples, device: torch.device | str) -> SteeringBatch:
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        original_padding_side = getattr(tokenizer, "padding_side", None)
        tokenizer.padding_side = "right"
        try:
            prompts = [example.prompt for example in examples]
            texts = [example.prompt + example.response for example in examples]
            prompt_ids = tokenizer(prompts, add_special_tokens=False).input_ids
            encoded = tokenizer(texts, add_special_tokens=False, padding=True, return_tensors="pt")
        finally:
            if original_padding_side is not None:
                tokenizer.padding_side = original_padding_side

        input_ids = encoded.input_ids.to(device)
        attention_mask = encoded.attention_mask.to(device)
        with torch.no_grad():
            hidden = self._last_hidden_state(input_ids, attention_mask)

        token_rows: list[int] = []
        token_positions: list[int] = []
        for row, _example in enumerate(examples):
            prompt_len = len(prompt_ids[row])
            seq_len = int(attention_mask[row].sum().item())
            start = max(prompt_len - 1, 0)
            end = max(seq_len - 1, start)
            token_rows.extend([row] * max(end - start, 0))
            token_positions.extend(range(start, end))

        if not token_rows:
            return self._empty_batch(device)

        token_rows_tensor = torch.tensor(token_rows, dtype=torch.long, device=device)
        token_positions_tensor = torch.tensor(token_positions, dtype=torch.long, device=device)
        hidden_tokens = hidden[token_rows_tensor, token_positions_tensor]
        target_ids = input_ids[token_rows_tensor, token_positions_tensor + 1]

        with torch.no_grad():
            base_logits = self._logits_from_hidden(hidden_tokens).detach()
            top_values, top_ids = torch.topk(base_logits, k=min(self.top_k, base_logits.shape[-1]), dim=-1)

        matches = top_ids.eq(target_ids.unsqueeze(-1))
        valid_mask = matches.any(dim=-1)
        if not bool(valid_mask.any().item()):
            return self._empty_batch(device)

        match_indices = matches.to(torch.long).argmax(dim=-1)[valid_mask]
        valid_rows = token_rows_tensor[valid_mask]
        valid_hidden = hidden_tokens[valid_mask]
        valid_top_ids = top_ids[valid_mask]
        steering_dtype = self.state_projector[0].weight.dtype
        valid_top_values = top_values[valid_mask].to(dtype=steering_dtype)
        states = valid_hidden.to(dtype=steering_dtype)
        u = self.state_projector(states)
        steering = (self.token_basis(valid_top_ids) * u.unsqueeze(1)).sum(dim=-1)
        steered_logits = valid_top_values + self.alpha * steering
        token_log_pi = torch.log_softmax(steered_logits, dim=-1).gather(1, match_indices.unsqueeze(1)).squeeze(1)
        token_log_ref = torch.log_softmax(valid_top_values, dim=-1).gather(1, match_indices.unsqueeze(1)).squeeze(1).detach()
        token_values = self.value_head(states).squeeze(-1)

        per_sequence: OrderedDict[int, list[int]] = OrderedDict()
        for token_index, row in enumerate(valid_rows.tolist()):
            per_sequence.setdefault(row, []).append(token_index)

        log_pi_values: list[torch.Tensor] = []
        log_ref_values: list[torch.Tensor] = []
        value_values: list[torch.Tensor] = []
        rewards: list[float] = []
        group_ids: list[str] = []
        valid_counts: list[int] = []
        token_to_sequence_values: list[int] = [0] * token_log_pi.numel()
        token_is_terminal_values: list[float] = [0.0] * token_log_pi.numel()

        for sequence_index, (row, token_indices) in enumerate(per_sequence.items()):
            index_tensor = torch.tensor(token_indices, dtype=torch.long, device=device)
            log_pi_values.append(token_log_pi[index_tensor].sum())
            log_ref_values.append(token_log_ref[index_tensor].sum())
            value_values.append(token_values[index_tensor].mean())
            rewards.append(float(examples[row].reward))
            group_ids.append(examples[row].group_id)
            valid_counts.append(len(token_indices))
            for token_index in token_indices:
                token_to_sequence_values[token_index] = sequence_index
            token_is_terminal_values[token_indices[-1]] = 1.0

        return SteeringBatch(
            log_pi=torch.stack(log_pi_values),
            log_ref=torch.stack(log_ref_values).detach(),
            values=torch.stack(value_values),
            rewards=torch.tensor(rewards, dtype=torch.float32, device=device),
            group_ids=group_ids,
            num_valid_tokens=torch.tensor(valid_counts, dtype=torch.float32, device=device),
            token_log_pi=token_log_pi,
            token_log_ref=token_log_ref,
            token_values=token_values,
            token_to_sequence=torch.tensor(token_to_sequence_values, dtype=torch.long, device=device),
            token_is_terminal=torch.tensor(token_is_terminal_values, dtype=torch.float32, device=device),
        )

    def _empty_batch(self, device: torch.device | str) -> SteeringBatch:
        trainable_param = next(param for param in self.parameters() if param.requires_grad)
        zero = trainable_param.sum() * 0.0
        return SteeringBatch(
            log_pi=zero.reshape(1),
            log_ref=zero.detach().reshape(1),
            values=zero.reshape(1),
            rewards=torch.zeros(1, device=device),
            group_ids=["empty"],
            num_valid_tokens=torch.zeros(1, device=device),
            token_log_pi=zero.reshape(1),
            token_log_ref=zero.detach().reshape(1),
            token_values=zero.reshape(1),
            token_to_sequence=torch.zeros(1, dtype=torch.long, device=device),
            token_is_terminal=torch.ones(1, device=device),
        )
