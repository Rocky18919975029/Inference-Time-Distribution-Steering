from __future__ import annotations

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


class TopKLowRankSteering(nn.Module):
    def __init__(
        self,
        model_name_or_path: str,
        *,
        top_k: int = 64,
        rank: int = 32,
        alpha: float = 1.0,
        torch_dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.top_k = top_k
        self.rank = rank
        self.alpha = alpha
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )
        for param in self.base_model.parameters():
            param.requires_grad_(False)
        hidden_size = self.base_model.config.hidden_size
        vocab_size = self.base_model.config.vocab_size
        self.state_projector = nn.Sequential(
            nn.Linear(hidden_size, rank),
            nn.Tanh(),
            nn.Linear(rank, rank),
        )
        self.token_basis = nn.Embedding(vocab_size, rank)
        self.value_head = nn.Linear(hidden_size, 1)

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
            output = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
            logits = output.logits[:, :-1, :].detach()
            hidden = output.hidden_states[-1][:, :-1, :].detach()

        log_pi_values: list[torch.Tensor] = []
        log_ref_values: list[torch.Tensor] = []
        value_values: list[torch.Tensor] = []
        rewards: list[float] = []
        group_ids: list[str] = []
        valid_counts: list[int] = []

        for row, example in enumerate(examples):
            prompt_len = len(prompt_ids[row])
            seq_len = int(attention_mask[row].sum().item())
            start = max(prompt_len - 1, 0)
            end = max(seq_len - 1, start)
            row_log_pi: list[torch.Tensor] = []
            row_log_ref: list[torch.Tensor] = []
            row_values: list[torch.Tensor] = []
            for pos in range(start, end):
                target_id = input_ids[row, pos + 1]
                base_logits = logits[row, pos]
                top_values, top_ids = torch.topk(base_logits, k=min(self.top_k, base_logits.shape[-1]))
                match = top_ids.eq(target_id).nonzero(as_tuple=False)
                if match.numel() == 0:
                    continue
                match_index = int(match[0].item())
                steering_dtype = self.state_projector[0].weight.dtype
                state = hidden[row, pos].to(dtype=steering_dtype)
                top_values = top_values.to(dtype=steering_dtype)
                u = self.state_projector(state)
                steering = (self.token_basis(top_ids) * u.unsqueeze(0)).sum(dim=-1)
                steered_logits = top_values + self.alpha * steering
                row_log_pi.append(torch.log_softmax(steered_logits, dim=-1)[match_index])
                row_log_ref.append(torch.log_softmax(top_values, dim=-1)[match_index])
                row_values.append(self.value_head(state).squeeze(-1))

            if row_log_pi:
                log_pi_values.append(torch.stack(row_log_pi).sum())
                log_ref_values.append(torch.stack(row_log_ref).sum())
                value_values.append(torch.stack(row_values).mean())
                rewards.append(float(example.reward))
                group_ids.append(example.group_id)
                valid_counts.append(len(row_log_pi))

        if not log_pi_values:
            trainable_param = next(param for param in self.parameters() if param.requires_grad)
            zero = trainable_param.sum() * 0.0
            return SteeringBatch(
                log_pi=zero.reshape(1),
                log_ref=zero.detach().reshape(1),
                values=zero.reshape(1),
                rewards=torch.zeros(1, device=device),
                group_ids=["empty"],
                num_valid_tokens=torch.zeros(1, device=device),
            )

        return SteeringBatch(
            log_pi=torch.stack(log_pi_values),
            log_ref=torch.stack(log_ref_values).detach(),
            values=torch.stack(value_values),
            rewards=torch.tensor(rewards, dtype=torch.float32, device=device),
            group_ids=group_ids,
            num_valid_tokens=torch.tensor(valid_counts, dtype=torch.float32, device=device),
        )
