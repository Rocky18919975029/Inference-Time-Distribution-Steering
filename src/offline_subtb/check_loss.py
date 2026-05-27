from __future__ import annotations

import random
from dataclasses import dataclass

import torch
from torch import nn

from .dataset import TrainingExample
from .loss import compute_newline_subtb_loss


@dataclass
class TinyEncoding:
    input_ids: list[int] | torch.Tensor
    attention_mask: torch.Tensor | None = None


class CharacterTokenizer:
    def __call__(self, text: str, add_special_tokens: bool = False, return_tensors: str | None = None) -> TinyEncoding:
        ids = [ord(char) % 127 for char in text]
        if return_tensors == "pt":
            tensor = torch.tensor([ids], dtype=torch.long)
            return TinyEncoding(input_ids=tensor, attention_mask=torch.ones_like(tensor))
        return TinyEncoding(input_ids=ids)

    def decode(self, token_ids: list[int]) -> str:
        return "".join(chr(token_id) for token_id in token_ids)


class TinyPolicy(nn.Module):
    def __init__(self, vocab_size: int = 127, hidden_size: int = 16):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size)
        self.flow_head = nn.Linear(hidden_size, 1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None):
        hidden = self.embed(input_ids)
        return type(
            "TinyOutput",
            (),
            {
                "logits": self.lm_head(hidden),
                "flow_values": self.flow_head(hidden).squeeze(-1),
            },
        )()


def main() -> None:
    tokenizer = CharacterTokenizer()
    model = TinyPolicy()
    example = TrainingExample(
        prompt="Question: 1+1?\nAnswer:",
        response="We compute 1+1.\n\\boxed{2}",
        reward=1.0,
        ref_logprob_sum=-10.0,
        ref_logprob_mean=-0.5,
        response_num_tokens=20,
        metadata={"problem_id": 0, "sample_id": 0},
    )
    output = compute_newline_subtb_loss(
        model=model,
        tokenizer=tokenizer,
        examples=[example],
        alpha=1.0,
        beta=1.0,
        lambda_subtb=1.0,
        max_pairs_per_trace=16,
        rng=random.Random(1),
        device="cpu",
    )
    output.loss.backward()
    print(f"loss={output.diagnostics['loss']:.6f}")
    for key, value in output.diagnostics.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
