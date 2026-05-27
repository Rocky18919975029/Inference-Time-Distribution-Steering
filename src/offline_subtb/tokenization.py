from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Protocol


class TokenizerLike(Protocol):
    def __call__(self, text: str, add_special_tokens: bool = False) -> object: ...

    def decode(self, token_ids: list[int]) -> str: ...


@dataclass(frozen=True)
class BoundaryResult:
    blocks: list[str]
    boundary_positions: list[int]
    prompt_len: int
    full_len: int


def _input_ids(encoded: object) -> list[int]:
    ids = getattr(encoded, "input_ids", None)
    if ids is None and isinstance(encoded, dict):
        ids = encoded["input_ids"]
    if ids is None:
        raise TypeError("Tokenizer output must expose input_ids")
    return list(ids)


def split_response_blocks(response: str) -> list[str]:
    blocks = response.split("\n")
    assert "\n".join(blocks) == response
    return blocks


def compute_boundary_positions(prompt: str, response: str, tokenizer: TokenizerLike) -> BoundaryResult:
    blocks = split_response_blocks(response)
    prompt_len = len(_input_ids(tokenizer(prompt, add_special_tokens=False)))
    full_len = len(_input_ids(tokenizer(prompt + response, add_special_tokens=False)))

    boundary_positions: list[int] = []
    for i in range(len(blocks) + 1):
        response_prefix = "\n".join(blocks[:i])
        prefix_ids = _input_ids(tokenizer(prompt + response_prefix, add_special_tokens=False))
        boundary_positions.append(len(prefix_ids))

    if boundary_positions[0] != prompt_len:
        raise ValueError(f"Boundary[0]={boundary_positions[0]} does not match prompt_len={prompt_len}")
    if boundary_positions[-1] != full_len:
        raise ValueError(f"Boundary[-1]={boundary_positions[-1]} does not match full_len={full_len}")
    if any(a > b for a, b in zip(boundary_positions, boundary_positions[1:])):
        raise ValueError(f"Boundary positions are not monotonic: {boundary_positions}")

    return BoundaryResult(blocks=blocks, boundary_positions=boundary_positions, prompt_len=prompt_len, full_len=full_len)


def enumerate_valid_pairs(boundary_positions: list[int]) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    n_blocks = len(boundary_positions) - 1
    for m in range(n_blocks + 1):
        for n in range(m + 1, n_blocks + 1):
            if boundary_positions[n] > boundary_positions[m]:
                pairs.append((m, n))
    return pairs


def sample_uniform_pairs(
    boundary_positions: list[int],
    max_pairs_per_trace: int,
    rng: random.Random | None = None,
) -> list[tuple[int, int]]:
    valid_pairs = enumerate_valid_pairs(boundary_positions)
    if not valid_pairs:
        return []
    rng = rng or random
    num_pairs = min(max_pairs_per_trace, len(valid_pairs))
    return rng.sample(valid_pairs, num_pairs)


def expected_text_span(blocks: list[str], m: int, n: int) -> str:
    return "\n".join(blocks[m:n])
