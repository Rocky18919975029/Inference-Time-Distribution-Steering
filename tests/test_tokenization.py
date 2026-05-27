from dataclasses import dataclass

from offline_subtb.tokenization import (
    compute_boundary_positions,
    enumerate_valid_pairs,
    expected_text_span,
    sample_uniform_pairs,
    split_response_blocks,
)


@dataclass
class Encoded:
    input_ids: list[int]


class CharTokenizer:
    def __call__(self, text: str, add_special_tokens: bool = False):
        return Encoded(input_ids=[ord(char) for char in text])

    def decode(self, token_ids):
        return "".join(chr(token_id) for token_id in token_ids)


def test_response_preservation_with_empty_blocks():
    response = "first\n\nthird\n"
    blocks = split_response_blocks(response)
    assert blocks == ["first", "", "third", ""]
    assert "\n".join(blocks) == response


def test_boundary_monotonicity_and_pairs():
    tokenizer = CharTokenizer()
    result = compute_boundary_positions("prompt:", "a\nbb", tokenizer)

    assert result.boundary_positions[0] == len("prompt:")
    assert result.boundary_positions[-1] == len("prompt:a\nbb")
    assert all(a <= b for a, b in zip(result.boundary_positions, result.boundary_positions[1:]))

    pairs = enumerate_valid_pairs(result.boundary_positions)
    assert (0, 1) in pairs
    assert (0, 2) in pairs
    assert (1, 2) in pairs
    assert expected_text_span(result.blocks, 0, 2) == "a\nbb"


def test_uniform_pair_sampling_is_bounded():
    tokenizer = CharTokenizer()
    result = compute_boundary_positions("p", "a\nb\nc", tokenizer)
    pairs = sample_uniform_pairs(result.boundary_positions, max_pairs_per_trace=2)
    assert len(pairs) == 2
    assert all(m < n for m, n in pairs)
