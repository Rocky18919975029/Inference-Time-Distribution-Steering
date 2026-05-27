from __future__ import annotations

import argparse
import random

from transformers import AutoTokenizer

from .tokenization import compute_boundary_positions, expected_text_span, sample_uniform_pairs
from .utils import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect newline boundaries and sampled SubTB pairs.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--max-pairs-per-trace", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    rows = list(read_jsonl(args.data))[: args.limit]
    tokenizer_name = args.tokenizer or rows[0].get("model_name_or_path")
    if not tokenizer_name:
        raise ValueError("Provide --tokenizer or include model_name_or_path in the data rows")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    rng = random.Random(args.seed)

    for row in rows:
        boundaries = compute_boundary_positions(row["prompt"], row["response"], tokenizer)
        pairs = sample_uniform_pairs(boundaries.boundary_positions, args.max_pairs_per_trace, rng)
        print("=" * 80)
        print(f"problem_id={row.get('problem_id')} sample_id={row.get('sample_id')}")
        print(f"reward={row.get('reward')} ref_logprob_mean={row.get('ref_logprob_mean')}")
        print(f"num_blocks={len(boundaries.blocks)} boundary_positions={boundaries.boundary_positions}")
        print("prompt:")
        print(row["prompt"])
        print("response:")
        print(row["response"])
        for m, n in pairs:
            start = boundaries.boundary_positions[m]
            end = boundaries.boundary_positions[n]
            print("-" * 40)
            print(f"pair=({m},{n}) token_span=[{start},{end})")
            print(expected_text_span(boundaries.blocks, m, n))


if __name__ == "__main__":
    main()
