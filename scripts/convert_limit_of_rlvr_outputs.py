#!/usr/bin/env python
from __future__ import annotations

import argparse

from itds.jsonl import write_jsonl
from itds.limit_of_rlvr_io import load_and_flatten_limit_of_rlvr_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert limit-of-RLVR saved JSONL outputs to SubTB rows.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-name-or-path", default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--n-sampling", type=int, default=None)
    args = parser.parse_args()

    rows = load_and_flatten_limit_of_rlvr_jsonl(
        args.input,
        model_name_or_path=args.model_name_or_path,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        n_sampling=args.n_sampling,
    )
    count = write_jsonl(args.output, rows)
    print(f"wrote {count} flattened rows to {args.output}")


if __name__ == "__main__":
    main()
