#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm
from transformers import AutoTokenizer

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from offline_subtb.tokenization import compute_boundary_positions, enumerate_valid_pairs


def percentile(sorted_values: list[int], pct: float) -> int:
    if not sorted_values:
        return 0
    index = round((len(sorted_values) - 1) * pct / 100.0)
    return sorted_values[index]


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect token length and newline SubTB pair distribution.")
    parser.add_argument("--input", required=True, help="Training JSONL, e.g. data/full_train_subtb_with_ref.jsonl")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B", help="Tokenizer name or path.")
    parser.add_argument(
        "--thresholds",
        default="2048,4096,8192,12288,16384,32768",
        help="Comma-separated token length thresholds to count.",
    )
    parser.add_argument("--max-rows", type=int, default=0, help="Inspect only the first N rows. 0 means all rows.")
    parser.add_argument("--output", default="", help="Optional JSON summary path.")
    args = parser.parse_args()

    input_path = Path(args.input)
    thresholds = [int(item) for item in args.thresholds.split(",") if item.strip()]
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    total_lengths: list[int] = []
    prompt_lengths: list[int] = []
    response_lengths: list[int] = []
    valid_pair_counts: list[int] = []
    over_threshold = {threshold: 0 for threshold in thresholds}
    zero_pair_rows = 0
    empty_response_rows = 0
    total_rows = 0

    with input_path.open("r", encoding="utf-8") as handle:
        for line in tqdm(handle, desc="inspect training rows"):
            if not line.strip():
                continue
            row = json.loads(line)
            prompt = str(row["prompt"])
            response = str(row["response"])
            if not response:
                empty_response_rows += 1

            boundaries = compute_boundary_positions(prompt, response, tokenizer)
            valid_pairs = enumerate_valid_pairs(boundaries.boundary_positions)
            total_len = boundaries.full_len
            prompt_len = boundaries.prompt_len
            response_len = total_len - prompt_len

            total_lengths.append(total_len)
            prompt_lengths.append(prompt_len)
            response_lengths.append(response_len)
            valid_pair_counts.append(len(valid_pairs))
            if not valid_pairs:
                zero_pair_rows += 1
            for threshold in thresholds:
                if total_len > threshold:
                    over_threshold[threshold] += 1

            total_rows += 1
            if args.max_rows and total_rows >= args.max_rows:
                break

    sorted_total = sorted(total_lengths)
    sorted_prompt = sorted(prompt_lengths)
    sorted_response = sorted(response_lengths)
    sorted_pairs = sorted(valid_pair_counts)

    summary = {
        "input": str(input_path),
        "model": args.model,
        "num_rows": total_rows,
        "total_length": {
            "min": sorted_total[0] if sorted_total else 0,
            "p50": percentile(sorted_total, 50),
            "p75": percentile(sorted_total, 75),
            "p90": percentile(sorted_total, 90),
            "p95": percentile(sorted_total, 95),
            "p99": percentile(sorted_total, 99),
            "max": sorted_total[-1] if sorted_total else 0,
        },
        "prompt_length": {
            "p50": percentile(sorted_prompt, 50),
            "p95": percentile(sorted_prompt, 95),
            "p99": percentile(sorted_prompt, 99),
            "max": sorted_prompt[-1] if sorted_prompt else 0,
        },
        "response_length": {
            "p50": percentile(sorted_response, 50),
            "p95": percentile(sorted_response, 95),
            "p99": percentile(sorted_response, 99),
            "max": sorted_response[-1] if sorted_response else 0,
        },
        "valid_pairs": {
            "zero_pair_rows": zero_pair_rows,
            "zero_pair_fraction": zero_pair_rows / total_rows if total_rows else 0.0,
            "empty_response_rows": empty_response_rows,
            "p50": percentile(sorted_pairs, 50),
            "p95": percentile(sorted_pairs, 95),
            "p99": percentile(sorted_pairs, 99),
            "max": sorted_pairs[-1] if sorted_pairs else 0,
        },
        "over_total_length_thresholds": {
            str(threshold): {
                "count": count,
                "fraction": count / total_rows if total_rows else 0.0,
            }
            for threshold, count in over_threshold.items()
        },
    }

    print(json.dumps(summary, indent=2))
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
