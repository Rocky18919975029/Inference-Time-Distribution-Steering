#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from tqdm import tqdm
from transformers import AutoTokenizer


def percentile(sorted_values: list[int], q: float) -> float:
    if not sorted_values:
        return 0.0
    if q <= 0:
        return float(sorted_values[0])
    if q >= 100:
        return float(sorted_values[-1])
    pos = (len(sorted_values) - 1) * q / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = pos - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


def summarize(values: list[int]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "mean": 0.0,
            "min": 0,
            "p01": 0.0,
            "p05": 0.0,
            "p10": 0.0,
            "p25": 0.0,
            "p50": 0.0,
            "p75": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "max": 0,
        }
    xs = sorted(values)
    return {
        "count": len(xs),
        "mean": sum(xs) / len(xs),
        "min": xs[0],
        "p01": percentile(xs, 1),
        "p05": percentile(xs, 5),
        "p10": percentile(xs, 10),
        "p25": percentile(xs, 25),
        "p50": percentile(xs, 50),
        "p75": percentile(xs, 75),
        "p90": percentile(xs, 90),
        "p95": percentile(xs, 95),
        "p99": percentile(xs, 99),
        "max": xs[-1],
    }


def bucketize(values: list[int], buckets: list[int]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for value in values:
        previous = 0
        for bucket in buckets:
            if value <= bucket:
                counts[f"{previous + 1}-{bucket}"] += 1
                break
            previous = bucket
        else:
            counts[f">{buckets[-1]}"] += 1
    return dict(counts)


def read_response(row: dict[str, Any]) -> str:
    value = row.get("response", "")
    if isinstance(value, list):
        return str(value[0]) if value else ""
    return str(value)


def row_key(row: dict[str, Any], row_index: int) -> str:
    return str(row.get("problem_id", row.get("row_index", row.get("idx", row_index))))


def inspect(path: Path, tokenizer, max_rows: int, max_examples: int) -> dict[str, Any]:
    prompt_tokens: list[int] = []
    response_tokens: list[int] = []
    train_target_tokens: list[int] = []
    full_tokens: list[int] = []
    response_chars: list[int] = []
    prompt_chars: list[int] = []
    boundary_merge_deltas: list[int] = []
    reward_to_train_tokens: dict[str, list[int]] = defaultdict(list)

    empty_prompt = 0
    empty_response = 0
    bad_json = 0
    processed = 0

    shortest: list[dict[str, Any]] = []
    longest: list[dict[str, Any]] = []
    boundary_examples: list[dict[str, Any]] = []

    def add_ranked(target: list[dict[str, Any]], item: dict[str, Any], *, reverse: bool) -> None:
        target.append(item)
        target.sort(key=lambda entry: entry["train_target_tokens"], reverse=reverse)
        del target[max_examples:]

    with path.open("r", encoding="utf-8") as handle:
        iterator = tqdm(handle, desc="inspect training rows")
        for row_index, line in enumerate(iterator):
            if max_rows > 0 and processed >= max_rows:
                break
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                bad_json += 1
                continue

            prompt = str(row.get("prompt", ""))
            response = read_response(row)
            if not prompt:
                empty_prompt += 1
            if not response:
                empty_response += 1

            prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
            response_ids = tokenizer(response, add_special_tokens=False).input_ids
            full_ids = tokenizer(prompt + response, add_special_tokens=False).input_ids

            prompt_len = len(prompt_ids)
            response_len = len(response_ids)
            full_len = len(full_ids)
            train_len = max(full_len - prompt_len, 0)
            boundary_delta = response_len - train_len

            prompt_tokens.append(prompt_len)
            response_tokens.append(response_len)
            full_tokens.append(full_len)
            train_target_tokens.append(train_len)
            response_chars.append(len(response))
            prompt_chars.append(len(prompt))
            boundary_merge_deltas.append(boundary_delta)
            reward_to_train_tokens[str(row.get("reward", row.get("is_correct", "missing")))].append(train_len)

            example = {
                "row_index": row_index,
                "problem_id": row_key(row, row_index),
                "reward": row.get("reward", row.get("is_correct")),
                "prompt_tokens": prompt_len,
                "response_tokens_tokenized_alone": response_len,
                "full_tokens": full_len,
                "train_target_tokens": train_len,
                "boundary_merge_delta": boundary_delta,
                "response_chars": len(response),
                "response_preview": response[:300],
            }
            add_ranked(shortest, example, reverse=False)
            add_ranked(longest, example, reverse=True)
            if boundary_delta != 0 and len(boundary_examples) < max_examples:
                boundary_examples.append(example)

            processed += 1

    return {
        "input": str(path),
        "rows": processed,
        "bad_json_rows": bad_json,
        "empty_prompt_rows": empty_prompt,
        "empty_response_rows": empty_response,
        "prompt_tokens": summarize(prompt_tokens),
        "response_tokens_tokenized_alone": summarize(response_tokens),
        "train_target_tokens": summarize(train_target_tokens),
        "full_tokens": summarize(full_tokens),
        "prompt_chars": summarize(prompt_chars),
        "response_chars": summarize(response_chars),
        "boundary_merge_delta": summarize(boundary_merge_deltas),
        "train_target_token_buckets": bucketize(
            train_target_tokens,
            [0, 1, 2, 3, 5, 10, 20, 50, 100, 256, 512, 1024, 2048, 4096, 8192, 16384],
        ),
        "by_reward_train_target_tokens": {
            key: summarize(values) for key, values in sorted(reward_to_train_tokens.items())
        },
        "shortest_examples": shortest,
        "longest_examples": longest,
        "boundary_merge_examples": boundary_examples,
    }


def print_stats(name: str, stats: dict[str, Any]) -> None:
    print(f"\n--- {name} ---")
    for key, value in stats.items():
        print(f"{key}: {value}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect ITDS training JSONL token lengths.")
    parser.add_argument("--input", required=True, help="Training JSONL path.")
    parser.add_argument("--model", required=True, help="Tokenizer/model name or path.")
    parser.add_argument("--max-rows", type=int, default=0, help="Maximum non-empty rows to inspect; 0 means all.")
    parser.add_argument("--max-examples", type=int, default=10)
    parser.add_argument("--output", default="", help="Optional JSON summary path.")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    result = inspect(Path(args.input), tokenizer, args.max_rows, args.max_examples)

    print(f"\n=== {result['input']} ===")
    print(f"rows: {result['rows']}")
    print(f"bad_json_rows: {result['bad_json_rows']}")
    print(f"empty_prompt_rows: {result['empty_prompt_rows']}")
    print(f"empty_response_rows: {result['empty_response_rows']}")
    print_stats("train_target_tokens (what ITDS trains on)", result["train_target_tokens"])
    print_stats("response_tokens_tokenized_alone", result["response_tokens_tokenized_alone"])
    print_stats("full_tokens", result["full_tokens"])
    print_stats("boundary_merge_delta = response_tokens_alone - train_target_tokens", result["boundary_merge_delta"])

    print("\n--- train_target_token_buckets ---")
    for key, value in result["train_target_token_buckets"].items():
        print(f"{key}: {value}")

    print("\n--- by_reward train_target p50/p95/max ---")
    for key, stats in result["by_reward_train_target_tokens"].items():
        print(f"reward={key}: count={stats['count']} p50={stats['p50']} p95={stats['p95']} max={stats['max']}")

    print("\n--- shortest_examples ---")
    for item in result["shortest_examples"]:
        print(json.dumps(item, ensure_ascii=False))

    print("\n--- longest_examples ---")
    for item in result["longest_examples"]:
        print(json.dumps(item, ensure_ascii=False))

    print("\n--- boundary_merge_examples ---")
    for item in result["boundary_merge_examples"]:
        print(json.dumps(item, ensure_ascii=False))

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nwrote summary to {output}")


if __name__ == "__main__":
    main()
