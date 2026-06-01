#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if q <= 0:
        return sorted_values[0]
    if q >= 100:
        return sorted_values[-1]
    pos = (len(sorted_values) - 1) * q / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = pos - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


def summarize(values: list[int]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "mean": 0,
            "min": 0,
            "p01": 0,
            "p05": 0,
            "p10": 0,
            "p25": 0,
            "p50": 0,
            "p75": 0,
            "p90": 0,
            "p95": 0,
            "p99": 0,
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

    for x in values:
        placed = False
        prev = 0
        for b in buckets:
            if x <= b:
                counts[f"{prev + 1}-{b}"] += 1
                placed = True
                break
            prev = b
        if not placed:
            counts[f">{buckets[-1]}"] += 1

    return dict(counts)


def as_boolish(value: Any) -> str:
    if value is True:
        return "true"
    if value is False:
        return "false"
    if value is None:
        return "none"
    return str(value)


def inspect_file(path: Path, max_examples: int, short_token_threshold: int, short_char_threshold: int) -> dict[str, Any]:
    char_lengths: list[int] = []
    token_lengths: list[int] = []
    missing_token_count = 0
    empty_response_count = 0
    bad_json_count = 0

    by_reward_chars: dict[str, list[int]] = defaultdict(list)
    by_reward_tokens: dict[str, list[int]] = defaultdict(list)

    by_correct_chars: dict[str, list[int]] = defaultdict(list)
    by_correct_tokens: dict[str, list[int]] = defaultdict(list)

    short_examples: list[dict[str, Any]] = []
    response_counter: Counter[str] = Counter()

    total = 0

    with path.open("r", encoding="utf-8") as handle:
        for line_idx, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue

            try:
                row = json.loads(line)
            except Exception as exc:
                bad_json_count += 1
                if len(short_examples) < max_examples:
                    short_examples.append(
                        {
                            "line_idx": line_idx,
                            "error": f"bad json: {exc}",
                        }
                    )
                continue

            total += 1

            response = str(row.get("response", ""))
            char_len = len(response)
            char_lengths.append(char_len)
            response_counter[response] += 1

            if not response:
                empty_response_count += 1

            token_len = row.get("response_num_tokens")
            has_token_len = isinstance(token_len, int)

            if has_token_len:
                token_lengths.append(token_len)
            else:
                missing_token_count += 1

            reward_key = str(row.get("reward", "missing"))
            correct_key = as_boolish(row.get("is_correct", row.get("score", None)))

            by_reward_chars[reward_key].append(char_len)
            by_correct_chars[correct_key].append(char_len)

            if has_token_len:
                by_reward_tokens[reward_key].append(token_len)
                by_correct_tokens[correct_key].append(token_len)

            is_short = False
            if has_token_len and token_len <= short_token_threshold:
                is_short = True
            if char_len <= short_char_threshold:
                is_short = True

            if is_short and len(short_examples) < max_examples:
                short_examples.append(
                    {
                        "line_idx": line_idx,
                        "problem_id": row.get("problem_id"),
                        "sample_id": row.get("sample_id"),
                        "reward": row.get("reward"),
                        "is_correct": row.get("is_correct"),
                        "response_num_tokens": token_len,
                        "response_char_length": char_len,
                        "response_preview": response[:300],
                        "extracted_answer": row.get("extracted_answer"),
                        "ground_truth": row.get("ground_truth"),
                    }
                )

    duplicate_top = [
        {
            "response": resp,
            "count": count,
            "char_length": len(resp),
        }
        for resp, count in response_counter.most_common(20)
    ]

    result = {
        "path": str(path),
        "total_rows": total,
        "bad_json_rows": bad_json_count,
        "empty_response_rows": empty_response_count,
        "missing_response_num_tokens_rows": missing_token_count,
        "char_length": summarize(char_lengths),
        "token_length": summarize(token_lengths),
        "char_length_buckets": bucketize(
            char_lengths,
            [0, 1, 2, 3, 5, 10, 20, 50, 100, 256, 512, 1024, 2048, 4096, 8192],
        ),
        "token_length_buckets": bucketize(
            token_lengths,
            [0, 1, 2, 3, 5, 10, 20, 50, 100, 256, 512, 1024, 2048, 4096, 8192],
        ),
        "by_reward": {
            reward: {
                "char_length": summarize(vals),
                "token_length": summarize(by_reward_tokens.get(reward, [])),
            }
            for reward, vals in sorted(by_reward_chars.items())
        },
        "by_correctness": {
            key: {
                "char_length": summarize(vals),
                "token_length": summarize(by_correct_tokens.get(key, [])),
            }
            for key, vals in sorted(by_correct_chars.items())
        },
        "top_duplicate_responses": duplicate_top,
        "short_examples": short_examples,
    }

    return result


def print_summary(result: dict[str, Any]) -> None:
    print(f"\n=== {result['path']} ===")
    print(f"total_rows: {result['total_rows']}")
    print(f"bad_json_rows: {result['bad_json_rows']}")
    print(f"empty_response_rows: {result['empty_response_rows']}")
    print(f"missing_response_num_tokens_rows: {result['missing_response_num_tokens_rows']}")

    print("\n--- char_length ---")
    for k, v in result["char_length"].items():
        print(f"{k}: {v}")

    print("\n--- token_length ---")
    for k, v in result["token_length"].items():
        print(f"{k}: {v}")

    print("\n--- token_length_buckets ---")
    for k, v in result["token_length_buckets"].items():
        print(f"{k}: {v}")

    print("\n--- char_length_buckets ---")
    for k, v in result["char_length_buckets"].items():
        print(f"{k}: {v}")

    print("\n--- by_reward token p50/p95/max ---")
    for reward, item in result["by_reward"].items():
        stats = item["token_length"]
        print(
            f"reward={reward}: "
            f"count={stats['count']} "
            f"p50={stats['p50']} "
            f"p95={stats['p95']} "
            f"max={stats['max']}"
        )

    print("\n--- top_duplicate_responses ---")
    for item in result["top_duplicate_responses"][:10]:
        print(
            f"count={item['count']} "
            f"char_length={item['char_length']} "
            f"response={item['response'][:120]!r}"
        )

    print("\n--- short_examples ---")
    for ex in result["short_examples"][:10]:
        print(json.dumps(ex, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect response length distribution for converted training JSONL.")
    parser.add_argument("--input", required=True, help="JSONL file to inspect.")
    parser.add_argument("--compare", default="", help="Optional second JSONL file to compare.")
    parser.add_argument("--output", default="", help="Optional JSON summary output path.")
    parser.add_argument("--max-examples", type=int, default=20)
    parser.add_argument("--short-token-threshold", type=int, default=10)
    parser.add_argument("--short-char-threshold", type=int, default=50)

    args = parser.parse_args()

    results = [
        inspect_file(
            Path(args.input),
            max_examples=args.max_examples,
            short_token_threshold=args.short_token_threshold,
            short_char_threshold=args.short_char_threshold,
        )
    ]

    if args.compare:
        results.append(
            inspect_file(
                Path(args.compare),
                max_examples=args.max_examples,
                short_token_threshold=args.short_token_threshold,
                short_char_threshold=args.short_char_threshold,
            )
        )

    for result in results:
        print_summary(result)

    if len(results) == 2:
        a, b = results
        print("\n=== comparison ===")
        print(f"input:   {a['path']}")
        print(f"compare: {b['path']}")
        print(f"rows: {a['total_rows']} vs {b['total_rows']}")
        print(f"token p50: {a['token_length']['p50']} vs {b['token_length']['p50']}")
        print(f"token p95: {a['token_length']['p95']} vs {b['token_length']['p95']}")
        print(f"token mean: {a['token_length']['mean']} vs {b['token_length']['mean']}")
        print(f"char p50: {a['char_length']['p50']} vs {b['char_length']['p50']}")
        print(f"char p95: {a['char_length']['p95']} vs {b['char_length']['p95']}")

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nwrote summary to {out}")


if __name__ == "__main__":
    main()