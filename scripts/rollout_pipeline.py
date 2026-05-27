#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from offline_subtb.limit_of_rlvr_io import build_qwen_boxed_prompt
from offline_subtb.utils import read_jsonl, write_jsonl

def _loads_if_json(value: Any) -> Any:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("[") or stripped.startswith("{"):
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                return value
    return value


def normalize_prompt(value: Any, question: str) -> str:
    value = _loads_if_json(value)
    if isinstance(value, str) and value:
        return value
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list) and value:
        first = value[0]
        if isinstance(first, dict) and isinstance(first.get("content"), str):
            return first["content"]
        if isinstance(first, str):
            return first
    if isinstance(value, dict) and isinstance(value.get("content"), str):
        return value["content"]
    return build_qwen_boxed_prompt(question)


def normalize_ground_truth(row: pd.Series) -> str:
    for column in ("gt_answer", "answer", "target"):
        if column in row and pd.notna(row[column]):
            return str(row[column])
    reward_model = _loads_if_json(row.get("reward_model"))
    if isinstance(reward_model, dict) and reward_model.get("ground_truth") is not None:
        return str(reward_model["ground_truth"])
    extra_info = _loads_if_json(row.get("extra_info"))
    if isinstance(extra_info, dict) and extra_info.get("answer") is not None:
        return str(extra_info["answer"])
    return ""


def normalize_problem_id(row: pd.Series, fallback: int) -> int:
    extra_info = _loads_if_json(row.get("extra_info"))
    if isinstance(extra_info, dict) and extra_info.get("index") is not None:
        try:
            return int(extra_info["index"])
        except (TypeError, ValueError):
            pass
    return int(fallback)


def prepare_mini_and_shards(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    output_path = Path(args.output)
    shard_dir = Path(args.shard_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shard_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(input_path)
    mini = df.sample(frac=args.fraction, random_state=args.seed).sort_index().copy()
    mini["__problem_id"] = [normalize_problem_id(row, int(index)) for index, row in mini.iterrows()]
    mini.to_parquet(output_path, index=False)

    shard_size = math.ceil(len(mini) / args.num_shards)
    manifest = {
        "input": str(input_path),
        "mini_output": str(output_path),
        "seed": args.seed,
        "fraction": args.fraction,
        "num_rows_raw": int(len(df)),
        "num_rows_mini": int(len(mini)),
        "num_shards": args.num_shards,
        "shards": [],
    }
    for shard_id in range(args.num_shards):
        start = shard_id * shard_size
        end = min(start + shard_size, len(mini))
        shard = mini.iloc[start:end].copy()
        shard_path = shard_dir / f"mini_train_shard_{shard_id}.parquet"
        shard.to_parquet(shard_path, index=False)
        manifest["shards"].append({"shard_id": shard_id, "path": str(shard_path), "num_rows": int(len(shard))})

    manifest_path = shard_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


def rows_from_parquet(path: str | Path) -> Iterable[dict[str, Any]]:
    df = pd.read_parquet(path)
    for row_index, row in df.iterrows():
        question = str(row.get("question", ""))
        prompt = normalize_prompt(row.get("prompt"), question)
        ground_truth = normalize_ground_truth(row)
        problem_id = int(row["__problem_id"]) if "__problem_id" in row and pd.notna(row["__problem_id"]) else int(row_index)
        yield {
            "problem_id": problem_id,
            "question": question,
            "prompt": prompt,
            "ground_truth": ground_truth,
            "subject": row.get("subject"),
            "level": row.get("level"),
            "data_source": row.get("data_source"),
        }


def export_limit_of_rlvr_data(args: argparse.Namespace) -> None:
    output_data_dir = Path(args.output_data_dir)
    manifest = {
        "output_data_dir": str(output_data_dir),
        "benchmark_prefix": args.benchmark_prefix,
        "num_shards": args.num_shards,
        "shards": [],
    }
    for shard_id in range(args.num_shards):
        shard_path = Path(args.shard_dir) / f"mini_train_shard_{shard_id}.parquet"
        benchmark = f"{args.benchmark_prefix}_{shard_id}"
        benchmark_dir = output_data_dir / benchmark
        benchmark_dir.mkdir(parents=True, exist_ok=True)
        output_path = benchmark_dir / "test.jsonl"

        count = 0
        with output_path.open("w", encoding="utf-8") as handle:
            for row in rows_from_parquet(shard_path):
                item = {
                    "idx": row["problem_id"],
                    "question": row["question"],
                    "gt_cot": None,
                    "gt": row["ground_truth"],
                    "answer": row["ground_truth"],
                    "level": row["level"],
                    "subject": row["subject"],
                    "data_source": row["data_source"],
                }
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
                count += 1
        manifest["shards"].append(
            {
                "shard_id": shard_id,
                "benchmark": benchmark,
                "path": str(output_path),
                "num_rows": count,
            }
        )

    manifest_path = output_data_dir / f"{args.benchmark_prefix}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


def merge_rollout_shards(args: argparse.Namespace) -> None:
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shard_paths = sorted(Path(args.rollout_dir).glob("shard_*.jsonl"))
    rows: list[dict] = []
    for shard_path in shard_paths:
        rows.extend(read_jsonl(shard_path))
    rows.sort(key=lambda row: int(row["idx"]))
    count = write_jsonl(output_path, rows)
    print(f"merged {len(shard_paths)} shards and wrote {count} problem rows to {output_path}")


def merge_limit_of_rlvr_outputs(args: argparse.Namespace) -> None:
    output_root = Path(args.output_root)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    shard_paths: list[str] = []
    for shard_id in range(args.num_shards):
        benchmark = f"{args.benchmark_prefix}_{shard_id}"
        shard_path = (
            output_root
            / benchmark
            / f"test_{args.template}_-1_seed{args.seed}_t{args.temperature}_s0_e-1.jsonl"
        )
        if not shard_path.exists():
            raise FileNotFoundError(f"Missing limit-of-RLVR output shard: {shard_path}")
        rows.extend(read_jsonl(shard_path))
        shard_paths.append(str(shard_path))
    rows.sort(key=lambda row: int(row["idx"]))
    count = write_jsonl(output_path, rows)
    print(json.dumps({"merged_rows": count, "output": str(output_path), "shards": shard_paths}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare, shard, rollout, and merge mini training data.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--input", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--shard-dir", required=True)
    prepare.add_argument("--fraction", type=float, default=0.05)
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--num-shards", type=int, default=4)
    prepare.set_defaults(func=prepare_mini_and_shards)

    export_limit = subparsers.add_parser("export-limit-of-rlvr-data")
    export_limit.add_argument("--shard-dir", required=True)
    export_limit.add_argument("--output-data-dir", required=True)
    export_limit.add_argument("--benchmark-prefix", default="mini_train_shard")
    export_limit.add_argument("--num-shards", type=int, default=4)
    export_limit.set_defaults(func=export_limit_of_rlvr_data)

    merge = subparsers.add_parser("merge")
    merge.add_argument("--rollout-dir", required=True)
    merge.add_argument("--output", required=True)
    merge.set_defaults(func=merge_rollout_shards)

    merge_limit = subparsers.add_parser("merge-limit-of-rlvr-outputs")
    merge_limit.add_argument("--output-root", required=True)
    merge_limit.add_argument("--output", required=True)
    merge_limit.add_argument("--benchmark-prefix", default="mini_train_shard")
    merge_limit.add_argument("--num-shards", type=int, default=4)
    merge_limit.add_argument("--template", default="qwen-boxed")
    merge_limit.add_argument("--seed", type=int, default=42)
    merge_limit.add_argument("--temperature", type=str, default="0.6")
    merge_limit.set_defaults(func=merge_limit_of_rlvr_outputs)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
