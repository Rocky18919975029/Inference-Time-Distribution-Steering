#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from offline_subtb.limit_of_rlvr_io import build_qwen_boxed_prompt


def _ground_truth(row: dict[str, Any]) -> str:
    for key in ("gt", "ground_truth", "gt_answer", "answer", "target"):
        value = row.get(key)
        if value is not None:
            return str(value)
    return ""


def _prompt(row: dict[str, Any], question: str) -> str:
    value = row.get("prompt")
    return value if isinstance(value, str) and value else build_qwen_boxed_prompt(question)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a prompt-level eval JSONL from rollout training JSONL.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-prompts", type=int, default=0)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    seen: set[str] = set()
    written = 0
    read_rows = 0
    skipped_missing = 0

    with input_path.open("r", encoding="utf-8") as source, output_path.open("w", encoding="utf-8") as sink:
        for line in source:
            if not line.strip():
                continue
            read_rows += 1
            row = json.loads(line)
            problem_id = str(row.get("problem_id", row.get("idx", row.get("row_index", read_rows - 1))))
            if problem_id in seen:
                continue
            question = str(row.get("question", row.get("problem", row.get("input", ""))))
            gt = _ground_truth(row)
            if not question or not gt:
                skipped_missing += 1
                continue
            seen.add(problem_id)
            sink.write(
                json.dumps(
                    {
                        "idx": problem_id,
                        "question": question,
                        "prompt": _prompt(row, question),
                        "gt": gt,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            written += 1
            if args.max_prompts > 0 and written >= args.max_prompts:
                break

    print(
        json.dumps(
            {
                "input": str(input_path),
                "output": str(output_path),
                "read_rows": read_rows,
                "unique_prompts_written": written,
                "skipped_missing_question_or_gt": skipped_missing,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
