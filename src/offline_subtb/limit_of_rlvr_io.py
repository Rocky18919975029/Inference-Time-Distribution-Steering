from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .utils import read_jsonl


QWEN_BOXED_TEMPLATE = (
    "<|im_start|>system\n"
    "You are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\n"
    "{input}\n"
    "Please reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n"
    "<|im_start|>assistant\n"
)


@dataclass(frozen=True)
class RolloutMetadata:
    benchmark: str | None
    template: str | None
    seed: int | None
    temperature: float | None


def build_qwen_boxed_prompt(question: str) -> str:
    return QWEN_BOXED_TEMPLATE.format(input=question)


def parse_limit_of_rlvr_path(path: str | Path) -> RolloutMetadata:
    file_path = Path(path)
    benchmark = file_path.parent.name if file_path.parent.name else None
    match = re.match(
        r"test_(?P<template>.+?)_-1_seed(?P<seed>\d+)_t(?P<temperature>[-+0-9.]+)_s\d+_e-?\d+\.jsonl$",
        file_path.name,
    )
    if not match:
        return RolloutMetadata(benchmark=benchmark, template=None, seed=None, temperature=None)
    return RolloutMetadata(
        benchmark=benchmark,
        template=match.group("template"),
        seed=int(match.group("seed")),
        temperature=float(match.group("temperature")),
    )


def _as_list(value: Any, field: str, row_id: Any) -> list:
    if isinstance(value, list):
        return value
    raise ValueError(f"Row {row_id!r} field {field!r} must be a list, got {type(value).__name__}")


def _optional_sample_list(value: Any, field: str, row_id: Any, expected_len: int) -> list[Any] | None:
    if value is None:
        return None
    items = _as_list(value, field, row_id)
    if len(items) != expected_len:
        raise ValueError(
            f"Row {row_id!r} field {field!r} has {len(items)} samples, expected {expected_len}"
        )
    return items


def flatten_limit_of_rlvr_rows(
    rows: Iterable[dict],
    *,
    source_path: str | Path | None = None,
    model_name_or_path: str | None = None,
    top_p: float | None = None,
    max_tokens: int | None = None,
    n_sampling: int | None = None,
) -> list[dict]:
    metadata = parse_limit_of_rlvr_path(source_path) if source_path else RolloutMetadata(None, None, None, None)
    flattened: list[dict] = []

    for row_index, row in enumerate(rows):
        problem_id = row.get("idx", row_index)
        question = row.get("question")
        if not isinstance(question, str):
            raise ValueError(f"Row {problem_id!r} is missing string field 'question'")

        predictions = _as_list(row.get("pred"), "pred", problem_id)
        scores = _as_list(row.get("score"), "score", problem_id)
        full_responses = _optional_sample_list(row.get("code"), "code", problem_id, len(predictions))
        finish_reasons = row.get("finish_reason", [None] * len(predictions))
        finish_reasons = _as_list(finish_reasons, "finish_reason", problem_id)

        if not (len(predictions) == len(scores) == len(finish_reasons)):
            raise ValueError(
                f"Row {problem_id!r} has mismatched sample counts: "
                f"pred={len(predictions)} score={len(scores)} finish_reason={len(finish_reasons)}"
            )

        prompt = row.get("prompt")
        if not isinstance(prompt, str):
            prompt = build_qwen_boxed_prompt(question)

        for sample_id, (prediction, score, finish_reason) in enumerate(
            zip(predictions, scores, finish_reasons, strict=True)
        ):
            full_response = full_responses[sample_id] if full_responses is not None else prediction
            reward = float(bool(score)) if isinstance(score, bool) else float(score)
            flattened.append(
                {
                    "problem_id": problem_id,
                    "row_index": row_index,
                    "sample_id": sample_id,
                    "benchmark": metadata.benchmark,
                    "question": question,
                    "prompt": prompt,
                    "response": "" if full_response is None else str(full_response),
                    "extracted_answer": "" if prediction is None else str(prediction),
                    "ground_truth": row.get("gt"),
                    "is_correct": bool(score),
                    "reward": reward,
                    "verify_error": None,
                    "finish_reason": finish_reason,
                    "sampling_seed": metadata.seed,
                    "n_sampling": n_sampling or len(predictions),
                    "temperature": metadata.temperature,
                    "top_p": top_p,
                    "max_tokens": max_tokens,
                    "template": metadata.template or "qwen-boxed",
                    "model_name_or_path": model_name_or_path,
                }
            )

    return flattened


def load_and_flatten_limit_of_rlvr_jsonl(path: str | Path, **kwargs: Any) -> list[dict]:
    return flatten_limit_of_rlvr_rows(read_jsonl(path), source_path=path, **kwargs)
