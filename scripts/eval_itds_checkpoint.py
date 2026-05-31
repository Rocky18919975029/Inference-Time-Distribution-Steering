#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))
MATH_EVAL_DIR = ROOT_DIR / "limit-of-RLVR" / "math" / "examples" / "math_eval"
sys.path.insert(0, str(MATH_EVAL_DIR))

from itds.generate import generate_one
from itds.model import TopKLowRankSteering
from offline_subtb.limit_of_rlvr_io import build_qwen_boxed_prompt
from grader import math_equal
from parser import extract_answer, strip_string


def _load_eval_rows(path: Path, max_samples: int) -> list[dict]:
    if path.suffix == ".parquet":
        rows = [_normalize_parquet_row(row, int(row_index)) for row_index, row in pd.read_parquet(path).iterrows()]
    else:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows[:max_samples] if max_samples > 0 else rows


def _loads_if_json(value: Any) -> Any:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("[") or stripped.startswith("{"):
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                return value
    return value


def _normalize_question(row: pd.Series) -> str:
    for key in ("question", "problem", "input", "Question"):
        if key in row and pd.notna(row[key]):
            return str(row[key])
    extra_info = _loads_if_json(row.get("extra_info"))
    if isinstance(extra_info, dict) and extra_info.get("question") is not None:
        return str(extra_info["question"])
    return ""


def _normalize_prompt(value: Any, question: str) -> str:
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


def _normalize_ground_truth(row: pd.Series) -> str:
    for column in ("gt", "gt_answer", "answer", "target"):
        if column in row and pd.notna(row[column]):
            return strip_string(str(row[column]))
    reward_model = _loads_if_json(row.get("reward_model"))
    if isinstance(reward_model, dict) and reward_model.get("ground_truth") is not None:
        return strip_string(str(reward_model["ground_truth"]))
    extra_info = _loads_if_json(row.get("extra_info"))
    if isinstance(extra_info, dict) and extra_info.get("answer") is not None:
        return strip_string(str(extra_info["answer"]))
    return ""


def _normalize_problem_id(row: pd.Series, fallback: int) -> int:
    if "__problem_id" in row and pd.notna(row["__problem_id"]):
        return int(row["__problem_id"])
    extra_info = _loads_if_json(row.get("extra_info"))
    if isinstance(extra_info, dict) and extra_info.get("index") is not None:
        try:
            return int(extra_info["index"])
        except (TypeError, ValueError):
            pass
    return int(fallback)


def _normalize_parquet_row(row: pd.Series, row_index: int) -> dict:
    question = _normalize_question(row)
    return {
        "idx": _normalize_problem_id(row, row_index),
        "question": question,
        "prompt": _normalize_prompt(row.get("prompt"), question),
        "gt": _normalize_ground_truth(row),
        "subject": row.get("subject"),
        "level": row.get("level"),
        "data_source": row.get("data_source"),
    }


def _question(row: dict) -> str:
    for key in ("question", "problem", "input", "Question"):
        if isinstance(row.get(key), str):
            return row[key]
    return ""


def _ground_truth(row: dict) -> str:
    for key in ("gt", "gt_answer", "answer", "target"):
        if row.get(key) is not None:
            return strip_string(str(row[key]))
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an ITDS steering checkpoint with limit-of-RLVR grading.")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--base-only", action="store_true", help="Evaluate the frozen base model through the ITDS decoding path.")
    parser.add_argument("--eval-data", default=str(ROOT_DIR / "data" / "test.parquet"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name-or-path", default="")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--actor-depth", type=int, default=None)
    parser.add_argument("--critic-depth", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--token-basis-init-std", type=float, default=None)
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    args = parser.parse_args()

    if not args.base_only and not args.checkpoint:
        parser.error("--checkpoint is required unless --base-only is set")

    checkpoint = Path(args.checkpoint).resolve() if args.checkpoint else None
    payload: dict[str, Any] = {}
    config: dict[str, Any] = {}
    if checkpoint is not None:
        payload = torch.load(checkpoint / "steering.pt", map_location="cpu")
        config = payload.get("config", {})
    model_name = args.model_name_or_path or config.get("model_name_or_path") or "Qwen/Qwen2.5-7B"
    top_k = args.top_k if args.top_k is not None else int(config.get("top_k", 64))
    rank = args.rank if args.rank is not None else int(config.get("rank", 32))
    actor_depth = args.actor_depth if args.actor_depth is not None else int(config.get("actor_depth", 10))
    critic_depth = args.critic_depth if args.critic_depth is not None else int(config.get("critic_depth", 10))
    alpha = 0.0 if args.base_only else (args.alpha if args.alpha is not None else float(config.get("alpha", 1.0)))
    token_basis_init_std = (
        args.token_basis_init_std if args.token_basis_init_std is not None else float(config.get("token_basis_init_std", 1e-3))
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else None
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = TopKLowRankSteering(
        model_name,
        top_k=top_k,
        rank=rank,
        actor_depth=actor_depth,
        critic_depth=critic_depth,
        alpha=alpha,
        token_basis_init_std=token_basis_init_std,
        torch_dtype=dtype,
    )
    if not args.base_only:
        model.load_steering_state_dict(payload["steering"])
    model.to(device)
    model.eval()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _load_eval_rows(Path(args.eval_data), args.max_samples)
    eval_name = "base_only" if args.base_only else checkpoint.name
    output_path = output_dir / f"{eval_name}_itds_eval.jsonl"
    correct = 0

    with output_path.open("w", encoding="utf-8") as handle:
        for idx, row in enumerate(tqdm(rows, desc="itds eval")):
            question = _question(row)
            gt = _ground_truth(row)
            prompt = row.get("prompt") if isinstance(row.get("prompt"), str) and row.get("prompt") else build_qwen_boxed_prompt(question)
            if not question or not gt:
                raise ValueError(
                    f"Eval row {row.get('idx', idx)!r} normalized to empty question or ground truth. "
                    f"Available keys: {sorted(row.keys())}"
                )
            response = generate_one(
                model,
                tokenizer,
                prompt,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                device=device,
            )
            pred = strip_string(extract_answer(response, "math-oai"))
            score = math_equal(pred, gt)
            correct += int(score)
            handle.write(
                json.dumps(
                    {
                        "idx": row.get("idx", idx),
                        "question": question,
                        "gt": gt,
                        "pred": [pred],
                        "score": [bool(score)],
                        "code": [response],
                        "finish_reason": ["stop"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            handle.flush()

    summary = {
        "checkpoint": "base_only" if args.base_only else str(checkpoint),
        "model_name_or_path": model_name,
        "top_k": top_k,
        "rank": rank,
        "actor_depth": actor_depth,
        "critic_depth": critic_depth,
        "alpha": alpha,
        "token_basis_init_std": token_basis_init_std,
        "accuracy": correct / len(rows) if rows else 0.0,
        "correct": correct,
        "total": len(rows),
        "output_path": str(output_path),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
