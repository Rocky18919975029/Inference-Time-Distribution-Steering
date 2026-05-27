#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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
        rows = pd.read_parquet(path).to_dict("records")
    else:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows[:max_samples] if max_samples > 0 else rows


def _question(row: dict) -> str:
    for key in ("question", "problem", "input", "Question"):
        if isinstance(row.get(key), str):
            return row[key]
    return ""


def _ground_truth(row: dict) -> str:
    value = row.get("gt", row.get("answer", ""))
    return strip_string(str(value))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an ITDS steering checkpoint with limit-of-RLVR grading.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--eval-data", default=str(ROOT_DIR / "data" / "test.parquet"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name-or-path", default="")
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint).resolve()
    payload = torch.load(checkpoint / "steering.pt", map_location="cpu")
    config = payload.get("config", {})
    model_name = args.model_name_or_path or config.get("model_name_or_path")
    top_k = args.top_k or int(config.get("top_k", 64))
    rank = args.rank or int(config.get("rank", 32))
    alpha = args.alpha if args.alpha is not None else float(config.get("alpha", 1.0))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else None
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = TopKLowRankSteering(model_name, top_k=top_k, rank=rank, alpha=alpha, torch_dtype=dtype)
    model.load_steering_state_dict(payload["steering"])
    model.to(device)
    model.eval()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _load_eval_rows(Path(args.eval_data), args.max_samples)
    output_path = output_dir / f"{checkpoint.name}_itds_eval.jsonl"
    correct = 0

    with output_path.open("w", encoding="utf-8") as handle:
        for idx, row in enumerate(tqdm(rows, desc="itds eval")):
            question = _question(row)
            gt = _ground_truth(row)
            prompt = build_qwen_boxed_prompt(question)
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

    summary = {
        "checkpoint": str(checkpoint),
        "accuracy": correct / len(rows) if rows else 0.0,
        "correct": correct,
        "total": len(rows),
        "output_path": str(output_path),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
