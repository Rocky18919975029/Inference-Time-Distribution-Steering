#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import subprocess
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
from itds.limit_of_rlvr_io import build_qwen_boxed_prompt
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


def _eval_name(args: argparse.Namespace, checkpoint: Path | None) -> str:
    return "base_only" if args.base_only else checkpoint.name


def _shard_ranges(total: int, num_shards: int) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    for shard_index in range(num_shards):
        start = total * shard_index // num_shards
        end = total * (shard_index + 1) // num_shards
        ranges.append((start, end))
    return ranges


def _optional_cli_args(args: argparse.Namespace) -> list[str]:
    cli: list[str] = []
    optional_values = {
        "--model-name-or-path": args.model_name_or_path,
        "--top-k": args.top_k,
        "--rank": args.rank,
        "--actor-depth": args.actor_depth,
        "--critic-depth": args.critic_depth,
        "--alpha": args.alpha,
        "--token-basis-init-std": args.token_basis_init_std,
    }
    for key, value in optional_values.items():
        if value is not None and value != "":
            cli.extend([key, str(value)])
    return cli


def _gpu_ids_for_shards(num_gpus: int) -> list[str]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible:
        gpu_ids = [item.strip() for item in visible.split(",") if item.strip()]
        if len(gpu_ids) >= num_gpus:
            return gpu_ids[:num_gpus]
    return [str(index) for index in range(num_gpus)]


def _merge_shards(output_dir: Path, eval_name: str, num_shards: int, summary_base: dict[str, Any]) -> dict[str, Any]:
    merged_output = output_dir / f"{eval_name}_itds_eval.jsonl"
    correct = 0
    total = 0
    shard_summaries: list[dict[str, Any]] = []

    with merged_output.open("w", encoding="utf-8") as out_handle:
        for shard_index in range(num_shards):
            shard_summary_path = output_dir / f"summary_shard_{shard_index}.json"
            if not shard_summary_path.exists():
                raise FileNotFoundError(f"Missing shard summary: {shard_summary_path}")
            shard_summary = json.loads(shard_summary_path.read_text(encoding="utf-8"))
            shard_summaries.append(shard_summary)
            correct += int(shard_summary.get("correct", 0))
            total += int(shard_summary.get("total", 0))

            shard_output = Path(shard_summary["output_path"])
            if not shard_output.exists():
                raise FileNotFoundError(f"Missing shard output: {shard_output}")
            with shard_output.open("r", encoding="utf-8") as shard_handle:
                for line in shard_handle:
                    out_handle.write(line)

    first_shard = shard_summaries[0] if shard_summaries else {}
    summary = {
        **first_shard,
        **summary_base,
        "accuracy": correct / total if total else 0.0,
        "correct": correct,
        "total": total,
        "invalid_rows": sum(int(item.get("invalid_rows", 0)) for item in shard_summaries),
        "output_path": str(merged_output),
        "shards": shard_summaries,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _run_parallel_eval(args: argparse.Namespace, checkpoint: Path | None, eval_name: str) -> None:
    rows = _load_eval_rows(Path(args.eval_data), args.max_samples)
    ranges = _shard_ranges(len(rows), args.num_gpus)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    gpu_ids = _gpu_ids_for_shards(args.num_gpus)
    script_path = Path(__file__).resolve()

    print(
        f"[itds-eval] launching {args.num_gpus} shards for {len(rows)} rows "
        f"with GPUs {','.join(gpu_ids)}",
        flush=True,
    )
    processes: list[tuple[int, subprocess.Popen]] = []
    for shard_index, (start, end) in enumerate(ranges):
        gpu_id = gpu_ids[shard_index]
        cmd = [
            sys.executable,
            str(script_path),
            "--eval-data",
            args.eval_data,
            "--output-dir",
            str(output_dir),
            "--max-samples",
            str(args.max_samples),
            "--max-tokens",
            str(args.max_tokens),
            "--temperature",
            str(args.temperature),
            "--top-p",
            str(args.top_p),
            "--num-gpus",
            "1",
            "--shard-index",
            str(shard_index),
            "--start",
            str(start),
            "--end",
            str(end),
            "--progress-position",
            str(shard_index),
        ]
        if args.checkpoint:
            cmd.extend(["--checkpoint", args.checkpoint])
        if args.base_only:
            cmd.append("--base-only")
        cmd.extend(_optional_cli_args(args))

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
        print(
            f"[itds-eval] shard {shard_index} on GPU {gpu_id}: rows [{start}, {end})",
            flush=True,
        )
        processes.append((shard_index, subprocess.Popen(cmd, cwd=str(ROOT_DIR), env=env)))

    failed: tuple[int, int] | None = None
    for shard_index, process in processes:
        return_code = process.wait()
        if return_code != 0 and failed is None:
            failed = (shard_index, return_code)
            for other_index, other_process in processes:
                if other_index != shard_index and other_process.poll() is None:
                    other_process.terminate()

    if failed is not None:
        shard_index, return_code = failed
        raise subprocess.CalledProcessError(return_code, f"eval shard {shard_index}")

    summary_base = {
        "checkpoint": "base_only" if args.base_only else str(checkpoint),
        "steering_path": "" if checkpoint is None else str(checkpoint / "steering.pt"),
        "num_gpus": args.num_gpus,
        "max_samples": args.max_samples,
    }
    summary = _merge_shards(output_dir, eval_name, args.num_gpus, summary_base)
    print("[itds-eval] merged shard outputs", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


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
    parser.add_argument("--num-gpus", type=int, default=1, help="Run eval shards in parallel across this many GPUs.")
    parser.add_argument("--start", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--end", type=int, default=-1, help=argparse.SUPPRESS)
    parser.add_argument("--shard-index", type=int, default=-1, help=argparse.SUPPRESS)
    parser.add_argument("--progress-position", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if not args.base_only and not args.checkpoint:
        parser.error("--checkpoint is required unless --base-only is set")

    checkpoint = Path(args.checkpoint).resolve() if args.checkpoint else None
    if args.num_gpus > 1 and args.shard_index < 0:
        _run_parallel_eval(args, checkpoint, _eval_name(args, checkpoint))
        return

    steering_path = checkpoint / "steering.pt" if checkpoint is not None else None
    payload: dict[str, Any] = {}
    config: dict[str, Any] = {}
    if checkpoint is not None:
        if not steering_path.exists():
            raise FileNotFoundError(f"Steering checkpoint not found: {steering_path}")
        print(f"[itds-eval] checkpoint_dir: {checkpoint}", flush=True)
        print(f"[itds-eval] steering_file:  {steering_path}", flush=True)
        payload = torch.load(steering_path, map_location="cpu")
        config = payload.get("config", {})
    elif args.base_only:
        print("[itds-eval] mode: base-only; no steering checkpoint will be loaded", flush=True)
    model_name = args.model_name_or_path or config.get("model_name_or_path") or "Qwen/Qwen2.5-7B"
    top_k = args.top_k if args.top_k is not None else int(config.get("top_k", 64))
    rank = args.rank if args.rank is not None else int(config.get("rank", 32))
    actor_depth = args.actor_depth if args.actor_depth is not None else int(config.get("actor_depth", 10))
    critic_depth = args.critic_depth if args.critic_depth is not None else int(config.get("critic_depth", 10))
    alpha = 0.0 if args.base_only else (args.alpha if args.alpha is not None else float(config.get("alpha", 1.0)))
    token_basis_init_std = (
        args.token_basis_init_std if args.token_basis_init_std is not None else float(config.get("token_basis_init_std", 1e-3))
    )
    print(f"[itds-eval] base_model: {model_name}", flush=True)
    print(
        "[itds-eval] steering_config: "
        f"base_only={args.base_only} top_k={top_k} rank={rank} "
        f"actor_depth={actor_depth} critic_depth={critic_depth} "
        f"alpha={alpha} token_basis_init_std={token_basis_init_std}",
        flush=True,
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
        print(f"[itds-eval] loaded steering weights from: {steering_path}", flush=True)
    model.to(device)
    model.eval()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _load_eval_rows(Path(args.eval_data), args.max_samples)
    if args.start or args.end >= 0:
        rows = rows[args.start : args.end if args.end >= 0 else None]
    eval_name = _eval_name(args, checkpoint)
    shard_suffix = f"_shard_{args.shard_index}" if args.shard_index >= 0 else ""
    output_path = output_dir / f"{eval_name}{shard_suffix}_itds_eval.jsonl"
    print(f"[itds-eval] eval_data: {Path(args.eval_data).resolve()}", flush=True)
    if args.shard_index >= 0:
        print(f"[itds-eval] shard_index: {args.shard_index} rows [{args.start}, {args.end})", flush=True)
    print(f"[itds-eval] output_path: {output_path}", flush=True)
    correct = 0
    invalid_rows = 0

    with output_path.open("w", encoding="utf-8") as handle:
        desc = f"itds eval shard {args.shard_index}" if args.shard_index >= 0 else "itds eval"
        progress = tqdm(rows, desc=desc, position=args.progress_position, leave=True, dynamic_ncols=True)
        for local_idx, row in enumerate(progress):
            idx = args.start + local_idx
            question = _question(row)
            gt = _ground_truth(row)
            prompt = row.get("prompt") if isinstance(row.get("prompt"), str) and row.get("prompt") else build_qwen_boxed_prompt(question)
            if not question and prompt:
                question = prompt
            if not prompt or not gt:
                invalid_rows += 1
                print(
                    f"[itds-eval] warning: skipping invalid eval row {row.get('idx', idx)!r}; "
                    f"empty prompt/question or ground truth. keys={sorted(row.keys())}",
                    flush=True,
                )
                handle.write(
                    json.dumps(
                        {
                            "idx": row.get("idx", idx),
                            "question": question,
                            "gt": gt,
                            "pred": [""],
                            "score": [False],
                            "code": [""],
                            "finish_reason": ["invalid_eval_row"],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                handle.flush()
                continue
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
        "steering_path": "" if steering_path is None else str(steering_path),
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
        "invalid_rows": invalid_rows,
        "output_path": str(output_path),
    }
    summary_name = f"summary_shard_{args.shard_index}.json" if args.shard_index >= 0 else "summary.json"
    (output_dir / summary_name).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
