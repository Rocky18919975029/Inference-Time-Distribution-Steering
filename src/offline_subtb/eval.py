from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from offline_subtb.limit_of_rlvr_io import build_qwen_boxed_prompt
from offline_subtb.utils import read_jsonl


def _loads_if_json(value: Any) -> Any:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("[") or stripped.startswith("{"):
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                return value
    return value


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


def _normalize_question(row: pd.Series) -> str:
    if "question" in row and pd.notna(row["question"]):
        return str(row["question"])
    extra_info = _loads_if_json(row.get("extra_info"))
    if isinstance(extra_info, dict) and extra_info.get("question") is not None:
        return str(extra_info["question"])
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


def _rows_from_parquet(path: str | Path) -> Iterable[dict[str, Any]]:
    df = pd.read_parquet(path)
    for row_index, row in df.iterrows():
        question = _normalize_question(row)
        problem_id = _normalize_problem_id(row, int(row_index))
        yield {
            "problem_id": problem_id,
            "question": question,
            "prompt": _normalize_prompt(row.get("prompt"), question),
            "ground_truth": _normalize_ground_truth(row),
            "subject": row.get("subject"),
            "level": row.get("level"),
            "data_source": row.get("data_source"),
        }


def _wandb_image(path: Path) -> Any:
    try:
        import wandb

        return wandb.Image(str(path))
    except ImportError:
        return str(path)


def _plot_accuracy(metrics_path: Path, plot_path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    steps: list[int] = []
    accuracies: list[float] = []
    if metrics_path.exists():
        for row in read_jsonl(metrics_path):
            steps.append(int(row["step"]))
            accuracies.append(float(row["accuracy"]))

    if not steps:
        return

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4))
    plt.plot(steps, accuracies, marker="o")
    plt.xlabel("step")
    plt.ylabel("accuracy")
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=160)
    plt.close()


def _export_eval_data(eval_data_path: str | Path, data_dir: Path, benchmark: str, max_samples: int) -> tuple[Path, int]:
    benchmark_dir = data_dir / benchmark
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    output_path = benchmark_dir / "test.jsonl"
    rows = list(_rows_from_parquet(eval_data_path))
    if max_samples > 0:
        rows = rows[:max_samples]
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            item = {
                "idx": row["problem_id"],
                "question": row["question"],
                "problem": row["question"],
                "input": row["question"],
                "gt_cot": None,
                "gt": row["ground_truth"],
                "answer": row["ground_truth"],
                "level": row["level"],
                "subject": row["subject"],
                "data_source": row["data_source"],
            }
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    return output_path, len(rows)


def _score_metrics(output_jsonl: Path) -> tuple[int, int, float]:
    correct = 0
    total = 0
    for row in read_jsonl(output_jsonl):
        scores = row.get("score", [])
        if not isinstance(scores, list):
            scores = [scores]
        for score in scores:
            correct += int(bool(score))
            total += 1
    accuracy = correct / total if total else 0.0
    return correct, total, accuracy


def _math_eval_cmd(
    *,
    model_name_or_path: str | Path,
    eval_data_dir: str | Path,
    eval_output_root: str | Path,
    benchmark: str,
    prompt_type: str,
    max_tokens: int,
    n_sampling: int,
    temperature: float,
    top_p: float,
    seed: int,
    start: int,
    end: int,
) -> list[str]:
    return [
        "python",
        "-u",
        "math_eval.py",
        "--model_name_or_path",
        str(model_name_or_path),
        "--data_names",
        benchmark,
        "--data_dir",
        str(eval_data_dir),
        "--output_dir",
        str(eval_output_root),
        "--split",
        "test",
        "--prompt_type",
        prompt_type,
        "--num_test_sample",
        "-1",
        "--max_tokens_per_call",
        str(max_tokens),
        "--seed",
        str(seed),
        "--temperature",
        str(temperature),
        "--n_sampling",
        str(n_sampling),
        "--top_p",
        str(top_p),
        "--start",
        str(start),
        "--end",
        str(end),
        "--use_vllm",
        "--save_outputs",
        "--overwrite",
    ]


def _output_jsonl_path(
    eval_output_root: str | Path,
    benchmark: str,
    prompt_type: str,
    seed: int,
    temperature: float,
    start: int,
    end: int,
) -> Path:
    return (
        Path(eval_output_root)
        / benchmark
        / f"test_{prompt_type}_-1_seed{seed}_t{temperature}_s{start}_e{end}.jsonl"
    )


def _write_metrics(
    *,
    output_dir: Path,
    step: int,
    eval_data_path_written: Path,
    output_jsonl: Path,
) -> dict[str, Any]:
    correct, total, accuracy = _score_metrics(output_jsonl)
    metrics_path = output_dir / "eval_metrics.jsonl"
    plot_path = output_dir / "eval_accuracy.png"
    metrics = {
        "step": step,
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "data_path": str(eval_data_path_written),
        "output_path": str(output_jsonl),
        "plot_path": str(plot_path),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(metrics, ensure_ascii=False) + "\n")
    _plot_accuracy(metrics_path, plot_path)
    if plot_path.exists():
        metrics["plot"] = _wandb_image(plot_path)
    return metrics


def run_limit_of_rlvr_eval(
    *,
    model_name_or_path: str | Path,
    eval_data_path: str | Path,
    output_dir: str | Path,
    step: int,
    limit_of_rlvr_dir: str | Path,
    limit_data_dir: str | Path | None,
    benchmark: str,
    prompt_type: str,
    max_samples: int,
    n_sampling: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
) -> dict[str, Any]:
    limit_root = Path(limit_of_rlvr_dir)
    math_eval_dir = limit_root / "math" / "examples" / "math_eval"
    math_eval_py = math_eval_dir / "math_eval.py"
    if not math_eval_py.exists():
        raise FileNotFoundError(f"Cannot find limit-of-RLVR math_eval.py: {math_eval_py}")

    output_dir = Path(output_dir)
    eval_output_root = output_dir / "limit_of_rlvr_outputs" / f"step_{step}"
    eval_data_dir = Path(limit_data_dir) if limit_data_dir else math_eval_dir / "data"
    eval_data_path_written, _ = _export_eval_data(eval_data_path, eval_data_dir, benchmark, max_samples)

    cmd = _math_eval_cmd(
        model_name_or_path=model_name_or_path,
        eval_data_dir=eval_data_dir,
        eval_output_root=eval_output_root,
        benchmark=benchmark,
        prompt_type=prompt_type,
        max_tokens=max_tokens,
        n_sampling=n_sampling,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        start=0,
        end=-1,
    )
    env = os.environ.copy()
    subprocess.run(cmd, cwd=math_eval_dir, env=env, check=True)

    output_jsonl = _output_jsonl_path(eval_output_root, benchmark, prompt_type, seed, temperature, 0, -1)
    if not output_jsonl.exists():
        raise FileNotFoundError(f"limit-of-RLVR did not produce expected eval output: {output_jsonl}")

    return _write_metrics(
        output_dir=output_dir,
        step=step,
        eval_data_path_written=eval_data_path_written,
        output_jsonl=output_jsonl,
    )


def run_limit_of_rlvr_eval_parallel(
    *,
    model_name_or_path: str | Path,
    eval_data_path: str | Path,
    output_dir: str | Path,
    step: int,
    limit_of_rlvr_dir: str | Path,
    limit_data_dir: str | Path | None,
    benchmark: str,
    prompt_type: str,
    max_samples: int,
    n_sampling: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
    gpu_ids: list[int],
) -> dict[str, Any]:
    if len(gpu_ids) <= 1:
        return run_limit_of_rlvr_eval(
            model_name_or_path=model_name_or_path,
            eval_data_path=eval_data_path,
            output_dir=output_dir,
            step=step,
            limit_of_rlvr_dir=limit_of_rlvr_dir,
            limit_data_dir=limit_data_dir,
            benchmark=benchmark,
            prompt_type=prompt_type,
            max_samples=max_samples,
            n_sampling=n_sampling,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
        )

    limit_root = Path(limit_of_rlvr_dir)
    math_eval_dir = limit_root / "math" / "examples" / "math_eval"
    math_eval_py = math_eval_dir / "math_eval.py"
    if not math_eval_py.exists():
        raise FileNotFoundError(f"Cannot find limit-of-RLVR math_eval.py: {math_eval_py}")

    output_dir = Path(output_dir)
    eval_output_root = output_dir / "limit_of_rlvr_outputs" / f"step_{step}"
    eval_output_root.mkdir(parents=True, exist_ok=True)
    eval_data_dir = Path(limit_data_dir) if limit_data_dir else math_eval_dir / "data"
    eval_data_path_written, num_rows = _export_eval_data(eval_data_path, eval_data_dir, benchmark, max_samples)
    if num_rows == 0:
        raise ValueError(f"No eval rows exported from {eval_data_path}")

    shard_size = (num_rows + len(gpu_ids) - 1) // len(gpu_ids)
    processes: list[tuple[int, int, int, Path, subprocess.Popen, Any]] = []
    for shard_id, gpu_id in enumerate(gpu_ids):
        start = shard_id * shard_size
        end = min(start + shard_size, num_rows)
        if start >= end:
            continue
        cmd = _math_eval_cmd(
            model_name_or_path=model_name_or_path,
            eval_data_dir=eval_data_dir,
            eval_output_root=eval_output_root,
            benchmark=benchmark,
            prompt_type=prompt_type,
            max_tokens=max_tokens,
            n_sampling=n_sampling,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            start=start,
            end=end,
        )
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        log_path = eval_output_root / f"shard_{shard_id}_gpu_{gpu_id}.log"
        log_handle = log_path.open("w", encoding="utf-8")
        print(f"  shard {shard_id} on GPU {gpu_id}: rows [{start}, {end}) -> {log_path}")
        process = subprocess.Popen(
            cmd,
            cwd=math_eval_dir,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        processes.append((shard_id, start, end, log_path, process, log_handle))

    shard_outputs: list[Path] = []
    for shard_id, start, end, log_path, process, log_handle in processes:
        return_code = process.wait()
        log_handle.close()
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, process.args, output=f"See log: {log_path}")
        output_jsonl = _output_jsonl_path(eval_output_root, benchmark, prompt_type, seed, temperature, start, end)
        if not output_jsonl.exists():
            raise FileNotFoundError(f"Shard {shard_id} did not produce expected eval output: {output_jsonl}")
        shard_outputs.append(output_jsonl)

    merged_output = _output_jsonl_path(eval_output_root, benchmark, prompt_type, seed, temperature, 0, -1)
    merged_rows = []
    for shard_output in shard_outputs:
        merged_rows.extend(read_jsonl(shard_output))
    merged_rows.sort(key=lambda row: int(row.get("idx", 0)))
    with merged_output.open("w", encoding="utf-8") as handle:
        for row in merged_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    return _write_metrics(
        output_dir=output_dir,
        step=step,
        eval_data_path_written=eval_data_path_written,
        output_jsonl=merged_output,
    )
