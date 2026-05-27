#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from offline_subtb.eval import run_limit_of_rlvr_eval, run_limit_of_rlvr_eval_parallel


def _step_from_checkpoint(checkpoint: str, fallback: int) -> int:
    match = re.search(r"step_(\d+)", checkpoint)
    return int(match.group(1)) if match else fallback


def _has_tokenizer(path: Path) -> bool:
    return any((path / name).exists() for name in ("tokenizer.json", "tokenizer.model", "vocab.json"))


def _link_tree_contents(source: Path, target: Path) -> None:
    for item in source.iterdir():
        link = target / item.name
        if link.exists() or link.is_symlink():
            continue
        link.symlink_to(item.resolve(), target_is_directory=item.is_dir())


def _training_state(checkpoint: Path) -> dict:
    state_path = checkpoint / "training_state.json"
    if not state_path.exists():
        return {}
    return json.loads(state_path.read_text(encoding="utf-8"))


def _merge_lora_model(checkpoint: Path, eval_models_dir: Path, step: int) -> Path:
    adapter_dir = checkpoint / "lora_adapter"
    state = _training_state(checkpoint)
    config = state.get("config", {}) if isinstance(state, dict) else {}
    base_model_name = config.get("model_name_or_path")
    if not base_model_name:
        raise ValueError(f"Cannot infer base model for LoRA checkpoint: {checkpoint}")

    merged_model = eval_models_dir / f"step_{step}_merged_lora"
    if _has_tokenizer(merged_model) and any(merged_model.glob("*.safetensors")):
        return merged_model

    try:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ImportError("Evaluating LoRA checkpoints requires peft. Install it with: pip install peft") from exc

    merged_model.mkdir(parents=True, exist_ok=True)
    model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype="auto", trust_remote_code=True)
    model = PeftModel.from_pretrained(model, adapter_dir)
    model = model.merge_and_unload()
    model.save_pretrained(merged_model, safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    tokenizer.save_pretrained(merged_model)
    return merged_model


def _model_path(checkpoint: Path, eval_models_dir: Path, step: int) -> Path:
    if (checkpoint / "lora_adapter").exists():
        return _merge_lora_model(checkpoint, eval_models_dir, step)

    base_model = checkpoint / "base_model"
    if not base_model.exists():
        return checkpoint
    if _has_tokenizer(base_model):
        return base_model

    tokenizer_dir = checkpoint / "tokenizer"
    if not tokenizer_dir.exists():
        raise FileNotFoundError(
            f"{base_model} does not contain tokenizer files, and no sibling tokenizer dir was found: {tokenizer_dir}"
        )

    merged_model = eval_models_dir / f"step_{step}"
    merged_model.mkdir(parents=True, exist_ok=True)
    _link_tree_contents(base_model, merged_model)
    _link_tree_contents(tokenizer_dir, merged_model)
    return merged_model


def _parse_gpu_ids(value: str, num_gpus: int) -> list[int]:
    if value:
        gpu_ids = [int(item.strip()) for item in value.split(",") if item.strip()]
        if not gpu_ids:
            raise ValueError("--gpu-ids was provided but no GPU ids were parsed.")
        return gpu_ids
    if num_gpus < 1:
        raise ValueError("--num-gpus must be at least 1.")
    return list(range(num_gpus))


def _resolve_checkpoint(value: str, eval_models_dir: Path, step: int) -> str:
    path = Path(value)
    if path.exists():
        return str(_model_path(path.resolve(), eval_models_dir, step))
    if value.startswith(".") or value.startswith("/") or value.startswith("~"):
        raise FileNotFoundError(f"Checkpoint not found: {Path(value).expanduser().resolve()}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one or more saved training checkpoints.")
    parser.add_argument("--checkpoint", action="append", default=[], help="Checkpoint dir. May be repeated.")
    parser.add_argument("--checkpoint-glob", default="", help="Optional glob for checkpoint dirs.")
    parser.add_argument("--eval-data", default=str(ROOT_DIR / "data" / "test.parquet"))
    parser.add_argument("--output-dir", default=str(ROOT_DIR / "outputs" / "checkpoint_eval"))
    parser.add_argument("--limit-of-rlvr-dir", default=str(ROOT_DIR / "limit-of-RLVR"))
    parser.add_argument("--limit-data-dir", default="")
    parser.add_argument("--benchmark-prefix", default="eval_test")
    parser.add_argument("--prompt-type", default="qwen-boxed")
    parser.add_argument("--max-samples", type=int, default=128)
    parser.add_argument("--n-sampling", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num-gpus", type=int, default=1, help="Number of GPUs used to shard each checkpoint eval.")
    parser.add_argument("--gpu-ids", default="", help="Comma-separated GPU ids. Overrides --num-gpus when set.")
    args = parser.parse_args()

    checkpoints = list(args.checkpoint)
    if args.checkpoint_glob:
        checkpoints.extend(str(item) for item in sorted(Path().glob(args.checkpoint_glob)))
    if not checkpoints:
        raise ValueError("Provide at least one --checkpoint or --checkpoint-glob.")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_models_dir = output_dir / "eval_model_views"
    summary_path = output_dir / "checkpoint_eval_summary.jsonl"
    eval_data_path = Path(args.eval_data).resolve()
    limit_of_rlvr_dir = Path(args.limit_of_rlvr_dir).resolve()
    limit_data_dir = Path(args.limit_data_dir).resolve() if args.limit_data_dir else None
    gpu_ids = _parse_gpu_ids(args.gpu_ids, args.num_gpus)

    for index, checkpoint in enumerate(checkpoints):
        step = _step_from_checkpoint(checkpoint, fallback=index)
        benchmark = f"{args.benchmark_prefix}_step_{step}"
        model_name_or_path = _resolve_checkpoint(checkpoint, eval_models_dir, step)
        print(f"Evaluating step {step}: {model_name_or_path}")
        eval_fn = run_limit_of_rlvr_eval_parallel if len(gpu_ids) > 1 else run_limit_of_rlvr_eval
        eval_kwargs = dict(
            model_name_or_path=model_name_or_path,
            eval_data_path=eval_data_path,
            output_dir=output_dir,
            step=step,
            limit_of_rlvr_dir=limit_of_rlvr_dir,
            limit_data_dir=limit_data_dir,
            benchmark=benchmark,
            prompt_type=args.prompt_type,
            max_samples=args.max_samples,
            n_sampling=args.n_sampling,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            seed=args.seed,
        )
        if len(gpu_ids) > 1:
            eval_kwargs["gpu_ids"] = gpu_ids
        metrics = eval_fn(**eval_kwargs)
        row = {
            "checkpoint": checkpoint,
            "model_name_or_path": str(model_name_or_path),
            **{key: value for key, value in metrics.items() if key != "plot"},
        }
        with summary_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(json.dumps(row, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
