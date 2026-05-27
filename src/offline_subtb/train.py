from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
from datetime import timedelta
from dataclasses import fields, replace
from pathlib import Path
from typing import Any

import torch
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import TrainConfig, load_config
from .dataset import JsonlSubTBDataset, collate_examples
from .eval import run_limit_of_rlvr_eval
from .loss import compute_newline_subtb_loss
from .model import PolicyWithFlow
from .utils import set_seed


def _coerce_value(value: str, current: Any) -> Any:
    if current is None:
        if value.lower() in {"none", "null"}:
            return None
        return value
    if isinstance(current, bool):
        return value.lower() in {"1", "true", "yes", "on"}
    if isinstance(current, int) and not isinstance(current, bool):
        return int(value)
    if isinstance(current, float):
        return float(value)
    return value


def _apply_overrides(config: TrainConfig, overrides: list[str]) -> TrainConfig:
    data = {field.name: getattr(config, field.name) for field in fields(config)}
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got {item!r}")
        key, value = item.split("=", 1)
        if key not in data:
            raise ValueError(f"Unknown config override {key!r}")
        data[key] = _coerce_value(value, data[key])
    return replace(config, **data)


def _latest_checkpoint(output_dir: str | Path) -> Path | None:
    root = Path(output_dir)
    if not root.exists():
        return None
    candidates: list[tuple[int, Path]] = []
    for path in root.glob("step_*"):
        match = re.match(r"step_(\d+)$", path.name)
        if match and (path / "training_state.json").exists():
            candidates.append((int(match.group(1)), path))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def _resolve_resume_path(config: TrainConfig) -> Path | None:
    if not config.resume_from_checkpoint:
        return None
    if config.resume_from_checkpoint == "auto":
        return _latest_checkpoint(config.output_dir)
    path = Path(config.resume_from_checkpoint)
    return path if path.exists() else None


def _load_training_state(checkpoint_dir: Path | None) -> dict[str, Any]:
    if checkpoint_dir is None:
        return {}
    state_path = checkpoint_dir / "training_state.json"
    if not state_path.exists():
        return {}
    return json.loads(state_path.read_text(encoding="utf-8"))


def _wandb_run_id_from_runtime() -> str | None:
    try:
        import wandb

        if wandb.run is not None:
            return wandb.run.id
    except ImportError:
        return None
    return None


def _wandb_init_kwargs(config: TrainConfig, resumed_state: dict[str, Any]) -> dict[str, Any]:
    init_kwargs: dict[str, Any] = {
        "name": config.wandb_run_name,
        "entity": config.wandb_entity,
    }
    run_id = config.wandb_run_id or resumed_state.get("wandb_run_id")
    if run_id:
        init_kwargs["id"] = run_id
        init_kwargs["resume"] = "allow"
        os.environ.setdefault("WANDB_RUN_ID", run_id)
        os.environ.setdefault("WANDB_RESUME", "allow")
    if config.wandb_mode:
        os.environ["WANDB_MODE"] = config.wandb_mode
    return {"wandb": {key: value for key, value in init_kwargs.items() if value is not None}}


def _reduce_diagnostics(accelerator: Accelerator, diagnostics: dict[str, float]) -> dict[str, float]:
    reduced: dict[str, float] = {}
    for key, value in diagnostics.items():
        tensor = torch.tensor(float(value), device=accelerator.device)
        reduced[key] = float(accelerator.reduce(tensor, reduction="mean").detach().cpu())
    return reduced


def _as_float(value: Any) -> float:
    if hasattr(value, "detach"):
        return float(value.detach().cpu())
    return float(value)


def _lora_target_modules(config: TrainConfig) -> list[str]:
    return [item.strip() for item in config.lora_target_modules.split(",") if item.strip()]


def _is_deepspeed(accelerator: Accelerator) -> bool:
    return "deepspeed" in str(accelerator.distributed_type).lower()


def _deepspeed_tag(global_step: int) -> str:
    return f"global_step{global_step}"


def _save_logical_checkpoint_artifacts(
    *,
    accelerator: Accelerator,
    model,
    tokenizer,
    checkpoint_dir: Path,
    global_step: int,
    micro_step: int,
    epoch: int,
    config: TrainConfig,
) -> None:
    if not accelerator.is_main_process:
        return

    unwrapped = accelerator.unwrap_model(model)
    if config.finetune_mode == "lora":
        adapter_dir = checkpoint_dir / "lora_adapter"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        unwrapped.base_model.save_pretrained(adapter_dir)
    else:
        (checkpoint_dir / "base_model").mkdir(parents=True, exist_ok=True)
        unwrapped.base_model.save_pretrained(checkpoint_dir / "base_model")
    tokenizer.save_pretrained(checkpoint_dir / "tokenizer")
    torch.save(unwrapped.flow_head.state_dict(), checkpoint_dir / "flow_head.pt")
    state = {
        "global_step": global_step,
        "micro_step": micro_step,
        "epoch": epoch,
        "deepspeed_tag": _deepspeed_tag(global_step),
        "wandb_run_id": _wandb_run_id_from_runtime() or config.wandb_run_id,
        "config": config.__dict__,
    }
    (checkpoint_dir / "training_state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")


def _save_checkpoint(
    *,
    accelerator: Accelerator,
    model,
    tokenizer,
    optimizer,
    checkpoint_dir: Path,
    global_step: int,
    micro_step: int,
    epoch: int,
    config: TrainConfig,
) -> None:
    accelerator.print(f"Saving checkpoint to {checkpoint_dir}")
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()

    if _is_deepspeed(accelerator):
        client_state = {
            "global_step": global_step,
            "micro_step": micro_step,
            "epoch": epoch,
            "wandb_run_id": _wandb_run_id_from_runtime() or config.wandb_run_id,
        }
        model.save_checkpoint(
            str(checkpoint_dir / "deepspeed"),
            tag=_deepspeed_tag(global_step),
            client_state=client_state,
        )
    elif accelerator.is_main_process:
        torch.save(optimizer.state_dict(), checkpoint_dir / "optimizer.pt")

    accelerator.wait_for_everyone()
    _save_logical_checkpoint_artifacts(
        accelerator=accelerator,
        model=model,
        tokenizer=tokenizer,
        checkpoint_dir=checkpoint_dir,
        global_step=global_step,
        micro_step=micro_step,
        epoch=epoch,
        config=config,
    )
    accelerator.wait_for_everyone()
    accelerator.print(f"Saved checkpoint to {checkpoint_dir}")


def _load_checkpoint_state(
    accelerator: Accelerator,
    model,
    optimizer,
    checkpoint_dir: Path | None,
    device: torch.device,
    resumed_state: dict[str, Any],
) -> None:
    if checkpoint_dir is None:
        return
    deepspeed_dir = checkpoint_dir / "deepspeed"
    if _is_deepspeed(accelerator) and deepspeed_dir.exists():
        tag = resumed_state.get("deepspeed_tag")
        accelerator.print(f"Loading DeepSpeed checkpoint state: {deepspeed_dir} tag={tag}")
        model.load_checkpoint(str(deepspeed_dir), tag=tag)
        return
    optimizer_path = checkpoint_dir / "optimizer.pt"
    if optimizer_path.exists():
        optimizer.load_state_dict(torch.load(optimizer_path, map_location=device))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline Newline-SubTB trainer.")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Override config values with key=value. May be repeated.",
    )
    return parser.parse_args()


def _resolve_project_path(path_value: str | None) -> str | None:
    if not path_value:
        return path_value
    path = Path(path_value)
    if path.is_absolute():
        return str(path)
    return str(Path.cwd() / path)


def _write_resolved_config(accelerator: Accelerator, config: TrainConfig) -> None:
    if not accelerator.is_main_process:
        return
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": config.__dict__,
        "cwd": str(Path.cwd()),
    }
    (output_dir / "resolved_config.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _save_eval_model_for_vllm(
    *,
    unwrapped_model,
    tokenizer,
    eval_model_dir: Path,
    config: TrainConfig,
    dtype: torch.dtype | None,
) -> None:
    eval_model_dir.mkdir(parents=True, exist_ok=True)
    if config.finetune_mode == "lora":
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise ImportError("LoRA eval requires peft. Install it with: pip install peft") from exc

        adapter_dir = eval_model_dir / "lora_adapter"
        unwrapped_model.base_model.save_pretrained(adapter_dir)
        merge_base = AutoModelForCausalLM.from_pretrained(
            config.model_name_or_path,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        merged = PeftModel.from_pretrained(merge_base, adapter_dir)
        merged = merged.merge_and_unload()
        merged.save_pretrained(eval_model_dir, safe_serialization=True)
        tokenizer.save_pretrained(eval_model_dir)
        del merged, merge_base
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return

    unwrapped_model.base_model.save_pretrained(eval_model_dir)
    tokenizer.save_pretrained(eval_model_dir)


def main() -> None:
    args = _parse_args()
    config = _apply_overrides(load_config(args.config), args.override)
    resume_path = _resolve_resume_path(config)
    resumed_state = _load_training_state(resume_path)

    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="wandb" if config.wandb_project else None,
        kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=3600))],
    )
    set_seed(config.seed + accelerator.process_index)
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    _write_resolved_config(accelerator, config)

    if config.wandb_project:
        accelerator.init_trackers(
            project_name=config.wandb_project,
            config=config.__dict__,
            init_kwargs=_wandb_init_kwargs(config, resumed_state),
        )

    dtype = torch.bfloat16 if config.bf16 and torch.cuda.is_available() else None
    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path, trust_remote_code=True)
    model_source = (
        str(resume_path / "base_model")
        if config.finetune_mode == "full" and resume_path and (resume_path / "base_model").exists()
        else config.model_name_or_path
    )
    model = PolicyWithFlow(
        model_source,
        torch_dtype=dtype,
        finetune_mode=config.finetune_mode,
        lora_r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        lora_target_modules=_lora_target_modules(config),
        lora_adapter_path=(
            str(resume_path / "lora_adapter")
            if config.finetune_mode == "lora" and resume_path and (resume_path / "lora_adapter").exists()
            else None
        ),
    )
    if resume_path and (resume_path / "flow_head.pt").exists():
        model.flow_head.load_state_dict(torch.load(resume_path / "flow_head.pt", map_location="cpu"))
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    trainable_params, total_params = model.trainable_parameter_counts()
    accelerator.print(
        {
            "finetune_mode": config.finetune_mode,
            "trainable_params": trainable_params,
            "total_params": total_params,
            "trainable_ratio": trainable_params / total_params,
        }
    )

    train_data_path = Path(config.train_data_path)
    index_cache_path = train_data_path.with_suffix(train_data_path.suffix + ".offsets")
    accelerator.print(f"Loading training data index: {index_cache_path}")
    dataset = JsonlSubTBDataset(
        config.train_data_path,
        index_cache_path=index_cache_path if index_cache_path.exists() else None,
    )
    accelerator.print(f"Loaded training index rows: {len(dataset)}")
    dataloader = DataLoader(
        dataset,
        batch_size=config.per_device_train_batch_size,
        shuffle=True,
        collate_fn=collate_examples,
        num_workers=config.dataloader_num_workers,
    )
    optimizer = AdamW(
        (param for param in model.parameters() if param.requires_grad),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    _load_checkpoint_state(accelerator, model, optimizer, resume_path, accelerator.device, resumed_state)

    rng = random.Random(config.seed + accelerator.process_index)
    global_step = int(resumed_state.get("global_step", 0))
    micro_step = int(resumed_state.get("micro_step", 0))
    epoch = int(resumed_state.get("epoch", 0))
    steps_per_epoch = math.ceil(len(dataloader) / config.gradient_accumulation_steps)
    if config.num_epochs > 0:
        epoch_target_steps = steps_per_epoch * max(config.num_epochs - epoch, 0)
        progress_total = global_step + epoch_target_steps
        if config.max_steps > 0:
            progress_total = min(config.max_steps, progress_total)
    else:
        progress_total = config.max_steps

    last_saved_step = -1
    skipped_invalid_microbatches = 0
    model.train()
    optimizer.zero_grad(set_to_none=True)
    if config.max_steps <= 0 and config.num_epochs <= 0:
        raise ValueError("At least one of max_steps or num_epochs must be positive.")
    progress_bar = tqdm(
        total=progress_total if progress_total > 0 else None,
        initial=global_step if progress_total > 0 else 0,
        desc="training",
        dynamic_ncols=True,
        disable=not accelerator.is_main_process,
    )

    while (config.max_steps <= 0 or global_step < config.max_steps) and (
        config.num_epochs <= 0 or epoch < config.num_epochs
    ):
        epoch += 1
        epoch_label = f"{epoch}/{config.num_epochs}" if config.num_epochs > 0 else str(epoch)
        if accelerator.is_main_process:
            progress_bar.set_description(f"training epoch {epoch_label}")
        for batch in dataloader:
            with accelerator.accumulate(model):
                output = compute_newline_subtb_loss(
                    model=model,
                    tokenizer=tokenizer,
                    examples=batch,
                    alpha=config.alpha,
                    beta=config.beta,
                    lambda_subtb=config.lambda_subtb,
                    max_pairs_per_trace=config.max_pairs_per_trace,
                    rng=rng,
                    device=accelerator.device,
                )
                invalid_batch = torch.tensor(
                    float(output.diagnostics.get("num_no_valid_pair_batches", 0.0)),
                    device=accelerator.device,
                )
                invalid_batches = accelerator.reduce(invalid_batch, reduction="sum")
                if float(invalid_batches.detach().cpu()) > 0.0:
                    skipped_invalid_microbatches += 1
                    if accelerator.is_main_process:
                        progress_bar.set_postfix(
                            epoch=epoch_label,
                            skipped=skipped_invalid_microbatches,
                            refresh=False,
                        )
                    continue

                accelerator.backward(output.loss)
                micro_step += 1

                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), config.gradient_clipping)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    progress_bar.update(1)

                    diagnostics = _reduce_diagnostics(accelerator, output.diagnostics)
                    diagnostics["grad_norm"] = _as_float(grad_norm)
                    diagnostics["learning_rate"] = config.learning_rate
                    if accelerator.is_main_process:
                        progress_bar.set_postfix(
                            epoch=epoch_label,
                            loss=f"{_as_float(output.loss):.4g}",
                            grad_norm=f"{diagnostics['grad_norm']:.4g}",
                            refresh=False,
                        )

                    if global_step % config.logging_steps == 0:
                        accelerator.print({"step": global_step, **diagnostics})
                        accelerator.log({f"train/{key}": value for key, value in diagnostics.items()}, step=global_step)

                    if global_step > 0 and global_step % config.save_steps == 0:
                        _save_checkpoint(
                            accelerator=accelerator,
                            model=model,
                            tokenizer=tokenizer,
                            optimizer=optimizer,
                            checkpoint_dir=Path(config.output_dir) / f"step_{global_step}",
                            global_step=global_step,
                            micro_step=micro_step,
                            epoch=epoch,
                            config=config,
                        )
                        last_saved_step = global_step

                    if (
                        config.eval_every_steps > 0
                        and config.eval_data_path
                        and global_step > 0
                        and global_step % config.eval_every_steps == 0
                    ):
                        accelerator.wait_for_everyone()
                        if accelerator.is_main_process:
                            eval_output_dir = config.eval_output_dir or str(Path(config.output_dir) / "eval")
                            unwrapped = accelerator.unwrap_model(model)
                            eval_model_dir = Path(eval_output_dir) / "model_checkpoints" / f"step_{global_step}"
                            _save_eval_model_for_vllm(
                                unwrapped_model=unwrapped,
                                tokenizer=tokenizer,
                                eval_model_dir=eval_model_dir,
                                config=config,
                                dtype=dtype,
                            )
                            metrics = run_limit_of_rlvr_eval(
                                model_name_or_path=eval_model_dir,
                                eval_data_path=config.eval_data_path,
                                output_dir=eval_output_dir,
                                step=global_step,
                                limit_of_rlvr_dir=_resolve_project_path(
                                    config.limit_of_rlvr_dir or os.environ.get("LIMIT_OF_RLVR_DIR", "")
                                ),
                                limit_data_dir=_resolve_project_path(
                                    config.limit_data_dir or os.environ.get("LIMIT_DATA_DIR")
                                ),
                                benchmark=f"{config.eval_benchmark}_step_{global_step}",
                                prompt_type=config.eval_prompt_type,
                                max_samples=config.eval_max_samples,
                                n_sampling=config.eval_n_sampling,
                                max_tokens=config.eval_max_tokens,
                                temperature=config.eval_temperature,
                                top_p=config.eval_top_p,
                                seed=config.eval_seed,
                            )
                            loggable = {
                                "eval/accuracy": metrics["accuracy"],
                                "eval/correct": metrics["correct"],
                                "eval/total": metrics["total"],
                            }
                            if "plot" in metrics:
                                loggable["eval/accuracy_plot"] = metrics["plot"]
                            accelerator.log(loggable, step=global_step)
                        accelerator.wait_for_everyone()
                        model.train()

                    if config.max_steps > 0 and global_step >= config.max_steps:
                        break

            if config.max_steps > 0 and global_step >= config.max_steps:
                break

    progress_bar.close()

    if global_step != last_saved_step:
        _save_checkpoint(
            accelerator=accelerator,
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            checkpoint_dir=Path(config.output_dir) / f"step_{global_step}",
            global_step=global_step,
            micro_step=micro_step,
            epoch=epoch,
            config=config,
        )
    accelerator.end_training()


if __name__ == "__main__":
    main()
