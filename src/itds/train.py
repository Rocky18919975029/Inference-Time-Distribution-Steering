from __future__ import annotations

import argparse
import json
import math
import os
import re
from dataclasses import fields, replace
from datetime import timedelta
from pathlib import Path
from typing import Any

import torch
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from .config import ITDSConfig, load_config
from .dataset import GroupBatchSampler, RolloutJsonlDataset, collate_examples
from .model import TopKLowRankSteering
from .objectives import actor_critic_loss, grpo_loss, tb_vargrad_loss
from .utils import set_seed


def _coerce_value(value: str, current: Any) -> Any:
    if current is None:
        return None if value.lower() in {"none", "null"} else value
    if isinstance(current, bool):
        return value.lower() in {"1", "true", "yes", "on"}
    if isinstance(current, int) and not isinstance(current, bool):
        return int(value)
    if isinstance(current, float):
        return float(value)
    return value


def _apply_overrides(config: ITDSConfig, overrides: list[str]) -> ITDSConfig:
    data = {field.name: getattr(config, field.name) for field in fields(config)}
    for item in overrides:
        key, value = item.split("=", 1)
        if key not in data:
            raise ValueError(f"Unknown config override {key!r}")
        data[key] = _coerce_value(value, data[key])
    return replace(config, **data)


def _latest_checkpoint(output_dir: str | Path) -> Path | None:
    candidates: list[tuple[int, Path]] = []
    for path in Path(output_dir).glob("step_*"):
        match = re.match(r"step_(\d+)$", path.name)
        if match and (path / "steering.pt").exists():
            candidates.append((int(match.group(1)), path))
    return max(candidates, key=lambda item: item[0])[1] if candidates else None


def _resolve_resume_path(config: ITDSConfig) -> Path | None:
    if not config.resume_from_checkpoint:
        return None
    if config.resume_from_checkpoint == "auto":
        return _latest_checkpoint(config.output_dir)
    path = Path(config.resume_from_checkpoint)
    return path if path.exists() else None


def _save_checkpoint(accelerator: Accelerator, model, optimizer, checkpoint_dir: Path, step: int, epoch: int, config: ITDSConfig) -> None:
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        unwrapped = accelerator.unwrap_model(model)
        torch.save(
            {
                "steering": unwrapped.steering_state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
                "epoch": epoch,
                "wandb_run_id": _wandb_run_id_from_runtime() or config.wandb_run_id,
                "config": config.__dict__,
            },
            checkpoint_dir / "steering.pt",
        )
        (checkpoint_dir / "training_state.json").write_text(
            json.dumps(
                {
                    "global_step": step,
                    "epoch": epoch,
                    "wandb_run_id": _wandb_run_id_from_runtime() or config.wandb_run_id,
                    "config": config.__dict__,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    accelerator.wait_for_everyone()


def _load_training_state(checkpoint_dir: Path | None) -> dict[str, Any]:
    if checkpoint_dir is None:
        return {}
    state_path = checkpoint_dir / "training_state.json"
    if not state_path.exists():
        return {}
    return json.loads(state_path.read_text(encoding="utf-8"))


def _load_checkpoint(accelerator: Accelerator, model, optimizer, checkpoint_dir: Path | None, device: torch.device) -> tuple[int, int]:
    if checkpoint_dir is None:
        return 0, 0
    payload = torch.load(checkpoint_dir / "steering.pt", map_location=device)
    accelerator.unwrap_model(model).load_steering_state_dict(payload["steering"])
    optimizer.load_state_dict(payload["optimizer"])
    return int(payload.get("step", 0)), int(payload.get("epoch", 0))


def _wandb_run_id_from_runtime() -> str | None:
    try:
        import wandb

        if wandb.run is not None:
            return wandb.run.id
    except ImportError:
        return None
    return None


def _wandb_init_kwargs(config: ITDSConfig, resumed_state: dict[str, Any]) -> dict[str, Any]:
    run_id = config.wandb_run_id or resumed_state.get("wandb_run_id")
    init_kwargs: dict[str, Any] = {
        "name": config.wandb_run_name,
        "entity": config.wandb_entity,
    }
    if run_id:
        init_kwargs["id"] = run_id
        init_kwargs["resume"] = "allow"
        os.environ.setdefault("WANDB_RUN_ID", run_id)
        os.environ.setdefault("WANDB_RESUME", "allow")
    return {"wandb": {key: value for key, value in init_kwargs.items() if value is not None}}


def _loss_for_objective(config: ITDSConfig, batch):
    if config.objective == "tb":
        return tb_vargrad_loss(batch, beta=config.beta)
    if config.objective == "grpo":
        return grpo_loss(batch, beta=config.beta, clip_epsilon=config.clip_epsilon)
    if config.objective == "ac":
        return actor_critic_loss(batch, beta=config.beta, value_loss_weight=config.value_loss_weight)
    raise ValueError(f"Unknown objective={config.objective!r}; expected tb, grpo, or ac.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TopK low-rank inference-time steering.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
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
    if accelerator.is_main_process:
        (Path(config.output_dir) / "resolved_config.json").write_text(
            json.dumps({"config": config.__dict__, "cwd": str(Path.cwd())}, indent=2),
            encoding="utf-8",
        )

    if config.wandb_mode:
        os.environ["WANDB_MODE"] = config.wandb_mode
    if config.wandb_project:
        accelerator.init_trackers(
            project_name=config.wandb_project,
            config=config.__dict__,
            init_kwargs=_wandb_init_kwargs(config, resumed_state),
        )

    dtype = torch.bfloat16 if config.bf16 and torch.cuda.is_available() else None
    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path, trust_remote_code=True)
    model = TopKLowRankSteering(
        config.model_name_or_path,
        top_k=config.top_k,
        rank=config.rank,
        actor_depth=config.actor_depth,
        critic_depth=config.critic_depth,
        alpha=config.alpha,
        torch_dtype=dtype,
    )
    trainable, total = model.trainable_parameter_counts()
    accelerator.print({"trainable_params": trainable, "total_params": total, "trainable_ratio": trainable / total})

    dataset = RolloutJsonlDataset(config.train_data_path)
    batch_sampler = GroupBatchSampler(
        dataset.groups,
        batch_size=config.per_device_train_batch_size,
        seed=config.seed + accelerator.process_index,
    )
    dataloader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        collate_fn=collate_examples,
        num_workers=config.dataloader_num_workers,
    )
    optimizer = AdamW((p for p in model.parameters() if p.requires_grad), lr=config.learning_rate, weight_decay=config.weight_decay)
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    global_step, epoch = _load_checkpoint(accelerator, model, optimizer, resume_path, accelerator.device)

    steps_per_epoch = math.ceil(len(dataloader) / config.gradient_accumulation_steps)
    total = config.max_steps if config.max_steps > 0 else steps_per_epoch * max(config.num_epochs - epoch, 0)
    progress = tqdm(total=total if total > 0 else None, initial=min(global_step, total) if total > 0 else 0, disable=not accelerator.is_main_process, dynamic_ncols=True, desc="itds")

    model.train()
    while (config.max_steps <= 0 or global_step < config.max_steps) and (config.num_epochs <= 0 or epoch < config.num_epochs):
        epoch += 1
        for examples in dataloader:
            with accelerator.accumulate(model):
                steering_batch = accelerator.unwrap_model(model).encode_batch(tokenizer, examples, accelerator.device)
                loss, diagnostics = _loss_for_objective(config, steering_batch)
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), config.gradient_clipping)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    progress.update(1)
                    diagnostics["grad_norm"] = float(grad_norm.detach().cpu()) if hasattr(grad_norm, "detach") else float(grad_norm)
                    diagnostics["learning_rate"] = config.learning_rate
                    if accelerator.is_main_process:
                        progress.set_postfix(loss=f"{diagnostics['loss']:.4g}", reward=f"{diagnostics['reward_mean']:.3g}", refresh=False)
                    if global_step % config.logging_steps == 0:
                        accelerator.print({"step": global_step, **diagnostics})
                        accelerator.log({f"train/{k}": v for k, v in diagnostics.items()}, step=global_step)
                    if global_step > 0 and global_step % config.save_steps == 0:
                        _save_checkpoint(accelerator, model, optimizer, Path(config.output_dir) / f"step_{global_step}", global_step, epoch, config)
                    if config.max_steps > 0 and global_step >= config.max_steps:
                        break
            if config.max_steps > 0 and global_step >= config.max_steps:
                break

    progress.close()
    _save_checkpoint(accelerator, model, optimizer, Path(config.output_dir) / f"step_{global_step}", global_step, epoch, config)
    if config.wandb_project:
        accelerator.end_training()


if __name__ == "__main__":
    main()
