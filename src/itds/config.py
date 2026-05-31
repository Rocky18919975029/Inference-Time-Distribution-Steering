from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ITDSConfig:
    model_name_or_path: str
    train_data_path: str
    output_dir: str
    objective: str = "tb"
    top_k: int = 64
    rank: int = 32
    actor_depth: int = 10
    critic_depth: int = 10
    alpha: float = 1.0
    token_basis_init_std: float = 1e-3
    beta: float = 0.1
    clip_epsilon: float = 0.2
    value_loss_weight: float = 0.1
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    max_steps: int = 1000
    num_epochs: int = 0
    save_steps: int = 500
    logging_steps: int = 10
    gradient_clipping: float = 1.0
    bf16: bool = True
    seed: int = 1
    dataloader_num_workers: int = 0
    resume_from_checkpoint: str | None = None
    wandb_project: str | None = "itds"
    wandb_entity: str | None = None
    wandb_run_name: str | None = None
    wandb_run_id: str | None = None
    wandb_mode: str | None = None


def load_config(path: str | Path) -> ITDSConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        data: dict[str, Any] = yaml.safe_load(handle) or {}
    return ITDSConfig(**data)
