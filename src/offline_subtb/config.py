from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class TrainConfig:
    model_name_or_path: str
    train_data_path: str
    output_dir: str
    template: str = "qwen-boxed"
    alpha: float = 1.0
    beta: float = 1.0
    lambda_subtb: float = 1.0
    max_pairs_per_trace: int = 16
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 2e-6
    weight_decay: float = 0.0
    max_steps: int = 1000
    num_epochs: int = 0
    save_steps: int = 100
    logging_steps: int = 10
    gradient_clipping: float = 1.0
    bf16: bool = True
    gradient_checkpointing: bool = True
    finetune_mode: str = "full"
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: str = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
    seed: int = 1
    dataloader_num_workers: int = 0
    resume_from_checkpoint: str | None = None
    wandb_project: str | None = "one_step_posttrain"
    wandb_entity: str | None = None
    wandb_run_name: str | None = None
    wandb_run_id: str | None = None
    wandb_mode: str | None = None
    limit_of_rlvr_dir: str | None = None
    limit_data_dir: str | None = None
    eval_every_steps: int = 0
    eval_data_path: str | None = "data/test.parquet"
    eval_output_dir: str | None = None
    eval_benchmark: str = "eval_test"
    eval_prompt_type: str = "qwen-boxed"
    eval_seed: int = 1
    eval_max_samples: int = 128
    eval_n_sampling: int = 1
    eval_temperature: float = 0.0
    eval_top_p: float = 1.0
    eval_max_tokens: int = 1024


def load_config(path: str | Path) -> TrainConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        data: dict[str, Any] = yaml.safe_load(handle) or {}
    return TrainConfig(**data)
