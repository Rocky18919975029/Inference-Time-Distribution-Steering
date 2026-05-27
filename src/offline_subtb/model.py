from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from transformers import AutoModelForCausalLM


@dataclass
class PolicyWithFlowOutput:
    logits: torch.Tensor
    final_hidden_state: torch.Tensor
    flow_values: torch.Tensor


class PolicyWithFlow(nn.Module):
    def __init__(
        self,
        model_name_or_path: str,
        *,
        torch_dtype: torch.dtype | None = None,
        finetune_mode: str = "full",
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules: list[str] | None = None,
        lora_adapter_path: str | None = None,
    ):
        super().__init__()
        self.finetune_mode = finetune_mode
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )
        if finetune_mode == "lora":
            try:
                from peft import LoraConfig, PeftModel, TaskType, get_peft_model
            except ImportError as exc:
                raise ImportError("LoRA training requires peft. Install it with: pip install peft") from exc

            if lora_adapter_path:
                self.base_model = PeftModel.from_pretrained(self.base_model, lora_adapter_path, is_trainable=True)
            else:
                if not lora_target_modules:
                    lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
                lora_config = LoraConfig(
                    r=lora_r,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    bias="none",
                    task_type=TaskType.CAUSAL_LM,
                    target_modules=lora_target_modules,
                )
                self.base_model = get_peft_model(self.base_model, lora_config)
        elif finetune_mode != "full":
            raise ValueError(f"Unknown finetune_mode={finetune_mode!r}; expected 'full' or 'lora'.")
        hidden_size = self.base_model.config.hidden_size
        self.flow_head = nn.Linear(hidden_size, 1)

    def trainable_parameter_counts(self) -> tuple[int, int]:
        trainable = sum(param.numel() for param in self.parameters() if param.requires_grad)
        total = sum(param.numel() for param in self.parameters())
        return trainable, total

    def gradient_checkpointing_enable(self) -> None:
        self.base_model.gradient_checkpointing_enable()
        if self.finetune_mode == "lora" and hasattr(self.base_model, "enable_input_require_grads"):
            self.base_model.enable_input_require_grads()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> PolicyWithFlowOutput:
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        final_hidden_state = outputs.hidden_states[-1]
        flow_values = self.flow_head(final_hidden_state).squeeze(-1)
        return PolicyWithFlowOutput(
            logits=outputs.logits,
            final_hidden_state=final_hidden_state,
            flow_values=flow_values,
        )
