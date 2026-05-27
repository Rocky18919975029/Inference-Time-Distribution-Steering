from __future__ import annotations

import random
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .dataset import TrainingExample
from .tokenization import compute_boundary_positions, sample_uniform_pairs


@dataclass
class SubTBLossOutput:
    loss: torch.Tensor
    diagnostics: dict[str, float]


def token_logprobs_from_logits(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
    target_ids = input_ids[:, 1:]
    return log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def compute_newline_subtb_loss(
    *,
    model,
    tokenizer,
    examples: list[TrainingExample],
    alpha: float,
    beta: float,
    lambda_subtb: float,
    max_pairs_per_trace: int,
    rng: random.Random | None = None,
    device: torch.device | str | None = None,
) -> SubTBLossOutput:
    rng = rng or random.Random()
    device = torch.device(device or next(model.parameters()).device)

    per_pair_losses: list[torch.Tensor] = []
    deltas: list[float] = []
    segment_logprobs: list[float] = []
    logf_starts: list[float] = []
    logf_ends: list[float] = []
    log_rs: list[float] = []
    rewards: list[float] = []
    ref_means: list[float] = []
    num_blocks: list[float] = []
    num_pairs: list[float] = []
    skipped_zero_spans = 0
    response_tokens: list[float] = []
    valid_items = []

    for example in examples:
        boundaries = compute_boundary_positions(example.prompt, example.response, tokenizer)
        pairs = sample_uniform_pairs(boundaries.boundary_positions, max_pairs_per_trace, rng)
        all_pairs_count = len(pairs)
        if not pairs:
            continue
        valid_items.append((example, boundaries, pairs, all_pairs_count))

    if valid_items:
        full_texts = [example.prompt + example.response for example, _, _, _ in valid_items]
        original_padding_side = getattr(tokenizer, "padding_side", None)
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        try:
            encoded = tokenizer(
                full_texts,
                add_special_tokens=False,
                padding=True,
                return_tensors="pt",
            )
        finally:
            if original_padding_side is not None:
                tokenizer.padding_side = original_padding_side
        input_ids = encoded.input_ids.to(device)
        attention_mask = getattr(encoded, "attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        output = model(input_ids=input_ids, attention_mask=attention_mask)
        batch_token_logprobs = token_logprobs_from_logits(output.logits, input_ids)

    for item_index, (example, boundaries, pairs, all_pairs_count) in enumerate(valid_items):
        token_logprobs = batch_token_logprobs[item_index]
        flow_values = output.flow_values[item_index]
        log_r = torch.as_tensor(
            alpha * example.ref_logprob_mean + example.reward / beta,
            dtype=flow_values.dtype,
            device=device,
        )

        for m, n in pairs:
            start = boundaries.boundary_positions[m]
            end = boundaries.boundary_positions[n]
            if end <= start:
                skipped_zero_spans += 1
                continue
            segment_logprob = token_logprobs[start - 1 : end - 1].sum()

            logf_m = flow_values[start - 1]
            logf_n = log_r if n == len(boundaries.blocks) else flow_values[end - 1]
            delta = logf_m + segment_logprob - logf_n
            weight = lambda_subtb ** (n - m)
            per_pair_losses.append(weight * delta.square())

            deltas.append(float(delta.detach().cpu()))
            segment_logprobs.append(float(segment_logprob.detach().cpu()))
            logf_starts.append(float(logf_m.detach().cpu()))
            logf_ends.append(float(logf_n.detach().cpu()))

        log_rs.append(float(log_r.detach().cpu()))
        rewards.append(float(example.reward))
        ref_means.append(float(example.ref_logprob_mean))
        num_blocks.append(float(len(boundaries.blocks)))
        num_pairs.append(float(all_pairs_count))
        response_tokens.append(float(example.response_num_tokens))

    if not per_pair_losses:
        loss = next(model.parameters()).sum() * 0.0
        diagnostics = {
            "loss": 0.0,
            "delta_mean": 0.0,
            "delta_std": 0.0,
            "delta_abs_mean": 0.0,
            "segment_logprob_mean": 0.0,
            "logF_start_mean": 0.0,
            "logF_end_mean": 0.0,
            "log_R_mean": _mean(log_rs),
            "reward_mean": _mean(rewards),
            "ref_logprob_mean_mean": _mean(ref_means),
            "num_blocks_mean": _mean(num_blocks),
            "num_pairs_mean": _mean(num_pairs),
            "num_zero_span_skipped": float(skipped_zero_spans),
            "response_num_tokens_mean": _mean(response_tokens),
            "num_no_valid_pair_batches": 1.0,
        }
        return SubTBLossOutput(loss=loss, diagnostics=diagnostics)

    loss = torch.stack(per_pair_losses).mean()
    delta_tensor = torch.tensor(deltas) if deltas else torch.tensor([0.0])
    diagnostics = {
        "loss": float(loss.detach().cpu()),
        "delta_mean": float(delta_tensor.mean()),
        "delta_std": float(delta_tensor.std(unbiased=False)),
        "delta_abs_mean": float(delta_tensor.abs().mean()),
        "segment_logprob_mean": _mean(segment_logprobs),
        "logF_start_mean": _mean(logf_starts),
        "logF_end_mean": _mean(logf_ends),
        "log_R_mean": _mean(log_rs),
        "reward_mean": _mean(rewards),
        "ref_logprob_mean_mean": _mean(ref_means),
        "num_blocks_mean": _mean(num_blocks),
        "num_pairs_mean": _mean(num_pairs),
        "num_zero_span_skipped": float(skipped_zero_spans),
        "response_num_tokens_mean": _mean(response_tokens),
        "num_no_valid_pair_batches": 0.0,
    }
    return SubTBLossOutput(loss=loss, diagnostics=diagnostics)
