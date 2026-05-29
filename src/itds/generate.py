from __future__ import annotations

import torch

from .model import TopKLowRankSteering


@torch.no_grad()
def generate_one(
    model: TopKLowRankSteering,
    tokenizer,
    prompt: str,
    *,
    max_new_tokens: int = 1024,
    temperature: float = 0.0,
    top_p: float = 1.0,
    device: torch.device | str,
) -> str:
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    input_ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    generated: list[int] = []
    past_key_values = None
    next_input_ids = input_ids
    for _ in range(max_new_tokens):
        output = model.base_model(
            input_ids=next_input_ids,
            past_key_values=past_key_values,
            output_hidden_states=True,
            use_cache=True,
        )
        past_key_values = output.past_key_values
        base_logits = output.logits[0, -1]
        hidden = output.hidden_states[-1][0, -1]
        top_values, top_ids = torch.topk(base_logits, k=min(model.top_k, base_logits.shape[-1]))
        steering_dtype = model.state_projector[0].weight.dtype
        top_values = top_values.to(dtype=steering_dtype)
        u = model.state_projector(hidden.to(dtype=steering_dtype))
        steering = (model.token_basis(top_ids) * u.unsqueeze(0)).sum(dim=-1)
        logits = top_values + model.alpha * steering
        if temperature and temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)
            if top_p < 1.0:
                sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                cumulative = sorted_probs.cumsum(dim=-1)
                keep = cumulative <= top_p
                keep[0] = True
                filtered = sorted_probs * keep
                filtered = filtered / filtered.sum()
                choice = sorted_idx[torch.multinomial(filtered, 1)]
            else:
                choice = torch.multinomial(probs, 1)
            next_id = top_ids[int(choice.item())]
        else:
            next_id = top_ids[int(torch.argmax(logits).item())]
        token_id = int(next_id.item())
        if tokenizer.eos_token_id is not None and token_id == tokenizer.eos_token_id:
            break
        generated.append(token_id)
        next_input_ids = next_id.reshape(1, 1)
    return tokenizer.decode(generated, skip_special_tokens=True)
