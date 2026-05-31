#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _load_row(path: Path, row_index: int) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            if index == row_index:
                return json.loads(line)
    raise IndexError(f"row {row_index} not found in {path}")


def _entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    return -(probs * log_probs).sum(dim=-1)


def _decode_token(tokenizer, token_id: int) -> str:
    return tokenizer.decode([token_id], skip_special_tokens=False).replace("\n", "\\n")


def _context(text: str, char_start: int, char_end: int, window: int) -> str:
    left = max(0, char_start - window)
    right = min(len(text), char_end + window)
    return text[left:char_start] + "<<<" + text[char_start:char_end] + ">>>" + text[char_end:right]


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect per-token base top-k entropy for one prompt/response trajectory.")
    parser.add_argument("--input", help="Training JSONL containing prompt and response.")
    parser.add_argument("--row", type=int, default=0, help="JSONL row index to inspect.")
    parser.add_argument("--prompt", default="", help="Prompt text. Used when --input is omitted.")
    parser.add_argument("--response", default="", help="Response text. Used when --input is omitted.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B")
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--context-chars", type=int, default=80)
    parser.add_argument("--max-length", type=int, default=0, help="Optional left-truncation length for model input.")
    parser.add_argument("--output", default="", help="Optional JSON output path.")
    args = parser.parse_args()

    if args.input:
        row = _load_row(Path(args.input), args.row)
        prompt = str(row.get("prompt", ""))
        response = str(row.get("response", ""))
    else:
        row = {}
        prompt = args.prompt
        response = args.response
    if not prompt or not response:
        raise ValueError("Both prompt and response must be non-empty.")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    dtype = torch.bfloat16 if torch.cuda.is_available() else None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, trust_remote_code=True).to(device)
    model.eval()

    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    full_text = prompt + response
    encoded = tokenizer(
        full_text,
        add_special_tokens=False,
        return_tensors="pt",
        return_offsets_mapping=True,
        truncation=bool(args.max_length),
        max_length=args.max_length if args.max_length else None,
    )
    input_ids = encoded.input_ids.to(device)
    offsets = encoded.offset_mapping[0].tolist()
    if input_ids.shape[1] < 2:
        raise ValueError("Need at least two tokens to inspect next-token entropy.")

    with torch.no_grad():
        logits = model(input_ids=input_ids).logits[0, :-1, :]

    response_start = len(prompt)
    candidates = []
    for pos in range(logits.shape[0]):
        target_pos = pos + 1
        char_start, char_end = offsets[target_pos]
        if char_end <= response_start:
            continue
        target_id = int(input_ids[0, target_pos].item())
        top_values, top_ids = torch.topk(logits[pos], k=min(args.top_k, logits.shape[-1]))
        entropy_nats = float(_entropy_from_logits(top_values.float()).cpu())
        entropy_bits = entropy_nats / math.log(2)
        rank_matches = (top_ids == target_id).nonzero(as_tuple=False)
        target_rank = int(rank_matches[0].item()) + 1 if rank_matches.numel() else None
        target_logprob = None
        if target_rank is not None:
            target_logprob = float(torch.log_softmax(top_values.float(), dim=-1)[target_rank - 1].cpu())
        candidates.append(
            {
                "position": int(target_pos),
                "response_token_index": int(target_pos - len(prompt_ids)),
                "char_span": [int(char_start), int(char_end)],
                "entropy_nats": entropy_nats,
                "entropy_bits": entropy_bits,
                "target_token_id": target_id,
                "target_token": _decode_token(tokenizer, target_id),
                "target_rank_in_top_k": target_rank,
                "target_logprob_top_k": target_logprob,
                "top_tokens": [
                    {
                        "rank": rank + 1,
                        "token_id": int(token_id),
                        "token": _decode_token(tokenizer, int(token_id)),
                        "prob": float(prob),
                    }
                    for rank, (token_id, prob) in enumerate(
                        zip(top_ids[:10].tolist(), torch.softmax(top_values.float(), dim=-1)[:10].cpu().tolist(), strict=True)
                    )
                ],
                "context": _context(full_text, int(char_start), int(char_end), args.context_chars),
            }
        )

    top = sorted(candidates, key=lambda item: item["entropy_nats"], reverse=True)[: args.top_n]
    result = {
        "input": args.input,
        "row": args.row if args.input else None,
        "model": args.model,
        "top_k": args.top_k,
        "num_response_tokens_scored": len(candidates),
        "top_entropy_positions": top,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
