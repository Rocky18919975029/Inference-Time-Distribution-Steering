from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import load_config
from .utils import read_jsonl


def response_logprobs(model, tokenizer, prompt: str, response: str, device: torch.device) -> tuple[float, float, int]:
    full_text = prompt + response
    encoded_full = tokenizer(full_text, add_special_tokens=False, return_tensors="pt").to(device)
    prompt_len = len(tokenizer(prompt, add_special_tokens=False).input_ids)
    full_len = encoded_full.input_ids.shape[1]
    response_num_tokens = full_len - prompt_len
    if response_num_tokens <= 0:
        raise ValueError("Response has no tokens under this tokenizer")

    with torch.no_grad():
        logits = model(**encoded_full).logits
        log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
        target_ids = encoded_full.input_ids[:, 1:]
        token_logprobs = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)[0]
        response_lp = token_logprobs[prompt_len - 1 : full_len - 1].sum()

    total = float(response_lp.detach().cpu())
    return total, total / response_num_tokens, response_num_tokens


def main() -> None:
    parser = argparse.ArgumentParser(description="Add reference response log-probabilities to converted JSONL rows.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--start-index", type=int, default=0, help="Inclusive 0-based input row offset.")
    parser.add_argument("--end-index", type=int, default=-1, help="Exclusive 0-based input row offset. -1 means EOF.")
    args = parser.parse_args()

    if args.start_index < 0:
        raise ValueError("--start-index must be non-negative")
    if args.end_index != -1 and args.end_index < args.start_index:
        raise ValueError("--end-index must be -1 or greater than --start-index")

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if config.bf16 and torch.cuda.is_available() else None
    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name_or_path,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_rows = None if args.end_index == -1 else args.end_index - args.start_index
    count = 0
    empty_response_count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        rows = enumerate(read_jsonl(args.input))
        rows = (
            (row_index, row)
            for row_index, row in rows
            if row_index >= args.start_index and (args.end_index == -1 or row_index < args.end_index)
        )
        for row_index, row in tqdm(rows, desc="reference logprobs", total=total_rows):
            row = dict(row)
            try:
                total, mean, n_tokens = response_logprobs(model, tokenizer, row["prompt"], row["response"], device)
            except ValueError as exc:
                if "Response has no tokens" not in str(exc):
                    raise
                empty_response_count += 1
                total = 0.0
                mean = 0.0
                n_tokens = 0
                row["ref_logprob_warning"] = "empty_response"
                row["ref_logprob_warning_row_index"] = row_index
            row["ref_logprob_sum"] = total
            row["ref_logprob_mean"] = mean
            row["response_num_tokens"] = n_tokens
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    print(f"wrote {count} rows to {args.output}")
    if empty_response_count:
        print(f"empty responses with zero ref logprob: {empty_response_count}")


if __name__ == "__main__":
    main()
