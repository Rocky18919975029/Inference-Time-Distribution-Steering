#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _load_jsonl_row(path: Path, row_index: int) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        for i, line in enumerate(handle):
            if not line.strip():
                continue
            if i == row_index:
                return json.loads(line)
    raise IndexError(f"Row {row_index} not found in {path}")


def _sample_value(value: Any, sample_id: int | None):
    if isinstance(value, list):
        if sample_id is not None and 0 <= sample_id < len(value):
            return value[sample_id]
        return value[0] if value else ""
    return value


def _select_response(row: dict[str, Any], fallback_response: str = "") -> tuple[str, str]:
    sample_id = row.get("sample_id")
    try:
        sample_id = int(sample_id) if sample_id is not None else None
    except (TypeError, ValueError):
        sample_id = None

    for field in ("full_response", "code", "completion", "generated_text", "text", "response"):
        if field not in row:
            continue
        value = _sample_value(row.get(field), sample_id)
        if value is not None and str(value):
            return str(value), field

    return fallback_response, "arg:response" if fallback_response else "missing"


def _entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    return -(probs * log_probs).sum(dim=-1)


def _decode_token(tokenizer, token_id: int) -> str:
    return tokenizer.decode([token_id], skip_special_tokens=False).replace("\n", "\\n")


def _extract_boxed_answer(text: str) -> str:
    """
    Lightweight extractor for convenience only.
    The official verifier should still be used for final scoring.
    """
    matches = re.findall(r"\\boxed\{([^{}]*)\}", text)
    if matches:
        return matches[-1].strip()

    # Fallback: "final answer is: 118" style.
    m = re.search(r"final answer is[:\s]+([^\n\.]+)", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()

    return ""


def _normalize_answer(x: Any) -> str:
    return str(x).strip().replace(" ", "")


def _lightweight_score(pred: str, gt: Any) -> bool | None:
    if pred == "":
        return None
    if gt is None:
        return None
    return _normalize_answer(pred) == _normalize_answer(gt)


def _get_problem_fields(row: dict[str, Any]) -> dict[str, Any]:
    problem_id = row.get("problem_id", row.get("idx", row.get("row_index", 0)))
    question = row.get("question", "")

    gt = row.get("ground_truth", row.get("gt", row.get("answer", "")))
    answer = row.get("answer", gt)

    level = row.get("level", "")
    gt_cot = row.get("gt_cot", None)

    return {
        "idx": problem_id,
        "question": question,
        "gt_cot": gt_cot,
        "gt": gt,
        "level": level,
        "answer": answer,
    }


def _find_top_entropy_positions(
    *,
    model,
    tokenizer,
    device: torch.device,
    prompt: str,
    response: str,
    top_k: int,
    top_n: int,
    max_length: int,
) -> list[dict[str, Any]]:
    if not prompt or not response:
        raise ValueError("Both prompt and response must be non-empty.")

    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    full_text = prompt + response

    encoded = tokenizer(
        full_text,
        add_special_tokens=False,
        return_tensors="pt",
        return_offsets_mapping=True,
        truncation=bool(max_length),
        max_length=max_length if max_length else None,
    )

    input_ids = encoded.input_ids.to(device)
    offsets = encoded.offset_mapping[0].tolist()

    if input_ids.shape[1] < 2:
        raise ValueError("Need at least two tokens to inspect next-token entropy.")

    with torch.no_grad():
        logits = model(input_ids=input_ids).logits[0, :-1, :]

    response_start = len(prompt)
    candidates: list[dict[str, Any]] = []

    for pos in range(logits.shape[0]):
        target_pos = pos + 1
        char_start, char_end = offsets[target_pos]

        if char_end <= response_start:
            continue

        target_id = int(input_ids[0, target_pos].item())

        top_values, top_ids = torch.topk(logits[pos], k=min(top_k, logits.shape[-1]))
        entropy_nats = float(_entropy_from_logits(top_values.float()).cpu())
        entropy_bits = entropy_nats / math.log(2)

        rank_matches = (top_ids == target_id).nonzero(as_tuple=False)
        target_rank = int(rank_matches[0].item()) + 1 if rank_matches.numel() else None

        candidates.append(
            {
                "position": int(target_pos),
                "response_token_index": int(target_pos - len(prompt_ids)),
                "char_span": [int(char_start), int(char_end)],
                "response_char_span": [
                    int(char_start - response_start),
                    int(char_end - response_start),
                ],
                "entropy_nats": entropy_nats,
                "entropy_bits": entropy_bits,
                "target_token_id": target_id,
                "target_token": _decode_token(tokenizer, target_id),
                "target_rank_in_top_k": target_rank,
                "top_tokens": [
                    {
                        "rank": rank + 1,
                        "token_id": int(token_id),
                        "token": _decode_token(tokenizer, int(token_id)),
                        "prob": float(prob),
                    }
                    for rank, (token_id, prob) in enumerate(
                        zip(
                            top_ids[:10].tolist(),
                            torch.softmax(top_values.float(), dim=-1)[:10].cpu().tolist(),
                            strict=True,
                        )
                    )
                ],
            }
        )

    return sorted(candidates, key=lambda item: item["entropy_nats"], reverse=True)[:top_n]


def _truncate_input_ids_from_left(input_ids: torch.Tensor, max_input_tokens: int) -> torch.Tensor:
    if max_input_tokens <= 0:
        return input_ids
    if input_ids.shape[1] <= max_input_tokens:
        return input_ids
    return input_ids[:, -max_input_tokens:]


def _generate_continuation(
    *,
    model,
    tokenizer,
    device: torch.device,
    text_prefix: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k_sampling: int,
    seed: int,
    max_input_tokens: int,
) -> tuple[str, str]:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    encoded = tokenizer(text_prefix, add_special_tokens=False, return_tensors="pt")
    input_ids = encoded.input_ids.to(device)
    input_ids = _truncate_input_ids_from_left(input_ids, max_input_tokens)

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    gen_kwargs = {
        "input_ids": input_ids,
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "temperature": temperature if temperature > 0 else None,
        "top_p": top_p,
        "pad_token_id": pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    if top_k_sampling > 0:
        gen_kwargs["top_k"] = top_k_sampling

    # Remove None values because generate may reject temperature=None.
    gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

    with torch.no_grad():
        output_ids = model.generate(**gen_kwargs)

    new_token_ids = output_ids[0, input_ids.shape[1] :]
    continuation = tokenizer.decode(new_token_ids, skip_special_tokens=True)

    finish_reason = "stop"
    if max_new_tokens > 0 and len(new_token_ids) >= max_new_tokens:
        finish_reason = "length"

    return continuation, finish_reason


def _build_output_row(
    *,
    source_row: dict[str, Any],
    original_response: str,
    branch_responses: list[str],
    branch_metadata: list[dict[str, Any]],
    model_name: str,
    entropy_top_k: int,
    rerollout_temperature: float,
    rerollout_top_p: float,
    rerollout_max_new_tokens: int,
    finish_reasons: list[str],
) -> dict[str, Any]:
    fields = _get_problem_fields(source_row)
    all_responses = [original_response] + branch_responses

    gt = fields["gt"]
    preds = [_extract_boxed_answer(resp) for resp in all_responses]
    scores = [_lightweight_score(pred, gt) for pred in preds]

    report = [
        {
            "source": "original",
            "branch_id": 0,
            "cut_response_char": None,
            "cut_response_token_index": None,
            "entropy_nats": None,
            "target_token": None,
        }
    ]

    for i, meta in enumerate(branch_metadata, start=1):
        start, _ = meta["response_char_span"]
        report.append(
            {
                "source": "entropy_rerollout",
                "branch_id": i,
                "cut_response_char": start,
                "cut_response_token_index": meta["response_token_index"],
                "entropy_nats": meta["entropy_nats"],
                "entropy_bits": meta["entropy_bits"],
                "target_token": meta["target_token"],
                "target_rank_in_top_k": meta["target_rank_in_top_k"],
                "top_tokens": meta["top_tokens"],
            }
        )

    return {
        **fields,
        "code": all_responses,
        "pred": preds,
        "report": report,
        "finish_reason": ["original"] + finish_reasons,
        "score": scores,
        "rerollout_metadata": {
            "model_name_or_path": model_name,
            "entropy_top_k": entropy_top_k,
            "num_entropy_branches": len(branch_responses),
            "temperature": rerollout_temperature,
            "top_p": rerollout_top_p,
            "max_new_tokens": rerollout_max_new_tokens,
            "note": (
                "score is lightweight exact-string comparison from locally extracted boxed answer; "
                "use the repository verifier for official scoring."
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Given one prompt/response row, find top-N high-entropy response token positions, "
            "rerollout one continuation from each position, and save original + N branched responses "
            "in limit-of-RLVR-style JSONL."
        )
    )

    parser.add_argument("--input", required=True, help="Input JSONL, e.g. data/full_train_subtb.jsonl")
    parser.add_argument("--row", type=int, default=0, help="Row index in input JSONL.")
    parser.add_argument("--output", required=True, help="Output JSONL path.")

    parser.add_argument("--model", default="Qwen/Qwen2.5-7B")
    parser.add_argument("--entropy-top-k", type=int, default=20, help="Top-k logits used to compute entropy.")
    parser.add_argument("--top-n", type=int, default=5, help="Number of entropy positions to branch from.")

    parser.add_argument("--max-length", type=int, default=0, help="Optional truncation length for entropy forward pass.")
    parser.add_argument("--max-input-tokens", type=int, default=0, help="Left-truncate generation input to this many tokens.")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k-sampling", type=int, default=0, help="Generation top-k sampling. 0 disables it.")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--cut-after-token",
        action="store_true",
        help=(
            "Default cuts before the high-entropy token, so that token is resampled. "
            "If set, cuts after that token instead."
        ),
    )

    args = parser.parse_args()

    row = _load_jsonl_row(Path(args.input), args.row)

    prompt = str(row.get("prompt", ""))
    if not prompt:
        raise ValueError("Input row must contain a non-empty 'prompt' field.")

    original_response, response_source = _select_response(row)
    if not original_response:
        raise ValueError("Could not find non-empty response from row.")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    dtype = torch.bfloat16 if torch.cuda.is_available() else None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    top_positions = _find_top_entropy_positions(
        model=model,
        tokenizer=tokenizer,
        device=device,
        prompt=prompt,
        response=original_response,
        top_k=args.entropy_top_k,
        top_n=args.top_n,
        max_length=args.max_length,
    )

    branch_responses: list[str] = []
    branch_metadata: list[dict[str, Any]] = []
    finish_reasons: list[str] = []

    for branch_idx, pos in enumerate(top_positions, start=1):
        start, end = pos["response_char_span"]
        cut_char = end if args.cut_after_token else start

        if cut_char < 0 or cut_char > len(original_response):
            continue

        response_prefix = original_response[:cut_char]
        generation_prefix = prompt + response_prefix

        continuation, finish_reason = _generate_continuation(
            model=model,
            tokenizer=tokenizer,
            device=device,
            text_prefix=generation_prefix,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k_sampling=args.top_k_sampling,
            seed=args.seed + branch_idx,
            max_input_tokens=args.max_input_tokens,
        )

        new_response = response_prefix + continuation
        branch_responses.append(new_response)
        branch_metadata.append(pos)
        finish_reasons.append(finish_reason)

        print(
            json.dumps(
                {
                    "branch": branch_idx,
                    "cut_char": cut_char,
                    "response_token_index": pos["response_token_index"],
                    "entropy_nats": pos["entropy_nats"],
                    "target_token": pos["target_token"],
                    "finish_reason": finish_reason,
                    "new_response_chars": len(new_response),
                },
                ensure_ascii=False,
            )
        )

    output_row = _build_output_row(
        source_row=row,
        original_response=original_response,
        branch_responses=branch_responses,
        branch_metadata=branch_metadata,
        model_name=args.model,
        entropy_top_k=args.entropy_top_k,
        rerollout_temperature=args.temperature,
        rerollout_top_p=args.top_p,
        rerollout_max_new_tokens=args.max_new_tokens,
        finish_reasons=finish_reasons,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(output_row, ensure_ascii=False) + "\n")

    print(f"wrote {1 + len(branch_responses)} responses to {output_path}")
    print(f"response_source={response_source}")


if __name__ == "__main__":
    main()