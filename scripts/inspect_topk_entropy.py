#!/usr/bin/env python
from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _load_rows(path: Path, row_indices: set[int], max_row: int) -> list[tuple[int, dict]]:
    rows: list[tuple[int, dict]] = []
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            if index in row_indices:
                rows.append((index, json.loads(line)))
            if index >= max_row:
                break
    return rows


def _parse_rows(value: str, start_row: int, num_rows: int) -> list[int]:
    if value:
        return [int(item.strip()) for item in value.split(",") if item.strip()]
    return list(range(start_row, start_row + num_rows))


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


def _inspect_response(
    *,
    model,
    tokenizer,
    device: torch.device,
    prompt: str,
    response: str,
    row_index: int | None,
    row: dict,
    top_k: int,
    top_n: int,
    context_chars: int,
    max_length: int,
) -> dict:
    if not prompt or not response:
        raise ValueError(f"Both prompt and response must be non-empty for row {row_index}.")

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
        raise ValueError(f"Need at least two tokens to inspect next-token entropy for row {row_index}.")

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
        top_values, top_ids = torch.topk(logits[pos], k=min(top_k, logits.shape[-1]))
        entropy_nats = float(_entropy_from_logits(top_values.float()).cpu())
        entropy_bits = entropy_nats / math.log(2)
        rank_matches = (top_ids == target_id).nonzero(as_tuple=False)
        target_rank = int(rank_matches[0].item()) + 1 if rank_matches.numel() else None
        target_logprob = None
        if target_rank is not None:
            target_logprob = float(torch.log_softmax(top_values.float(), dim=-1)[target_rank - 1].cpu())
        candidates.append(
            {
                "row": row_index,
                "problem_id": row.get("problem_id", row.get("idx", row_index)),
                "position": int(target_pos),
                "response_token_index": int(target_pos - len(prompt_ids)),
                "char_span": [int(char_start), int(char_end)],
                "response_char_span": [int(char_start - response_start), int(char_end - response_start)],
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
                "context": _context(full_text, int(char_start), int(char_end), context_chars),
            }
        )

    top = sorted(candidates, key=lambda item: item["entropy_nats"], reverse=True)[:top_n]
    return {
        "row": row_index,
        "problem_id": row.get("problem_id", row.get("idx", row_index)),
        "reward": row.get("reward", row.get("is_correct")),
        "response": response,
        "num_response_tokens_scored": len(candidates),
        "top_entropy_positions": top,
    }


def _highlighted_response(response: str, positions: list[dict]) -> str:
    highlights = []
    entropies = [item["entropy_nats"] for item in positions]
    min_entropy = min(entropies) if entropies else 0.0
    max_entropy = max(entropies) if entropies else 1.0
    denom = max(max_entropy - min_entropy, 1e-9)
    for rank, item in enumerate(positions, start=1):
        start, end = item["response_char_span"]
        if start < 0 or end <= start:
            continue
        intensity = (item["entropy_nats"] - min_entropy) / denom
        alpha = 0.25 + 0.65 * intensity
        title = (
            f"rank #{rank} | entropy={item['entropy_nats']:.4f} nats / {item['entropy_bits']:.4f} bits"
            f" | token={item['target_token']!r} | top-k-rank={item['target_rank_in_top_k']}"
        )
        top_tokens = ", ".join(f"{tok['rank']}:{tok['token']}({tok['prob']:.3f})" for tok in item["top_tokens"][:5])
        highlights.append(
            {
                "start": start,
                "end": end,
                "rank": rank,
                "style": f"background: rgba(255, 80, 0, {alpha:.3f}); border-bottom: 2px solid rgba(150, 0, 0, 0.55);",
                "title": title + " | top tokens: " + top_tokens,
            }
        )
    highlights.sort(key=lambda item: (item["start"], item["end"]))

    chunks = []
    cursor = 0
    for item in highlights:
        start = max(cursor, min(len(response), item["start"]))
        end = max(start, min(len(response), item["end"]))
        chunks.append(html.escape(response[cursor:start]))
        token_text = html.escape(response[start:end])
        chunks.append(
            f'<span class="hot-token" style="{item["style"]}" title="{html.escape(item["title"])}">'
            f'<sup>{item["rank"]}</sup>{token_text}</span>'
        )
        cursor = end
    chunks.append(html.escape(response[cursor:]))
    return "".join(chunks)


def _write_html(result: dict, path: Path) -> None:
    blocks = []
    for item in result["per_response"]:
        highlighted = _highlighted_response(item["response"], item["top_entropy_positions"])
        rows = []
        for rank, pos in enumerate(item["top_entropy_positions"], start=1):
            top_tokens = " ".join(
                f"<code>{html.escape(tok['token'])}</code>:{tok['prob']:.3f}" for tok in pos["top_tokens"][:5]
            )
            rows.append(
                "<tr>"
                f"<td>{rank}</td>"
                f"<td>{pos['response_token_index']}</td>"
                f"<td>{pos['entropy_nats']:.4f}</td>"
                f"<td>{pos['entropy_bits']:.4f}</td>"
                f"<td><code>{html.escape(pos['target_token'])}</code></td>"
                f"<td>{pos['target_rank_in_top_k']}</td>"
                f"<td>{top_tokens}</td>"
                "</tr>"
            )
        blocks.append(
            f"""
            <section>
              <h2>row={html.escape(str(item["row"]))} problem_id={html.escape(str(item["problem_id"]))} reward={html.escape(str(item["reward"]))}</h2>
              <div class="response">{highlighted}</div>
              <table>
                <thead><tr><th>#</th><th>response token</th><th>entropy nats</th><th>entropy bits</th><th>target</th><th>target rank</th><th>top candidates</th></tr></thead>
                <tbody>{''.join(rows)}</tbody>
              </table>
            </section>
            """
        )
    document = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Top-k Entropy Heatmap</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; color: #222; }}
    h1 {{ margin-bottom: 0; }}
    .meta {{ color: #666; margin: 8px 0 24px; }}
    section {{ border-top: 1px solid #ddd; padding-top: 20px; margin-top: 24px; }}
    .response {{ white-space: pre-wrap; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; line-height: 1.65; border: 1px solid #ddd; padding: 16px; border-radius: 6px; background: #fafafa; }}
    .hot-token {{ border-radius: 3px; padding: 1px 2px; }}
    sup {{ font-size: 10px; color: #7a0000; margin-right: 1px; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 12px; font-size: 13px; }}
    th, td {{ border: 1px solid #ddd; padding: 6px 8px; vertical-align: top; }}
    th {{ background: #f2f2f2; text-align: left; }}
    code {{ background: #f4f4f4; padding: 1px 3px; border-radius: 3px; }}
  </style>
</head>
<body>
  <h1>Top-k Entropy Heatmap</h1>
  <div class="meta">model={html.escape(str(result["model"]))} | top-k={result["top_k"]} | responses={result["num_responses"]} | scored tokens={result["num_response_tokens_scored"]}</div>
  {''.join(blocks)}
</body>
</html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect per-token base top-k entropy for prompt/response trajectories.")
    parser.add_argument("--input", help="Training JSONL containing prompt and response.")
    parser.add_argument("--row", type=int, default=0, help="JSONL row index to inspect.")
    parser.add_argument("--rows", default="", help="Comma-separated JSONL row indices to inspect, e.g. 0,5,9.")
    parser.add_argument("--start-row", type=int, default=0)
    parser.add_argument("--num-rows", type=int, default=1)
    parser.add_argument("--prompt", default="", help="Prompt text. Used when --input is omitted.")
    parser.add_argument("--response", default="", help="Response text. Used when --input is omitted.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B")
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--context-chars", type=int, default=80)
    parser.add_argument("--max-length", type=int, default=0, help="Optional left-truncation length for model input.")
    parser.add_argument("--output", default="", help="Optional JSON output path.")
    parser.add_argument("--html-output", default="", help="Optional HTML heatmap output path.")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    dtype = torch.bfloat16 if torch.cuda.is_available() else None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, trust_remote_code=True).to(device)
    model.eval()

    if args.input:
        row_indices = _parse_rows(args.rows, args.start_row if args.rows else args.row, args.num_rows)
        loaded_rows = _load_rows(Path(args.input), set(row_indices), max(row_indices))
    else:
        loaded_rows = [(None, {"prompt": args.prompt, "response": args.response})]

    per_response = []
    global_positions = []
    for row_index, row in loaded_rows:
        item = _inspect_response(
            model=model,
            tokenizer=tokenizer,
            device=device,
            prompt=str(row.get("prompt", "")),
            response=str(row.get("response", "")),
            row_index=row_index,
            row=row,
            top_k=args.top_k,
            top_n=args.top_n,
            context_chars=args.context_chars,
            max_length=args.max_length,
        )
        per_response.append(item)
        global_positions.extend(item["top_entropy_positions"])

    result = {
        "input": args.input,
        "rows": [item[0] for item in loaded_rows],
        "model": args.model,
        "top_k": args.top_k,
        "num_responses": len(per_response),
        "num_response_tokens_scored": sum(item["num_response_tokens_scored"] for item in per_response),
        "global_top_entropy_positions": sorted(global_positions, key=lambda item: item["entropy_nats"], reverse=True)[: args.top_n],
        "per_response": per_response,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.html_output:
        _write_html(result, Path(args.html_output))


if __name__ == "__main__":
    main()
