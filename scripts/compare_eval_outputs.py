#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            idx = str(row.get("idx", line_number - 1))
            if idx in rows:
                raise ValueError(f"Duplicate idx={idx!r} in {path} at line {line_number}")
            rows[idx] = row
    return rows


def _first(value: Any, default: Any = "") -> Any:
    if isinstance(value, list):
        return value[0] if value else default
    return value if value is not None else default


def _is_correct(row: dict[str, Any]) -> bool:
    return bool(_first(row.get("score"), False))


def _is_invalid(row: dict[str, Any]) -> bool:
    return _first(row.get("finish_reason")) == "invalid_eval_row"


def _format_example(idx: str, base: dict[str, Any], itds: dict[str, Any]) -> str:
    question = itds.get("question") or base.get("question") or ""
    gt = itds.get("gt") or base.get("gt") or ""
    return "\n".join(
        [
            "=" * 100,
            f"idx: {idx}",
            f"question: {question}",
            f"gt: {gt}",
            f"base_pred: {_first(base.get('pred'))}",
            f"itds_pred: {_first(itds.get('pred'))}",
            "-" * 40 + " BASE RESPONSE " + "-" * 45,
            str(_first(base.get("code"))),
            "-" * 40 + " ITDS RESPONSE " + "-" * 45,
            str(_first(itds.get("code"))),
        ]
    )


def _print_group(
    title: str,
    indices: list[str],
    base_rows: dict[str, dict[str, Any]],
    itds_rows: dict[str, dict[str, Any]],
    max_examples: int,
) -> None:
    shown = indices if max_examples == 0 else indices[:max_examples]
    print(f"\n{'#' * 30} {title}: {len(indices)} {'#' * 30}")
    for idx in shown:
        print(_format_example(idx, base_rows[idx], itds_rows[idx]))
    if len(shown) < len(indices):
        print(f"\n... omitted {len(indices) - len(shown)} examples; use --max-examples 0 to print all.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare aligned base and ITDS eval JSONL outputs and print changed outcomes."
    )
    parser.add_argument("--base", type=Path, required=True, help="Base-model eval JSONL.")
    parser.add_argument("--itds", type=Path, required=True, help="ITDS checkpoint eval JSONL.")
    parser.add_argument(
        "--max-examples",
        type=int,
        default=20,
        help="Maximum examples printed per direction; use 0 to print all.",
    )
    args = parser.parse_args()
    if args.max_examples < 0:
        parser.error("--max-examples must be >= 0")

    base_rows = _load_rows(args.base)
    itds_rows = _load_rows(args.itds)
    shared = sorted(set(base_rows) & set(itds_rows), key=lambda value: (not value.isdigit(), int(value) if value.isdigit() else value))
    valid = [idx for idx in shared if not _is_invalid(base_rows[idx]) and not _is_invalid(itds_rows[idx])]

    base_wrong_itds_right = [
        idx for idx in valid if not _is_correct(base_rows[idx]) and _is_correct(itds_rows[idx])
    ]
    base_right_itds_wrong = [
        idx for idx in valid if _is_correct(base_rows[idx]) and not _is_correct(itds_rows[idx])
    ]
    both_right = sum(_is_correct(base_rows[idx]) and _is_correct(itds_rows[idx]) for idx in valid)
    both_wrong = sum(not _is_correct(base_rows[idx]) and not _is_correct(itds_rows[idx]) for idx in valid)

    print(
        json.dumps(
            {
                "base_path": str(args.base),
                "itds_path": str(args.itds),
                "base_rows": len(base_rows),
                "itds_rows": len(itds_rows),
                "shared_rows": len(shared),
                "valid_shared_rows": len(valid),
                "invalid_shared_rows": len(shared) - len(valid),
                "only_in_base": len(set(base_rows) - set(itds_rows)),
                "only_in_itds": len(set(itds_rows) - set(base_rows)),
                "both_right": both_right,
                "both_wrong": both_wrong,
                "base_wrong_itds_right": len(base_wrong_itds_right),
                "base_right_itds_wrong": len(base_right_itds_wrong),
                "net_gain": len(base_wrong_itds_right) - len(base_right_itds_wrong),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    _print_group("BASE WRONG -> ITDS RIGHT", base_wrong_itds_right, base_rows, itds_rows, args.max_examples)
    _print_group("BASE RIGHT -> ITDS WRONG", base_right_itds_wrong, base_rows, itds_rows, args.max_examples)


if __name__ == "__main__":
    main()
