#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path


def build_offsets(input_path: Path, output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    offset = 0
    with input_path.open("rb") as source, output_path.open("w", encoding="utf-8") as target:
        for line in source:
            if line.strip():
                target.write(f"{offset}\n")
                count += 1
            offset += len(line)
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a byte-offset index for a JSONL file.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    count = build_offsets(Path(args.input), Path(args.output))
    print(f"wrote {count} offsets to {args.output}")


if __name__ == "__main__":
    main()
