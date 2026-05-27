from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset


REQUIRED_FIELDS = {
    "prompt",
    "response",
    "reward",
    "ref_logprob_sum",
    "ref_logprob_mean",
    "response_num_tokens",
}


@dataclass(frozen=True)
class TrainingExample:
    prompt: str
    response: str
    reward: float
    ref_logprob_sum: float
    ref_logprob_mean: float
    response_num_tokens: int
    metadata: dict[str, Any]


class JsonlSubTBDataset(Dataset):
    def __init__(self, path: str | Path, *, index_cache_path: str | Path | None = None):
        self.path = Path(path)
        self.index_cache_path = Path(index_cache_path) if index_cache_path else None
        self.offsets = self._load_offsets() if self.index_cache_path and self.index_cache_path.exists() else self._build_offsets()
        if not self.offsets:
            raise ValueError(f"{self.path} is empty")
        self._validate_required_fields()

    @staticmethod
    def build_index_cache(path: str | Path, index_cache_path: str | Path) -> int:
        dataset_path = Path(path)
        cache_path = Path(index_cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        offsets: list[int] = []
        offset = 0
        with dataset_path.open("rb") as handle:
            for line in handle:
                if line.strip():
                    offsets.append(offset)
                offset += len(line)
        with cache_path.open("w", encoding="utf-8") as handle:
            for item in offsets:
                handle.write(f"{item}\n")
        return len(offsets)

    def _build_offsets(self) -> list[int]:
        offsets: list[int] = []
        offset = 0
        with self.path.open("rb") as handle:
            for line in handle:
                if line.strip():
                    offsets.append(offset)
                offset += len(line)
        return offsets

    def _load_offsets(self) -> list[int]:
        assert self.index_cache_path is not None
        with self.index_cache_path.open("r", encoding="utf-8") as handle:
            return [int(line) for line in handle if line.strip()]

    def _read_row(self, index: int) -> dict[str, Any]:
        with self.path.open("rb") as handle:
            handle.seek(self.offsets[index])
            line = handle.readline()
        try:
            return json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {self.path}:{index + 1}: {exc}") from exc

    def _validate_required_fields(self) -> None:
        row = self._read_row(0)
        missing = REQUIRED_FIELDS.difference(row)
        if missing:
            missing_text = ", ".join(sorted(missing))
            raise ValueError(f"{self.path}:1 missing required field(s): {missing_text}")

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, index: int) -> TrainingExample:
        row = self._read_row(index)
        metadata = {key: value for key, value in row.items() if key not in REQUIRED_FIELDS}
        return TrainingExample(
            prompt=str(row["prompt"]),
            response=str(row["response"]),
            reward=float(row["reward"]),
            ref_logprob_sum=float(row["ref_logprob_sum"]),
            ref_logprob_mean=float(row["ref_logprob_mean"]),
            response_num_tokens=int(row["response_num_tokens"]),
            metadata=metadata,
        )


def collate_examples(examples: list[TrainingExample]) -> list[TrainingExample]:
    return examples
