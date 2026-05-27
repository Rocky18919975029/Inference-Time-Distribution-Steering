from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset, Sampler


@dataclass(frozen=True)
class RolloutExample:
    prompt: str
    response: str
    reward: float
    group_id: str
    metadata: dict[str, Any]


class RolloutJsonlDataset(Dataset):
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.offsets = self._build_offsets()
        if not self.offsets:
            raise ValueError(f"{self.path} is empty")
        self.groups = self._build_groups()

    def _build_offsets(self) -> list[int]:
        offsets: list[int] = []
        offset = 0
        with self.path.open("rb") as handle:
            for line in handle:
                if line.strip():
                    offsets.append(offset)
                offset += len(line)
        return offsets

    def _read_row(self, index: int) -> dict[str, Any]:
        with self.path.open("rb") as handle:
            handle.seek(self.offsets[index])
            return json.loads(handle.readline())

    def _build_groups(self) -> dict[str, list[int]]:
        groups: dict[str, list[int]] = {}
        for index in range(len(self.offsets)):
            row = self._read_row(index)
            group_id = str(row.get("problem_id", row.get("row_index", index)))
            groups.setdefault(group_id, []).append(index)
        return groups

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, index: int) -> RolloutExample:
        row = self._read_row(index)
        prompt = str(row.get("prompt", ""))
        response = str(row.get("response", ""))
        reward = float(row.get("reward", row.get("is_correct", 0.0)))
        group_id = str(row.get("problem_id", row.get("row_index", index)))
        return RolloutExample(prompt=prompt, response=response, reward=reward, group_id=group_id, metadata=row)


def collate_examples(examples: list[RolloutExample]) -> list[RolloutExample]:
    return examples


class GroupBatchSampler(Sampler[list[int]]):
    def __init__(self, groups: dict[str, list[int]], *, batch_size: int, seed: int = 1, drop_last: bool = False):
        self.groups = {key: list(value) for key, value in groups.items() if value}
        self.batch_size = batch_size
        self.seed = seed
        self.drop_last = drop_last

    def __iter__(self):
        rng = random.Random(self.seed)
        group_ids = list(self.groups)
        rng.shuffle(group_ids)
        for group_id in group_ids:
            indices = list(self.groups[group_id])
            rng.shuffle(indices)
            for start in range(0, len(indices), self.batch_size):
                batch = indices[start : start + self.batch_size]
                if len(batch) == self.batch_size or (batch and not self.drop_last):
                    yield batch

    def __len__(self) -> int:
        total = 0
        for indices in self.groups.values():
            full, remainder = divmod(len(indices), self.batch_size)
            total += full
            if remainder and not self.drop_last:
                total += 1
        return total
