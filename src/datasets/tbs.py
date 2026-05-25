"""TBS dataset loader — 4-choice MCQ (strongest binding affinity).

Loads pre-rendered prompts from data/tbs/{base,fewshot,cot}.jsonl.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from .base import Example, SupraDataset, register_dataset


@register_dataset("tbs")
class Task2Dataset(SupraDataset):
    def __iter__(self) -> Iterator[Example]:
        path = Path(self.config["data_path"])
        with path.open("r", encoding="utf-8") as fh:
            rows = [json.loads(line) for line in fh if line.strip()]

        for idx, row in enumerate(rows):
            if self.limit is not None and idx >= self.limit:
                break
            yield Example(
                id=row["id"],
                prompt=row["question"],
                reference=str(row["answer"]),
                metadata={
                    "host_name":        row.get("host_name", ""),
                    "correct_molecule": row.get("correct_molecule", ""),
                    "options":          row.get("options", []),
                    "version":          row.get("version", ""),
                },
            )
