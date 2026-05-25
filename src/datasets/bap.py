"""BAP dataset loader — logKa prediction.

Loads pre-rendered prompts from data/bap/{base,fewshot,cot}.jsonl.
Each record's ``question`` field is a fully rendered prompt produced by
``tools/generate_questions.py``; it is passed through as-is.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from .base import Example, SupraDataset, register_dataset


@register_dataset("bap")
class Task1Dataset(SupraDataset):
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
                reference=float(row["answer"]),
                metadata={
                    "host_name": row.get("host_name", ""),
                    "molecule":  row.get("molecule", ""),
                    "version":   row.get("version", ""),
                },
            )
