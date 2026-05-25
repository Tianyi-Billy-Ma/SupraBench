"""Example dataset loader — reference implementation for new tasks.

Reads a single JSON file containing a list of objects with the schema
used by ``data/bap/sample.json``::

    [
      {"question": str, "host": str (SMILES), "guest": str (SMILES), "answer": Any},
      ...
    ]

The user-facing query is composed as:

    {question}
    Host SMILES: {host}
    Guest SMILES: {guest}

and rendered through :func:`templates.generate_prompt`. **Every dataset
loader must do this**; hand-formatted prompts break cross-task /
cross-model comparability.

Row indices become example ids (``example_0000``, ``example_0001``, ...)
since the source records have no ``id`` field.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from templates import generate_prompt

from .base import Example, SupraDataset, register_dataset


@register_dataset("example")
class ExampleDataset(SupraDataset):
    def __iter__(self) -> Iterator[Example]:
        path = Path(self.config["data_path"])
        prompt_cfg = self.config.get("prompt", {}) or {}
        thinking = bool(prompt_cfg.get("thinking", False))

        with path.open("r", encoding="utf-8") as fh:
            rows = json.load(fh)
        if not isinstance(rows, list):
            raise ValueError(
                f"{path} must contain a JSON array of objects; got {type(rows).__name__}"
            )

        for idx, row in enumerate(rows):
            if self.limit is not None and idx >= self.limit:
                break

            query = (
                f"{row['question']}\n"
                f"Host SMILES: {row['host']}\n"
                f"Guest SMILES: {row['guest']}"
            )
            prompt = generate_prompt(
                query,
                fewshot_examples=row.get("fewshot"),
                thinking=thinking,
            )

            reserved = {"question", "host", "guest", "answer", "fewshot"}
            yield Example(
                id=f"example_{idx:04d}",
                prompt=prompt,
                reference=row["answer"],
                metadata={k: v for k, v in row.items() if k not in reserved},
            )
