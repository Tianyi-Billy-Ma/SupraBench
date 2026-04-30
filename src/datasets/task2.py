"""Task 2 dataset loader.

TODO: implement once task 2 is finalized. The implementation **must**
render each example's prompt through :func:`templates.generate_prompt`
(optionally composed with :func:`templates.generate_options`) — never
hand-format prompts here. See :mod:`datasets.example` for the canonical
integration.
"""

from __future__ import annotations

from typing import Iterator

from .base import Example, SupraDataset, register_dataset


@register_dataset("task2")
class Task2Dataset(SupraDataset):
    """TODO: implement the task 2 dataset loader."""

    def __iter__(self) -> Iterator[Example]:
        raise NotImplementedError(
            "Task2Dataset is not implemented yet. "
            "See src/datasets/example.py for a reference implementation."
        )
