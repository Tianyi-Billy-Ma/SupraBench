"""Task 7 evaluator.

TODO: implement once task 7's evaluation metric is finalized. See
:mod:`eval.example` for a reference implementation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import Evaluator, register_evaluator


@register_evaluator("task7")
class Task7Evaluator(Evaluator):
    """TODO: implement the task 7 evaluator."""

    def evaluate(self, predictions_path: Path) -> dict[str, Any]:
        raise NotImplementedError(
            "Task7Evaluator is not implemented yet. "
            "See src/eval/example.py for a reference implementation."
        )
