"""Task 5 evaluator.

TODO: implement once task 5's evaluation metric is finalized. See
:mod:`eval.example` for a reference implementation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import Evaluator, register_evaluator


@register_evaluator("task5")
class Task5Evaluator(Evaluator):
    """TODO: implement the task 5 evaluator."""

    def evaluate(self, predictions_path: Path) -> dict[str, Any]:
        raise NotImplementedError(
            "Task5Evaluator is not implemented yet. "
            "See src/eval/example.py for a reference implementation."
        )
