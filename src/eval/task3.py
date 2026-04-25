"""Task 3 evaluator.

TODO: implement once task 3's evaluation metric is finalized. See
:mod:`eval.example` for a reference implementation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import Evaluator, register_evaluator


@register_evaluator("task3")
class Task3Evaluator(Evaluator):
    """TODO: implement the task 3 evaluator."""

    def evaluate(self, predictions_path: Path) -> dict[str, Any]:
        raise NotImplementedError(
            "Task3Evaluator is not implemented yet. "
            "See src/eval/example.py for a reference implementation."
        )
