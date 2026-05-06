"""Task 1 evaluator.

TODO: implement once task 1's evaluation metric is finalized. See
:mod:`eval.example` for a reference implementation that composes shared
metrics from :mod:`eval.metrics`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import Evaluator, register_evaluator


@register_evaluator("task1")
class Task1Evaluator(Evaluator):
    """TODO: implement the task 1 evaluator."""

    def evaluate(self, predictions_path: Path) -> dict[str, Any]:
        raise NotImplementedError(
            "Task1Evaluator is not implemented yet. "
            "See src/eval/example.py for a reference implementation."
        )
