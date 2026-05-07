"""Evaluation logic for SupraBench tasks.

Task-specific evaluators (``task1`` … ``task7``) are imported here so
their ``@register_evaluator`` decorators run at package-import time.
Shared metric helpers live under :mod:`eval.metrics`.
:mod:`eval.example` is the canonical reference implementation — copy it
when filling in a task evaluator.
"""

from .base import Evaluator, build_evaluator, register_evaluator

# Side-effect imports — these populate the evaluator registry.
from . import example, task1, task2, task3, task4, task5, task6, task7  # noqa: F401
from . import vqa_identification, vqa_logka  # noqa: F401

__all__ = ["Evaluator", "build_evaluator", "register_evaluator"]
