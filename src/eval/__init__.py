"""Evaluation logic for SupraBench tasks.

Task-specific evaluators (``bap``, ``tbs``, ``hgd``, ``mi``, ``sid``) are
imported here so their ``@register_evaluator`` decorators run at
package-import time. Shared metric helpers live under :mod:`eval.metrics`.
:mod:`eval.example` is the canonical reference implementation — copy it
when filling in a task evaluator.
"""

from .base import Evaluator, build_evaluator, register_evaluator

# Side-effect imports — these populate the evaluator registry.
from . import example, bap, tbs, hgd, mi, sid  # noqa: F401

__all__ = ["Evaluator", "build_evaluator", "register_evaluator"]
