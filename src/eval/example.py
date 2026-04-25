"""Example evaluator — reference implementation for new tasks.

Pairs with :mod:`datasets.example`: the dataset hands the model a
host/guest binding-affinity question and the reference is a numeric
``logKa`` value, so the evaluator parses each prediction back into a
float and reports MAE / RMSE.

Float parsing is regex-based and intentionally permissive: it picks up
the **first** number it sees in the prediction (whether the model wrote
``<answer>2.05</answer>``, ``logKa = 2.05``, or just ``2.05``). When no
number is found, the prediction is recorded as ``None`` and the metric
modules report ``n_parsed`` < ``n_total`` so partial parse failures stay
visible.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .base import Evaluator, register_evaluator
from .metrics import compute_mae, compute_rmse

# Matches integers, decimals, and scientific notation (e.g. "1.7e-3").
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def _parse_float(text: str) -> float | None:
    match = _NUMBER_RE.search(text)
    if match is None:
        return None
    try:
        return float(match.group())
    except ValueError:
        return None


@register_evaluator("example")
class ExampleEvaluator(Evaluator):
    def evaluate(self, predictions_path: Path) -> dict[str, Any]:
        rows = list(self._load_predictions(predictions_path))
        preds: list[float | None] = [_parse_float(str(r["prediction"])) for r in rows]
        refs: list[float] = [float(r["reference"]) for r in rows]

        mae = compute_mae(preds, refs)
        rmse = compute_rmse(preds, refs)
        # Both metrics report n_total / n_parsed; they agree, so collapse.
        return {
            "mae": mae["mae"],
            "rmse": rmse["rmse"],
            "n_total": mae["n_total"],
            "n_parsed": mae["n_parsed"],
        }
