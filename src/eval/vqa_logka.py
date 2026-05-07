"""Evaluator for the VQA logKa task.

Pairs with :mod:`datasets.vqa_logka`. The reference is a float
``logKa``; the prediction is parsed back into a float (first numeric
token in the model output) and compared via regression metrics.

Reports ``mae`` / ``rmse`` (composed from :mod:`eval.metrics`) plus
``r2``, ``pearson_r``, ``spearman_rho``, ``within_0.5``, ``within_1.0``,
and ``bias`` (computed locally — these are not used by other tasks so
they don't belong in :mod:`eval.metrics`).
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

from .base import Evaluator, register_evaluator
from .metrics import compute_mae, compute_rmse


_FLOAT_RE = re.compile(r"-?\d+(?:\.\d+)?")
_ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)


def _parse_float(text: str) -> float | None:
    """Extract a float from raw model output.

    Prefer content inside ``<answer>...</answer>`` (the BASE_TEMPLATE
    convention) over the full response, so a number from the model's
    reasoning prose can't shadow the actual answer.
    """
    if not text:
        return None
    answer_match = _ANSWER_TAG_RE.search(text)
    search_in = answer_match.group(1) if answer_match else text
    match = _FLOAT_RE.search(search_in)
    if match is None:
        return None
    try:
        return float(match.group())
    except ValueError:
        return None


def _average_ranks(values: list[float]) -> list[float]:
    n = len(values)
    order = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = (i + j + 2) / 2  # 1-indexed average rank for the tie group
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def _pearson(x: list[float], y: list[float]) -> float:
    n = len(x)
    if n < 2:
        return float("nan")
    mx = sum(x) / n
    my = sum(y) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(x, y))
    var_x = sum((a - mx) ** 2 for a in x)
    var_y = sum((b - my) ** 2 for b in y)
    if var_x == 0 or var_y == 0:
        return float("nan")
    return cov / math.sqrt(var_x * var_y)


@register_evaluator("vqa_logka")
class VQALogKaEvaluator(Evaluator):
    def evaluate(self, predictions_path: Path) -> dict[str, Any]:
        rows = list(self._load_predictions(predictions_path))

        preds: list[float | None] = [_parse_float(str(r["prediction"])) for r in rows]
        refs: list[float] = [float(r["reference"]) for r in rows]

        mae = compute_mae(preds, refs)
        rmse = compute_rmse(preds, refs)

        valid_pairs = [(p, r) for p, r in zip(preds, refs)
                       if p is not None and not math.isnan(p)]
        n_total = len(rows)
        n_valid = len(valid_pairs)

        metrics: dict[str, Any] = {
            "n_total": n_total,
            "n_valid": n_valid,
            "valid_rate": n_valid / n_total if n_total else 0.0,
            "mae": mae["mae"],
            "rmse": rmse["rmse"],
        }

        if n_valid >= 2:
            pred_vals = [p for p, _ in valid_pairs]
            ref_vals = [r for _, r in valid_pairs]
            errors = [p - r for p, r in valid_pairs]
            abs_errors = [abs(e) for e in errors]

            ss_res = sum(e ** 2 for e in errors)
            ref_mean = sum(ref_vals) / n_valid
            ss_tot = sum((r - ref_mean) ** 2 for r in ref_vals)
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

            metrics.update({
                "r2": r2,
                "pearson_r": _pearson(ref_vals, pred_vals),
                "spearman_rho": _pearson(_average_ranks(ref_vals), _average_ranks(pred_vals)),
                "within_0.5": sum(1 for e in abs_errors if e <= 0.5) / n_valid,
                "within_1.0": sum(1 for e in abs_errors if e <= 1.0) / n_valid,
                "bias": sum(errors) / n_valid,
            })
        else:
            for k in ("r2", "pearson_r", "spearman_rho",
                      "within_0.5", "within_1.0", "bias"):
                metrics[k] = float("nan")

        return metrics
