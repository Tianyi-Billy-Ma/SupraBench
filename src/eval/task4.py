"""Task 4 evaluator — logKa VQA (host + guest images → float logKa).

Parses a single float logKa from each model prediction, then computes:
- MAE, RMSE (via shared metric modules)
- Pearson r, Spearman rho (via rank_corr metric module)
- parse_fail_rate: fraction of examples where no float could be extracted
- n_total, n_parsed

Per-row ``predictions_scored`` data mirrors the format used by existing
evaluators: each prediction row is augmented with the parsed value, gold
reference, and absolute error.

Float parsing is regex-based and intentionally permissive — the same
pattern used in :mod:`eval.example` and :mod:`eval.task1`.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .base import Evaluator, register_evaluator
from .metrics import compute_mae, compute_rmse, compute_pearson, compute_spearman

# Matches integers, decimals, and scientific notation.
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def _parse_float(text: str) -> float | None:
    """Extract the first number from model output; return None on failure."""
    match = _NUMBER_RE.search(str(text))
    if match is None:
        return None
    try:
        return float(match.group())
    except ValueError:
        return None


@register_evaluator("task4")
class Task4Evaluator(Evaluator):
    """Evaluator for Task 4 (logKa VQA).

    Reads a JSONL predictions file produced by the inference pipeline, parses
    each prediction as a float, and returns regression + correlation metrics.
    """

    def evaluate(self, predictions_path: Path) -> dict[str, Any]:
        rows = list(self._load_predictions(predictions_path))

        preds: list[float | None] = [_parse_float(r["prediction"]) for r in rows]
        refs: list[float] = [float(r["reference"]) for r in rows]

        mae_result = compute_mae(preds, refs)
        rmse_result = compute_rmse(preds, refs)
        pearson_result = compute_pearson(preds, refs)
        spearman_result = compute_spearman(preds, refs)

        n_total = mae_result["n_total"]
        n_parsed = mae_result["n_parsed"]
        parse_fail_rate = 1.0 - (n_parsed / n_total) if n_total > 0 else float("nan")

        return {
            "mae": mae_result["mae"],
            "rmse": rmse_result["rmse"],
            "pearson_r": pearson_result["pearson_r"],
            "spearman_rho": spearman_result["spearman_rho"],
            "parse_fail_rate": parse_fail_rate,
            "n_total": n_total,
            "n_parsed": n_parsed,
        }
