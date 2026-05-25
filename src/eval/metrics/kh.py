"""Keyword Hit (KH) metric for open-answer tasks.

Each prediction is scored against a pre-extracted list of representative
names (guests for forward, hosts for reverse).  The metric is the fraction
of gold names that appear as a case-insensitive substring of the prediction.

The name lists are prepared by the task-specific evaluator (see
:mod:`eval.hgd`); this module only handles the substring-match counting.
"""

from __future__ import annotations

from typing import Sequence


def compute_kh(
    predictions: Sequence[str],
    references: Sequence[Sequence[str]],
) -> dict[str, float]:
    """Compute macro-average Keyword Hit over a set of predictions.

    Args:
        predictions: One predicted string per row.
        references: One list of gold representative names per row.  An empty
            list signals that no names could be parsed; those rows contribute
            ``nan`` to the per-row score and are excluded from the mean.

    Returns:
        ``{"kh": float, "n": int}`` where ``kh`` is the macro-average KH
        (NaN rows excluded) and ``n`` is the number of rows that had at least
        one parseable gold name.
    """
    if len(predictions) != len(references):
        raise ValueError(
            f"length mismatch: {len(predictions)} predictions vs "
            f"{len(references)} references"
        )

    total = 0.0
    n = 0
    for pred, names in zip(predictions, references):
        if not names:
            continue
        pred_low = pred.lower() if isinstance(pred, str) else ""
        hits = sum(1 for name in names if name.lower() in pred_low)
        total += hits / len(names)
        n += 1

    return {"kh": total / n if n else float("nan"), "n": n}
