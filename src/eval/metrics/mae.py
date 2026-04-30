"""Mean Absolute Error.

Inputs are sequences of ``float | None``. ``None`` predictions (e.g.
unparseable model output) are excluded from the average; the count of
parsed-vs-total examples is reported alongside the score so partial
failures stay visible.
"""

from __future__ import annotations

import math
from typing import Sequence


def compute_mae(
    predictions: Sequence[float | None],
    references: Sequence[float],
) -> dict[str, float]:
    if len(predictions) != len(references):
        raise ValueError(
            f"length mismatch: {len(predictions)} predictions vs {len(references)} references"
        )

    pairs = [
        (p, r)
        for p, r in zip(predictions, references)
        if p is not None and not math.isnan(p)
    ]
    n_total = len(predictions)
    n_parsed = len(pairs)
    if n_parsed == 0:
        return {"mae": float("nan"), "n_total": n_total, "n_parsed": 0}

    err = sum(abs(p - r) for p, r in pairs) / n_parsed
    return {"mae": err, "n_total": n_total, "n_parsed": n_parsed}
