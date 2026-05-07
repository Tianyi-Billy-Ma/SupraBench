"""Pearson r and Spearman rho rank-correlation metrics.

Both functions accept sequences of ``float | None`` predictions and
``float`` references. ``None`` predictions are excluded from the
computation; the count of parsed examples is reported in the result dict.

Requires only stdlib + numpy (already a base dependency).
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def compute_pearson(
    predictions: Sequence[float | None],
    references: Sequence[float],
) -> dict[str, float]:
    """Compute Pearson r between predictions and references.

    Args:
        predictions: Model-predicted values; ``None`` entries are skipped.
        references: Gold-standard float values.

    Returns:
        Dict with keys ``pearson_r`` (float, NaN when undefined) and ``n``
        (number of valid pairs used).
    """
    if len(predictions) != len(references):
        raise ValueError(
            f"length mismatch: {len(predictions)} predictions vs {len(references)} references"
        )
    pairs = [
        (p, r)
        for p, r in zip(predictions, references)
        if p is not None and not math.isnan(p)
    ]
    n = len(pairs)
    if n < 2:
        return {"pearson_r": float("nan"), "n": n}

    g = np.array([r for _, r in pairs], dtype=float)
    p = np.array([pp for pp, _ in pairs], dtype=float)
    if g.std() == 0 or p.std() == 0:
        return {"pearson_r": float("nan"), "n": n}
    r = float(np.corrcoef(g, p)[0, 1])
    return {"pearson_r": r, "n": n}


def compute_spearman(
    predictions: Sequence[float | None],
    references: Sequence[float],
) -> dict[str, float]:
    """Compute Spearman rho between predictions and references.

    Uses average-rank tie-breaking (matching scipy.stats.spearmanr).

    Args:
        predictions: Model-predicted values; ``None`` entries are skipped.
        references: Gold-standard float values.

    Returns:
        Dict with keys ``spearman_rho`` (float, NaN when undefined) and
        ``n`` (number of valid pairs used).
    """
    if len(predictions) != len(references):
        raise ValueError(
            f"length mismatch: {len(predictions)} predictions vs {len(references)} references"
        )
    pairs = [
        (p, r)
        for p, r in zip(predictions, references)
        if p is not None and not math.isnan(p)
    ]
    n = len(pairs)
    if n < 2:
        return {"spearman_rho": float("nan"), "n": n}

    def _average_ranks(x: np.ndarray) -> np.ndarray:
        order = np.argsort(x, kind="mergesort")
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(1, len(x) + 1, dtype=float)
        sorted_x = x[order]
        i, total = 0, len(x)
        while i < total:
            j = i
            while j + 1 < total and sorted_x[j + 1] == sorted_x[i]:
                j += 1
            if j > i:
                ranks[order[i : j + 1]] = (i + j + 2) / 2.0
            i = j + 1
        return ranks

    g = np.array([r for _, r in pairs], dtype=float)
    p = np.array([pp for pp, _ in pairs], dtype=float)
    gr = _average_ranks(g)
    pr = _average_ranks(p)
    if gr.std() == 0 or pr.std() == 0:
        return {"spearman_rho": float("nan"), "n": n}
    rho = float(np.corrcoef(gr, pr)[0, 1])
    return {"spearman_rho": rho, "n": n}
