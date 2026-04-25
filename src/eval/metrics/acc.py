"""Accuracy: fraction of predictions whose normalised string equals the reference.

Use for multiple-choice or single-label tasks where the only thing that
matters is whether the model picked the right answer. For free-form
generation prefer :mod:`eval.metrics.em` (exact match with light
normalisation) or :mod:`eval.metrics.f1` (token overlap).
"""

from __future__ import annotations

from typing import Sequence


def _normalise(text: str) -> str:
    return text.strip().lower()


def compute_acc(
    predictions: Sequence[str],
    references: Sequence[str],
) -> dict[str, float]:
    if len(predictions) != len(references):
        raise ValueError(
            f"length mismatch: {len(predictions)} predictions vs {len(references)} references"
        )
    if not predictions:
        return {"accuracy": 0.0, "n": 0}

    correct = sum(
        int(_normalise(pred) == _normalise(ref))
        for pred, ref in zip(predictions, references)
    )
    return {"accuracy": correct / len(predictions), "n": len(predictions)}
