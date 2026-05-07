"""Multiclass macro- and weighted-F1 with per-class precision/recall breakdown.

Follows the same interface convention as other metrics in this package:
``compute_multiclass_f1(predictions, references, labels)`` returns a
``dict[str, ...]``.

This is a pure-Python implementation that avoids scikit-learn so the base
install stays lightweight.
"""

from __future__ import annotations

from typing import Sequence


def compute_multiclass_f1(
    predictions: Sequence[str],
    references: Sequence[str],
    labels: Sequence[str],
) -> dict:
    """Compute macro-F1, weighted-F1, and per-class statistics.

    Args:
        predictions: Predicted class labels (already decoded from letters).
        references:  Ground-truth class labels.
        labels:      The ordered list of all valid class labels.

    Returns:
        A dict with keys:
          ``macro_f1``   – unweighted mean F1 over classes that have support.
          ``weighted_f1``– support-weighted mean F1.
          ``per_class``  – mapping from label to
                           ``{precision, recall, f1, support}``.
    """
    if len(predictions) != len(references):
        raise ValueError(
            f"length mismatch: {len(predictions)} predictions vs "
            f"{len(references)} references"
        )

    per_class: dict[str, dict[str, float]] = {}
    f1s_for_macro: list[float] = []
    total_support = 0

    for label in labels:
        tp = sum(1 for t, p in zip(references, predictions) if t == label and p == label)
        fp = sum(1 for t, p in zip(references, predictions) if t != label and p == label)
        fn = sum(1 for t, p in zip(references, predictions) if t == label and p != label)
        support = sum(1 for t in references if t == label)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
        total_support += support
        if support > 0:
            f1s_for_macro.append(f1)

    macro_f1 = sum(f1s_for_macro) / len(f1s_for_macro) if f1s_for_macro else 0.0
    weighted_f1 = (
        sum(per_class[lbl]["f1"] * per_class[lbl]["support"] for lbl in labels)
        / total_support
        if total_support > 0
        else 0.0
    )

    return {
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class": per_class,
    }
