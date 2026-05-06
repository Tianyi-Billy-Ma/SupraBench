"""ROUGE-1 / ROUGE-2 / ROUGE-L via the ``rouge_score`` package.

Requires ``rouge_score`` (not in the base install). Add it to the
``dev`` extras or install ad-hoc with ``uv pip install rouge-score``.
"""

from __future__ import annotations

from typing import Sequence

_ROUGE_TYPES = ("rouge1", "rouge2", "rougeL")


def compute_rouge(
    predictions: Sequence[str],
    references: Sequence[str],
    rouge_types: Sequence[str] = _ROUGE_TYPES,
) -> dict[str, float]:
    if len(predictions) != len(references):
        raise ValueError(
            f"length mismatch: {len(predictions)} predictions vs {len(references)} references"
        )
    if not predictions:
        return {f"{r}_f": 0.0 for r in rouge_types} | {"n": 0}

    try:
        from rouge_score import rouge_scorer
    except ImportError as err:
        raise RuntimeError(
            "rouge metric requires the `rouge_score` package: "
            "`uv pip install rouge-score`"
        ) from err

    scorer = rouge_scorer.RougeScorer(list(rouge_types), use_stemmer=True)
    totals = {r: 0.0 for r in rouge_types}
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        for r in rouge_types:
            totals[r] += scores[r].fmeasure

    n = len(predictions)
    return {f"{r}_f": totals[r] / n for r in rouge_types} | {"n": n}
