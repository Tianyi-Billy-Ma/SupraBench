"""Token-level F1 with SQuAD-style normalisation.

For each (prediction, reference) pair we tokenise on whitespace after
normalisation, then compute precision / recall / F1 from the multiset of
shared tokens. The returned ``f1`` is the macro-average across pairs.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import Sequence

_ARTICLES = re.compile(r"\b(a|an|the)\b", flags=re.IGNORECASE)
_PUNCT = str.maketrans("", "", string.punctuation)
_WS = re.compile(r"\s+")


def _normalise(text: str) -> list[str]:
    text = text.lower()
    text = text.translate(_PUNCT)
    text = _ARTICLES.sub(" ", text)
    text = _WS.sub(" ", text).strip()
    return text.split()


def _pair_f1(pred_tokens: list[str], ref_tokens: list[str]) -> float:
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(ref_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_f1(
    predictions: Sequence[str],
    references: Sequence[str],
) -> dict[str, float]:
    if len(predictions) != len(references):
        raise ValueError(
            f"length mismatch: {len(predictions)} predictions vs {len(references)} references"
        )
    if not predictions:
        return {"f1": 0.0, "n": 0}

    scores = [
        _pair_f1(_normalise(pred), _normalise(ref))
        for pred, ref in zip(predictions, references)
    ]
    return {"f1": sum(scores) / len(scores), "n": len(predictions)}
