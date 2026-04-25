"""Exact-match with SQuAD-style normalisation.

Lower-cases, strips articles, collapses whitespace, and removes punctuation
before comparing. Slightly more forgiving than raw string equality; useful
for short free-form answers.
"""

from __future__ import annotations

import re
import string
from typing import Sequence

_ARTICLES = re.compile(r"\b(a|an|the)\b", flags=re.IGNORECASE)
_PUNCT = str.maketrans("", "", string.punctuation)
_WS = re.compile(r"\s+")


def _normalise(text: str) -> str:
    text = text.lower()
    text = text.translate(_PUNCT)
    text = _ARTICLES.sub(" ", text)
    text = _WS.sub(" ", text).strip()
    return text


def compute_em(
    predictions: Sequence[str],
    references: Sequence[str],
) -> dict[str, float]:
    if len(predictions) != len(references):
        raise ValueError(
            f"length mismatch: {len(predictions)} predictions vs {len(references)} references"
        )
    if not predictions:
        return {"exact_match": 0.0, "n": 0}

    matches = sum(
        int(_normalise(pred) == _normalise(ref))
        for pred, ref in zip(predictions, references)
    )
    return {"exact_match": matches / len(predictions), "n": len(predictions)}
