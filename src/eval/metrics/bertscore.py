"""BERTScore via the ``bert_score`` package.

Requires ``bert_score`` + a backing transformer model (not in base install).
Install ad-hoc with ``uv pip install bert-score``; the first call also
downloads the tokenizer/model weights.
"""

from __future__ import annotations

from typing import Sequence


def compute_bertscore(
    predictions: Sequence[str],
    references: Sequence[str],
    lang: str = "en",
    model_type: str | None = None,
    rescale_with_baseline: bool = True,
) -> dict[str, float]:
    if len(predictions) != len(references):
        raise ValueError(
            f"length mismatch: {len(predictions)} predictions vs {len(references)} references"
        )
    if not predictions:
        return {"bertscore_p": 0.0, "bertscore_r": 0.0, "bertscore_f1": 0.0, "n": 0}

    try:
        from bert_score import score as bert_score
    except ImportError as err:
        raise RuntimeError(
            "bertscore metric requires the `bert_score` package: "
            "`uv pip install bert-score`"
        ) from err

    p, r, f1 = bert_score(
        list(predictions),
        list(references),
        lang=lang,
        model_type=model_type,
        rescale_with_baseline=rescale_with_baseline,
        verbose=False,
    )
    return {
        "bertscore_p": float(p.mean().item()),
        "bertscore_r": float(r.mean().item()),
        "bertscore_f1": float(f1.mean().item()),
        "n": len(predictions),
    }
