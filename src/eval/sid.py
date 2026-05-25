"""SID evaluator — solvent compatibility classification (6-class MCQ).

Parses the predicted answer letter (A–F) from ``<answer>…</answer>`` tags,
maps it to the corresponding solvent label, then computes:

  * accuracy           — over successfully parsed rows only
  * macro_f1           — unweighted mean F1 across classes with support
  * weighted_f1        — support-weighted F1
  * parse_fail_rate    — fraction of rows where no valid letter was found
  * per_class          — per-label {precision, recall, f1, support}
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .base import Evaluator, register_evaluator
from .metrics import compute_multiclass_f1

# ---------------------------------------------------------------------------
# Label set — must match Task7Dataset exactly
# ---------------------------------------------------------------------------

LABELS_ORDER = ["water", "DMSO", "MeCN", "MeOH", "CHCl3", "CH2Cl2"]
LETTER_TO_LABEL: dict[str, str] = {
    chr(ord("A") + i): lab for i, lab in enumerate(LABELS_ORDER)
}

# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _after_think(text: str) -> str:
    if "</think>" in text:
        return text.split("</think>", 1)[1]
    return text


def _extract_answer_tag(text: str) -> str:
    m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else _after_think(text).strip()


def parse_letter(text: str) -> str | None:
    """Return an uppercase letter A–F, or None if not found / out of range."""
    ans = _extract_answer_tag(text)
    # Prefer a standalone letter (word boundary)
    m = re.search(r"\b([A-Fa-f])\b", ans)
    if m:
        letter = m.group(1).upper()
        if letter in LETTER_TO_LABEL:
            return letter
    # Fallback: first letter anywhere in the answer string
    m = re.search(r"([A-Fa-f])", ans)
    if m:
        letter = m.group(1).upper()
        if letter in LETTER_TO_LABEL:
            return letter
    return None


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


@register_evaluator("sid")
class Task7Evaluator(Evaluator):
    """Evaluate solvent classification predictions."""

    def evaluate(self, predictions_path: Path) -> dict[str, Any]:
        rows = list(self._load_predictions(predictions_path))
        n_total = len(rows)

        parsed_preds: list[str] = []
        parsed_refs: list[str] = []
        n_parse_fail = 0

        for row in rows:
            letter = parse_letter(str(row["prediction"]))
            if letter is None:
                n_parse_fail += 1
                continue
            pred_label = LETTER_TO_LABEL[letter]
            ref_label = str(row["reference"])
            parsed_preds.append(pred_label)
            parsed_refs.append(ref_label)

        n_parsed = len(parsed_preds)
        parse_fail_rate = n_parse_fail / n_total if n_total else 0.0

        if n_parsed == 0:
            return {
                "accuracy": float("nan"),
                "macro_f1": 0.0,
                "weighted_f1": 0.0,
                "parse_fail_rate": parse_fail_rate,
                "n_total": n_total,
                "n_parsed": 0,
                "per_class": {
                    lbl: {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0}
                    for lbl in LABELS_ORDER
                },
            }

        accuracy = sum(
            1 for p, r in zip(parsed_preds, parsed_refs) if p == r
        ) / n_parsed

        f1_result = compute_multiclass_f1(parsed_preds, parsed_refs, LABELS_ORDER)

        return {
            "accuracy": accuracy,
            "macro_f1": f1_result["macro_f1"],
            "weighted_f1": f1_result["weighted_f1"],
            "parse_fail_rate": parse_fail_rate,
            "n_total": n_total,
            "n_parsed": n_parsed,
            "per_class": f1_result["per_class"],
        }
