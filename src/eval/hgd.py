"""HGD evaluator — host/guest property explanation (open QA).

Metrics
-------
ROUGE-L F1
    Computed between the extracted ``<answer>`` text and the gold answer
    using :func:`eval.metrics.compute_rouge` (stemmer on).

Keyword Hit (KH)
    Fraction of representative guest/host names from the gold answer that
    appear as case-insensitive substrings of the prediction.  Names are
    extracted with the same regexes used in the original evaluation script
    (``experiments/ziming/hgd/scripts/04_evaluate.py``).

The final metrics dict contains overall aggregates plus a per-subtype
breakdown for ``forward`` and ``reverse`` rows.
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from .base import Evaluator, register_evaluator
from .metrics import compute_kh, compute_rouge

# ---------------------------------------------------------------------------
# Gold-answer name extraction (mirrors 04_evaluate.py)
# ---------------------------------------------------------------------------

# forward: "Representative guests: G1, G2, ..." (ends with ".")
RE_FWD = re.compile(
    r"Representative (?:guests?|hosts?):\s*(.+?)\s*\.?\s*$",
    re.IGNORECASE | re.DOTALL,
)
# reverse: "include: H1 (logKa=N1), H2 (logKa=N2), ... . The highest"
RE_REV_INCLUDE = re.compile(
    r"include:\s*(.+?)\.\s*The highest",
    re.IGNORECASE | re.DOTALL,
)
RE_REV_INCLUDE_FALLBACK = re.compile(
    r"include:\s*(.+)$",
    re.IGNORECASE | re.DOTALL,
)
RE_REV_ENTRY = re.compile(
    r"(.+?)\s*\(\s*logKa\s*=\s*[\d.\-]+\s*\)",
    re.IGNORECASE | re.DOTALL,
)


def _split_at_depth_zero(s: str) -> list[str]:
    """Split by ', ' at parenthesis-depth 0, preserving inner commas."""
    parts: list[str] = []
    buf: list[str] = []
    depth = 0
    i = 0
    while i < len(s):
        c = s[i]
        if c in "([{":
            depth += 1
            buf.append(c)
        elif c in ")]}":
            depth = max(0, depth - 1)
            buf.append(c)
        elif depth == 0 and c == "," and i + 1 < len(s) and s[i + 1] == " ":
            parts.append("".join(buf).strip())
            buf = []
            i += 2
            continue
        else:
            buf.append(c)
        i += 1
    if buf:
        parts.append("".join(buf).strip())
    return [p.strip(" .,;") for p in parts if p.strip(" .,;")]


def _extract_keywords(gold: str, subtype: str) -> list[str]:
    """Return the list of representative names from the gold answer."""
    if not isinstance(gold, str) or not gold:
        return []
    if subtype == "forward":
        m = RE_FWD.search(gold)
        if not m:
            return []
        return _split_at_depth_zero(m.group(1))
    elif subtype == "reverse":
        m = RE_REV_INCLUDE.search(gold) or RE_REV_INCLUDE_FALLBACK.search(gold)
        if not m:
            return []
        tail = m.group(1)
        names = []
        for em in RE_REV_ENTRY.finditer(tail):
            name = em.group(1).strip(" ,.;")
            name = re.sub(r"^[,;]\s*", "", name).strip()
            if name:
                names.append(name)
        return names
    return []


# ---------------------------------------------------------------------------
# Prediction parsing
# ---------------------------------------------------------------------------

def _after_think(text: str) -> str:
    if "</think>" in text:
        return text.split("</think>", 1)[1]
    return text


def _extract_answer_tag(text: str) -> str:
    m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else _after_think(text).strip()


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

@register_evaluator("hgd")
class Task3Evaluator(Evaluator):
    def evaluate(self, predictions_path: Path) -> dict[str, Any]:
        rows = list(self._load_predictions(predictions_path))

        # Collect per-row data.
        pred_texts: list[str] = []
        gold_texts: list[str] = []
        kw_lists: list[list[str]] = []
        subtypes: list[str] = []

        for row in rows:
            raw_pred = str(row.get("prediction", "") or "")
            pred_answer = _extract_answer_tag(raw_pred)
            gold = str(row.get("reference", "") or "")
            subtype = ""
            meta = row.get("metadata")
            if isinstance(meta, dict):
                subtype = str(meta.get("subtype", ""))

            pred_texts.append(pred_answer)
            gold_texts.append(gold)
            kw_lists.append(_extract_keywords(gold, subtype))
            subtypes.append(subtype)

        n_total = len(rows)

        # Overall ROUGE-L F1.
        rouge_result = compute_rouge(pred_texts, gold_texts, rouge_types=("rougeL",))
        overall_rougeL = rouge_result["rougeL_f"]

        # Overall KH.
        kh_result = compute_kh(pred_texts, kw_lists)
        overall_kh = kh_result["kh"]
        n_parsed = kh_result["n"]

        # Per-subtype breakdown.
        by_st: dict[str, dict[str, list]] = defaultdict(lambda: {
            "preds": [], "golds": [], "kws": []
        })
        for pred, gold, kws, st in zip(pred_texts, gold_texts, kw_lists, subtypes):
            by_st[st]["preds"].append(pred)
            by_st[st]["golds"].append(gold)
            by_st[st]["kws"].append(kws)

        subtype_metrics: dict[str, dict[str, Any]] = {}
        for st, data in by_st.items():
            st_rouge = compute_rouge(
                data["preds"], data["golds"], rouge_types=("rougeL",)
            )
            st_kh = compute_kh(data["preds"], data["kws"])
            subtype_metrics[st] = {
                "rougeL_f": st_rouge["rougeL_f"],
                "kh": st_kh["kh"],
                "n": len(data["preds"]),
            }

        return {
            "rougeL_f": overall_rougeL,
            "kh": overall_kh,
            "n_total": n_total,
            "n_parsed": n_parsed,
            "forward": subtype_metrics.get("forward", {}),
            "reverse": subtype_metrics.get("reverse", {}),
        }
