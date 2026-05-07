"""Task 2 evaluator — 4-choice MCQ accuracy.

Parses the predicted answer letter (A/B/C/D) from model output and
computes accuracy.  Also reports per-host accuracy breakdown.
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from .base import Evaluator, register_evaluator


def _after_think(text: str) -> str:
    if "</think>" in text:
        return text.split("</think>", 1)[1]
    return text


def _extract_answer_tag(text: str) -> str:
    m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else _after_think(text).strip()


def parse_mcq(text: str) -> str | None:
    ans = _extract_answer_tag(text)
    m = re.search(r"\b([A-Da-d])\b", ans)
    if m:
        return m.group(1).upper()
    m = re.search(r"([A-Da-d])", ans)
    return m.group(1).upper() if m else None


@register_evaluator("task2")
class Task2Evaluator(Evaluator):
    def evaluate(self, predictions_path: Path) -> dict[str, Any]:
        rows = list(self._load_predictions(predictions_path))

        preds = [parse_mcq(str(r["prediction"])) for r in rows]
        refs  = [str(r["reference"]).strip().upper() for r in rows]

        n_total  = len(rows)
        parsed   = [(p, r) for p, r in zip(preds, refs) if p is not None]
        n_parsed = len(parsed)
        accuracy = sum(1 for p, r in parsed if p == r) / n_parsed if n_parsed else float("nan")

        # Per-host breakdown
        host_results: dict[str, list[int]] = defaultdict(list)
        for row, pred in zip(rows, preds):
            if pred is not None:
                host = row.get("metadata", {}).get("host_name", "unknown") if isinstance(row.get("metadata"), dict) else "unknown"
                host_results[host].append(int(pred == str(row["reference"]).strip().upper()))

        per_host = {
            h: {"accuracy": sum(hits) / len(hits), "n": len(hits)}
            for h, hits in sorted(host_results.items(), key=lambda x: -len(x[1]))
        }

        return {
            "accuracy":  accuracy,
            "n_total":   n_total,
            "n_parsed":  n_parsed,
            "per_host":  per_host,
        }
