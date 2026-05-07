"""Task 1 evaluator — logKa prediction.

Parses a float logKa from the model's prediction text, then computes
MAE, RMSE, Acc@0.5, and Acc@1.0.  Also reports per-host MAE breakdown
via ``metadata.host_name`` when available.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from .base import Evaluator, register_evaluator
from .metrics import compute_mae, compute_rmse


def _after_think(text: str) -> str:
    if "</think>" in text:
        return text.split("</think>", 1)[1]
    return text


def _extract_answer_tag(text: str) -> str:
    m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else _after_think(text).strip()


def parse_logka(text: str) -> float | None:
    ans = _extract_answer_tag(text)
    m = re.search(r"\*{1,2}([-+]?\d+\.?\d*)\*{1,2}", ans)
    if m:
        return float(m.group(1))
    cleaned = re.sub(r"\b25\s*[°˚]C\b|\b298\.?1?5?\s*K\b", "", ans)
    m = re.search(r"[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?", cleaned)
    return float(m.group()) if m else None


@register_evaluator("task1")
class Task1Evaluator(Evaluator):
    def evaluate(self, predictions_path: Path) -> dict[str, Any]:
        rows = list(self._load_predictions(predictions_path))

        preds: list[float | None] = [parse_logka(str(r["prediction"])) for r in rows]
        refs:  list[float]        = [float(r["reference"]) for r in rows]

        mae  = compute_mae(preds, refs)
        rmse = compute_rmse(preds, refs)

        # Acc@0.5 and Acc@1.0
        pairs = [(p, r) for p, r in zip(preds, refs) if p is not None and not math.isnan(p)]
        n_parsed = len(pairs)
        acc05 = sum(1 for p, r in pairs if abs(p - r) <= 0.5) / n_parsed if n_parsed else float("nan")
        acc10 = sum(1 for p, r in pairs if abs(p - r) <= 1.0) / n_parsed if n_parsed else float("nan")

        # Per-host breakdown
        host_errors: dict[str, list[float]] = defaultdict(list)
        for row, pred in zip(rows, preds):
            if pred is not None and not math.isnan(pred):
                host = row.get("metadata", {}).get("host_name", "unknown") if isinstance(row.get("metadata"), dict) else "unknown"
                host_errors[host].append(abs(pred - float(row["reference"])))

        per_host = {
            h: {"mae": sum(errs) / len(errs), "n": len(errs)}
            for h, errs in sorted(host_errors.items(), key=lambda x: -len(x[1]))
        }

        return {
            "mae":      mae["mae"],
            "rmse":     rmse["rmse"],
            "acc@0.5":  acc05,
            "acc@1.0":  acc10,
            "n_total":  mae["n_total"],
            "n_parsed": mae["n_parsed"],
            "per_host": per_host,
        }
