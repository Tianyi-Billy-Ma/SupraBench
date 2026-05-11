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


def _coerce_logka(value: float, ans: str) -> float | None:
    """Map a parsed numeric answer onto a plausible logKa, or reject it.

    Some completions emit raw association constants (``Ka = 2.5e3``) instead
    of the requested log10. We accept those when the surrounding text has
    no ``log`` qualifier and the value is large enough to look like a Ka,
    converting via ``log10``. Anything still outside ``[-10, 30]`` after
    coercion is treated as a parse failure.
    """
    if value > 50 and "log" not in ans.lower():
        try:
            value = math.log10(value)
        except ValueError:
            return None
    if value < -10 or value > 30:
        return None
    return value


def parse_logka(text: str) -> float | None:
    ans = _extract_answer_tag(text)
    # Bolded form: **N**, **-1.5**, **2.3e3** — also handles sci notation now.
    m = re.search(
        r"\*{1,2}([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\*{1,2}",
        ans,
    )
    if m:
        try:
            return _coerce_logka(float(m.group(1)), ans)
        except ValueError:
            pass
    # Fallback: strip common temperature mentions, take the first number.
    cleaned = re.sub(r"\b25\s*[°˚]C\b|\b298\.?1?5?\s*K\b", "", ans)
    m = re.search(r"[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?", cleaned)
    if not m:
        return None
    try:
        return _coerce_logka(float(m.group()), ans)
    except ValueError:
        return None


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

        # Robust statistics. The reference logKa range is [0, 12]; the parser
        # accepts the looser [-10, 30] window, so a handful of extreme outliers
        # can dominate the headline MAE. Emit alongside the regular metrics
        # so the paper can report tail-robust numbers without re-running eval.
        errs = sorted(abs(p - r) for p, r in pairs)
        if n_parsed:
            medae = errs[n_parsed // 2]
            cut95 = max(1, int(n_parsed * 0.95))
            mae_5pct_trimmed = sum(errs[:cut95]) / cut95
            in_range = [abs(p - r) for p, r in pairs if -2.0 <= p <= 15.0]
            mae_clipped = (sum(in_range) / len(in_range)) if in_range else float("nan")
            n_in_clip   = len(in_range)
        else:
            medae = mae_5pct_trimmed = mae_clipped = float("nan")
            n_in_clip = 0

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
            "mae":              mae["mae"],
            "rmse":             rmse["rmse"],
            "acc@0.5":          acc05,
            "acc@1.0":          acc10,
            "medae":            medae,
            "mae_5pct_trimmed": mae_5pct_trimmed,
            "mae_clipped":      mae_clipped,
            "n_in_clip":        n_in_clip,
            "n_total":          mae["n_total"],
            "n_parsed":         mae["n_parsed"],
            "per_host":         per_host,
        }
