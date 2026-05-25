"""MI evaluator — molecular-structure identification VQA.

Dispatches to name-mode or SMILES-mode scoring based on ``config['mode']``.

**Name mode** (``mode: name``):
    Normalise prediction and every alias to lowercase with punctuation and
    whitespace stripped, then check for exact match against any alias.
    Also reports rapidfuzz fuzzy-match ratio (threshold ≥ 85) and mean
    fuzzy score.

    Requires ``rapidfuzz`` (lazy import, not in the base install)::

        uv pip install rapidfuzz

    Returns: {exact, norm_exact, fuzzy_at_85, mean_fuzzy, empty_rate, n}

**SMILES mode** (``mode: smiles``, default):
    Canonicalise both predicted and gold SMILES via RDKit, check for exact
    canonical match, and compute Tanimoto similarity via Morgan fingerprints
    (radius=2, nBits=2048).

    Requires ``rdkit`` (lazy import, not in the base install)::

        uv pip install rdkit

    Returns: {exact_canonical, mean_tanimoto, parse_fail_rate, n}
"""

from __future__ import annotations

import string
from pathlib import Path
from typing import Any

from .base import Evaluator, register_evaluator
from .metrics import compute_tanimoto

# Punctuation + whitespace table for name normalisation.
_PUNCT_TBL = str.maketrans("", "", string.punctuation + " \t")


def _norm_name(s: str) -> str:
    return s.lower().translate(_PUNCT_TBL)


def _score_name(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Name-mode scoring with rapidfuzz."""
    try:
        from rapidfuzz import fuzz
    except ImportError as err:
        raise RuntimeError(
            "name-mode evaluation requires rapidfuzz: `uv pip install rapidfuzz`"
        ) from err

    n = exact = norm_exact = fuzzy85 = empty = 0
    fuzzy_sum = 0.0

    for row in rows:
        meta = row.get("metadata") or {}
        aliases: list[str] = meta.get("aliases", [])
        # Fall back to reference itself if no aliases stored.
        if not aliases:
            aliases = [str(row.get("reference", ""))]

        pred_raw = str(row.get("prediction", "")).strip()
        n += 1

        if not pred_raw:
            empty += 1
            continue

        pred_norm = _norm_name(pred_raw)
        e = any(pred_raw == a for a in aliases)
        ne = any(pred_norm == _norm_name(a) for a in aliases)
        best_ratio = 0.0
        for a in aliases:
            r = fuzz.ratio(pred_norm, _norm_name(a))
            if r > best_ratio:
                best_ratio = r
        fz85 = best_ratio >= 85

        exact += e
        norm_exact += ne
        fuzzy85 += fz85
        fuzzy_sum += best_ratio / 100.0

    return {
        "exact": exact / n if n else float("nan"),
        "norm_exact": norm_exact / n if n else float("nan"),
        "fuzzy_at_85": fuzzy85 / n if n else float("nan"),
        "mean_fuzzy": fuzzy_sum / n if n else float("nan"),
        "empty_rate": empty / n if n else float("nan"),
        "n": n,
    }


def _score_smiles(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """SMILES-mode scoring with RDKit canonical match + Tanimoto."""
    try:
        from rdkit import Chem, RDLogger
    except ImportError as err:
        raise RuntimeError(
            "smiles-mode evaluation requires rdkit: `uv pip install rdkit`"
        ) from err

    RDLogger.DisableLog("rdApp.*")

    n = exact_canonical = 0
    pred_smiles_list: list[str] = []
    ref_smiles_list: list[str] = []

    for row in rows:
        ref_raw = str(row.get("reference", "")).strip()
        pred_raw = str(row.get("prediction", "")).strip()
        n += 1

        # Canonicalise for exact-match check.
        rmol = Chem.MolFromSmiles(ref_raw) if ref_raw else None
        pmol = Chem.MolFromSmiles(pred_raw) if pred_raw else None
        if rmol is not None and pmol is not None:
            if Chem.MolToSmiles(rmol) == Chem.MolToSmiles(pmol):
                exact_canonical += 1

        pred_smiles_list.append(pred_raw)
        ref_smiles_list.append(ref_raw)

    tanimoto_result = compute_tanimoto(pred_smiles_list, ref_smiles_list)
    parse_fail_rate = (
        1.0 - (tanimoto_result["n_parsed"] / n) if n > 0 else float("nan")
    )

    return {
        "exact_canonical": exact_canonical / n if n else float("nan"),
        "mean_tanimoto": tanimoto_result["tanimoto"],
        "parse_fail_rate": parse_fail_rate,
        "n": n,
    }


@register_evaluator("mi")
class Task5Evaluator(Evaluator):
    """Evaluator for MI (molecular-structure identification VQA).

    Reads ``config['mode']`` to dispatch to name or SMILES scoring.
    """

    def evaluate(self, predictions_path: Path) -> dict[str, Any]:
        mode = self.config.get("mode", "smiles")
        rows = list(self._load_predictions(predictions_path))

        if mode == "name":
            return _score_name(rows)
        elif mode == "smiles":
            return _score_smiles(rows)
        else:
            raise ValueError(
                f"MI 'mode' must be 'name' or 'smiles'; got {mode!r}"
            )
