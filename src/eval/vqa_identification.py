"""Evaluator for the VQA identification (SMILES) task.

Pairs with :mod:`datasets.vqa_identification`. The reference is the
gold canonical SMILES; the prediction is the model's raw text output,
which is first cleaned to extract a SMILES token, then parsed and
compared with RDKit (canonical match, InChIKey first-block match,
formula match, Morgan fingerprint Tanimoto, heavy-atom diff).

Requires the ``vqa`` extra (RDKit)::

    uv sync --extra vqa
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .base import Evaluator, register_evaluator


_SMILES_TOKEN_RE = re.compile(r"\S+")
_ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)


def _postprocess_smiles(raw: str) -> str:
    """Pull a candidate SMILES token out of a free-form model response.

    Order of operations:
    1. If the response wraps a payload in ``<answer>...</answer>`` (the
       format the BASE_TEMPLATE explicitly asks for), use only that payload.
    2. Strip fenced code blocks and leading ``smiles:`` prefixes.
    3. Take the first non-whitespace token from the first line — same
       cleanup the legacy ``run_identification`` script did before scoring.

    Returns the empty string if nothing plausible is found.
    """
    if not raw:
        return ""
    answer_match = _ANSWER_TAG_RE.search(raw)
    if answer_match:
        raw = answer_match.group(1)
    s = raw.strip().replace("```", " ").replace("`", " ")
    s = re.sub(r"(?i)^\s*smiles\s*[:=]\s*", "", s)
    line = s.splitlines()[0].strip() if s else ""
    match = _SMILES_TOKEN_RE.search(line)
    return match.group(0) if match else ""


@register_evaluator("vqa_identification")
class VQAIdentificationEvaluator(Evaluator):
    def evaluate(self, predictions_path: Path) -> dict[str, Any]:
        from rdkit import Chem, RDLogger
        from rdkit.Chem import rdFingerprintGenerator, rdMolDescriptors
        from rdkit.DataStructs import TanimotoSimilarity

        RDLogger.DisableLog("rdApp.*")
        fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

        def _mol(smiles: str):
            return Chem.MolFromSmiles(smiles) if smiles else None

        rows = list(self._load_predictions(predictions_path))

        n = 0
        valid = 0
        canonical_exact = 0
        inchi_match = 0
        formula_match = 0
        tanimoto_sum = 0.0
        tanimoto_n = 0
        tanimoto_ge_0_7 = 0
        heavy_abs_diff_sum = 0.0
        heavy_n = 0

        for row in rows:
            gold_smiles = (row.get("reference") or "").strip()
            gold_mol = _mol(gold_smiles)
            if gold_mol is None:
                continue
            n += 1

            pred_smiles = _postprocess_smiles(str(row.get("prediction") or ""))
            pred_mol = _mol(pred_smiles)
            if pred_mol is None:
                continue
            valid += 1

            g_canon = Chem.MolToSmiles(gold_mol)
            p_canon = Chem.MolToSmiles(pred_mol)
            if g_canon == p_canon:
                canonical_exact += 1

            g_ik = Chem.MolToInchiKey(gold_mol).split("-")[0]
            p_ik = Chem.MolToInchiKey(pred_mol).split("-")[0]
            if g_ik and g_ik == p_ik:
                inchi_match += 1

            if rdMolDescriptors.CalcMolFormula(gold_mol) == rdMolDescriptors.CalcMolFormula(pred_mol):
                formula_match += 1

            tan = TanimotoSimilarity(fpgen.GetFingerprint(gold_mol),
                                     fpgen.GetFingerprint(pred_mol))
            tanimoto_sum += tan
            tanimoto_n += 1
            if tan >= 0.7:
                tanimoto_ge_0_7 += 1

            heavy_abs_diff_sum += abs(gold_mol.GetNumHeavyAtoms() - pred_mol.GetNumHeavyAtoms())
            heavy_n += 1

        return {
            "n": n,
            "validity": valid / n if n else 0.0,
            "canonical_exact": canonical_exact / n if n else 0.0,
            "inchikey_first_block_match": inchi_match / n if n else 0.0,
            "formula_match": formula_match / n if n else 0.0,
            "tanimoto_mean_over_valid": tanimoto_sum / tanimoto_n if tanimoto_n else 0.0,
            "tanimoto_ge_0.7": tanimoto_ge_0_7 / n if n else 0.0,
            "heavy_atom_abs_diff_mean": heavy_abs_diff_sum / heavy_n if heavy_n else 0.0,
        }
