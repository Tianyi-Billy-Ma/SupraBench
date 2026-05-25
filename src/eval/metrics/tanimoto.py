"""Tanimoto similarity via RDKit Morgan fingerprints.

RDKit is an optional dependency used only by the MI (SMILES-mode)
evaluator. Install it before running that evaluator::

    uv pip install rdkit

or add it to the ``hf`` extra in ``pyproject.toml`` (already done as of
this commit). The import is deferred inside ``compute_tanimoto`` so that
the base install (which lacks rdkit) can still import this module without
errors.
"""

from __future__ import annotations

from typing import Sequence


def compute_tanimoto(
    pred_smiles: Sequence[str],
    ref_smiles: Sequence[str],
) -> dict[str, float]:
    """Compute mean Tanimoto similarity between SMILES pairs.

    Uses Morgan fingerprints (radius=2, nBits=2048). Pairs where either
    SMILES fails to parse are counted in ``n`` but not in ``n_parsed``;
    their contribution to the mean is zero (conservative choice).

    Args:
        pred_smiles: Predicted SMILES strings.
        ref_smiles: Gold-standard SMILES strings.

    Returns:
        Dict with keys:
        - ``tanimoto``: mean Tanimoto over all *n* pairs (unparseable → 0).
        - ``n_parsed``: pairs where both SMILES parsed successfully.
        - ``n``: total pairs.
    """
    if len(pred_smiles) != len(ref_smiles):
        raise ValueError(
            f"length mismatch: {len(pred_smiles)} predictions vs {len(ref_smiles)} references"
        )

    try:
        from rdkit import Chem, RDLogger
        from rdkit.Chem import rdFingerprintGenerator
        from rdkit.DataStructs import TanimotoSimilarity
    except ImportError as err:
        raise RuntimeError(
            "tanimoto metric requires rdkit: `uv pip install rdkit`"
        ) from err

    RDLogger.DisableLog("rdApp.*")
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

    n = len(pred_smiles)
    n_parsed = 0
    total_sim = 0.0

    for ps, rs in zip(pred_smiles, ref_smiles):
        pmol = Chem.MolFromSmiles(ps) if ps else None
        rmol = Chem.MolFromSmiles(rs) if rs else None
        if pmol is not None and rmol is not None:
            sim = TanimotoSimilarity(fpgen.GetFingerprint(pmol), fpgen.GetFingerprint(rmol))
            total_sim += sim
            n_parsed += 1
        # unparseable pairs contribute 0 to the mean

    mean_tan = total_sim / n if n > 0 else float("nan")
    return {"tanimoto": mean_tan, "n_parsed": n_parsed, "n": n}
