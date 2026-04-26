"""
Score identification (name / SMILES) predictions for one run against
supra_vqa/identification.csv.

Run:
    python3 supra_vqa/scripts/evaluate_identification.py --run-id <id>
    python3 supra_vqa/scripts/evaluate_identification.py --latest name
    python3 supra_vqa/scripts/evaluate_identification.py --latest smiles

Outputs (inside supra_vqa/results/<run_id>/):
    predictions_scored.csv
    summary.json
"""

import argparse
import csv
import json
import string
import sys
from pathlib import Path

SUPRA_VQA_ROOT = Path(__file__).resolve().parent.parent
EVAL_CSV = SUPRA_VQA_ROOT / "identification.csv"
RESULTS_DIR = SUPRA_VQA_ROOT / "results"


def load_eval_index() -> dict[str, dict]:
    with EVAL_CSV.open() as f:
        return {r["molecule_id"]: r for r in csv.DictReader(f)}


def load_predictions(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


# -------------------- name metrics --------------------

_PUNCT_TBL = str.maketrans("", "", string.punctuation + " \t")


def norm_name(s: str) -> str:
    return s.lower().translate(_PUNCT_TBL)


def score_name(preds: list[dict], idx: dict[str, dict]) -> tuple[dict, list[dict]]:
    try:
        from rapidfuzz import fuzz
    except ImportError:
        sys.exit("rapidfuzz not installed. run: uv pip install rapidfuzz")

    n = exact = norm_exact = fuzzy85 = empty = 0
    fuzzy_sum = 0.0
    scored_rows = []
    for p in preds:
        gold = idx.get(p["molecule_id"])
        if not gold:
            continue
        aliases = [a for a in gold["names_alias_set"].split("|") if a]
        if not aliases:
            continue
        n += 1
        pred_raw = p["pred"]
        if not pred_raw:
            empty += 1
            scored_rows.append({**p, "best_alias": "",
                                "exact": False, "normalized_exact": False,
                                "fuzzy_best": 0.0, "fuzzy_ge_0.85": False})
            continue
        pred_norm = norm_name(pred_raw)
        e = any(pred_raw == a for a in aliases)
        ne = any(pred_norm == norm_name(a) for a in aliases)
        best_alias = ""
        best_ratio = 0.0
        for a in aliases:
            r = fuzz.ratio(pred_norm, norm_name(a))
            if r > best_ratio:
                best_ratio = r
                best_alias = a
        fz85 = best_ratio >= 85
        exact += e
        norm_exact += ne
        fuzzy85 += fz85
        fuzzy_sum += best_ratio / 100.0
        scored_rows.append({**p, "best_alias": best_alias,
                            "exact": e, "normalized_exact": ne,
                            "fuzzy_best": round(best_ratio / 100.0, 3),
                            "fuzzy_ge_0.85": fz85})
    metrics = {
        "n": n,
        "exact": exact / n if n else 0.0,
        "normalized_exact": norm_exact / n if n else 0.0,
        "fuzzy_ge_0.85": fuzzy85 / n if n else 0.0,
        "fuzzy_mean": fuzzy_sum / n if n else 0.0,
        "empty_rate": empty / n if n else 0.0,
    }
    return metrics, scored_rows


# -------------------- smiles metrics --------------------

def score_smiles(preds: list[dict], idx: dict[str, dict]) -> tuple[dict, list[dict]]:
    try:
        from rdkit import Chem, RDLogger
        from rdkit.Chem import rdFingerprintGenerator, rdMolDescriptors
        from rdkit.DataStructs import TanimotoSimilarity
    except ImportError:
        sys.exit("rdkit not installed. run: uv pip install rdkit")
    RDLogger.DisableLog("rdApp.*")

    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

    def mol_of(s: str):
        if not s:
            return None
        return Chem.MolFromSmiles(s)

    n = valid = canonical_exact = inchi_match = formula_match = 0
    tanimoto_sum = 0.0
    tanimoto_ge_0_7 = 0
    tanimoto_n = 0
    heavy_abs_diff_sum = 0.0
    heavy_n = 0
    scored_rows = []

    for p in preds:
        gold = idx.get(p["molecule_id"])
        if not gold or not gold["cano_smiles"]:
            continue
        gmol = mol_of(gold["cano_smiles"])
        if gmol is None:
            continue
        n += 1
        pmol = mol_of(p["pred"])
        row = {**p, "gold_smiles": gold["cano_smiles"]}
        if pmol is None:
            row.update({"valid": False, "canonical_exact": False,
                        "inchikey_first_block_match": False,
                        "formula_match": False, "tanimoto": 0.0})
            scored_rows.append(row)
            continue
        valid += 1
        g_canon = Chem.MolToSmiles(gmol)
        p_canon = Chem.MolToSmiles(pmol)
        ce = g_canon == p_canon
        g_ik = Chem.MolToInchiKey(gmol).split("-")[0]
        p_ik = Chem.MolToInchiKey(pmol).split("-")[0]
        ikm = bool(g_ik) and g_ik == p_ik
        g_f = rdMolDescriptors.CalcMolFormula(gmol)
        p_f = rdMolDescriptors.CalcMolFormula(pmol)
        fm = g_f == p_f
        tan = TanimotoSimilarity(fpgen.GetFingerprint(gmol), fpgen.GetFingerprint(pmol))
        canonical_exact += ce
        inchi_match += ikm
        formula_match += fm
        tanimoto_sum += tan
        tanimoto_n += 1
        if tan >= 0.7:
            tanimoto_ge_0_7 += 1
        heavy_abs_diff_sum += abs(gmol.GetNumHeavyAtoms() - pmol.GetNumHeavyAtoms())
        heavy_n += 1
        row.update({"valid": True, "canonical_exact": ce,
                    "inchikey_first_block_match": ikm, "formula_match": fm,
                    "tanimoto": round(tan, 3)})
        scored_rows.append(row)

    metrics = {
        "n": n,
        "validity": valid / n if n else 0.0,
        "canonical_exact": canonical_exact / n if n else 0.0,
        "inchikey_first_block_match": inchi_match / n if n else 0.0,
        "formula_match": formula_match / n if n else 0.0,
        "tanimoto_mean_over_valid": tanimoto_sum / tanimoto_n if tanimoto_n else 0.0,
        "tanimoto_ge_0.7": tanimoto_ge_0_7 / n if n else 0.0,
        "heavy_atom_abs_diff_mean": heavy_abs_diff_sum / heavy_n if heavy_n else 0.0,
    }
    return metrics, scored_rows


# -------------------- runner --------------------

def write_scored(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    cols = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def write_summary(run_dir: Path, run_info: dict, metrics: dict) -> Path:
    summary = {
        "run_id": run_info.get("run_id"),
        "model": run_info.get("model"),
        "mode": run_info.get("mode"),
        "timestamp": run_info.get("timestamp"),
        "n": metrics.get("n"),
        "metrics": metrics,
    }
    path = run_dir / "summary.json"
    path.write_text(json.dumps(summary, indent=2))
    return path


def resolve_run_dir(run_id: str | None, latest_mode: str | None) -> Path:
    if run_id:
        d = RESULTS_DIR / run_id
        if not d.is_dir():
            sys.exit(f"no such run dir: {d}")
        return d
    if latest_mode:
        candidates = sorted(
            (p for p in RESULTS_DIR.glob(f"*_{latest_mode}_*") if p.is_dir()),
            key=lambda p: p.name,
        )
        if not candidates:
            sys.exit(f"no runs for mode={latest_mode} under {RESULTS_DIR}")
        return candidates[-1]
    sys.exit("pass --run-id <id> or --latest {name,smiles}")


def score_run(run_dir: Path) -> dict:
    pred_path = run_dir / "predictions.csv"
    info_path = run_dir / "run_info.json"
    if not pred_path.exists():
        sys.exit(f"missing predictions: {pred_path}")
    if not info_path.exists():
        sys.exit(f"missing run_info.json: {info_path}")

    run_info = json.loads(info_path.read_text())
    mode = run_info["mode"]
    idx = load_eval_index()
    preds = load_predictions(pred_path)

    if mode == "name":
        metrics, scored = score_name(preds, idx)
    elif mode == "smiles":
        metrics, scored = score_smiles(preds, idx)
    else:
        sys.exit(f"unknown mode in run_info.json: {mode}")

    scored_path = run_dir / "predictions_scored.csv"
    write_scored(scored_path, scored)
    summary_path = write_summary(run_dir, run_info, metrics)

    print(f"run_id: {run_info['run_id']}  (mode={mode})")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:32s} {v:.4f}")
        else:
            print(f"  {k:32s} {v}")
    print(f"scored rows -> {scored_path}")
    print(f"summary     -> {summary_path}")
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", type=str, default=None)
    ap.add_argument("--latest", choices=["name", "smiles"], default=None)
    args = ap.parse_args()
    score_run(resolve_run_dir(args.run_id, args.latest))


if __name__ == "__main__":
    main()
