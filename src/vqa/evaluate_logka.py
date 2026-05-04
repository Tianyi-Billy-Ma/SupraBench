"""
Score logKa predictions for a given run against supra_vqa/logka.csv.

Run:
    python3 supra_vqa/scripts/evaluate_logka.py --run-id <id>
    python3 supra_vqa/scripts/evaluate_logka.py --latest

Outputs (inside supra_vqa/results/<run_id>/):
    predictions_scored.csv
    summary.json
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

SUPRA_VQA_ROOT = Path(__file__).resolve().parents[2] / "supra-vqa"
EVAL_CSV = SUPRA_VQA_ROOT / "logka.csv"
RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"


def load_eval_index() -> dict[str, dict]:
    with EVAL_CSV.open() as f:
        return {r["pair_id"]: r for r in csv.DictReader(f)}


def load_predictions(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def parse_float(s: str) -> float | None:
    s = (s or "").strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def average_ranks(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(x) + 1)
    sorted_x = x[order]
    i, n = 0, len(x)
    while i < n:
        j = i
        while j + 1 < n and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2
        i = j + 1
    return ranks


def regression_metrics(gold: np.ndarray, pred: np.ndarray) -> dict:
    err = pred - gold
    abs_err = np.abs(err)
    mae = float(abs_err.mean())
    rmse = float(np.sqrt((err ** 2).mean()))
    bias = float(err.mean())
    ss_res = float(((gold - pred) ** 2).sum())
    ss_tot = float(((gold - gold.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    pearson_r = (float(np.corrcoef(gold, pred)[0, 1])
                 if gold.std() > 0 and pred.std() > 0 else float("nan"))
    g_ranks = average_ranks(gold)
    p_ranks = average_ranks(pred)
    spearman_rho = (float(np.corrcoef(g_ranks, p_ranks)[0, 1])
                    if g_ranks.std() > 0 and p_ranks.std() > 0 else float("nan"))
    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "pearson_r": pearson_r,
        "spearman_rho": spearman_rho,
        "within_0.5": float((abs_err <= 0.5).mean()),
        "within_1.0": float((abs_err <= 1.0).mean()),
        "bias": bias,
    }


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


def resolve_run_dir(run_id: str | None, latest: bool) -> Path:
    if run_id:
        d = RESULTS_DIR / run_id
        if not d.is_dir():
            sys.exit(f"no such run dir: {d}")
        return d
    if latest:
        candidates = sorted(
            (p for p in RESULTS_DIR.glob("*_logka_*") if p.is_dir()),
            key=lambda p: p.name,
        )
        if not candidates:
            sys.exit(f"no logka runs found under {RESULTS_DIR}")
        return candidates[-1]
    sys.exit("pass --run-id <id> or --latest")


def score_run(run_dir: Path) -> dict:
    pred_path = run_dir / "predictions.csv"
    info_path = run_dir / "run_info.json"
    if not pred_path.exists():
        sys.exit(f"missing predictions: {pred_path}")
    if not info_path.exists():
        sys.exit(f"missing run_info.json: {info_path}")

    run_info = json.loads(info_path.read_text())
    idx = load_eval_index()
    preds = load_predictions(pred_path)

    gold_list, pred_list = [], []
    scored_rows = []
    n_seen = 0
    n_missing_eval = 0
    for p in preds:
        pid = p["pair_id"]
        gold_row = idx.get(pid)
        if gold_row is None:
            n_missing_eval += 1
            continue
        n_seen += 1
        gold = float(gold_row["logka_standard"])
        pred_val = parse_float(p.get("pred"))
        valid = pred_val is not None
        abs_err = abs(pred_val - gold) if valid else ""
        scored_rows.append({
            "pair_id": pid,
            "host_name": gold_row["host_name"],
            "guest_name": gold_row["guest_name"],
            "logka_gold": f"{gold:.4f}",
            "logka_pred": f"{pred_val:.4f}" if valid else "",
            "abs_error": f"{abs_err:.4f}" if valid else "",
            "valid": valid,
        })
        if valid:
            gold_list.append(gold)
            pred_list.append(pred_val)

    metrics: dict = {
        "n": n_seen,
        "n_valid": len(gold_list),
        "valid_rate": (len(gold_list) / n_seen) if n_seen else 0.0,
    }
    if len(gold_list) >= 2:
        metrics.update(regression_metrics(np.array(gold_list), np.array(pred_list)))
    else:
        for k in ("mae", "rmse", "r2", "pearson_r", "spearman_rho",
                  "within_0.5", "within_1.0", "bias"):
            metrics[k] = float("nan")

    if n_missing_eval:
        print(f"[warn] {n_missing_eval} predictions had pair_id not in eval set", file=sys.stderr)

    scored_path = run_dir / "predictions_scored.csv"
    write_scored(scored_path, scored_rows)
    summary_path = write_summary(run_dir, run_info, metrics)

    print(f"run_id: {run_info['run_id']}  (mode={run_info.get('mode')})")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:24s} {'nan' if math.isnan(v) else f'{v:.4f}'}")
        else:
            print(f"  {k:24s} {v}")
    print(f"scored rows -> {scored_path}")
    print(f"summary     -> {summary_path}")
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", type=str, default=None)
    ap.add_argument("--latest", action="store_true")
    args = ap.parse_args()
    score_run(resolve_run_dir(args.run_id, args.latest))


if __name__ == "__main__":
    main()
