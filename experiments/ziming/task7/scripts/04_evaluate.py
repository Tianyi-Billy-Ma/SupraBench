"""
Task 3 评测: accuracy / macro-F1 / weighted-F1 / parse_fail_rate.

读 outputs/task3/{out_name}/{setting}/predictions.csv,
按 interaction_id 去重 (resume 写入靠后的更新) 后评.

输出:
  outputs/task3_eval/{out_name}_{setting}_per_class.csv  (per-class P/R/F1)
  outputs/task3_eval/leaderboard.csv                      (总表)

用法:
    python scripts/04_evaluate_task3.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TASK3_DIR = ROOT / "outputs" / "task3"
EVAL_DIR = ROOT / "outputs" / "task3_eval"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

SETTINGS = ["base", "fewshot", "cot"]
LABELS = ["water", "DMSO", "MeCN", "MeOH", "CHCl3", "CH2Cl2"]


def macro_f1(y_true: list[str], y_pred: list[str]) -> tuple[float, dict]:
    per_class: dict = {}
    f1s: list[float] = []
    for L in LABELS:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == L and p == L)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != L and p == L)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == L and p != L)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        support = sum(1 for t in y_true if t == L)
        per_class[L] = {"precision": prec, "recall": rec, "f1": f1, "support": support}
        if support > 0:
            f1s.append(f1)
    return (sum(f1s) / len(f1s) if f1s else 0.0), per_class


def weighted_f1(per_class: dict) -> float:
    total = sum(pc["support"] for pc in per_class.values())
    if total == 0:
        return 0.0
    return sum(pc["f1"] * pc["support"] for pc in per_class.values()) / total


def evaluate_one(csv_path: Path) -> dict:
    df = pd.read_csv(csv_path)
    df = df.drop_duplicates(subset=["interaction_id"], keep="last").reset_index(drop=True)

    n_total = len(df)
    pf_mask = df["pred_label"].isna() | (df["pred_label"] == "parse_fail")
    pf = int(pf_mask.sum())
    valid = df[~pf_mask].copy()

    if len(valid):
        acc = float((valid["pred_label"] == valid["true_label"]).mean())
        mf1, per_class = macro_f1(
            valid["true_label"].tolist(), valid["pred_label"].tolist()
        )
        wf1 = weighted_f1(per_class)
    else:
        acc = 0.0
        mf1 = 0.0
        wf1 = 0.0
        per_class = {L: {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0}
                     for L in LABELS}

    return {
        "n_total": n_total,
        "parse_fail": pf,
        "parse_fail_rate": pf / n_total if n_total else 0.0,
        "accuracy": acc,
        "macro_f1": mf1,
        "weighted_f1": wf1,
        "per_class": per_class,
    }


def main():
    leaderboard = []
    for model_dir in sorted(TASK3_DIR.iterdir()):
        if not model_dir.is_dir():
            continue
        out_name = model_dir.name
        for setting in SETTINGS:
            csv_path = model_dir / setting / "predictions.csv"
            if not csv_path.exists():
                continue
            print(f"[{out_name} / {setting}] ...", end=" ")
            r = evaluate_one(csv_path)

            # per-class 明细
            pc_df = pd.DataFrame([
                {"label": L, **r["per_class"][L]} for L in LABELS
            ])
            pc_df.to_csv(EVAL_DIR / f"{out_name}_{setting}_per_class.csv",
                         index=False)

            leaderboard.append({
                "out_name": out_name,
                "setting": setting,
                "n": r["n_total"],
                "parse_fail": r["parse_fail"],
                "parse_fail_rate": r["parse_fail_rate"],
                "accuracy": r["accuracy"],
                "macro_f1": r["macro_f1"],
                "weighted_f1": r["weighted_f1"],
            })
            print(f"n={r['n_total']}  acc={r['accuracy']:.3f}  "
                  f"mF1={r['macro_f1']:.3f}  wF1={r['weighted_f1']:.3f}  "
                  f"pf={r['parse_fail_rate']:.2%}")

    lb = pd.DataFrame(leaderboard)
    lb.to_csv(EVAL_DIR / "leaderboard.csv", index=False)

    print("\n" + "=" * 100)
    print("TASK3 LEADERBOARD (按 macro_f1 降序)")
    print("=" * 100)
    print(lb.sort_values("macro_f1", ascending=False)
            .to_string(index=False, float_format="%.3f"))

    print("\n" + "=" * 100)
    print("TASK3 LEADERBOARD (按 accuracy 降序)")
    print("=" * 100)
    print(lb.sort_values("accuracy", ascending=False)
            .to_string(index=False, float_format="%.3f"))

    print(f"\n保存: {EVAL_DIR / 'leaderboard.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
