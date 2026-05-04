"""
Task 1 评测: ROUGE-L F1 + Keyword Hit (KH).

KH 关键词 = gold answer 里的 representative guests / hosts 名字.
- forward: "Representative guests: G1, G2, ..." 后面用句号结尾
- reverse: "include: H1 (logKa=...), H2 (logKa=...)" — 去掉 (logKa=) 后的名字

ROUGE-L F1: pred_answer vs gold_answer, 整段比较, stemmer on.

读 outputs/task1/{out_name}/{setting}/predictions.csv,
按 id 去重 (qwen3.5-27b base 有 dup) 后评.
输出:
  outputs/task1_eval/{out_name}_{setting}_per_row.csv  (每行明细)
  outputs/task1_eval/leaderboard.csv                    (总表)

用法:
    python scripts/04_evaluate_task1.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd
from rouge_score import rouge_scorer

ROOT = Path(__file__).resolve().parent.parent
TASK1_DIR = ROOT / "outputs" / "task1"
EVAL_DIR = ROOT / "outputs" / "task1_eval"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

SETTINGS = ["base", "fewshot", "cot"]

# forward gold 模板: "Representative guests: A, B, C." (entries 之间 ", " 切)
RE_FWD = re.compile(r"Representative (?:guests?|hosts?):\s*(.+?)\s*\.?\s*$",
                    re.IGNORECASE | re.DOTALL)
# reverse gold 模板: "include: H1 (logKa=N1), H2 (logKa=N2), ... . The highest..."
RE_REV_INCLUDE = re.compile(r"include:\s*(.+?)\.\s*The highest",
                            re.IGNORECASE | re.DOTALL)
RE_REV_INCLUDE_FALLBACK = re.compile(r"include:\s*(.+)$",
                                     re.IGNORECASE | re.DOTALL)
# 每个 reverse entry 形如 "<name> (logKa=N)" — 用 logKa 当分隔锚
RE_REV_ENTRY = re.compile(r"(.+?)\s*\(\s*logKa\s*=\s*[\d.\-]+\s*\)",
                          re.IGNORECASE | re.DOTALL)


def _split_at_depth_zero(s: str) -> list[str]:
    """按 ', ' (逗号+空格) 切, 但只在括号/方括号深度=0 时切.

    化学名内部的逗号 (e.g. 'tetradecane-4,9-diol') 没空格, 不切;
    括号内的 ', ' (e.g. '(R, S)-isomer') 也保留.
    """
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
    # 去空, 去首尾标点
    return [p.strip(" .,;") for p in parts if p.strip(" .,;")]


def extract_keywords(gold: str, subtype: str) -> list[str]:
    """从 gold 抽 KH 关键词列表 (guest/host 名字)."""
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
        # 用 (logKa=N) 当锚, 每个 entry 是 anchor 前的 name 部分
        names = []
        for em in RE_REV_ENTRY.finditer(tail):
            name = em.group(1).strip(" ,.;")
            # 如果 name 以 ", " 开头 (entry 之间分隔), 去掉
            name = re.sub(r"^[,;]\s*", "", name).strip()
            if name:
                names.append(name)
        return names
    return []


def kh_score(pred: str, keywords: list[str]) -> tuple[float, int, int]:
    """
    返回 (kh_recall, n_hit, n_total).
    匹配: lowercase substring (整名子串命中).
    """
    if not keywords:
        return float("nan"), 0, 0
    if not isinstance(pred, str) or not pred:
        return 0.0, 0, len(keywords)
    p_low = pred.lower()
    n_hit = sum(1 for k in keywords if k.lower() in p_low)
    return n_hit / len(keywords), n_hit, len(keywords)


def evaluate_one(csv_path: Path, scorer: rouge_scorer.RougeScorer) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # 去重: 同 id 保留最后一条 (resume 写入靠后的更新)
    df = df.drop_duplicates(subset=["id"], keep="last").reset_index(drop=True)

    rows = []
    for _, r in df.iterrows():
        gold = r.get("gold_answer", "") or ""
        pred = r.get("pred_answer", "") or ""
        gold = "" if pd.isna(gold) else str(gold)
        pred = "" if pd.isna(pred) else str(pred)
        subtype = r.get("subtype", "")

        # ROUGE-L F1 + ROUGE-1 recall (scaling-friendly, 见 06_explore_metrics_task1.py)
        if gold and pred:
            rouge = scorer.score(gold, pred)
            rougeL_f = rouge["rougeL"].fmeasure
            rouge1_r = rouge["rouge1"].recall
        else:
            rougeL_f = 0.0
            rouge1_r = 0.0

        # KH
        kws = extract_keywords(gold, subtype)
        kh, hit, total = kh_score(pred, kws)

        rows.append({
            "id": r.get("id"),
            "subtype": subtype,
            "rougeL_f": rougeL_f,
            "rouge1_r": rouge1_r,
            "kh": kh,
            "kh_hit": hit,
            "kh_total": total,
            "n_keywords_found": len(kws),
            "parse_status": r.get("parse_status"),
        })
    return pd.DataFrame(rows)


def main():
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)

    leaderboard = []
    for model_dir in sorted(TASK1_DIR.iterdir()):
        if not model_dir.is_dir():
            continue
        out_name = model_dir.name
        for setting in SETTINGS:
            csv_path = model_dir / setting / "predictions.csv"
            if not csv_path.exists():
                continue
            print(f"[{out_name} / {setting}] ...", end=" ")
            per_row = evaluate_one(csv_path, scorer)
            per_row.to_csv(EVAL_DIR / f"{out_name}_{setting}_per_row.csv",
                           index=False)

            # 过滤掉 parse_status='fallback_full' 的行 (parser 抽不出干净答案,
            # 通常是 reasoning leak / 长噪声; 退化输出会污染 recall 类指标).
            # n_total / n_valid / parse_fail_rate 都报出来, 透明.
            n_total = len(per_row)
            pf_mask = per_row["parse_status"].astype(str).str.lower() == "fallback_full"
            valid_row = per_row[~pf_mask].reset_index(drop=True)
            n_valid = len(valid_row)
            pf_rate = pf_mask.sum() / n_total if n_total else 0.0

            def _safe_mean(d, col):
                return d[col].mean() if len(d) else float("nan")

            # 总分: 在 valid_row 上算; 若 valid 为空, 用 NaN
            rougeL_mean = _safe_mean(valid_row, "rougeL_f")
            rouge1r_mean = _safe_mean(valid_row, "rouge1_r")
            kh_valid_rows = valid_row.dropna(subset=["kh"])
            kh_mean = _safe_mean(kh_valid_rows, "kh")

            # 按 subtype 拆开 (也在 valid 上)
            fwd = valid_row[valid_row["subtype"] == "forward"]
            rev = valid_row[valid_row["subtype"] == "reverse"]

            row = {
                "out_name": out_name,
                "setting": setting,
                "n": n_total,
                "n_valid": n_valid,
                "parse_fail_rate": pf_rate,
                "rougeL_all": rougeL_mean,
                "rouge1r_all": rouge1r_mean,
                "kh_all": kh_mean,
                "rougeL_fwd": _safe_mean(fwd, "rougeL_f"),
                "rouge1r_fwd": _safe_mean(fwd, "rouge1_r"),
                "kh_fwd": _safe_mean(fwd.dropna(subset=["kh"]), "kh"),
                "rougeL_rev": _safe_mean(rev, "rougeL_f"),
                "rouge1r_rev": _safe_mean(rev, "rouge1_r"),
                "kh_rev": _safe_mean(rev.dropna(subset=["kh"]), "kh"),
            }
            leaderboard.append(row)
            print(f"n={n_total}  valid={n_valid}  pf={pf_rate:.1%}  "
                  f"rougeL={rougeL_mean:.3f}  rouge1r={rouge1r_mean:.3f}  "
                  f"kh={kh_mean:.3f}")

    lb = pd.DataFrame(leaderboard)
    lb.to_csv(EVAL_DIR / "leaderboard.csv", index=False)

    # 打印排序后的总表
    print("\n" + "=" * 90)
    print("LEADERBOARD (按 rougeL_all 降序)")
    print("=" * 90)
    lb_sorted = lb.sort_values("rougeL_all", ascending=False)
    print(lb_sorted.to_string(index=False, float_format="%.3f"))

    print("\n" + "=" * 90)
    print("LEADERBOARD (按 kh_all 降序)")
    print("=" * 90)
    print(lb.sort_values("kh_all", ascending=False)
            .to_string(index=False, float_format="%.3f"))

    print(f"\n保存: {EVAL_DIR / 'leaderboard.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
