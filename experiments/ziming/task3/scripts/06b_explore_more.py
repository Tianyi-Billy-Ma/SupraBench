"""
Task 1 · 指标探索 v2: 加 rouge-2 / METEOR / chrF / IDF-weighted rouge1_r /
TF-IDF cosine / 不同 tol 的 num_rec / per-subtype 拆分.

只看 fewshot, 同样 EXCLUDE 集; 跟 v1 的 rouge1_r baseline 对比.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path

import nltk
import numpy as np
import pandas as pd
from rouge_score import rouge_scorer

# 静默 nltk 资源 (METEOR 需要 wordnet)
for res in ("wordnet", "omw-1.4", "punkt", "punkt_tab"):
    try:
        nltk.data.find(f"corpora/{res}")
    except LookupError:
        try:
            nltk.download(res, quiet=True)
        except Exception:
            pass
from nltk.translate.meteor_score import meteor_score

ROOT = Path(__file__).resolve().parent.parent
TASK1_DIR = ROOT / "outputs" / "task1"
OUT_DIR = ROOT / "outputs" / "task1_eval"

SETTINGS = ["base", "fewshot", "cot"]
EXCLUDE_MODELS = {
    "gpt-5.5_nothinking", "gpt-5.5_xhigh", "gemini-3-flash-preview_high",
    "gpt-5.4-nano_nothinking", "gpt-5.4-nano_xhigh",
}

MODEL_SIZE_B = {
    "deepseek-v4-pro": 670.0, "claude-sonnet-4.6": 200.0,
    "gemini-3-flash-preview_nothinking": 80.0, "llama-3.1-70b-instruct": 70.0,
    "qwen3.5-27b": 27.0, "qwen3.5-9b": 9.0, "llama-3.1-8b-instruct": 8.0,
}

RE_WORD = re.compile(r"[A-Za-z][A-Za-z0-9\-]+")
RE_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def tokens(s: str) -> list[str]:
    return [w.lower() for w in RE_WORD.findall(s or "")]


def numbers_in(s: str) -> list[float]:
    out = []
    for m in RE_NUM.finditer(s or ""):
        try: out.append(float(m.group()))
        except: pass
    return out


def num_rec(g: list[float], p: list[float], tol_rel: float) -> float:
    if not g: return float("nan")
    used = [False] * len(p); hit = 0
    for gv in g:
        for j, pv in enumerate(p):
            if used[j]: continue
            tol = max(abs(gv) * tol_rel, 1e-6 if tol_rel == 0 else 0.05)
            if abs(gv - pv) <= tol:
                used[j] = True; hit += 1; break
    return hit / len(g)


# --- chrF (sacrebleu-style char-ngram F1, default n=6) -----------------
def char_ngrams(s: str, n: int) -> Counter:
    s = re.sub(r"\s+", " ", s.strip())
    if len(s) < n: return Counter()
    return Counter(s[i:i+n] for i in range(len(s) - n + 1))


def chrF(g: str, p: str, beta: float = 2.0, max_n: int = 6) -> float:
    if not g or not p: return 0.0
    f_total = 0.0; cnt = 0
    for n in range(1, max_n + 1):
        gn, pn = char_ngrams(g, n), char_ngrams(p, n)
        if not pn: continue
        match = sum(min(pn[k], gn.get(k, 0)) for k in pn)
        prec = match / max(sum(pn.values()), 1)
        rec  = match / max(sum(gn.values()), 1)
        if prec + rec == 0: continue
        f = (1 + beta**2) * prec * rec / (beta**2 * prec + rec)
        f_total += f; cnt += 1
    return f_total / cnt if cnt else 0.0


# --- IDF-weighted rouge1 recall ---------------------------------------
def build_idf(gold_texts: list[str]) -> dict[str, float]:
    N = len(gold_texts)
    df = Counter()
    for g in gold_texts:
        for tok in set(tokens(g)):
            df[tok] += 1
    return {t: math.log(1 + N / df[t]) for t in df}


def idf_rouge1_r(gold: str, pred: str, idf: dict[str, float]) -> float:
    gtok = tokens(gold); ptok_set = set(tokens(pred))
    if not gtok: return float("nan")
    num = den = 0.0
    for t in gtok:
        w = idf.get(t, 0.0)
        den += w
        if t in ptok_set: num += w
    return num / den if den else 0.0


# --- TF-IDF cosine -----------------------------------------------------
def tfidf_cosine(gold: str, pred: str, idf: dict[str, float]) -> float:
    if not gold or not pred: return 0.0
    gc, pc = Counter(tokens(gold)), Counter(tokens(pred))
    if not gc or not pc: return 0.0
    vocab = set(gc) | set(pc)
    g_vec = np.array([gc[t] * idf.get(t, 0.0) for t in vocab])
    p_vec = np.array([pc[t] * idf.get(t, 0.0) for t in vocab])
    ng, np_ = np.linalg.norm(g_vec), np.linalg.norm(p_vec)
    if ng == 0 or np_ == 0: return 0.0
    return float((g_vec @ p_vec) / (ng * np_))


# --- METEOR (nltk) -----------------------------------------------------
def meteor(g: str, p: str) -> float:
    if not g or not p: return 0.0
    try:
        return float(meteor_score([tokens(g)], tokens(p)))
    except Exception:
        return float("nan")


# --- 主流程 -----------------------------------------------------------
def main() -> None:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    # 先收集所有 (model, setting) 的 (id, gold, pred), 用 gold 池建 IDF
    bucket = []  # rows
    gold_corpus: list[str] = []
    seen_gold_ids = set()
    for md in sorted(TASK1_DIR.iterdir()):
        if not md.is_dir() or md.name in EXCLUDE_MODELS: continue
        for s in SETTINGS:
            p = md / s / "predictions.csv"
            if not p.exists(): continue
            d = pd.read_csv(p).drop_duplicates(subset=["id"], keep="last")
            for _, r in d.iterrows():
                g = "" if pd.isna(r.get("gold_answer")) else str(r["gold_answer"])
                pr = "" if pd.isna(r.get("pred_answer")) else str(r["pred_answer"])
                bucket.append({
                    "out_name": md.name, "setting": s, "id": r["id"],
                    "subtype": r.get("subtype", ""), "gold": g, "pred": pr,
                })
                key = (r["id"], g)
                if key not in seen_gold_ids:
                    seen_gold_ids.add(key)
                    gold_corpus.append(g)
    print(f"loaded {len(bucket)} rows; built IDF from {len(gold_corpus)} unique gold texts")
    idf = build_idf(gold_corpus)

    # 算指标
    rows = []
    for i, r in enumerate(bucket):
        if i % 200 == 0: print(f"  {i}/{len(bucket)}")
        g, p = r["gold"], r["pred"]
        if g and p:
            sc = scorer.score(g, p)
            r1r = sc["rouge1"].recall
            r2r = sc["rouge2"].recall; r2f = sc["rouge2"].fmeasure; r2p = sc["rouge2"].precision
        else:
            r1r = r2r = r2f = r2p = 0.0
        gn, pn = numbers_in(g), numbers_in(p)
        rows.append({
            **{k: r[k] for k in ("out_name", "setting", "id", "subtype")},
            "rouge1_r":  r1r,
            "rouge2_r":  r2r, "rouge2_f": r2f, "rouge2_p": r2p,
            "meteor":    meteor(g, p),
            "chrF":      chrF(g, p, beta=2.0, max_n=6),
            "idf_r1r":   idf_rouge1_r(g, p, idf),
            "tfidf_cos": tfidf_cosine(g, p, idf),
            "num_rec_tol01":  num_rec(gn, pn, 0.01),
            "num_rec_tol05":  num_rec(gn, pn, 0.05),
            "num_rec_tol10":  num_rec(gn, pn, 0.10),
            "num_rec_tol20":  num_rec(gn, pn, 0.20),
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "metric_explore_v2_per_row.csv", index=False)

    metric_cols = [c for c in df.columns
                   if c not in ("out_name", "setting", "id", "subtype")]
    agg = df.groupby(["out_name", "setting"])[metric_cols].mean().reset_index()
    agg.to_csv(OUT_DIR / "metric_explore_v2_summary.csv", index=False)

    # ====== Scaling check ======
    fs = agg[agg.setting == "fewshot"].copy()
    fs["size_B"] = fs.out_name.map(MODEL_SIZE_B)
    fs = fs.dropna(subset=["size_B"])
    fs["log_size"] = fs.size_B.apply(math.log10)

    def corr_table(sub: pd.DataFrame, title: str) -> pd.DataFrame:
        out = []
        for c in metric_cols:
            x = sub[["log_size", c]].dropna()
            if len(x) < 4: continue
            out.append({
                "metric": c,
                "spearman": x.log_size.corr(x[c], method="spearman"),
                "pearson":  x.log_size.corr(x[c], method="pearson"),
                "n": len(x),
            })
        df_ = pd.DataFrame(out).sort_values("spearman", ascending=False)
        print("\n" + "=" * 80); print(title); print("=" * 80)
        print(df_.to_string(index=False, float_format="%.3f"))
        return df_

    cdf_full = corr_table(fs, "Spearman corr (fewshot, n=7 含 qwen-9b)")
    cdf_clean = corr_table(fs[fs.out_name != "qwen3.5-9b"],
                            "Spearman corr (fewshot, n=6 排除 qwen-9b)")
    cdf_full.to_csv(OUT_DIR / "metric_scaling_corr_v2_full.csv", index=False)
    cdf_clean.to_csv(OUT_DIR / "metric_scaling_corr_v2_clean.csv", index=False)

    # 实际数值
    print("\n" + "=" * 110)
    print("FEWSHOT 实际值 (按 size 降序, 排除 qwen-9b)")
    print("=" * 110)
    show = (fs[fs.out_name != "qwen3.5-9b"]
            .sort_values("size_B", ascending=False)
            [["out_name", "size_B"] + metric_cols])
    print(show.to_string(index=False, float_format="%.3f"))


if __name__ == "__main__":
    main()
