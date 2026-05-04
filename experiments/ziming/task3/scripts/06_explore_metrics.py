"""
Task 1 · 指标探索: 批量算多种 simple metric, 看哪个与模型大小(scaling)相关.

候选指标:
  ROUGE 系: rougeL_f, rougeL_p, rougeL_r, rouge1_f, rouge1_r
  KH 系   : kh_recall(原), kh_p_template, kh_f1_template
  数字系  : num_jaccard, num_recall (tol=10%)
  长度系  : len_ratio, brevity_penalty
  内容词  : content_jaccard, content_recall

只看 setting=fewshot, 排除 nano + gpt-5.5.
然后对每个 metric 求 Spearman corr(metric, log(model_size)),
正相关 = 满足 scaling.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import pandas as pd
from rouge_score import rouge_scorer

# 复用 04_evaluate_task1.py 的关键词抽取
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module
ev = import_module("04_evaluate_task1")
extract_keywords = ev.extract_keywords
_split_at_depth_zero = ev._split_at_depth_zero

ROOT = Path(__file__).resolve().parent.parent
TASK1_DIR = ROOT / "outputs" / "task1"
OUT_DIR = ROOT / "outputs" / "task1_eval"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SETTINGS = ["base", "fewshot", "cot"]

EXCLUDE_MODELS = {
    "gpt-5.5_nothinking", "gpt-5.5_xhigh", "gemini-3-flash-preview_high",
    "gpt-5.4-nano_nothinking", "gpt-5.4-nano_xhigh",
}

# 模型 → 参数量(B). MoE 用 total params (deepseek-v4-pro 估 670B).
# 闭源用粗估;同一量级 scaling 看趋势, 不看绝对值.
MODEL_SIZE_B = {
    "deepseek-v4-pro": 670.0,
    "claude-sonnet-4.6": 200.0,
    "gemini-3-flash-preview_nothinking": 80.0,
    "llama-3.1-70b-instruct": 70.0,
    "qwen3.5-27b": 27.0,
    "qwen3.5-9b": 9.0,
    "llama-3.1-8b-instruct": 8.0,
}

STOPWORDS = set("""
a an the and or of to in on for with by is are was were be been being
this that these those it its as at from into onto over under
have has had do does did will would can could should may might
not no nor but if then else than so such which who whom whose what when where why how
about above below across through between among per via vs versus
also typically generally often usually likely most many some few
""".split())


# --- 数字抽取 (含 ±, 范围) -------------------------------------------------
RE_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def numbers_in(s: str) -> list[float]:
    if not isinstance(s, str):
        return []
    out = []
    for m in RE_NUM.finditer(s):
        try:
            v = float(m.group())
            # 跳过明显是化合物名内编号 (e.g. 7.06, 9.10) 的小整数太多没意义,
            # 但也别过滤太狠 — 保留所有
            out.append(v)
        except ValueError:
            pass
    return out


def numbers_match(g: list[float], p: list[float], tol_rel: float = 0.10) -> tuple[float, float]:
    """返回 (jaccard_with_tol, recall_with_tol)."""
    if not g and not p:
        return float("nan"), float("nan")
    if not g:
        return 0.0, float("nan")
    used = [False] * len(p)
    hit = 0
    for gv in g:
        for j, pv in enumerate(p):
            if used[j]:
                continue
            tol = max(abs(gv) * tol_rel, 0.05)
            if abs(gv - pv) <= tol:
                used[j] = True
                hit += 1
                break
    rec = hit / len(g) if g else 0.0
    union = len(g) + len(p) - hit
    jac = hit / union if union else 0.0
    return jac, rec


# --- KH 模板抽取(对 pred 也用同样 anchor,失败 → 空) -----------------------
def extract_keywords_pred(pred: str, subtype: str) -> list[str] | None:
    """对 pred 用同 RE 抽 compound list. None = 抽不到 (template 不匹配)."""
    if not isinstance(pred, str) or not pred:
        return None
    if subtype == "forward":
        m = ev.RE_FWD.search(pred)
        if not m:
            return None
        return _split_at_depth_zero(m.group(1))
    elif subtype == "reverse":
        m = ev.RE_REV_INCLUDE.search(pred) or ev.RE_REV_INCLUDE_FALLBACK.search(pred)
        if not m:
            return None
        names = []
        for em in ev.RE_REV_ENTRY.finditer(m.group(1)):
            n = em.group(1).strip(" ,.;")
            n = re.sub(r"^[,;]\s*", "", n).strip()
            if n:
                names.append(n)
        return names if names else None
    return None


def kh_pf1(pred: str, gold_kws: list[str], subtype: str) -> tuple[float, float, float]:
    """
    template-based KH precision / recall / F1.
    pred 也用同 RE 抽 list, 然后双向 substring 匹配.
    pred 抽不到 → precision=0, recall 退化为原 substring KH.
    """
    if not gold_kws:
        return float("nan"), float("nan"), float("nan")
    pred_kws = extract_keywords_pred(pred, subtype)
    g_low = [k.lower() for k in gold_kws]
    if pred_kws is None:
        # recall 仍按 substring 给 (跟原 kh 一致), precision 给 0
        p_low_text = (pred or "").lower()
        rec = sum(1 for k in g_low if k in p_low_text) / len(g_low)
        return 0.0, rec, 0.0
    p_low = [k.lower() for k in pred_kws]
    if not p_low:
        return 0.0, 0.0, 0.0
    # recall: gold 中 % 在 pred list 内出现 (子串双向)
    def hit_in_list(needle: str, hay: list[str]) -> bool:
        return any(needle in h or h in needle for h in hay)
    rec = sum(1 for g in g_low if hit_in_list(g, p_low)) / len(g_low)
    prec = sum(1 for p in p_low if hit_in_list(p, g_low)) / len(p_low)
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1


# --- 长度系 ---------------------------------------------------------------
def length_ratio(g: str, p: str) -> float:
    """1 - |len_p - len_g| / max(...) — 完全等长=1, 差很多=0."""
    lg, lp = len(g or ""), len(p or "")
    M = max(lg, lp)
    if M == 0:
        return 0.0
    return 1.0 - abs(lp - lg) / M


def brevity_penalty(g: str, p: str) -> float:
    """BLEU 风格: pred 太短 → 罚, 太长 → 不罚 (= 1)."""
    lg, lp = len(g or ""), len(p or "")
    if lp == 0:
        return 0.0
    if lp >= lg:
        return 1.0
    return math.exp(1 - lg / lp)


# --- 内容词 ---------------------------------------------------------------
RE_WORD = re.compile(r"[A-Za-z][A-Za-z0-9\-]+")


def content_words(s: str) -> set[str]:
    if not isinstance(s, str):
        return set()
    return {w.lower() for w in RE_WORD.findall(s) if w.lower() not in STOPWORDS and len(w) > 2}


def content_metrics(g: str, p: str) -> tuple[float, float]:
    G, P = content_words(g), content_words(p)
    if not G:
        return float("nan"), float("nan")
    inter = len(G & P)
    rec = inter / len(G)
    union = len(G | P)
    jac = inter / union if union else 0.0
    return jac, rec


# --- 主流程 ---------------------------------------------------------------
def evaluate_one(csv_path: Path, scorer: rouge_scorer.RougeScorer) -> pd.DataFrame:
    df = pd.read_csv(csv_path).drop_duplicates(subset=["id"], keep="last").reset_index(drop=True)
    rows = []
    for _, r in df.iterrows():
        gold = "" if pd.isna(r.get("gold_answer")) else str(r["gold_answer"])
        pred = "" if pd.isna(r.get("pred_answer")) else str(r["pred_answer"])
        subtype = r.get("subtype", "")

        # ROUGE
        if gold and pred:
            sc = scorer.score(gold, pred)
            rL_f, rL_p, rL_r = sc["rougeL"].fmeasure, sc["rougeL"].precision, sc["rougeL"].recall
            r1_f, r1_r = sc["rouge1"].fmeasure, sc["rouge1"].recall
        else:
            rL_f = rL_p = rL_r = r1_f = r1_r = 0.0

        # KH
        kws = extract_keywords(gold, subtype)
        kh_p, kh_r, kh_f1 = kh_pf1(pred, kws, subtype)

        # 数字
        nj, nr = numbers_match(numbers_in(gold), numbers_in(pred), tol_rel=0.10)

        # 长度
        lr = length_ratio(gold, pred)
        bp = brevity_penalty(gold, pred)

        # 内容词
        cj, cr = content_metrics(gold, pred)

        rows.append({
            "id": r.get("id"), "subtype": subtype,
            "rougeL_f": rL_f, "rougeL_p": rL_p, "rougeL_r": rL_r,
            "rouge1_f": r1_f, "rouge1_r": r1_r,
            "kh_recall": kh_r, "kh_p_tpl": kh_p, "kh_f1_tpl": kh_f1,
            "num_jac": nj, "num_rec": nr,
            "len_ratio": lr, "brevity_pen": bp,
            "content_jac": cj, "content_rec": cr,
        })
    return pd.DataFrame(rows)


def main() -> None:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)

    big = []
    for model_dir in sorted(TASK1_DIR.iterdir()):
        if not model_dir.is_dir():
            continue
        out_name = model_dir.name
        if out_name in EXCLUDE_MODELS:
            continue
        for setting in SETTINGS:
            csv_path = model_dir / setting / "predictions.csv"
            if not csv_path.exists():
                continue
            print(f"[{out_name} / {setting}] computing...")
            per_row = evaluate_one(csv_path, scorer)
            per_row["out_name"] = out_name
            per_row["setting"] = setting
            big.append(per_row)
    all_rows = pd.concat(big, ignore_index=True)
    all_rows.to_csv(OUT_DIR / "metric_explore_per_row.csv", index=False)

    # 聚合: 每 (model, setting) 取均值
    metric_cols = [c for c in all_rows.columns
                   if c not in ("id", "subtype", "out_name", "setting")]
    agg = (all_rows.groupby(["out_name", "setting"])[metric_cols]
           .mean().reset_index())
    agg.to_csv(OUT_DIR / "metric_explore_summary.csv", index=False)

    # ====== Scaling 检验 ======
    print("\n" + "=" * 100)
    print("SCALING CHECK — Spearman corr(metric, log10(size_B)) on setting=fewshot")
    print("(positive = bigger model -> higher metric, 也即满足 scaling law)")
    print("=" * 100)

    fs = agg[agg["setting"] == "fewshot"].copy()
    fs["size_B"] = fs["out_name"].map(MODEL_SIZE_B)
    fs = fs.dropna(subset=["size_B"]).sort_values("size_B", ascending=False)
    fs["log_size"] = fs["size_B"].apply(math.log10)

    corrs = []
    for c in metric_cols:
        x = fs[["log_size", c]].dropna()
        if len(x) < 4:
            continue
        rho = x["log_size"].corr(x[c], method="spearman")
        pearson = x["log_size"].corr(x[c], method="pearson")
        corrs.append({"metric": c, "spearman": rho, "pearson": pearson, "n": len(x)})
    cdf = pd.DataFrame(corrs).sort_values("spearman", ascending=False)
    print(cdf.to_string(index=False, float_format="%.3f"))
    cdf.to_csv(OUT_DIR / "metric_scaling_corr.csv", index=False)

    print("\n" + "=" * 100)
    print("FEWSHOT 排名 (按 size 降序), 各 metric 的实际数值")
    print("=" * 100)
    show = fs[["out_name", "size_B"] + metric_cols].copy()
    print(show.to_string(index=False, float_format="%.3f"))

    # 每个 metric 的实际排名 (per metric 看 top 是谁)
    print("\n" + "=" * 100)
    print("FEWSHOT 各 metric 的 TOP-3 / BOTTOM-2 模型")
    print("=" * 100)
    for c in metric_cols:
        s = fs[["out_name", c]].dropna().sort_values(c, ascending=False)
        if len(s) < 3:
            continue
        top = ", ".join(f"{r.out_name}({getattr(r, c):.3f})" for r in s.head(3).itertuples())
        bot = ", ".join(f"{r.out_name}({getattr(r, c):.3f})" for r in s.tail(2).itertuples())
        print(f"  {c:<14}  TOP: {top}    |    BOT: {bot}")

    print(f"\n保存:")
    print(f"  {OUT_DIR / 'metric_explore_per_row.csv'}")
    print(f"  {OUT_DIR / 'metric_explore_summary.csv'}")
    print(f"  {OUT_DIR / 'metric_scaling_corr.csv'}")


if __name__ == "__main__":
    main()
