"""
Task 3 figures (源 task1, 已重命名):
  - fig_task3_rougeL.{png,pdf}: 12 model × 3 setting grouped bar (ROUGE-L F1)
  - fig_task3_kh.{png,pdf}    : 2 子面板 (forward / reverse) × 12 model × 3 setting grouped bar (KH)

按 setting=fewshot 的指标值降序排模型 (展示 best-case 排名).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
LB = ROOT / "outputs" / "task1_eval" / "leaderboard.csv"
FIG_DIR = ROOT / "outputs" / "task1_eval" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SETTINGS = ["base", "fewshot", "cot"]
COLORS = {"base": "#4C72B0", "fewshot": "#DD8452", "cot": "#55A868"}

EXCLUDE_MODELS = {
    "gpt-5.5_nothinking",
    "gpt-5.5_xhigh",
    "gemini-3-flash-preview_high",
}


def load_pivot(metric: str) -> pd.DataFrame:
    lb = pd.read_csv(LB)
    lb = lb[~lb["out_name"].isin(EXCLUDE_MODELS)]
    piv = lb.pivot(index="out_name", columns="setting", values=metric)[SETTINGS]
    return piv


def grouped_bar(ax, piv: pd.DataFrame, ylabel: str, title: str,
                ymax: float | None = None) -> None:
    models = piv.index.tolist()
    n_m = len(models)
    n_s = len(SETTINGS)
    x = np.arange(n_m)
    width = 0.26
    offsets = (np.arange(n_s) - (n_s - 1) / 2) * width

    for i, s in enumerate(SETTINGS):
        vals = piv[s].values
        bars = ax.bar(x + offsets[i], vals, width,
                      label=s, color=COLORS[s], edgecolor="white", linewidth=0.4)
        # 数值标签
        for b, v in zip(bars, vals):
            if not np.isnan(v) and v >= 0.02:
                ax.text(b.get_x() + b.get_width() / 2, v + 0.005,
                        f"{v:.2f}", ha="center", va="bottom",
                        fontsize=6.2, color="#333")

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    if ymax is not None:
        ax.set_ylim(0, ymax)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(title="setting", frameon=False, loc="upper right", fontsize=9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def fig_rouge() -> None:
    piv = load_pivot("rougeL_all")
    # 按 fewshot 降序排
    piv = piv.sort_values("fewshot", ascending=False)
    fig, ax = plt.subplots(figsize=(11, 5.2))
    grouped_bar(ax, piv,
                ylabel="ROUGE-L F1",
                title="Task 3: ROUGE-L F1 across models × prompting settings",
                ymax=0.75)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"fig_task3_rougeL.{ext}", dpi=300,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"saved fig_task3_rougeL.png/pdf  (12 models × 3 settings)")


def fig_kh() -> None:
    piv = load_pivot("kh_all")
    piv = piv.sort_values("fewshot", ascending=False)
    fig, ax = plt.subplots(figsize=(11, 5.2))
    grouped_bar(ax, piv,
                ylabel="Keyword Hit (recall, avg over fwd+rev)",
                title="Task 3: Keyword Hit across models × prompting settings",
                ymax=piv.max().max() * 1.18)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"fig_task3_kh.{ext}", dpi=300,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"saved fig_task3_kh.png/pdf   (single panel, kh_all)")


def fig_rouge1r() -> None:
    """ROUGE-1 recall (avg over fwd+rev): scaling-friendly (Spearman +0.75)."""
    piv = load_pivot("rouge1r_all")
    piv = piv.sort_values("fewshot", ascending=False)
    fig, ax = plt.subplots(figsize=(11, 5.2))
    grouped_bar(ax, piv,
                ylabel="ROUGE-1 recall",
                title="Task 3: ROUGE-1 recall across models × prompting settings "
                      "(scaling-friendly metric)",
                ymax=piv.max().max() * 1.18)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"fig_task3_rouge1r.{ext}", dpi=300,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"saved fig_task3_rouge1r.png/pdf  ({len(piv)} models × 3 settings)")


def fig_rouge1r_rev() -> None:
    """ROUGE-1 recall on reverse subtype only: best scaling signal (Spearman +0.89).
    Forward 让小模型抄 fewshot 模板作弊; reverse 含 logKa+host_name 结构, 真知识胜出."""
    piv = load_pivot("rouge1r_rev")
    piv = piv.sort_values("fewshot", ascending=False)
    fig, ax = plt.subplots(figsize=(11, 5.2))
    grouped_bar(ax, piv,
                ylabel="ROUGE-1 recall (reverse subtype)",
                title="Task 3: ROUGE-1 recall on REVERSE subtype "
                      "(strongest scaling signal)",
                ymax=piv.max().max() * 1.18)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"fig_task3_rouge1r_rev.{ext}", dpi=300,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"saved fig_task3_rouge1r_rev.png/pdf  ({len(piv)} models × 3 settings)")


def main() -> None:
    fig_rouge()
    fig_kh()
    fig_rouge1r()
    fig_rouge1r_rev()
    print(f"\n输出目录: {FIG_DIR}")


if __name__ == "__main__":
    main()
