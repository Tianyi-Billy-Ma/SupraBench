"""
Task 7 figure (源 task3, 已重命名): 9 model × 3 setting grouped bar (macro-F1).
排除 gpt-5.5_nothinking / gpt-5.5_xhigh / gemini-3-flash-preview_high
(同 05_plot_task1.py 的过滤集).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
LB = ROOT / "outputs" / "task3_eval" / "leaderboard.csv"
FIG_DIR = ROOT / "outputs" / "task3_eval" / "figures"
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
        for b, v in zip(bars, vals):
            if not np.isnan(v) and v >= 0.005:
                ax.text(b.get_x() + b.get_width() / 2, v + 0.005,
                        f"{v:.2f}", ha="center", va="bottom",
                        fontsize=6.5, color="#333")

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


def fig_macro_f1() -> None:
    piv = load_pivot("macro_f1")
    piv = piv.sort_values("fewshot", ascending=False)
    ymax = piv.max().max() * 1.18
    fig, ax = plt.subplots(figsize=(11, 5.2))
    grouped_bar(ax, piv,
                ylabel="macro-F1",
                title="Task 7: macro-F1 across models × prompting settings",
                ymax=ymax)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"fig_task7_macroF1.{ext}", dpi=300,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"saved fig_task7_macroF1.png/pdf  ({len(piv)} models × 3 settings)")


def main() -> None:
    fig_macro_f1()
    print(f"\n输出目录: {FIG_DIR}")


if __name__ == "__main__":
    main()
