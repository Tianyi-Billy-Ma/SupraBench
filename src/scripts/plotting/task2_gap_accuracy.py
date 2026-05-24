"""Difficulty-calibration curve for Task 2 (Top-binder selection).

Layout: 1 row × 4 panels (one per model), sharey.
X-axis: difficulty bins by gap = logKa(gold) - max(logKa[non-gold]).
Y-axis: top-binder accuracy (higher is better).
Lines: Base / Few-Shot / CoT with Wilson 95% CI error bars.
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

REPO = Path("/Users/billyma/Workspace/SupraBench")
OUT_PDF = REPO / "paper-overleaf/figures/task2_gap_accuracy.pdf"
OUT_PNG = REPO / "paper-overleaf/figures/task2_gap_accuracy.png"

MODEL_ORDER = [
    ("qwen3.5-27b",                       "Qwen3.5-27B"),
    ("gpt-5.4-mini_nothinking",           "GPT-5.4-Mini"),
    ("gemini-3-flash-preview_nothinking", "Gemini-3-Flash"),
    ("deepseek-v4-pro",                   "DeepSeek-v4"),
]
METHODS = [("base", "Base"), ("fewshot", "Few-Shot"), ("cot", "CoT")]

# Okabe-Ito palette keyed by method slug
COLORS = {"base": "#56B4E9", "fewshot": "#D55E00", "cot": "#009E73"}
MARKERS = {"base": "o", "fewshot": "s", "cot": "^"}

# Gap bins: [lo, hi) — last bin is [3, +inf)
BIN_EDGES = [0.0, 0.5, 1.0, 2.0, 3.0, math.inf]
BIN_MIDPOINTS = [0.22, 0.74, 1.42, 2.45, 4.16]
BIN_LABELS = ["[0,0.5)", "[0.5,1)", "[1,2)", "[2,3)", "[3,∞)"]
N_BINS = len(BIN_MIDPOINTS)


def letter_to_index(letter: str) -> int:
    return ord(letter.upper()) - ord("A")


def compute_gap(record: dict) -> float | None:
    opts = record.get("options_logka")
    ref = record.get("reference", {})
    letter = ref.get("letter")
    if not opts or not letter:
        return None
    try:
        gold_idx = letter_to_index(letter)
        gold_val = opts[gold_idx]
        non_gold = [v for i, v in enumerate(opts) if i != gold_idx]
        if not non_gold:
            return None
        return gold_val - max(non_gold)
    except (IndexError, TypeError):
        return None


def is_correct(record: dict) -> bool:
    ref = record.get("reference", {})
    pred = record.get("prediction", {})
    ref_letter = ref.get("letter")
    pred_letter = pred.get("letter")
    if pred_letter is None:
        return False
    return pred_letter.upper() == ref_letter.upper()


def bin_index(gap: float) -> int:
    for i in range(len(BIN_EDGES) - 1):
        if BIN_EDGES[i] <= gap < BIN_EDGES[i + 1]:
            return i
    return len(BIN_MIDPOINTS) - 1


def wilson_ci(n_correct: int, n_total: int, z: float = 1.96):
    """Wilson score interval for a Bernoulli proportion."""
    if n_total == 0:
        return float("nan"), float("nan"), float("nan")
    p = n_correct / n_total
    denom = 1 + z ** 2 / n_total
    centre = (p + z ** 2 / (2 * n_total)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n_total + z ** 2 / (4 * n_total ** 2))
    return p, max(0.0, centre - half), min(1.0, centre + half)


def load_records(model_slug: str, method: str) -> list[dict]:
    path = REPO / f"results/task2/{method}/{model_slug}.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.open() if line.strip()]


# ── Aggregate stats ──────────────────────────────────────────────────────────
# stats[(model_slug, method)][bin_idx] = (n_correct, n_total)
stats: dict[tuple, list[tuple[int, int]]] = {}

for model_slug, _ in MODEL_ORDER:
    for method, _ in METHODS:
        counts = [(0, 0)] * N_BINS
        bin_data: list[list[bool]] = [[] for _ in range(N_BINS)]
        for rec in load_records(model_slug, method):
            gap = compute_gap(rec)
            if gap is None:
                continue
            b = bin_index(gap)
            bin_data[b].append(is_correct(rec))
        counts = [(sum(bools), len(bools)) for bools in bin_data]
        stats[(model_slug, method)] = counts

# ── Stdout audit ─────────────────────────────────────────────────────────────
print("=" * 72)
print(f"{'Model':<35} {'Method':<10} " + "  ".join(f"{bl:>10}" for bl in BIN_LABELS))
print("=" * 72)
for model_slug, model_label in MODEL_ORDER:
    for method, method_label in METHODS:
        row_counts = stats[(model_slug, method)]
        cells = []
        for nc, nt in row_counts:
            if nt == 0:
                cells.append("     --   ")
            else:
                p, lo, hi = wilson_ci(nc, nt)
                cells.append(f"{p*100:5.1f}% n={nt}")
        print(f"{model_label:<35} {method_label:<10} " + "  ".join(cells))
print("=" * 72)

# ── Sanity checks ────────────────────────────────────────────────────────────
def check(model_slug, method, bin_idx, expected_pct, expected_n, label):
    nc, nt = stats[(model_slug, method)][bin_idx]
    actual_pct = nc / nt * 100 if nt else float("nan")
    ok = abs(actual_pct - expected_pct) < 0.5 and nt == expected_n
    flag = "OK" if ok else "MISMATCH"
    print(f"[{flag}] {label}: {actual_pct:.1f}% (n={nt})  expected {expected_pct}% (n={expected_n})")

check("qwen3.5-27b", "base", 0, 34.7, 805, "Qwen3.5-27B / Base / gap<0.5")
check("qwen3.5-27b", "base", 4, 60.0, 125, "Qwen3.5-27B / Base / gap>=3")
check("gemini-3-flash-preview_nothinking", "fewshot", 0, 44.2, 805,
      "Gemini-3-Flash / Few-Shot / gap<0.5")

# ── Plot ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(18, 5), sharey=True)

for ax_idx, (ax, (model_slug, model_label)) in enumerate(zip(axes, MODEL_ORDER)):

    for method, method_label in METHODS:
        row_counts = stats[(model_slug, method)]
        xs, ys, yerr_lo, yerr_hi = [], [], [], []
        for b, (nc, nt) in enumerate(row_counts):
            if nt == 0:
                continue
            p, lo, hi = wilson_ci(nc, nt)
            xs.append(BIN_MIDPOINTS[b])
            ys.append(p)
            yerr_lo.append(p - lo)
            yerr_hi.append(hi - p)

        ax.errorbar(
            xs, ys,
            yerr=[yerr_lo, yerr_hi],
            color=COLORS[method],
            marker=MARKERS[method],
            linewidth=1.4,
            markersize=6,
            alpha=0.95,
            capsize=3,
            label=method_label,
        )

    # Random baseline
    ax.axhline(0.25, color="grey", linestyle="--", linewidth=1.0, label="Random (1/4)")
    ax.text(
        3.6, 0.255, "Random (1/4)",
        fontsize=10, fontweight="bold", color="grey",
        ha="right", va="bottom",
    )

    # Panel title
    ax.set_title(model_label, fontsize=22, fontweight="bold", pad=8)

    # X-axis
    ax.set_xlabel(
        r"Gold $-$ runner-up $\log K_a$ (gap)",
        fontsize=20, fontweight="bold",
    )
    ax.set_xticks(BIN_MIDPOINTS)
    ax.set_xticklabels(
        [f"{m:.2f}" for m in BIN_MIDPOINTS],
        fontsize=17, fontweight="bold",
    )

    # Y-axis
    ax.set_ylim(0.20, 0.75)
    for lbl in ax.get_yticklabels():
        lbl.set_fontsize(17)
        lbl.set_fontweight("bold")

    if ax_idx == 0:
        ax.set_ylabel(r"Top-binder accuracy $\uparrow$", fontsize=20, fontweight="bold")

    ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.5)
    ax.set_axisbelow(True)

    # Legend — only on the first panel
    if ax_idx == 0:
        method_handles = [
            mlines.Line2D([], [],
                          color=COLORS[m], marker=MARKERS[m],
                          linewidth=1.4, markersize=6, alpha=0.95,
                          label=lbl)
            for m, lbl in METHODS
        ]
        random_handle = mlines.Line2D([], [],
                                      color="grey", linestyle="--",
                                      linewidth=1.0, label="Random (1/4)")
        leg = ax.legend(
            handles=method_handles + [random_handle],
            loc="upper left",
            ncol=2,
            frameon=True,
            framealpha=0.9,
            fontsize=13,
            edgecolor="black",
            fancybox=False,
        )
        for txt in leg.get_texts():
            txt.set_fontweight("bold")

    # CoT-collapse annotation — only on Qwen panel
    if ax_idx == 0:
        # Find CoT, gap>=3 point
        cot_counts = stats[(model_slug, "cot")]
        nc, nt = cot_counts[4]  # bin index 4 = [3, +inf)
        if nt:
            p_cot, _, _ = wilson_ci(nc, nt)
            ax.annotate(
                "CoT collapse",
                xy=(BIN_MIDPOINTS[4], p_cot),
                xytext=(BIN_MIDPOINTS[4] - 0.8, p_cot - 0.09),
                fontsize=10,
                fontweight="bold",
                color=COLORS["cot"],
                arrowprops=dict(
                    arrowstyle="->",
                    color=COLORS["cot"],
                    lw=1.2,
                ),
            )

plt.tight_layout()
OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT_PDF, bbox_inches="tight")
plt.savefig(OUT_PNG, dpi=180, bbox_inches="tight")
print(f"\nSaved: {OUT_PDF}")
print(f"Saved: {OUT_PNG}")
