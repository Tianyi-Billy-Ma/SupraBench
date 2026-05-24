"""Per-host error distribution on Task 1 (Drug Delivery), top 10 hosts by
sample count, 4 models x {Base, Few-Shot, CoT}.

Each cell is a box plot of |pred - ref| with two overlays:
  - black line in box  = median
  - white diamond      = mean (= MAE for that cell, matches main_results.tex)
"""
from __future__ import annotations
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, "/Users/billyma/Workspace/SupraBench")
from src.eval.task1 import parse_logka

REPO = Path("/Users/billyma/Workspace/SupraBench")
OUT_PDF = REPO / "paper-overleaf/figures/task1.pdf"
OUT_PNG = REPO / "paper-overleaf/figures/task1.png"

MODEL_ORDER = [
    ("qwen3.5-27b",                      "Qwen3.5-27B"),
    ("gpt-5.4-mini_nothinking",          "GPT-5.4-Mini"),
    ("gemini-3-flash-preview_nothinking","Gemini-3-Flash"),
    ("deepseek-v4-pro",                  "DeepSeek-v4"),
]
METHODS = [("base", "Base"), ("fewshot", "Few-Shot"), ("cot", "CoT")]
HOST_ABBREV = {
    "Cucurbit[8]uril":              "CB[8]",
    "β-Cyclodextrin":               "β-CD",
    "p-Sulfonatocalix[4]arene":     "p-SC4",
    "Cucurbit[6]uril":              "CB[6]",
    "α-Cyclodextrin":               "α-CD",
    "p-Sulfonatocalix[6]arene":     "p-SC6",
    "syn-Amide Naphthotube":        "syn-NT",
    "Octa acid cavitand":           "OA",
    "anti-Amide Naphthotube":       "anti-NT",
    "Calix[4]arene":                "CAL[4]",
}
YBOX_MAX = 12.0   # clip per-example errors above this for plot readability

def load_t1(model_slug, method):
    path = REPO / f"results/task1/{method}/{model_slug}.jsonl"
    return [json.loads(l) for l in path.open()] if path.exists() else []

# Top-10 hosts (Qwen-9B base used as the reference index — same id schema across models)
ref_rows = load_t1("qwen3.5-9b", "base")
host_counts = defaultdict(int)
for r in ref_rows:
    host_counts[r["host_name"]] += 1
TOP_HOSTS = [h for h, _ in sorted(host_counts.items(), key=lambda x: -x[1])[:8]]
HOST_LABELS = [HOST_ABBREV[h] for h in TOP_HOSTS]
print("Top 8 hosts:")
for h in TOP_HOSTS:
    print(f"  {h:40s} n={host_counts[h]:4d}  abbrev={HOST_ABBREV[h]}")

# Collect per-example absolute errors
errs = defaultdict(list)
for model_slug, _ in MODEL_ORDER:
    for method, _ in METHODS:
        for r in load_t1(model_slug, method):
            if r["host_name"] not in TOP_HOSTS:
                continue
            p = parse_logka(r.get("response") or "")
            if p is None or math.isnan(p):
                continue
            errs[(model_slug, method, r["host_name"])].append(abs(p - r["reference"]))


def mean(values):
    return sum(values) / len(values) if values else float("nan")


# Okabe-Ito palette (Wong, Nature Methods 2011) — colorblind-safe
colors = {"base": "#56B4E9", "fewshot": "#D55E00", "cot": "#009E73"}

fig, axes = plt.subplots(2, 2, figsize=(16, 8.5), sharey=True)
axes = axes.flatten()

# Position pattern: 3 boxes per host, 0.27 spacing between methods
positions_per_host = []
for h_idx in range(len(TOP_HOSTS)):
    for m_idx in range(len(METHODS)):
        positions_per_host.append(h_idx + (m_idx - 1) * 0.27)

for ax_idx, (ax, (model_slug, model_label)) in enumerate(zip(axes, MODEL_ORDER)):
    box_data, box_colors, box_means = [], [], []
    for h in TOP_HOSTS:
        for method, _ in METHODS:
            data = errs.get((model_slug, method, h), [])
            mu = mean(data)
            # Pass raw errors to boxplot so IQR/whiskers reflect the true data;
            # the y-axis is clipped purely for display via set_ylim below.
            box_data.append(data)
            box_colors.append(colors[method])
            box_means.append(mu)

    bp = ax.boxplot(box_data, positions=positions_per_host, widths=0.24,
                    patch_artist=True, showfliers=False,
                    medianprops=dict(color="black", linewidth=0.9),
                    whiskerprops=dict(color="black", linewidth=0.6),
                    capprops=dict(color="black", linewidth=0.6))
    for patch, c in zip(bp["boxes"], box_colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.75)
        patch.set_edgecolor("black")
        patch.set_linewidth(0.5)

    # Mean diamond (= per-cell MAE, matches main_results.tex)
    for x, mu in zip(positions_per_host, box_means):
        if mu is None or math.isnan(mu):
            continue
        ax.plot([x], [mu], marker="D", markerfacecolor="white",
                markeredgecolor="black", markersize=4.5, zorder=6)

    ax.set_title(model_label, fontsize=22, fontweight="bold", pad=8)
    ax.set_xticks(range(len(TOP_HOSTS)))
    if ax_idx < 2:  # top row: hide host labels (shared with bottom row)
        ax.set_xticklabels([])
    else:
        ax.set_xticklabels(HOST_LABELS, rotation=35, ha="right",
                           fontsize=18, fontweight="bold")
    ax.set_ylim(-0.3, YBOX_MAX * 1.02)
    if ax_idx % 2 == 0:  # left column only
        ax.set_ylabel(r"$\mathbf{Absolute\ Error\ \downarrow}$", fontsize=20)
    for lbl in ax.get_yticklabels():
        lbl.set_fontsize(17)
        lbl.set_fontweight("bold")
    ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.5)
    ax.set_axisbelow(True)

    method_handles = [mpatches.Patch(color=colors[m], alpha=0.75, label=lbl)
                      for m, lbl in METHODS]
    mean_handle = plt.Line2D([0], [0], marker="D", color="none",
                             markerfacecolor="white", markeredgecolor="black",
                             markersize=9, linestyle="None",
                             label="Mean (MAE)")
    leg = ax.legend(handles=method_handles + [mean_handle], loc="upper right",
                    ncol=2, frameon=True, framealpha=0.9, fontsize=13,
                    edgecolor="black", fancybox=False)
    for txt in leg.get_texts():
        txt.set_fontweight("bold")

plt.tight_layout()
OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT_PDF, bbox_inches="tight")
plt.savefig(OUT_PNG, dpi=180, bbox_inches="tight")
print(f"\nSaved: {OUT_PDF}\nSaved: {OUT_PNG}")
