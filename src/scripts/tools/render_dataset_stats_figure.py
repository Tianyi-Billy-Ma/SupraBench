"""Render the SupraBench dataset-statistics figure (Figure 2).

Layout: donut chart on the left (spans full height); on the right, top-10 hosts
bar chart stacked over a binding-affinity histogram.

Panel (a): Nested donut. Outer ring = 4 tasks. Inner ring = top-3 hosts plus
           an "Others" bucket per task.
Panel (b): Horizontal bar chart of the top-10 hosts in Binding-Affinity
           Prediction with an additional "Others" bar combining the long tail.
Panel (c): Histogram of the binding-affinity ($\\log K_a$) target distribution
           with a percentile summary inset.

Run:
  uv run python tools/render_dataset_stats_figure.py
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import numpy as np
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO     = Path(__file__).resolve().parents[3]
STATS    = REPO / "results/analysis/dataset_stats.json"
T1_JSONL = REPO / "results/task1/base/qwen3.5-9b.jsonl"
T7_PARQ  = REPO / "data/sid/eval.parquet"
OUT_DIR  = REPO / "paper-overleaf/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PDF  = OUT_DIR / "dataset_stats.pdf"
OUT_PNG  = OUT_DIR / "dataset_stats.png"

# ---------------------------------------------------------------------------
# Okabe-Ito palette (one color per task)
# ---------------------------------------------------------------------------
TASK_COLORS = {
    "T1": "#56B4E9",   # sky blue
    "T2": "#D55E00",   # vermillion
    "T7": "#CC79A7",   # reddish purple
    "T3": "#009E73",   # bluish green
}

TASK_NAMES = {
    "T1": "Binding-Affinity Prediction",
    "T2": "Top-Binder Selection",
    "T7": "Solvent Identification",
    "T3": "Sequestrant Discovery",
}

# Inner-ring shades per task: rank 1 darker, rank 2 base, rank 3 lighter, others gray
def shade(hex_color: str, lighten: float) -> str:
    """Lighten or darken a hex color. lighten=0 keeps base; +1 white; -1 black."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    if lighten >= 0:
        r = int(r + (255 - r) * lighten)
        g = int(g + (255 - g) * lighten)
        b = int(b + (255 - b) * lighten)
    else:
        f = 1.0 + lighten
        r = int(r * f); g = int(g * f); b = int(b * f)
    return f"#{r:02x}{g:02x}{b:02x}"

# ---------------------------------------------------------------------------
# Bold sans-serif globals
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "font.weight":        "bold",
    "axes.titleweight":   "bold",
    "axes.labelweight":   "bold",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.linewidth":     1.0,
})

# ---------------------------------------------------------------------------
# Host abbreviation map
# ---------------------------------------------------------------------------
HOST_ABBREV: dict[str, str] = {
    "Cucurbit[8]uril":          "CB[8]",
    "Cucurbit[7]uril":          "CB[7]",
    "Cucurbit[6]uril":          "CB[6]",
    "Cucurbit[5]uril":          "CB[5]",
    "β-Cyclodextrin":           r"$\beta$-CD",
    "α-Cyclodextrin":           r"$\alpha$-CD",
    "γ-Cyclodextrin":           r"$\gamma$-CD",
    "p-Sulfonatocalix[4]arene": r"$p$-SC4",
    "p-Sulfonatocalix[5]arene": r"$p$-SC5",
    "p-Sulfonatocalix[6]arene": r"$p$-SC6",
    "p-sulfonatocalix[8]arene": r"$p$-SC8",
    "syn-Amide Naphthotube":    "syn-NT",
    "anti-Amide Naphthotube":   "anti-NT",
    "Octa acid cavitand":       "OA",
    "Calix[4]arene":            "CAL[4]",
}

def abbrev_host(name: str) -> str:
    return HOST_ABBREV.get(name, name)

# ---------------------------------------------------------------------------
# Load + assemble data
# ---------------------------------------------------------------------------
with open(STATS) as fh:
    stats = json.load(fh)

def load_jsonl(p: Path) -> list[dict]:
    with open(p) as fh:
        return [json.loads(line) for line in fh if line.strip()]

t1_rows = load_jsonl(T1_JSONL)

# T1 host counts from dataset_stats.json
t1_hosts_full = dict(stats["bap"]["host_top20"])
T1_TOTAL = sum(t1_hosts_full.values()) + (2392 - sum(t1_hosts_full.values()))   # = 2392
t1_top3  = stats["bap"]["host_top20"][:3]
t1_others = 2392 - sum(c for _, c in t1_top3)

# T2 host counts from dataset_stats.json
t2_top3   = stats["tbs"]["host_top20"][:3]
t2_others = 2064 - sum(c for _, c in t2_top3)

# T7 host counts from source parquet
t7_df       = pq.read_table(T7_PARQ).to_pandas()
t7_counter  = Counter(t7_df["host"].tolist())
t7_top3     = t7_counter.most_common(3)
t7_others   = 1955 - sum(c for _, c in t7_top3)

# T3: tiny slice — treat as a single "All hosts" bucket (134 items)
t3_total = 134

# Outer ring (tasks)
OUTER_KEYS   = ["T1", "T2", "T7", "T3"]
OUTER_COUNTS = [2392, 2064, 1955, 134]
OUTER_COLORS = [TASK_COLORS[k] for k in OUTER_KEYS]
OUTER_TOTAL  = sum(OUTER_COUNTS)
OUTER_PCTS   = [100 * c / OUTER_TOTAL for c in OUTER_COUNTS]

# Inner ring (top-3 hosts + Others per task; T3 = single slice)
def shades_for(base: str) -> list[str]:
    """Returns [rank1, rank2, rank3, others] colors derived from base."""
    return [shade(base, -0.30), base, shade(base, 0.30), "#d0d0d0"]

INNER_COUNTS: list[int] = []
INNER_COLORS: list[str] = []
INNER_LABELS: list[str] = []
for key in OUTER_KEYS:
    base = TASK_COLORS[key]
    if key == "T1":
        for (h, c), col in zip(t1_top3, shades_for(base)[:3]):
            INNER_COUNTS.append(c); INNER_COLORS.append(col); INNER_LABELS.append(abbrev_host(h))
        INNER_COUNTS.append(t1_others); INNER_COLORS.append(shades_for(base)[3]); INNER_LABELS.append("Others")
    elif key == "T2":
        for (h, c), col in zip(t2_top3, shades_for(base)[:3]):
            INNER_COUNTS.append(c); INNER_COLORS.append(col); INNER_LABELS.append(abbrev_host(h))
        INNER_COUNTS.append(t2_others); INNER_COLORS.append(shades_for(base)[3]); INNER_LABELS.append("Others")
    elif key == "T7":
        for (h, c), col in zip(t7_top3, shades_for(base)[:3]):
            INNER_COUNTS.append(c); INNER_COLORS.append(col); INNER_LABELS.append(abbrev_host(h))
        INNER_COUNTS.append(t7_others); INNER_COLORS.append(shades_for(base)[3]); INNER_LABELS.append("Others")
    else:  # T3 — single slice (too narrow to subdivide)
        INNER_COUNTS.append(t3_total); INNER_COLORS.append(base); INNER_LABELS.append("All")

# Panel (b): top-10 hosts in T1 + Others
top10_t1      = stats["bap"]["host_top20"][:10]
top10_others  = 2392 - sum(c for _, c in top10_t1)
bar_labels    = [abbrev_host(h) for h, _ in top10_t1] + ["Others"]
bar_counts    = [c for _, c in top10_t1] + [top10_others]
# Flip for top-to-bottom display
bar_labels    = list(reversed(bar_labels))
bar_counts    = list(reversed(bar_counts))

# Panel (c): logKa values + percentiles
logka_vals = [r["reference"] for r in t1_rows
              if isinstance(r.get("reference"), (int, float))]
logka_arr  = np.array(logka_vals, dtype=float)
PCTS       = [5, 25, 50, 75, 95]
pct_vals   = {p: float(np.percentile(logka_arr, p)) for p in PCTS}

# ---------------------------------------------------------------------------
# Print summary
# ---------------------------------------------------------------------------
print("=== SupraBench dataset statistics ===")
print(f"Outer ring (tasks):  {list(zip(OUTER_KEYS, OUTER_COUNTS, [f'{p:.1f}%' for p in OUTER_PCTS]))}")
print(f"T1 top-3 + Others:   {t1_top3}, Others={t1_others}")
print(f"T2 top-3 + Others:   {t2_top3}, Others={t2_others}")
print(f"T7 top-3 + Others:   {t7_top3}, Others={t7_others}")
print(f"T3 single slice:     {t3_total}")
print(f"logKa n={len(logka_vals):,}  percentiles: {pct_vals}")
print(f"Bar chart (b): {list(zip(reversed(bar_labels), reversed(bar_counts)))}")

# ---------------------------------------------------------------------------
# Figure layout
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(14, 8))
gs = fig.add_gridspec(
    2, 2,
    width_ratios=[1.20, 1.00],
    height_ratios=[1.0, 1.0],
    hspace=0.55,
    wspace=0.22,
)
ax_donut  = fig.add_subplot(gs[:, 0])
ax_hosts  = fig.add_subplot(gs[0, 1])
ax_hist   = fig.add_subplot(gs[1, 1])

# ===========================================================================
# Panel (a) — Nested donut: outer=tasks, inner=top-3 hosts + Others
# ===========================================================================
OUTER_R = 0.85
OUTER_W = 0.26
INNER_W = 0.32

# Outer ring
ax_donut.pie(
    OUTER_COUNTS,
    radius=OUTER_R,
    colors=OUTER_COLORS,
    startangle=90,
    counterclock=False,
    wedgeprops=dict(width=OUTER_W, edgecolor="white", linewidth=1.8),
)

# Inner ring
ax_donut.pie(
    INNER_COUNTS,
    radius=OUTER_R - OUTER_W,
    colors=INNER_COLORS,
    startangle=90,
    counterclock=False,
    wedgeprops=dict(width=INNER_W, edgecolor="white", linewidth=1.0),
)

# Center text
ax_donut.text(
    0, 0,
    f"{OUTER_TOTAL:,}\nexamples",
    ha="center", va="center",
    fontsize=14, fontweight="bold", color="#1a1a1a", linespacing=1.4,
)

# Outer-ring percentage labels at slice midpoints
cumulative = 0.0
outer_mid_angles: list[float] = []
for c in OUTER_COUNTS:
    frac = c / OUTER_TOTAL
    mid  = 90 - (cumulative + frac / 2) * 360
    outer_mid_angles.append(mid)
    cumulative += frac

for angle, pct in zip(outer_mid_angles, OUTER_PCTS):
    r_label = OUTER_R - OUTER_W / 2
    if pct < 5:
        continue
    rad = np.deg2rad(angle)
    lx  = r_label * np.cos(rad)
    ly  = r_label * np.sin(rad)
    ax_donut.text(
        lx, ly, f"{pct:.1f}%",
        ha="center", va="center",
        fontsize=12, fontweight="bold", color="white",
        path_effects=[pe.withStroke(linewidth=1.5, foreground="black")],
    )

# T3 outer-ring arrow (2.0% is too narrow to label in slice)
det_angle  = outer_mid_angles[3]   # T3
det_rad    = np.deg2rad(det_angle)
r_mid_out  = OUTER_R - OUTER_W / 2
tip_x      = r_mid_out * np.cos(det_rad)
tip_y      = r_mid_out * np.sin(det_rad)
ax_donut.annotate(
    "2.0%",
    xy=(tip_x, tip_y),
    xytext=(tip_x * 1.65, tip_y * 1.65 - 0.05),
    fontsize=10.5, fontweight="bold", color="#1a1a1a",
    ha="center", va="center",
    arrowprops=dict(arrowstyle="-|>", color="#555555", lw=1.1,
                    connectionstyle="arc3,rad=0.15"),
)

ax_donut.set_aspect("equal")
ax_donut.axis("off")

# Legend below donut
task_handles = [
    mpatches.Patch(facecolor=TASK_COLORS[k], edgecolor="white",
                   label=f"{k} {TASK_NAMES[k]}")
    for k in OUTER_KEYS
]
rank_handles = [
    mpatches.Patch(facecolor="#444444", edgecolor="white", label="Rank 1 host"),
    mpatches.Patch(facecolor="#888888", edgecolor="white", label="Rank 2 host"),
    mpatches.Patch(facecolor="#bbbbbb", edgecolor="white", label="Rank 3 host"),
    mpatches.Patch(facecolor="#d0d0d0", edgecolor="white", label="Other hosts"),
]
leg = ax_donut.legend(
    handles=task_handles + rank_handles,
    loc="lower center",
    bbox_to_anchor=(0.50, -0.20),
    ncol=2,
    fontsize=10.5,
    frameon=True,
    framealpha=0.92,
    edgecolor="#bbbbbb",
    fancybox=False,
    columnspacing=1.4,
    handlelength=1.2,
    handletextpad=0.5,
)
for txt in leg.get_texts():
    txt.set_fontweight("bold")

ax_donut.set_title(
    r"$\mathbf{(a)\ Task\ composition}$",
    fontsize=20, pad=14, loc="center",
)

# ===========================================================================
# Panel (b) — Top-10 hosts + Others in Binding-Affinity Prediction
# ===========================================================================
ax = ax_hosts
ys = np.arange(len(bar_labels))
# "Others" bar gets neutral gray; named hosts use blue
others_index_in_flipped = 0   # we reversed: "Others" is now at index 0 (bottom)
bar_colors = ["#888888" if lbl == "Others" else TASK_COLORS["T1"]
              for lbl in bar_labels]
ax.barh(ys, bar_counts, color=bar_colors,
        edgecolor="black", linewidth=0.7, height=0.7)
ax.set_yticks(ys)
ax.set_yticklabels(bar_labels, fontsize=12.5, fontweight="bold")
ax.set_xlabel("# examples", fontsize=15, fontweight="bold")
for lbl in ax.get_xticklabels():
    lbl.set_fontsize(12); lbl.set_fontweight("bold")
ax.set_xlim(0, max(bar_counts) * 1.18)
ax.grid(axis="x", color="white", linestyle="-", linewidth=0.8, alpha=0.7)
ax.set_facecolor("#f5f5f5")
ax.set_axisbelow(True)
ax.spines["left"].set_visible(False)
ax.tick_params(axis="y", length=0)
for y, c in zip(ys, bar_counts):
    ax.text(c + max(bar_counts) * 0.012, y, f"{c:,}",
            ha="left", va="center", fontsize=11.5, fontweight="bold")
ax.set_title(
    r"$\mathbf{(b)\ Host\ distribution\ in\ Binding\text{-}Affinity\ Prediction}$",
    fontsize=16, pad=8,
)

# ===========================================================================
# Panel (c) — Binding Affinity Distribution histogram
# ===========================================================================
ax = ax_hist
bins = np.linspace(0, 12, 25)
ax.hist(logka_arr, bins=bins, color=TASK_COLORS["T1"],
        edgecolor="black", linewidth=0.55, alpha=0.92)
ax.set_xlabel(r"$\log K_a$", fontsize=16, fontweight="bold")
ax.set_ylabel("# examples", fontsize=15, fontweight="bold")
for lbl in ax.get_xticklabels() + ax.get_yticklabels():
    lbl.set_fontsize(12.5); lbl.set_fontweight("bold")
ax.set_xlim(0, 12)
ax.grid(axis="y", color="white", linestyle="-", linewidth=0.8, alpha=0.7)
ax.set_facecolor("#f5f5f5")
ax.set_axisbelow(True)
# Percentile box (top-right)
pct_text = "\n".join(
    f"$p_{{{p}}}$ $=$ {pct_vals[p]:.2f}" for p in PCTS
)
ax.text(
    0.975, 0.96,
    pct_text,
    transform=ax.transAxes, ha="right", va="top",
    fontsize=11, fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
              edgecolor="#aaaaaa", linewidth=0.9),
)
ax.set_title(
    r"$\mathbf{(c)\ Binding\ Affinity\ Distribution}$",
    fontsize=18, pad=8,
)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
fig.savefig(OUT_PDF, bbox_inches="tight")
fig.savefig(OUT_PNG, dpi=180, bbox_inches="tight")
print(f"\nSaved: {OUT_PDF}")
print(f"Saved: {OUT_PNG}")
print(f"PDF size: {OUT_PDF.stat().st_size / 1024:.1f} KB")
