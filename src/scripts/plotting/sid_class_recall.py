"""SID – per-class recall heatmap.

13 rows × 6 columns:
  Rows  0–11 : 4 models × 3 methods (base / fewshot / cot)
  Row   12   : Always-water Baseline (separated by extra whitespace)
  Cols       : water | DMSO | MeCN | MeOH | CHCl3 | CH2Cl2

Style matches per_host_mae_bap.py:
  bold sans-serif; title 22pt; tick labels 17pt; axis labels 20pt.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import numpy as np

matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.weight"] = "bold"

REPO = Path(__file__).resolve().parents[3]
OUT_PDF = REPO / "paper-overleaf/figures/sid_class_recall.pdf"
OUT_PNG = REPO / "paper-overleaf/figures/sid_class_recall.png"

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
LABELS_ORDER = ["water", "DMSO", "MeCN", "MeOH", "CHCl3", "CH2Cl2"]
COL_HEADERS = ["water", "DMSO", "MeCN", "MeOH", r"CHCl$_3$", r"CH$_2$Cl$_2$"]
CLASS_SUPPORT = [1819, 2, 10, 94, 27, 3]  # n= row annotation

MODEL_ORDER = [
    ("qwen3.5-27b",                       "Qwen3.5-27B"),
    ("gpt-5.4-mini_nothinking",           "GPT-5.4-Mini"),
    ("gemini-3-flash-preview_nothinking", "Gemini-3-Flash"),
    ("deepseek-v4-pro",                   "DeepSeek-v4"),
]
METHODS = [("base", "base"), ("fewshot", "fewshot"), ("cot", "cot")]

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_rows(model_slug: str, method: str) -> list[dict]:
    path = REPO / f"results/sid/{method}/{model_slug}.jsonl"
    if not path.exists():
        print(f"  WARN: missing {path}", file=sys.stderr)
        return []
    return [json.loads(line) for line in path.open()]


def compute_recall(rows: list[dict]) -> dict[str, float]:
    """Per-class recall: correct / total_ref for each class."""
    correct: dict[str, int] = {c: 0 for c in LABELS_ORDER}
    total:   dict[str, int] = {c: 0 for c in LABELS_ORDER}
    for row in rows:
        ref = row.get("reference", {})
        pred = row.get("prediction", {})
        ref_label = ref.get("label") if isinstance(ref, dict) else None
        pred_label = pred.get("label") if isinstance(pred, dict) else None
        if ref_label not in LABELS_ORDER:
            continue
        total[ref_label] += 1
        if pred_label == ref_label:
            correct[ref_label] += 1
    recall = {}
    for c in LABELS_ORDER:
        recall[c] = correct[c] / total[c] if total[c] > 0 else float("nan")
    return recall


# Build 12-row recall matrix
data_rows: list[dict[str, float]] = []
row_model_labels: list[str] = []   # model display name per row
row_method_labels: list[str] = []  # method per row

for model_slug, model_display in MODEL_ORDER:
    for method_key, method_display in METHODS:
        rows = load_rows(model_slug, method_key)
        recall = compute_recall(rows)
        data_rows.append(recall)
        row_model_labels.append(model_display)
        row_method_labels.append(method_display)

# Baseline row: always predict water
baseline_recall = {c: (1.0 if c == "water" else 0.0) for c in LABELS_ORDER}

# ---------------------------------------------------------------------------
# Printout of full 13×6 grid
# ---------------------------------------------------------------------------
print("=" * 72)
print(f"{'Row':<30s}  " + "  ".join(f"{c:>8s}" for c in LABELS_ORDER))
print("-" * 72)
for i, (recall, model_lbl, method_lbl) in enumerate(
    zip(data_rows, row_model_labels, row_method_labels)
):
    label = f"{model_lbl}/{method_lbl}"
    vals = "  ".join(
        f"{recall[c]:8.3f}" if not np.isnan(recall[c]) else "     nan"
        for c in LABELS_ORDER
    )
    print(f"{label:<30s}  {vals}")

print("-" * 72)
label = "Always-water Baseline"
vals = "  ".join(f"{baseline_recall[c]:8.3f}" for c in LABELS_ORDER)
print(f"{label:<30s}  {vals}")
print("=" * 72)

# Spot-check assertions
def _get(model_slug, method_key):
    for i, (rl, ml, meth) in enumerate(zip(data_rows, row_model_labels, row_method_labels)):
        mdisplay = dict(MODEL_ORDER)[model_slug]
        if ml == mdisplay and meth == method_key:
            return data_rows[i]
    return None

checks = [
    ("qwen3.5-27b",                       "cot",  "MeCN",  0.800),
    ("gemini-3-flash-preview_nothinking", "base", "MeOH",  0.894),
    ("deepseek-v4-pro",                   "cot",  "CHCl3", 0.889),
]
print("\nSpot checks:")
all_ok = True
for model_slug, method_key, col, expected in checks:
    rec = _get(model_slug, method_key)
    if rec is None:
        print(f"  MISSING: {model_slug}/{method_key}")
        all_ok = False
        continue
    got = rec[col]
    ok = abs(got - expected) < 0.015
    flag = "OK" if ok else "FAIL"
    print(f"  [{flag}] {model_slug}/{method_key}/{col}: got={got:.3f}, expected={expected:.3f}")
    if not ok:
        all_ok = False

# DMSO check: all 12 rows should be 0.0
print("  DMSO column (expect all 0.000):")
dmso_ok = True
for i, (recall, ml, meth) in enumerate(zip(data_rows, row_model_labels, row_method_labels)):
    v = recall["DMSO"]
    flag = "OK" if v == 0.0 or np.isnan(v) else "FAIL"
    if flag == "FAIL":
        print(f"    [{flag}] {ml}/{meth}: DMSO={v:.3f}")
        dmso_ok = False
if dmso_ok:
    print("    All DMSO = 0.000 ✓")

# ---------------------------------------------------------------------------
# Build numpy matrix for imshow  (13 rows × 6 cols)
# ---------------------------------------------------------------------------
matrix = np.zeros((13, 6), dtype=float)
for i, recall in enumerate(data_rows):
    for j, c in enumerate(LABELS_ORDER):
        v = recall[c]
        matrix[i, j] = 0.0 if np.isnan(v) else v

# Row 12: baseline
for j, c in enumerate(LABELS_ORDER):
    matrix[12, j] = baseline_recall[c]

# ---------------------------------------------------------------------------
# Figure layout
# ---------------------------------------------------------------------------
# GridSpec: 12 data rows + 1 gap spacer (0.5×) + 1 baseline row
fig = plt.figure(figsize=(11, 9))

row_heights = [1.0] * 12 + [0.5, 1.0]
gs = fig.add_gridspec(
    14, 1,
    height_ratios=row_heights,
    hspace=0.0,
    top=0.86, bottom=0.12, left=0.30, right=0.95,
)

ax_data = fig.add_subplot(gs[:12, 0])
ax_gap  = fig.add_subplot(gs[12,  0])
ax_base = fig.add_subplot(gs[13,  0])
ax_gap.set_visible(False)

cmap = plt.get_cmap("cividis")
norm = mcolors.Normalize(vmin=0.0, vmax=1.0)


def _draw_cells(ax, sub_matrix):
    """imshow + annotate cells. Returns the AxesImage."""
    nr, nc = sub_matrix.shape
    im = ax.imshow(sub_matrix, aspect="auto", cmap=cmap, norm=norm,
                   interpolation="nearest")
    ax.set_xlim(-0.5, nc - 0.5)
    ax.set_ylim(nr - 0.5, -0.5)
    for r in range(nr):
        for c in range(nc):
            val = sub_matrix[r, c]
            bold   = val >= 0.50
            white  = val >= 0.50
            ax.text(c, r, f"{val:.2f}",
                    ha="center", va="center",
                    fontsize=9.5,
                    color="white" if white else "black",
                    fontweight="bold" if bold else "normal",
                    fontfamily="sans-serif")
    return im


_draw_cells(ax_data, matrix[:12, :])
_draw_cells(ax_base, matrix[12:13, :])

# ---------------------------------------------------------------------------
# Spines
# ---------------------------------------------------------------------------
for ax in (ax_data, ax_base):
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)

# top of data panel: thin border; bottom: thick separator line
ax_data.spines["top"].set_linewidth(0.8)
ax_data.spines["bottom"].set_linewidth(1.5)
ax_data.spines["bottom"].set_color("black")

# baseline panel: thick top (matches data bottom), thin bottom
ax_base.spines["top"].set_linewidth(1.5)
ax_base.spines["top"].set_color("black")
ax_base.spines["bottom"].set_linewidth(0.8)

# ---------------------------------------------------------------------------
# Tick labels
# ---------------------------------------------------------------------------
# Data axes: column headers on TOP; no bottom ticks
ax_data.xaxis.set_label_position("top")
ax_data.xaxis.tick_top()
ax_data.set_xticks(range(6))
ax_data.set_xticklabels(COL_HEADERS, fontsize=14, fontweight="bold")
ax_data.tick_params(axis="x", length=0, pad=4)
ax_data.set_yticks([])

# Baseline axes: no x ticks (column headers already shown on top via ax_data)
ax_base.set_xticks([])
ax_base.set_yticks([])

# n= support annotation — one line above the column headers
support_labels = [f"n={s}" for s in CLASS_SUPPORT]
for j, slbl in enumerate(support_labels):
    ax_data.text(j, -0.55, slbl,
                 ha="center", va="bottom",
                 transform=ax_data.get_xaxis_transform(),
                 fontsize=8.5, fontstyle="italic",
                 color="gray", fontweight="normal")

# ---------------------------------------------------------------------------
# Vertical separator: between col 0 (water) and col 1 (DMSO)
# ---------------------------------------------------------------------------
for ax in (ax_data, ax_base):
    ax.axvline(x=0.5, color="black", linewidth=1.2, zorder=10)

# ---------------------------------------------------------------------------
# Two-tier Y-axis labels (placed in axes-fraction coords of ax_data / ax_base)
# ---------------------------------------------------------------------------
# We need pixel-stable positions. Use blended transforms:
#   x in axes fraction, y in data coords  →  ax.transData + offset_copy trick
# Simpler: use ax.transData for y, ax.transAxes for x via blended transform.

import matplotlib.transforms as mtransforms

def _blended(ax):
    """Return a transform: x in axes fraction, y in data coords."""
    return mtransforms.blended_transform_factory(ax.transAxes, ax.transData)

trans_data = _blended(ax_data)

INNER_XF = -0.04   # axes-fraction x for method labels (right-aligned)
OUTER_XF = -0.22   # axes-fraction x for model labels  (right-aligned)

for row_i in range(12):
    model_lbl  = row_model_labels[row_i]
    method_lbl = row_method_labels[row_i]

    # Method label (italic, every row)
    ax_data.text(INNER_XF, row_i, method_lbl,
                 ha="right", va="center",
                 transform=trans_data,
                 fontsize=11, fontstyle="italic",
                 fontfamily="sans-serif", fontweight="normal",
                 clip_on=False)

    # Model label (bold, center of its 3-row block)
    if row_i % 3 == 1:
        ax_data.text(OUTER_XF, row_i, model_lbl,
                     ha="right", va="center",
                     transform=trans_data,
                     fontsize=12, fontweight="bold",
                     fontfamily="sans-serif",
                     clip_on=False)

# Baseline label — use blended transform on ax_base
trans_base = _blended(ax_base)
ax_base.text(INNER_XF, 0, "Always-water\nBaseline",
             ha="right", va="center",
             transform=trans_base,
             fontsize=10, fontstyle="italic",
             fontfamily="sans-serif", fontweight="normal",
             clip_on=False)

# ---------------------------------------------------------------------------
# Colorbar (horizontal, below the baseline panel)
# ---------------------------------------------------------------------------
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar_ax = fig.add_axes([0.30, 0.03, 0.65, 0.025])
cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
cbar.set_ticks([0.0, 0.25, 0.50, 0.75, 1.0])
cbar.set_ticklabels(["0", "0.25", "0.50", "0.75", "1.0"])
cbar.set_label("Per-class recall", fontsize=14, fontweight="bold", labelpad=6)
for lbl in cbar.ax.get_xticklabels():
    lbl.set_fontsize(12)
    lbl.set_fontweight("bold")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT_PDF, bbox_inches="tight")
plt.savefig(OUT_PNG, dpi=180, bbox_inches="tight")
print(f"\nSaved: {OUT_PDF}")
print(f"Saved: {OUT_PNG}")
