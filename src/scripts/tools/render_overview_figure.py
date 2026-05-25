"""Render the SupraBench overview figure (Figure 1).

Conference-style schematic. Left side shows an abstract host-guest pair; four
arrows fan out into a 2x2 grid of labeled task panels, one per SupraBench task.
Pure-vector output (PDF + PNG) so downstream typography stays sharp.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import (
    Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Rectangle
)

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# canvas — leave a touch of right-margin so the right column doesn't get clipped
fig = plt.figure(figsize=(9.6, 4.6))
ax  = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 18.5)
ax.set_ylim(0, 9)
ax.set_aspect("equal")
ax.axis("off")

# colors / stroke
NEAR_BLACK = "#111111"
GRAY       = "#666666"
LIGHT      = "#f2f2f2"

# ----- header -----
ax.text(9.25, 8.45, "SupraBench",
        ha="center", va="center", fontsize=14, weight="bold",
        family="serif", color=NEAR_BLACK)

# ----- left side: host-guest pair (abstract) -----
HX, HY, HR = 2.6, 4.5, 1.45  # host ellipse center + radii (use width/2 -> diameter 2HR)
host = Ellipse((HX, HY), 2 * HR, 2 * HR * 0.85,
               edgecolor=NEAR_BLACK, facecolor="white", linewidth=1.6)
ax.add_patch(host)

# small four-tick indicators around the guest (binding interactions)
guest = Circle((HX, HY), 0.35, edgecolor=NEAR_BLACK, facecolor=NEAR_BLACK)
ax.add_patch(guest)
for dx, dy in [(0.85, 0), (-0.85, 0), (0, 0.65), (0, -0.65)]:
    ax.annotate("", xy=(HX + dx, HY + dy),
                xytext=(HX + dx*0.55, HY + dy*0.55),
                arrowprops=dict(arrowstyle="-", lw=0.8, color=GRAY))

# caption beneath host-guest
ax.text(HX, HY - 1.75, "host-guest pair",
        ha="center", va="top", fontsize=9.5, style="italic",
        family="serif", color=NEAR_BLACK)
ax.text(HX, HY - 2.25, r"(SMILES + name)",
        ha="center", va="top", fontsize=8, family="serif", color=GRAY)

# ----- panel grid coordinates -----
PANEL_W, PANEL_H = 4.8, 2.4
COL_LEFT_X = 7.8
COL_RIGHT_X = COL_LEFT_X + PANEL_W + 0.7
ROW_TOP_Y = 5.45
ROW_BOT_Y = 1.95

panels = [
    # x, y, title, output-icon-callback (drawn inside box)
    (COL_LEFT_X,  ROW_TOP_Y, "Binding-Affinity Regression",  "regress"),
    (COL_RIGHT_X, ROW_TOP_Y, "Top-Binder Selection",         "mcq4"),
    (COL_LEFT_X,  ROW_BOT_Y, "Solvent Classification",       "mcq6"),
    (COL_RIGHT_X, ROW_BOT_Y, "Open-Ended Host-Guest QA",     "qa"),
]

def draw_regress(x0, y0):
    # box w/ "log Ka = 5.3" text
    ax.text(x0 + PANEL_W / 2, y0 + PANEL_H * 0.42, r"$\log K_a = 5.3$",
            ha="center", va="center", fontsize=13, family="serif", color=NEAR_BLACK)
    ax.text(x0 + PANEL_W / 2, y0 + PANEL_H * 0.18, "(real value, MAE/RMSE)",
            ha="center", va="center", fontsize=8, family="serif", color=GRAY)

def draw_mcq4(x0, y0):
    # four small lettered boxes; B filled
    labels = ["A", "B", "C", "D"]
    n = 4
    bw, bh = 0.55, 0.55
    spacing = 0.18
    total = n * bw + (n - 1) * spacing
    start = x0 + (PANEL_W - total) / 2
    for i, lab in enumerate(labels):
        bx = start + i * (bw + spacing)
        by = y0 + PANEL_H * 0.40
        face = NEAR_BLACK if lab == "B" else "white"
        edge = NEAR_BLACK
        ax.add_patch(Rectangle((bx, by), bw, bh, facecolor=face, edgecolor=edge, linewidth=1.0))
        ax.text(bx + bw / 2, by + bh / 2, lab,
                ha="center", va="center", fontsize=9,
                color=("white" if lab == "B" else NEAR_BLACK), weight="bold")
    ax.text(x0 + PANEL_W / 2, y0 + PANEL_H * 0.18, "(4-way MCQ, accuracy)",
            ha="center", va="center", fontsize=8, family="serif", color=GRAY)

def draw_mcq6(x0, y0):
    labels = ["water", "DMSO", "MeCN", "MeOH", r"CHCl$_3$", r"CH$_2$Cl$_2$"]
    n = 6
    bw, bh = 0.55, 0.55
    spacing = 0.10
    total = n * bw + (n - 1) * spacing
    start = x0 + (PANEL_W - total) / 2
    for i, lab in enumerate(labels):
        bx = start + i * (bw + spacing)
        by = y0 + PANEL_H * 0.50
        # tiny "flask" = trapezoid-ish. use rectangle for simplicity.
        face = NEAR_BLACK if i == 0 else "white"
        ax.add_patch(Rectangle((bx, by), bw, bh, facecolor=face, edgecolor=NEAR_BLACK, linewidth=1.0))
        ax.text(bx + bw / 2, by - 0.22, lab,
                ha="center", va="top", fontsize=6.5, family="serif", color=NEAR_BLACK)
    ax.text(x0 + PANEL_W / 2, y0 + PANEL_H * 0.18, "(6-way MCQ, macro-F1)",
            ha="center", va="center", fontsize=8, family="serif", color=GRAY)

def draw_qa(x0, y0):
    # speech-bubble icon
    bx = x0 + PANEL_W * 0.32
    by = y0 + PANEL_H * 0.35
    bw, bh = 2.4, 0.95
    ax.add_patch(FancyBboxPatch((bx, by), bw, bh,
                                boxstyle="round,pad=0.05,rounding_size=0.12",
                                facecolor="white", edgecolor=NEAR_BLACK, linewidth=1.0))
    # three short horizontal lines inside
    for i, off in enumerate((0.65, 0.45, 0.25)):
        line_w = (0.85, 0.70, 0.55)[i]
        ax.plot([bx + 0.18, bx + 0.18 + bw * line_w],
                [by + bh * off, by + bh * off],
                color=NEAR_BLACK, lw=1.0)
    # tail of speech bubble
    ax.plot([bx + 0.25, bx + 0.10],
            [by, by - 0.18],
            color=NEAR_BLACK, lw=1.0)
    ax.text(x0 + PANEL_W / 2, y0 + PANEL_H * 0.13, "(free text, ROUGE-L + Keyword Hit)",
            ha="center", va="center", fontsize=8, family="serif", color=GRAY)

draw_map = {"regress": draw_regress, "mcq4": draw_mcq4, "mcq6": draw_mcq6, "qa": draw_qa}

for x0, y0, title, kind in panels:
    ax.add_patch(FancyBboxPatch(
        (x0, y0), PANEL_W, PANEL_H,
        boxstyle="round,pad=0.04,rounding_size=0.18",
        facecolor="white", edgecolor=NEAR_BLACK, linewidth=1.2,
    ))
    ax.text(x0 + PANEL_W / 2, y0 + PANEL_H - 0.30, title,
            ha="center", va="top", fontsize=10.5, weight="bold",
            family="serif", color=NEAR_BLACK)
    draw_map[kind](x0, y0)

# ----- arrows from host-guest -> panel left edges -----
for x0, y0, _, _ in panels:
    target_x = x0
    target_y = y0 + PANEL_H / 2
    arrow = FancyArrowPatch((HX + HR * 0.92, HY), (target_x - 0.05, target_y),
                            arrowstyle="-|>", mutation_scale=10,
                            linewidth=1.2, color=NEAR_BLACK,
                            connectionstyle="arc3,rad=0.05")
    ax.add_patch(arrow)

# ----- save -----
out_pdf = OUT / "suprabench_overview.pdf"
out_png = OUT / "suprabench_overview.png"
fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.05)
fig.savefig(out_png, dpi=240, bbox_inches="tight", pad_inches=0.05)
print(f"wrote {out_pdf}")
print(f"wrote {out_png}")
