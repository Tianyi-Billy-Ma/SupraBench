"""HGD canon-vs-gold scatter figure.

For each of 4 models × 3 methods, compute over the n=35 forward items
(dedup by id, keep last):

  canon_rate  — fraction of responses containing ≥1 of the 33 curated
                guest-like canonical tokens (after normalisation)
  gold_rate   — fraction of responses containing ≥1 gold guest name
                (head ≥4 chars, same normalisation)

Outputs
-------
  paper-overleaf/figures/hgd_canon_vs_gold.{pdf,png}

Style matches per_host_mae_bap.py (bold sans-serif, Okabe-Ito palette).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO = Path("/Users/anonuser/Workspace/SupraBench")
OUT_PDF = REPO / "paper-overleaf/figures/hgd_canon_vs_gold.pdf"
OUT_PNG = REPO / "paper-overleaf/figures/hgd_canon_vs_gold.png"

# ---------------------------------------------------------------------------
# Model / method config
# ---------------------------------------------------------------------------
MODEL_ORDER = [
    ("qwen3.5-27b",                       "Qwen3.5-27B",    "o"),
    ("gpt-5.4-mini_nothinking",           "GPT-5.4-Mini",   "s"),
    ("gemini-3-flash-preview_nothinking", "Gemini-3-Flash", "^"),
    ("deepseek-v4-pro",                   "DeepSeek-v4",    "D"),
]
METHODS = [("base", "Base"), ("fewshot", "Few-Shot"), ("cot", "CoT")]
METHOD_COLOR = {"base": "#56B4E9", "fewshot": "#D55E00", "cot": "#009E73"}

# ---------------------------------------------------------------------------
# 33 guest-like canonical tokens (lowercased)
# ---------------------------------------------------------------------------
CANON_TOKENS = [
    "adamantane", "1-adamantanol", "adamantylamine",
    "ferrocene", "ferrocenium",
    "methyl viologen", "viologen", "paraquat",
    "bipyridinium", "4,4'-bipyridine",
    "naphthalene", "anthracene", "pyrene", "perylene",
    "benzene", "toluene", "p-xylene", "p-cresol",
    "tetramethylammonium", "trimethylammonium", "ammonium",
    "pyridinium", "azobenzene", "stilbene",
    "porphyrin", "fullerene", "c60", "c70",
    "spermine", "spermidine", "putrescine",
    "methylene blue", "rhodamine b",
]

# ---------------------------------------------------------------------------
# Gold-guest extraction (mirrors src/eval/hgd.py RE_FWD)
# ---------------------------------------------------------------------------
RE_FWD = re.compile(
    r"Representative (?:guests?|hosts?):\s*(.+?)\s*\.?\s*$",
    re.IGNORECASE | re.DOTALL,
)


def _split_at_depth_zero(s: str) -> list[str]:
    parts: list[str] = []
    buf: list[str] = []
    depth = 0
    i = 0
    while i < len(s):
        c = s[i]
        if c in "([{":
            depth += 1; buf.append(c)
        elif c in ")]}":
            depth = max(0, depth - 1); buf.append(c)
        elif depth == 0 and c == "," and i + 1 < len(s) and s[i + 1] == " ":
            parts.append("".join(buf).strip())
            buf = []; i += 2; continue
        else:
            buf.append(c)
        i += 1
    if buf:
        parts.append("".join(buf).strip())
    return [p.strip(" .,;") for p in parts if p.strip(" .,;")]


def extract_gold_guests(reference: str) -> list[str]:
    """Return representative guest names from a forward gold answer."""
    m = RE_FWD.search(reference or "")
    if not m:
        return []
    return _split_at_depth_zero(m.group(1))


# ---------------------------------------------------------------------------
# Response normalisation
# ---------------------------------------------------------------------------
RE_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
RE_ANSWER = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)


def normalise(text: str) -> str:
    """Strip think tags, prefer answer tag content, lowercase, unify beta/dash."""
    text = RE_THINK.sub("", text)
    m = RE_ANSWER.search(text)
    if m:
        text = m.group(1)
    text = text.lower()
    text = text.replace("β", "beta")  # β → beta
    text = re.sub(r"[–—]", "-", text)  # en/em dash → hyphen
    return text


# ---------------------------------------------------------------------------
# Per-cell computation
# ---------------------------------------------------------------------------

def compute_cell(model_slug: str, method: str) -> dict:
    path = REPO / f"results/task3/{method}/{model_slug}.jsonl"
    if not path.exists():
        raise FileNotFoundError(path)

    # Dedup by id, keep last occurrence
    seen: dict[str, dict] = {}
    for line in path.open():
        row = json.loads(line)
        if row.get("subtype") != "forward":
            continue
        seen[row["id"]] = row
    rows = list(seen.values())
    assert len(rows) == 35, f"{method}/{model_slug}: expected 35 forward rows, got {len(rows)}"

    canon_hits = 0
    gold_hits = 0
    for row in rows:
        resp_norm = normalise(row.get("response", "") or "")
        gold_guests = extract_gold_guests(row.get("reference", "") or "")

        # Canon hit
        if any(token in resp_norm for token in CANON_TOKENS):
            canon_hits += 1

        # Gold hit: any gold guest head (≥4 chars) present in response
        hit = False
        for guest in gold_guests:
            head = guest.lower().strip()
            if len(head) >= 4 and head in resp_norm:
                hit = True
                break
        if hit:
            gold_hits += 1

    return {
        "canon_rate": canon_hits / len(rows),
        "gold_rate": gold_hits / len(rows),
        "n": len(rows),
    }


# ---------------------------------------------------------------------------
# Compute all 12 cells
# ---------------------------------------------------------------------------
print(f"{'Model':25s} {'Method':10s} {'Canon%':>8s} {'Gold%':>8s} {'Bias(pp)':>10s}")
print("-" * 65)

results: dict[tuple[str, str], dict] = {}
for model_slug, model_label, _ in MODEL_ORDER:
    for method, method_label in METHODS:
        cell = compute_cell(model_slug, method)
        results[(model_slug, method)] = cell
        bias_pp = (cell["canon_rate"] - cell["gold_rate"]) * 100
        print(f"{model_label:25s} {method_label:10s} "
              f"{cell['canon_rate']*100:7.1f}% {cell['gold_rate']*100:7.1f}% "
              f"{bias_pp:+9.1f}pp")

# Sanity-check: mean canon_rate under Base averaged over 4 models
base_canon = [results[(slug, "base")]["canon_rate"] for slug, _, _ in MODEL_ORDER]
base_gold  = [results[(slug, "base")]["gold_rate"]  for slug, _, _ in MODEL_ORDER]
print(f"\nSanity check (Base, mean over 4 models):")
print(f"  mean canon_rate = {sum(base_canon)/len(base_canon)*100:.1f}%")
print(f"  mean gold_rate  = {sum(base_gold)/len(base_gold)*100:.1f}%")

# Overall mean bias
all_biases = [
    (results[(slug, m)]["canon_rate"] - results[(slug, m)]["gold_rate"]) * 100
    for slug, _, _ in MODEL_ORDER
    for m, _ in METHODS
]
mean_bias = sum(all_biases) / len(all_biases)
print(f"  mean canon-gold bias (all 12 cells) = {mean_bias:+.1f}pp")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 7))

# Shade canon-saturated regime (x >= 0.9)
ax.axvspan(0.9, 1.0, color="lightgrey", alpha=0.35, zorder=0)
ax.text(0.905, 0.195, "canon-saturated\nregime",
        fontsize=8, fontweight="bold", color="grey",
        va="top", ha="left", transform=ax.get_xaxis_transform())

# Diagonal y = x reference line (clipped to visible range)
diag_x = [0.0, 0.20]
ax.plot(diag_x, diag_x, color="lightgrey", linestyle="--", linewidth=1.2,
        label="y=x equally textbook and literature", zorder=1)

# Scatter the 12 markers
for model_slug, model_label, marker in MODEL_ORDER:
    for method, method_label in METHODS:
        cell = results[(model_slug, method)]
        x = cell["canon_rate"]
        y = cell["gold_rate"]
        ax.scatter(x, y,
                   marker=marker,
                   s=150,
                   color=METHOD_COLOR[method],
                   edgecolors="black",
                   linewidths=0.8,
                   zorder=5)

# Annotate Few-Shot points with model name (to the right)
for model_slug, model_label, marker in MODEL_ORDER:
    cell = results[(model_slug, "fewshot")]
    x = cell["canon_rate"]
    y = cell["gold_rate"]
    ax.annotate(model_label,
                xy=(x, y),
                xytext=(6, 0),
                textcoords="offset points",
                fontsize=8,
                fontweight="bold",
                va="center")

# Infobox top-left
ax.text(0.03, 0.97,
        f"mean canon−gold bias = +88.9 pp\nacross 12 cells",
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor="black", linewidth=1.3))

# Axes formatting
ax.set_xlim(-0.02, 1.05)
ax.set_ylim(-0.005, 0.20)
ax.set_xlabel("Canon-hit rate (textbook vocabulary)", fontsize=20, fontweight="bold")
ax.set_ylabel("Gold-hit rate (literature reference)", fontsize=20, fontweight="bold")
ax.set_title("HGD: Canon vs. Gold Hit Rate", fontsize=22, fontweight="bold", pad=10)

for lbl in ax.get_xticklabels() + ax.get_yticklabels():
    lbl.set_fontsize(17)
    lbl.set_fontweight("bold")

ax.grid(linestyle="--", linewidth=0.4, alpha=0.5)
ax.set_axisbelow(True)

# Two-column legend in lower-right
# Left column: method color swatches (3)
method_handles = [
    mpatches.Patch(color=METHOD_COLOR[m], label=lbl, linewidth=0)
    for m, lbl in METHODS
]
# Right column: model shapes (4, black fill)
model_handles = [
    mlines.Line2D([], [], marker=marker, color="none",
                  markerfacecolor="grey", markeredgecolor="black",
                  markeredgewidth=0.8, markersize=9,
                  linestyle="None", label=model_label)
    for _, model_label, marker in MODEL_ORDER
]

# Combine: interleave so ncol=2 gives method | model layout
# matplotlib legend with ncol=2 fills row-by-row; we need 3 method rows
# then 4 model rows — use a single combined list with a spacer patch
spacer = mpatches.Patch(color="none", label=" ")
combined = method_handles + [spacer] + model_handles

leg = ax.legend(handles=combined,
                loc="lower right",
                ncol=2,
                frameon=True,
                framealpha=0.9,
                fontsize=13,
                edgecolor="black",
                fancybox=False)
for txt in leg.get_texts():
    txt.set_fontweight("bold")

plt.tight_layout()
OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT_PDF, bbox_inches="tight")
plt.savefig(OUT_PNG, dpi=180, bbox_inches="tight")
print(f"\nSaved: {OUT_PDF}")
print(f"Saved: {OUT_PNG}")
