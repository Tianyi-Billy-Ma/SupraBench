"""Walk ``results/metrics/`` and build a cross-(model, task) comparison table.

Each subdirectory under ``results/metrics/`` is named
``task<N>_base_<model-tag>/`` and contains a ``metrics.json``. This script
parses every such file, groups by task, and emits a markdown table per task
with one row per model.

Usage::

    uv run --extra hf python scripts/aggregate_results.py [--results-dir results/metrics] [--out results/comparison.md]

The headline metric per task is the same one used in the paper:

    BAP: MAE        (lower is better)
    TBS: accuracy   (higher is better)
    HGD: rougeL_f   (higher is better)
    SID: macro_f1   (higher is better)

For BAP the script also reports the robust statistics (medae,
mae_5pct_trimmed, mae_clipped) when present, since v2-and-later eval runs
emit them.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

# task -> (json key, label, direction-arrow, display-format)
HEADLINE_METRICS = {
    1: ("mae",       "MAE",       "↓", "{:.3f}"),
    2: ("accuracy",  "ACC",       "↑", "{:.3f}"),
    3: ("rougeL_f",  "Rouge-L",   "↑", "{:.3f}"),
    7: ("macro_f1",  "macro-F1",  "↑", "{:.3f}"),
}

# Stable column ordering: paper baselines first, then our v1/v2 variants.
MODEL_ORDER = [
    "qwen3.5-27b-base",
    "qwen3.5-27b-fewshot",
    "qwen3.5-27b-cot",
    "qwen3.5-27b-eupmc-lora",
    "qwen3.5-27b-supra-v2-lora",
    "qwen3.5-27b-supra-v2-lora-guided",
    "qwen3.5-9b-supra-v2-lora",
    "llama3.1-8b-supra-v2-lora",
]
MODEL_DISPLAY = {
    "qwen3.5-27b-base":                  "Qwen3.5-27B (Base)",
    "qwen3.5-27b-fewshot":               "Qwen3.5-27B (Few-Shot)",
    "qwen3.5-27b-cot":                   "Qwen3.5-27B (CoT)",
    "qwen3.5-27b-eupmc-lora":            "Qwen3.5-27B + v1 LoRA",
    "qwen3.5-27b-supra-v2-lora":         "Qwen3.5-27B + v2 LoRA",
    "qwen3.5-27b-supra-v2-lora-guided":  "Qwen3.5-27B + v2 LoRA (guided)",
    "qwen3.5-9b-supra-v2-lora":          "Qwen3.5-9B + v2 LoRA",
    "llama3.1-8b-supra-v2-lora":         "Llama-3.1-8B + v2 LoRA",
}

DIR_RE = re.compile(r"task(\d+)_base_(.+)")


def collect(results_dir: Path) -> dict[int, dict[str, dict]]:
    """Returns ``{task_id: {model_tag: metrics_dict}}``."""
    out: dict[int, dict[str, dict]] = defaultdict(dict)
    for d in sorted(results_dir.iterdir()):
        if not d.is_dir():
            continue
        m = DIR_RE.match(d.name)
        if not m:
            continue
        task = int(m.group(1))
        model_tag = m.group(2)
        mj = d / "metrics.json"
        if not mj.is_file():
            continue
        try:
            out[task][model_tag] = json.loads(mj.read_text())
        except Exception as e:
            print(f"warn: failed to parse {mj}: {e}")
    return out


def _model_order_key(tag: str) -> tuple[int, str]:
    if tag in MODEL_ORDER:
        return (MODEL_ORDER.index(tag), tag)
    return (len(MODEL_ORDER), tag)


def render_headline_table(metrics: dict[int, dict[str, dict]]) -> str:
    """A single wide table: rows = models, columns = (task, headline metric)."""
    # Union of model tags across all tasks
    all_tags = sorted({tag for t in metrics.values() for tag in t}, key=_model_order_key)
    headers = ["Model"]
    for t in sorted(metrics.keys()):
        key, label, arrow, _ = HEADLINE_METRICS[t]
        headers.append(f"T{t} {label} {arrow}")
    rows = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for tag in all_tags:
        display = MODEL_DISPLAY.get(tag, tag)
        cells = [display]
        for t in sorted(metrics.keys()):
            key, _, _, fmt = HEADLINE_METRICS[t]
            d = metrics[t].get(tag)
            if d and key in d and isinstance(d[key], (int, float)):
                cells.append(fmt.format(d[key]))
            else:
                cells.append("—")
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def render_bap_robust(metrics: dict[int, dict[str, dict]]) -> str:
    """Per-model robust-statistics table for BAP."""
    t1 = metrics.get(1, {})
    if not t1:
        return "_(no BAP metrics)_\n"
    cols = ["MAE", "MedAE", "MAE 5%-trim", "MAE clipped[-2,15]", "n_parsed/n_total"]
    headers = ["Model"] + cols
    rows = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for tag in sorted(t1, key=_model_order_key):
        d = t1[tag]
        display = MODEL_DISPLAY.get(tag, tag)
        def fmt(k, ff="{:.3f}"):
            v = d.get(k)
            return ff.format(v) if isinstance(v, (int, float)) else "—"
        n_parsed = d.get("n_parsed", "—")
        n_total  = d.get("n_total",  "—")
        rows.append("| " + " | ".join([
            display,
            fmt("mae"),
            fmt("medae"),
            fmt("mae_5pct_trimmed"),
            fmt("mae_clipped"),
            f"{n_parsed}/{n_total}",
        ]) + " |")
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("results/metrics"))
    parser.add_argument("--out", type=Path, default=Path("results/comparison.md"))
    args = parser.parse_args()

    metrics = collect(args.results_dir)
    tasks = sorted(metrics.keys())
    n_pairs = sum(len(t) for t in metrics.values())
    print(f"scanned {args.results_dir}: {n_pairs} (task, model) pairs across {len(tasks)} task(s)")

    body = []
    body.append("# Cross-model headline comparison\n")
    body.append("Auto-generated by `scripts/aggregate_results.py` from `results/metrics/*/metrics.json`.\n")
    body.append("Arrows mark optimization direction.  ↑ = higher is better, ↓ = lower is better.\n")
    body.append("## Headline metric per task\n")
    body.append(render_headline_table(metrics) + "\n")
    body.append("## BAP robust statistics\n")
    body.append("MedAE, 5%-trimmed MAE, and chemistry-plausible-range MAE expose ")
    body.append("how much of the headline gap is tail-driven.\n")
    body.append(render_bap_robust(metrics) + "\n")
    body.append("## Missing pairs\n")
    expected = sorted(MODEL_DISPLAY)
    missing_lines = []
    for t in sorted(HEADLINE_METRICS):
        for tag in expected:
            if tag not in metrics.get(t, {}):
                missing_lines.append(f"- Task {t} × {MODEL_DISPLAY[tag]}")
    body.append("\n".join(missing_lines) if missing_lines else "_(none — full grid)_")
    body.append("")

    text = "\n".join(body)
    args.out.write_text(text)
    print(f"wrote {args.out}")
    print()
    print(text)


if __name__ == "__main__":
    main()
